"""Compare fan-curve runs side by side (read-only, no hardware).

Does a duct fan curve stop the fan hunting? This tool prints the numbers that
answer it for two or more runs:

- the peak junction, memory, and edge temperature;
- the decode, prefill, lane sweep, and soak token rates;
- the minimum, maximum, and mean duct PWM and tach rpm during the soak;
- the hunting metric: direction changes of the mean duct PWM per minute of
  soak;
- the standard deviation of the mean duct PWM in the soak;
- the slope of the hottest sensor over the last 60 s of the soak;
- the settle time: seconds from the start of the last soak until the 60 s
  slope of the hottest sensor stays below 0.2 C/min to the end of the soak.

The mean duct PWM is the mean of the selected channels in one telemetry line.
A direction change is a turn from a rise to a fall, or from a fall to a rise.
A repeated value is not a turn.

Run it as `python -m inferference.experiment.analysis.curve_compare [RUN ...]`.
Each RUN is a run directory or an artifact root with one run directory.
Pass `--fan-channel` to select channels. The default is the comma list in
`INFERFERENCE_DUCT_FAN_CHANNELS` (declare it in `ci/runner/rig-local.sh`). With
neither, the tool refuses. The case exhaust and unused ports do not count as
the duct.

Add `--json` for the full records.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from inferference.telemetry import load_jsonl

from .load import LoadError, load_run
from .run_summary import (
    DUCT_CHANNELS_ENV,
    RunSummaryError,
    _is_soak,
    _stages,
    resolve_duct_channels,
    resolve_run_dir,
    summarize_run,
)
from .run_summary import _hottest_slope as _slope_over

SLOPE_WINDOW_S = 60.0
SETTLE_SLOPE_C_PER_MIN = 0.2


@dataclass(frozen=True, slots=True)
class Stats:
    """Minimum, maximum, and mean of one reading."""

    minimum: float
    maximum: float
    mean: float


@dataclass(frozen=True, slots=True)
class CurveMetrics:
    """The numbers this tool reports for one run."""

    run_dir: str
    experiment: str | None
    cap_w: int | None
    state: str
    failure_boundary: str | None
    peak_junction_c: float | None
    peak_mem_c: float | None
    peak_edge_c: float | None
    decode_tok_s: float | None
    prefill_tok_s: float | None
    lane_sweep_tok_s: float | None
    soak_tok_s: float | None
    soak_s: float
    duct_pwm: Stats | None
    duct_rpm: Stats | None
    pwm_stdev: float | None
    direction_changes: int | None
    direction_changes_per_min: float | None
    hottest_slope_c_per_min: float | None
    hottest_sensor: str | None
    settle_time_s: float | None = None


def direction_changes(values: Sequence[float]) -> int:
    """Count the turns of a series. A repeated value is not a turn."""
    turns = 0
    last_sign = 0
    previous: float | None = None
    for value in values:
        if previous is not None and value != previous:
            sign = 1 if value > previous else -1
            if last_sign and sign != last_sign:
                turns += 1
            last_sign = sign
        previous = value
    return turns


def _ols_slope(points: Sequence[tuple[float, float]]) -> float | None:
    """Least-squares slope in value units per second, or None for fewer than 3 points."""
    if len(points) < 3:
        return None
    mean_t = sum(t for t, _v in points) / len(points)
    mean_v = sum(v for _t, v in points) / len(points)
    denominator = sum((t - mean_t) ** 2 for t, _v in points)
    if denominator == 0:
        return None
    return sum((t - mean_t) * (v - mean_v) for t, v in points) / denominator


def settle_time_s(
    points: Sequence[tuple[float, float]],
    window_s: float = SLOPE_WINDOW_S,
    threshold_c_per_min: float = SETTLE_SLOPE_C_PER_MIN,
) -> float | None:
    """Seconds from the first point until the trailing slope stays below the threshold.

    `points` are `(time_s, temperature_c)` pairs in time order. The trailing
    slope at a point is the least-squares slope over the `window_s` before it.
    The first point with a full window counts. The result is the first time
    from which every later slope has an absolute value below the threshold.
    Return None when the slope never settles, or when the series is shorter
    than the window.
    """
    if not points:
        return None
    origin = points[0][0]
    slopes: list[tuple[float, float]] = []
    for index, (time_s, _value) in enumerate(points):
        if time_s - origin < window_s:
            continue
        window = [point for point in points[: index + 1] if point[0] >= time_s - window_s]
        slope = _ols_slope(window)
        if slope is not None:
            slopes.append((time_s, abs(slope) * 60.0))
    if not slopes:
        return None
    settled_at: float | None = None
    for time_s, slope_c_min in reversed(slopes):
        if slope_c_min >= threshold_c_per_min:
            break
        settled_at = time_s
    return None if settled_at is None else settled_at - origin


def _hottest_points(record: Any, soak: Mapping[str, Any]) -> list[tuple[float, float]]:
    """The series of the sensor that peaks highest in one soak stage, as (time, C) pairs."""
    start = float(soak["started_mono"])
    end = float(soak["ended_mono"])
    series: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for observation in record.observations:
        if not start <= observation.mono <= end:
            continue
        for card, reading in observation.cards.items():
            series.setdefault((card, "junction"), []).append((observation.mono, reading.junction_c))
            series.setdefault((card, "mem"), []).append((observation.mono, reading.mem_c))
    if not series:
        return []
    return max(series.values(), key=lambda points: max(value for _t, value in points))


def _stats(values: Sequence[float]) -> Stats | None:
    if not values:
        return None
    return Stats(min(values), max(values), sum(values) / len(values))


def soak_duct_series(
    lines: Sequence[Mapping[str, Any]], windows: Sequence[tuple[float, float]], channels: Sequence[int]
) -> tuple[list[float], list[float]]:
    """The mean duct PWM of each telemetry line in a soak window, and every duct rpm reading."""
    pwm_series: list[float] = []
    rpm_values: list[float] = []
    for line in lines:
        mono = line.get("monotonic")
        if not isinstance(mono, (int, float)) or not any(start <= mono <= end for start, end in windows):
            continue
        ducts = [d for d in line.get("fan_ducts", ()) if int(d["channel"]) in channels]
        if not ducts:
            continue
        pwm_series.append(sum(float(d["pwm"]) for d in ducts) / len(ducts))
        rpm_values.extend(float(d["tach_rpm"]) for d in ducts)
    return pwm_series, rpm_values


def curve_metrics(path: Path, *, fan_channels: Sequence[int] | None = None) -> CurveMetrics:
    """Measure the run at `path`. Raises `RunSummaryError` when it cannot.

    The duct series covers `fan_channels`. The default is `resolve_duct_channels()`.
    """
    fan_channels = resolve_duct_channels(fan_channels)
    summary = summarize_run(path)
    run_dir = resolve_run_dir(path)
    try:
        record = load_run(run_dir)
    except LoadError as exc:
        raise RunSummaryError(str(exc)) from exc
    soaks = [stage for stage in _stages(record) if _is_soak(stage)]
    windows = [(float(stage["started_mono"]), float(stage["ended_mono"])) for stage in soaks]
    soak_s = sum(end - start for start, end in windows)
    pwm_series, rpm_values = soak_duct_series(load_jsonl(run_dir / "telemetry.jsonl"), windows, fan_channels)
    turns = direction_changes(pwm_series) if pwm_series else None
    last_soak = max(soaks, key=lambda stage: float(stage["ended_mono"])) if soaks else None
    slope = _slope_over(record, last_soak, SLOPE_WINDOW_S) if last_soak is not None else None
    slope_c_min = slope.slope_c_per_s * 60.0 if slope is not None and slope.slope_c_per_s is not None else None
    settle = settle_time_s(_hottest_points(record, last_soak)) if last_soak is not None else None
    peaks = summary.peaks
    return CurveMetrics(
        run_dir=str(run_dir),
        experiment=summary.experiment,
        cap_w=summary.cap_w,
        state=summary.state,
        failure_boundary=summary.failure_boundary,
        peak_junction_c=max((p.junction_c for p in peaks), default=None),
        peak_mem_c=max((p.mem_c for p in peaks), default=None),
        peak_edge_c=max((p.edge_c for p in peaks), default=None),
        decode_tok_s=summary.decode_tok_s,
        prefill_tok_s=summary.prefill_tok_s,
        lane_sweep_tok_s=summary.lane_sweep_tok_s,
        soak_tok_s=summary.soak_tok_s,
        soak_s=soak_s,
        duct_pwm=_stats(pwm_series),
        duct_rpm=_stats(rpm_values),
        pwm_stdev=statistics.pstdev(pwm_series) if len(pwm_series) > 1 else None,
        direction_changes=turns,
        direction_changes_per_min=turns / (soak_s / 60.0) if turns is not None and soak_s > 0 else None,
        hottest_slope_c_per_min=slope_c_min,
        hottest_sensor=f"card{slope.card} {slope.sensor}" if slope is not None else None,
        settle_time_s=settle,
    )


def _num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _triple(value: Stats | None, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{value.minimum:.{digits}f}/{value.maximum:.{digits}f}/{value.mean:.{digits}f}"


def _rows(metrics: CurveMetrics) -> list[str]:
    outcome = metrics.state if not metrics.failure_boundary else f"{metrics.state} {metrics.failure_boundary}"
    return [
        outcome,
        str(metrics.cap_w) if metrics.cap_w is not None else "n/a",
        f"{_num(metrics.peak_junction_c)}/{_num(metrics.peak_mem_c)}/{_num(metrics.peak_edge_c)}",
        _num(metrics.decode_tok_s),
        _num(metrics.prefill_tok_s),
        _num(metrics.lane_sweep_tok_s),
        _num(metrics.soak_tok_s),
        _num(metrics.soak_s, 0),
        _triple(metrics.duct_pwm),
        _triple(metrics.duct_rpm),
        _num(metrics.direction_changes_per_min, 2),
        _num(metrics.pwm_stdev, 1),
        f"{metrics.hottest_slope_c_per_min:+.2f}" if metrics.hottest_slope_c_per_min is not None else "n/a",
        _num(metrics.settle_time_s, 0),
    ]


ROW_LABELS = (
    "outcome",
    "cap (W)",
    "peak junction/mem/edge (C)",
    "decode (tok/s)",
    "prefill (tok/s)",
    "lane_sweep (tok/s)",
    "soak (tok/s)",
    "soak length (s)",
    "duct PWM min/max/mean",
    "duct rpm min/max/mean",
    "PWM turns per min (hunting)",
    "PWM stdev",
    "hottest slope, last 60 s (C/min)",
    "settle time (s), slope < 0.2 C/min",
)


def render_table(columns: Sequence[tuple[str, CurveMetrics | str]]) -> str:
    """Render labeled runs side by side. A column that failed shows its error text."""
    cells: list[list[str]] = []
    for _label, item in columns:
        if isinstance(item, str):
            cells.append([f"error: {item}"] + [""] * (len(ROW_LABELS) - 1))
        else:
            cells.append(_rows(item))
    label_width = max(len(label) for label in ROW_LABELS)
    widths = [
        max(len(label), *(len(cell) for cell in column)) for (label, _), column in zip(columns, cells, strict=True)
    ]
    header = " | ".join(
        ["metric".ljust(label_width)] + [label.ljust(width) for (label, _), width in zip(columns, widths, strict=True)]
    )
    lines = [header, "-" * len(header)]
    for index, row_label in enumerate(ROW_LABELS):
        lines.append(
            " | ".join(
                [row_label.ljust(label_width)]
                + [column[index].ljust(width) for column, width in zip(cells, widths, strict=True)]
            ).rstrip()
        )
    return "\n".join(lines) + "\n"


def _measure(label: str, path: Path, fan_channels: Sequence[int] | None) -> tuple[str, CurveMetrics | str]:
    try:
        return label, curve_metrics(path, fan_channels=fan_channels)
    except (RunSummaryError, OSError, KeyError, ValueError) as exc:
        return label, str(exc)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curve_compare",
        description="Compare fan-curve runs side by side (read-only).",
    )
    parser.add_argument("runs", nargs="+", type=Path, help="run directories or artifact roots")
    parser.add_argument(
        "--fan-channel",
        type=int,
        action="append",
        help=f"duct fan channel; repeat to add more (default: ${DUCT_CHANNELS_ENV})",
    )
    parser.add_argument("--json", action="store_true", help="print the full records as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 when a table printed, 1 when a run cannot be measured, 2 when there is nothing to compare."""
    args = _build_parser().parse_args(argv)
    groups = [("runs", [(path.name[:34], path) for path in args.runs])]
    failed = False
    printed = 0
    records: dict[str, Any] = {}
    for title, members in groups:
        if not members:
            continue
        columns = [_measure(label, path, args.fan_channel) for label, path in members]
        failed = failed or any(isinstance(item, str) for _label, item in columns)
        printed += 1
        if args.json:
            records[title] = {label: (item if isinstance(item, str) else asdict(item)) for label, item in columns}
        else:
            print(f"== {title} ==")
            for label, path in members:
                print(f"  {label}: {path}")
            print(render_table(columns))
    if printed == 0:
        print("curve_compare: no runs to compare", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
