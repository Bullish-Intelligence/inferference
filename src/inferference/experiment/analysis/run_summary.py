"""Summarize one experiment run from its saved artifacts.

This module reads one run directory and reports:

- the outcome (`state`, `failure_boundary`);
- the peak junction, memory, and edge temperature and the peak power of each card;
- the prefill, decode, lane sweep, and soak token rates;
- the minimum and maximum PWM and tach reading of each ARCTIC fan channel while
  a soak stage ran;
- the slope of the hottest sensor over the last 120 s of the last soak stage
  (the plateau check).

It is read-only and needs no hardware. It accepts any `soak_<N>s` stage.

Run it as `python -m inferference.experiment.analysis.run_summary RUN_DIR`.
Add `--oneline` for one `key=value` line, or `--json` for the full record.
`RUN_DIR` may also be an artifact root that holds exactly one run directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inferference.telemetry import load_jsonl

from .estimators import EstimatorError, ols_slope, upper_confidence_bound
from .load import LoadError, RunRecord, load_run
from .stage_metrics import StageMetricsError, _stage_requests_throughput, _timing_rate

PLATEAU_WINDOW_S = 120.0
# Same tolerance as the plateau test in `plateau.py`.
PLATEAU_SLOPE_TOLERANCE_C_PER_S = 0.02
_SENSORS = ("junction", "mem")


class RunSummaryError(ValueError):
    """A run directory cannot be summarized."""


@dataclass(frozen=True, slots=True)
class CardPeaks:
    """The peak readings of one card over the whole run."""

    card: int
    junction_c: float
    mem_c: float
    edge_c: float
    power_w: float


@dataclass(frozen=True, slots=True)
class Range:
    """The minimum and maximum of one reading."""

    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class HottestSlope:
    """The slope of the hottest sensor over the last window of the last soak stage."""

    card: int
    sensor: str
    stage: str
    window_s: float
    samples: int
    mean_c: float
    slope_c_per_s: float | None
    upper_bound_c_per_s: float | None
    settled: bool | None
    note: str | None


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Everything this module reports for one run."""

    run_dir: str
    experiment: str | None
    cap_w: int | None
    lanes: str | None
    state: str
    failure_boundary: str | None
    detail: str | None
    peaks: tuple[CardPeaks, ...]
    prefill_tok_s: float | None
    decode_tok_s: float | None
    lane_sweep_tok_s: float | None
    soak_tok_s: float | None
    soak_stages: tuple[str, ...]
    fan_pwm: Mapping[int, Range] = field(default_factory=dict)
    fan_rpm: Mapping[int, Range] = field(default_factory=dict)
    watchdog_target_pwm: Range | None = None
    slope: HottestSlope | None = None
    notes: tuple[str, ...] = ()


def resolve_run_dir(path: Path) -> Path:
    """Return the run directory at `path`, or the one inside an artifact root.

    A run directory holds `outcome.json`. An artifact root holds run
    directories. A root with no run directory, or with more than one, is a
    refusal: the caller must name the run.
    """
    if (path / "outcome.json").is_file() or (path / "run.json").is_file():
        return path
    if not path.is_dir():
        raise RunSummaryError(f"{path} is not a directory")
    children = sorted(item for item in path.iterdir() if item.is_dir() and (item / "run.json").is_file())
    if len(children) != 1:
        raise RunSummaryError(f"{path} holds {len(children)} run directories, expected exactly 1")
    return children[0]


def _stages(record: RunRecord) -> list[Mapping[str, Any]]:
    if record.stages is None:
        return []
    return [item for item in record.stages.get("stages", ()) if isinstance(item, Mapping)]


def _is_soak(stage: Mapping[str, Any]) -> bool:
    return str(stage.get("workload", stage.get("name", ""))).startswith("soak_") or str(
        stage.get("name", "")
    ).startswith("soak_")


def _stage_by_workload(stages: Sequence[Mapping[str, Any]], workload: str) -> Mapping[str, Any] | None:
    for stage in stages:
        if stage.get("name") == workload or stage.get("workload") == workload:
            return stage
    return None


def _stage_completed(stage: Mapping[str, Any] | None) -> bool:
    return stage is not None and stage.get("completed") is True and stage.get("stop_reason") is None


def _rate(stage: Mapping[str, Any] | None, run_dir: Path, how: str) -> float | None:
    """Return a stage rate, or None when the stage did not finish."""
    if not _stage_completed(stage) or stage is None:
        return None
    try:
        if how == "prefill":
            return _timing_rate(
                stage, workload="prefill", count_key="prompt_n", duration_key="prompt_ms", run_dir=run_dir
            )
        if how == "decode":
            return _timing_rate(
                stage, workload="decode", count_key="predicted_n", duration_key="predicted_ms", run_dir=run_dir
            )
        return _stage_requests_throughput(stage, run_dir)
    except StageMetricsError:
        return None


def _soak_rate(stage: Mapping[str, Any]) -> float | None:
    """Completion tokens of valid requests over the stage wall time."""
    start = stage.get("started_mono")
    end = stage.get("ended_mono")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
        return None
    tokens = 0
    for request in stage.get("requests", ()):
        if not isinstance(request, Mapping) or request.get("valid") is not True:
            continue
        value = request.get("completion_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            tokens += value
    return tokens / (end - start) if tokens else None


def _peaks(record: RunRecord) -> tuple[CardPeaks, ...]:
    cards: dict[int, list[Any]] = {}
    for observation in record.observations:
        for card, reading in observation.cards.items():
            cards.setdefault(card, []).append(reading)
    return tuple(
        CardPeaks(
            card=card,
            junction_c=max(item.junction_c for item in readings),
            mem_c=max(item.mem_c for item in readings),
            edge_c=max(item.edge_c for item in readings),
            power_w=max(item.power_w for item in readings),
        )
        for card, readings in sorted(cards.items())
    )


def _fan_ranges(
    lines: Sequence[Mapping[str, Any]], windows: Sequence[tuple[float, float]]
) -> tuple[dict[int, Range], dict[int, Range], Range | None]:
    pwm: dict[int, list[float]] = {}
    rpm: dict[int, list[float]] = {}
    target: list[float] = []
    for line in lines:
        mono = line.get("monotonic")
        if not isinstance(mono, (int, float)) or not any(start <= mono <= end for start, end in windows):
            continue
        for duct in line.get("fan_ducts", ()):
            channel = int(duct["channel"])
            pwm.setdefault(channel, []).append(float(duct["pwm"]))
            rpm.setdefault(channel, []).append(float(duct["tach_rpm"]))
        watchdog = line.get("watchdog")
        if isinstance(watchdog, Mapping) and isinstance(watchdog.get("target_pwm"), (int, float)):
            target.append(float(watchdog["target_pwm"]))
    return (
        {channel: Range(min(values), max(values)) for channel, values in sorted(pwm.items())},
        {channel: Range(min(values), max(values)) for channel, values in sorted(rpm.items())},
        Range(min(target), max(target)) if target else None,
    )


def _hottest_slope(
    record: RunRecord, soak: Mapping[str, Any], window_s: float = PLATEAU_WINDOW_S
) -> HottestSlope | None:
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
        return None
    (card, sensor), points = max(series.items(), key=lambda item: max(value for _t, value in item[1]))
    window_start = max(start, end - window_s)
    window = [(t, value) for t, value in points if t >= window_start]
    stage_name = str(soak.get("name", "soak"))
    mean_c = sum(value for _t, value in window) / len(window) if window else math.nan
    base = {
        "card": card,
        "sensor": sensor,
        "stage": stage_name,
        "window_s": end - window_start,
        "samples": len(window),
        "mean_c": mean_c,
    }
    try:
        estimate = ols_slope(
            [t for t, _v in window],
            [v for _t, v in window],
            band_c=(min(v for _t, v in window), max(v for _t, v in window)),
        )
    except (EstimatorError, ValueError) as exc:
        return HottestSlope(**base, slope_c_per_s=None, upper_bound_c_per_s=None, settled=None, note=str(exc))
    upper = upper_confidence_bound(estimate)
    return HottestSlope(
        **base,
        slope_c_per_s=estimate.slope_c_per_s,
        upper_bound_c_per_s=upper,
        settled=upper < PLATEAU_SLOPE_TOLERANCE_C_PER_S,
        note=None,
    )


def summarize_run(path: Path) -> RunSummary:
    """Summarize the run at `path`. Raises `RunSummaryError` when it cannot."""
    run_dir = resolve_run_dir(path)
    try:
        record = load_run(run_dir)
    except LoadError as exc:
        raise RunSummaryError(str(exc)) from exc
    if record.outcome is None:
        raise RunSummaryError(f"{run_dir} has no outcome.json; the run did not finish")
    notes: list[str] = []
    stages = _stages(record)
    soaks = [stage for stage in stages if _is_soak(stage)]
    windows = [(float(stage["started_mono"]), float(stage["ended_mono"])) for stage in soaks]

    declaration = record.declaration or {}
    fixed = declaration.get("fixed", {}) if isinstance(declaration, Mapping) else {}
    control = fixed.get("control", {}) if isinstance(fixed, Mapping) else {}
    factor = declaration.get("factor", {}) if isinstance(declaration, Mapping) else {}
    cap_w = control.get("power_cap_w") if isinstance(control, Mapping) else None

    telemetry_lines = load_jsonl(run_dir / "telemetry.jsonl")
    fan_pwm, fan_rpm, target = _fan_ranges(telemetry_lines, windows)
    if not soaks:
        notes.append("no soak stage ran; the fan ranges are empty")
    elif not fan_pwm:
        notes.append("telemetry has no fan duct readings during the soak")

    last_soak = max(soaks, key=lambda stage: float(stage["ended_mono"])) if soaks else None
    slope = _hottest_slope(record, last_soak) if last_soak is not None else None

    return RunSummary(
        run_dir=str(run_dir),
        experiment=(record.run or {}).get("experiment"),
        cap_w=cap_w if isinstance(cap_w, int) else None,
        lanes=str(factor["value"]) if isinstance(factor, Mapping) and "value" in factor else None,
        state=str(record.outcome.get("state")),
        failure_boundary=record.outcome.get("failure_boundary"),
        detail=record.outcome.get("detail"),
        peaks=_peaks(record),
        prefill_tok_s=_rate(_stage_by_workload(stages, "prefill_7799"), run_dir, "prefill"),
        decode_tok_s=_rate(_stage_by_workload(stages, "decode_single_stream"), run_dir, "decode"),
        lane_sweep_tok_s=_rate(_stage_by_workload(stages, "lane_sweep"), run_dir, "lane_sweep"),
        soak_tok_s=_soak_rate(last_soak) if last_soak is not None else None,
        soak_stages=tuple(str(stage.get("name")) for stage in soaks),
        fan_pwm=fan_pwm,
        fan_rpm=fan_rpm,
        watchdog_target_pwm=target,
        slope=slope,
        notes=tuple(notes),
    )


def _fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _range_text(value: Range | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value.minimum:.{digits}f}-{value.maximum:.{digits}f}"


def render_oneline(summary: RunSummary, *, fan_channels: Sequence[int] | None = None) -> str:
    """One line of `key=value` words. The first word is `state=`; no value holds a space."""
    peak_j = max((item.junction_c for item in summary.peaks), default=None)
    peak_m = max((item.mem_c for item in summary.peaks), default=None)
    peak_e = max((item.edge_c for item in summary.peaks), default=None)
    duct_ranges = [item for channel, item in summary.fan_pwm.items() if fan_channels is None or channel in fan_channels]
    pwm_low = min((item.minimum for item in duct_ranges), default=None)
    pwm_high = max((item.maximum for item in duct_ranges), default=None)
    duct = "n/a" if pwm_low is None or pwm_high is None else f"{pwm_low:.0f}-{pwm_high:.0f}"
    slope = summary.slope
    slope_text = "n/a"
    if slope is not None and slope.slope_c_per_s is not None:
        slope_text = f"{slope.slope_c_per_s * 60.0:+.2f}C/min"
    return " ".join(
        (
            f"state={summary.state}",
            f"boundary={summary.failure_boundary or '-'}",
            f"cap_w={summary.cap_w if summary.cap_w is not None else 'n/a'}",
            f"peak_junction_c={_fmt(peak_j)}",
            f"peak_mem_c={_fmt(peak_m)}",
            f"peak_edge_c={_fmt(peak_e)}",
            f"duct_pwm={duct}",
            f"lane_sweep_tok_s={_fmt(summary.lane_sweep_tok_s)}",
            f"soak_slope={slope_text}",
        )
    )


def render_text(summary: RunSummary) -> str:
    """The full report, for a terminal or a log."""
    rows = [
        f"run: {summary.run_dir}",
        f"experiment: {summary.experiment or 'n/a'}  cap: {summary.cap_w if summary.cap_w is not None else 'n/a'} W"
        f"  lanes: {summary.lanes or 'n/a'}",
        f"outcome: {summary.state}  failure_boundary: {summary.failure_boundary or '-'}",
    ]
    if summary.detail:
        rows.append(f"detail: {summary.detail}")
    for peak in summary.peaks:
        rows.append(
            f"card {peak.card}: peak junction {peak.junction_c:.1f} C, mem {peak.mem_c:.1f} C, "
            f"edge {peak.edge_c:.1f} C, power {peak.power_w:.0f} W"
        )
    rows.append(
        f"rates (tok/s): prefill {_fmt(summary.prefill_tok_s)}, decode {_fmt(summary.decode_tok_s)}, "
        f"lane_sweep {_fmt(summary.lane_sweep_tok_s)}, soak {_fmt(summary.soak_tok_s)}"
    )
    stages = ", ".join(summary.soak_stages) or "none"
    rows.append(f"fan ducts during soak ({stages}):")
    for channel in summary.fan_pwm:
        rows.append(
            f"  channel {channel}: pwm {_range_text(summary.fan_pwm[channel])}, "
            f"rpm {_range_text(summary.fan_rpm.get(channel))}"
        )
    if not summary.fan_pwm:
        rows.append("  n/a")
    rows.append(f"watchdog target pwm during soak: {_range_text(summary.watchdog_target_pwm)}")
    slope = summary.slope
    if slope is None:
        rows.append("plateau: n/a (no soak samples)")
    elif slope.slope_c_per_s is None:
        rows.append(f"plateau: n/a for card {slope.card} {slope.sensor} ({slope.note})")
    else:
        verdict = "settled" if slope.settled else "still moving"
        rows.append(
            f"plateau: hottest sensor card {slope.card} {slope.sensor}, last {slope.window_s:.0f} s of {slope.stage}: "
            f"mean {slope.mean_c:.1f} C, slope {slope.slope_c_per_s * 60.0:+.2f} C/min "
            f"(95% upper {(slope.upper_bound_c_per_s or 0.0) * 60.0:+.2f} C/min), {verdict}"
        )
    rows.extend(f"note: {note}" for note in summary.notes)
    return "\n".join(rows) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_summary",
        description="Summarize one experiment run directory (read-only).",
    )
    parser.add_argument("run_dir", type=Path, help="a run directory, or an artifact root with one run directory")
    parser.add_argument("--fan-channel", type=int, action="append", help="include this channel in the one-line range")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--oneline", action="store_true", help="print one key=value line")
    mode.add_argument("--json", action="store_true", help="print the full record as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 when the summary printed, 1 when the run cannot be summarized."""
    args = _build_parser().parse_args(argv)
    try:
        summary = summarize_run(args.run_dir)
    except (RunSummaryError, OSError, KeyError, ValueError) as exc:
        print(f"run_summary: {exc}", file=sys.stderr)
        return 1
    if args.oneline:
        print(render_oneline(summary, fan_channels=args.fan_channel))
    elif args.json:
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    else:
        print(render_text(summary), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
