"""The plateau test: does one stage's final window show a settled card.

Four criteria must hold over a stage's final `window_s` of samples before a
measurement from it counts as taken at a plateau rather than mid-ramp. All four are checked on every call; none is
skipped, and none passes by default:

1. the one-sided 95% upper bound of the AR(1)-corrected slope is below
   `slope_tolerance_c_per_s`;
2. the window's mean junction temperature is at least `margin_c` below
   `arm_bound_c`;
3. the window's completions per minute sit within `load_tolerance` of the
   whole stage's completions per minute (a starved queue fails here, not
   because the card is unsettled but because the offered load dropped);
4. every lane was continuously busy in the window, with no gap longer than
   `max_lane_gap_s`.

A stage shorter than `window_s` cannot be evaluated at all: it fails with the
single reason `"stage shorter than window"`, and the window is only as wide
as the data that exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .estimators import EstimatorError, SlopeEstimate, mean, ols_slope, upper_confidence_bound
from .load import RunRecord
from .series import Point, card_series
from .series import window as window_points

_STAGE_SHORTER_THAN_WINDOW = "stage shorter than window"


@dataclass(frozen=True, slots=True)
class PlateauResult:
    """The plateau verdict for one stage, one card: every criterion checked."""

    passed: bool
    reasons: tuple[str, ...]
    window_s: float
    slope: SlopeEstimate | None
    upper_bound_c_per_s: float | None
    mean_junction_c: float | None
    arm_bound_c: float
    load_ratio: float | None


def _find_stage(record: RunRecord, stage_name: str) -> Mapping[str, Any]:
    """Find the named stage in `record.stages`. Raises `ValueError` if absent.

    A missing `stages.json`, or a stage name it does not list, is a caller
    error (the wrong name, or the wrong run), not a measured plateau
    failure, so it raises rather than returning a failed `PlateauResult`.
    """
    if record.stages is None:
        raise ValueError(f"{record.path} has no stages.json; cannot run plateau_test")
    for stage in record.stages.get("stages", ()):
        if stage.get("name") == stage_name:
            return stage
    raise ValueError(f"{record.path} has no stage named {stage_name!r}")


def _completions_per_minute(requests: Sequence[Mapping[str, Any]], *, start_mono: float, end_mono: float) -> int:
    """The count of requests whose `end_mono` falls inside `[start_mono, end_mono]`."""
    count = 0
    for request in requests:
        end = request.get("end_mono")
        if end is not None and start_mono <= end <= end_mono:
            count += 1
    return count


def _lane_gap_reasons(
    requests: Sequence[Mapping[str, Any]],
    *,
    concurrency: int,
    start_mono: float,
    end_mono: float,
    max_lane_gap_s: float,
) -> tuple[str, ...]:
    """Every lane whose busy intervals leave a gap in the window longer than `max_lane_gap_s`."""
    reasons: list[str] = []
    for lane in range(concurrency):
        intervals: list[tuple[float, float]] = []
        for request in requests:
            if request.get("lane") != lane:
                continue
            dispatch = request.get("dispatch_mono")
            end = request.get("end_mono")
            if dispatch is None or end is None:
                continue
            if end < start_mono or dispatch > end_mono:
                continue
            intervals.append((max(dispatch, start_mono), min(end, end_mono)))
        intervals.sort()

        cursor = start_mono
        for interval_start, interval_end in intervals:
            gap = interval_start - cursor
            if gap > max_lane_gap_s:
                reasons.append(
                    f"criterion 4 (lane busy): lane {lane} gap of {gap:.2f}s exceeds max_lane_gap_s {max_lane_gap_s}s"
                )
            cursor = max(cursor, interval_end)
        trailing_gap = end_mono - cursor
        if trailing_gap > max_lane_gap_s:
            reasons.append(
                f"criterion 4 (lane busy): lane {lane} gap of {trailing_gap:.2f}s "
                f"exceeds max_lane_gap_s {max_lane_gap_s}s"
            )
    return tuple(reasons)


def plateau_test(
    record: RunRecord,
    *,
    stage_name: str,
    card: int,
    arm_bound_c: float,
    window_s: float = 120.0,
    slope_tolerance_c_per_s: float = 0.02,
    margin_c: float = 5.0,
    load_tolerance: float = 0.10,
    max_lane_gap_s: float = 1.0,
) -> PlateauResult:
    """Run the four plateau criteria over the final `window_s` of `stage_name`.

    `band_c` on the returned `slope` is a judgment call: the plateau window
    has no declared temperature band, so `SlopeEstimate.band_c` is set to the
    window's own observed min and max, for description only — it plays no
    part in the fit.
    """
    stage = _find_stage(record, stage_name)
    started_mono = float(stage["started_mono"])
    ended_mono = float(stage["ended_mono"])
    requests: Sequence[Mapping[str, Any]] = stage.get("requests", ())

    if ended_mono - started_mono < window_s:
        return PlateauResult(
            passed=False,
            reasons=(_STAGE_SHORTER_THAN_WINDOW,),
            window_s=window_s,
            slope=None,
            upper_bound_c_per_s=None,
            mean_junction_c=None,
            arm_bound_c=arm_bound_c,
            load_ratio=None,
        )

    start_mono = ended_mono - window_s
    reasons: list[str] = []

    temp_points: tuple[Point, ...] = window_points(card_series(record, card, "junction_c"), start_mono, ended_mono)

    slope_estimate: SlopeEstimate | None = None
    upper_bound: float | None = None
    try:
        values = [point.value for point in temp_points]
        slope_estimate = ols_slope(
            [point.mono for point in temp_points],
            values,
            band_c=(min(values), max(values)) if values else (0.0, 0.0),
        )
    except EstimatorError as exc:
        reasons.append(f"criterion 1 (slope): {exc}")
    else:
        upper_bound = upper_confidence_bound(slope_estimate, confidence=0.95)
        if upper_bound >= slope_tolerance_c_per_s:
            reasons.append(
                f"criterion 1 (slope): upper bound {upper_bound:.4f} C/s >= tolerance {slope_tolerance_c_per_s} C/s"
            )

    mean_junction: float | None = None
    if temp_points:
        mean_junction = mean([point.value for point in temp_points])
        bound = arm_bound_c - margin_c
        if mean_junction > bound:
            reasons.append(
                f"criterion 2 (margin): mean junction {mean_junction:.2f} C above "
                f"{bound:.2f} C ({arm_bound_c} - {margin_c})"
            )
    else:
        reasons.append("criterion 2 (margin): no junction samples in window")

    window_duration_min = window_s / 60.0
    stage_duration_min = (ended_mono - started_mono) / 60.0
    window_completions = _completions_per_minute(requests, start_mono=start_mono, end_mono=ended_mono)
    stage_completions = _completions_per_minute(requests, start_mono=started_mono, end_mono=ended_mono)
    window_cpm = window_completions / window_duration_min if window_duration_min > 0 else 0.0
    stage_cpm = stage_completions / stage_duration_min if stage_duration_min > 0 else 0.0

    load_ratio: float | None = None
    if stage_cpm <= 0.0:
        reasons.append("criterion 3 (load): stage has zero completions per minute, cannot form a ratio")
    else:
        load_ratio = window_cpm / stage_cpm
        if abs(load_ratio - 1.0) > load_tolerance:
            reasons.append(
                f"criterion 3 (load): window completions/min {window_cpm:.3f} is "
                f"{load_ratio:.3f}x the stage mean {stage_cpm:.3f}, outside {load_tolerance:.0%} tolerance"
            )

    concurrency = int(stage.get("concurrency", 1))
    reasons.extend(
        _lane_gap_reasons(
            requests,
            concurrency=concurrency,
            start_mono=start_mono,
            end_mono=ended_mono,
            max_lane_gap_s=max_lane_gap_s,
        )
    )

    return PlateauResult(
        passed=not reasons,
        reasons=tuple(reasons),
        window_s=window_s,
        slope=slope_estimate,
        upper_bound_c_per_s=upper_bound,
        mean_junction_c=mean_junction,
        arm_bound_c=arm_bound_c,
        load_ratio=load_ratio,
    )
