"""Compare the two cards of a dual-card pair run, from saved artifacts.

`run_pair` writes `<artifact_root>/pair/<idx>-pair.json` after each pair. That
record names both run directories and the monotonic release time of the pair.
This module reads the record and the two run directories, and reports:

* per card: the lane-sweep aggregate, the decode and prefill rates, the peak
  junction and memory temperature inside the overlap window, the band power
  and band slope of the soak inside the overlap window, and the time to the
  arm bound measured from the pair release;
* for the pair: the summed lane-sweep throughput when both cards completed the
  stage, and the per-tick sum of both cards' power;
* the per-record difference card0 minus card1 for each number.

The **overlap window** runs from the gate release to the earlier of the two run
ends. Both runs share one monotonic clock, so a window is one pair of numbers.

The summed power is the sum of the telemetry `power_w` field of both cards
(the amdgpu `power1_average` reading). It is **not** wall power.

The module reuses the stage helpers. It reads only; it writes
nothing into a run directory.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .estimators import EstimatorError, mean, ols_slope
from .load import LoadError, RunRecord, load_run
from .series import Episode, Point, band_samples
from .stage_metrics import (
    StageMetricsError,
    _stage,
    _stage_requests_throughput,
    _timing_rate,
)

POWER_LABEL = "sum of power1_average"
BAND_C = (50.0, 70.0)
POWER_NOTE = "The sum of the telemetry power_w reading (power1_average) of both cards. It is not wall power."

__all__ = [
    "POWER_LABEL",
    "CardMetrics",
    "PairAnalysis",
    "PairedAnalysisError",
    "analyze_pair",
    "analyze_pair_root",
    "load_pair_records",
]


class PairedAnalysisError(ValueError):
    """A pair record or one of its run directories is missing, malformed, or inconsistent."""


@dataclass(frozen=True, slots=True)
class CardMetrics:
    """One card's numbers from one pair run. A number is `None` when its stage never completed."""

    card: int
    run_dir: Path
    state: str
    failure_boundary: str | None
    lane_sweep_tokens_per_second: float | None
    decode_tokens_per_second: float | None
    prefill_tokens_per_second: float | None
    peak_junction_c: float | None
    peak_mem_c: float | None
    band_power_w: float | None
    band_slope_c_per_s: float | None
    time_to_bound_s: float | None
    run_end_mono: float
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PairAnalysis:
    """One pair record, analyzed."""

    record_path: Path
    experiment: str
    condition: str
    replicate: int
    release_mono: float
    overlap_window: tuple[float, float]
    cards: Mapping[int, CardMetrics]
    summed_lane_sweep_tokens_per_second: float | None
    summed_power_label: str
    summed_power_overlap_mean_w: float | None
    summed_power_overlap_peak_w: float | None
    summed_power_soak_mean_w: float | None
    summed_power_ticks: int
    delta_card0_minus_card1: Mapping[str, float | None]
    notes: tuple[str, ...]


def _read_record(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedAnalysisError(f"could not read pair record {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise PairedAnalysisError(f"{path} is not a schema_version 1 pair record")
    return payload


def load_pair_records(path: Path) -> tuple[Path, ...]:
    """The pair record files at `path`: one record, a `pair/` directory, or an artifact root."""
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise PairedAnalysisError(f"pair input does not exist: {path}")
    directory = path / "pair" if (path / "pair").is_dir() else path
    records = tuple(sorted(directory.glob("*-pair.json")))
    if not records:
        raise PairedAnalysisError(f"no *-pair.json record found under {path}")
    return records


def _run_directory(record_path: Path, card: int, entry: Mapping[str, Any]) -> Path:
    named = Path(str(entry.get("run_dir", "")))
    if named.is_dir():
        return named
    # The artifact root may have moved since the run. Look beside the record.
    moved = record_path.parent.parent / f"card{card}" / named.name
    if moved.is_dir():
        return moved
    raise PairedAnalysisError(f"{record_path}: run directory for card {card} not found: {named}")


def _mapping(value: Any, label: str, where: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairedAnalysisError(f"{where}: {label} must be an object")
    return value


def _stages(record: RunRecord) -> Sequence[Mapping[str, Any]]:
    payload = _mapping(record.stages, "stages.json", record.path)
    return [item for item in payload.get("stages", ()) if isinstance(item, Mapping)]


def _stage_named(record: RunRecord, prefix: str) -> Mapping[str, Any] | None:
    """The one stage whose name equals `prefix` or starts with it. Not the residency probe."""
    found = [item for item in _stages(record) if str(item.get("name", "")).startswith(prefix)]
    return found[0] if len(found) == 1 else None


def _run_end_mono(record: RunRecord) -> float:
    ends = [float(item["ended_mono"]) for item in _stages(record) if item.get("ended_mono") is not None]
    if not ends:
        raise PairedAnalysisError(f"{record.path}: stages.json holds no stage end time")
    return max(ends)


def _card_points(record: RunRecord, card: int, field: str) -> tuple[Point, ...]:
    points = []
    for observation in record.observations:
        item = observation.cards.get(card)
        if item is not None:
            points.append(Point(observation.mono, float(getattr(item, field))))
    return tuple(points)


def _in_window(points: Sequence[Point], start: float, end: float) -> tuple[Point, ...]:
    return tuple(point for point in points if start <= point.mono <= end)


def _card_metrics(
    record: RunRecord,
    entry: Mapping[str, Any],
    card: int,
    *,
    release_mono: float,
    window: tuple[float, float],
) -> CardMetrics:
    run_dir = record.path
    outcome = _mapping(record.outcome, "outcome.json", run_dir)
    stages_payload = _mapping(record.stages, "stages.json", run_dir)
    notes: list[str] = []

    def stage_number(label: str, prefix: str, compute: Any) -> float | None:
        stage = _stage_named(record, prefix)
        if stage is None:
            notes.append(f"{label}: no {prefix!r} stage")
            return None
        try:
            return float(compute(stage))
        except (StageMetricsError, KeyError, TypeError, ValueError) as exc:
            notes.append(f"{label}: {exc}")
            return None

    def completed(stage: Mapping[str, Any]) -> Mapping[str, Any]:
        # A stage that did not finish cleanly gives no rate.
        return _stage(stages_payload, str(stage["name"]), run_dir)

    lane_sweep = stage_number(
        "lane_sweep",
        "lane_sweep",
        lambda stage: _stage_requests_throughput(completed(stage), run_dir),
    )
    decode = stage_number(
        "decode",
        "decode_single_stream",
        lambda stage: _timing_rate(
            completed(stage),
            workload=str(stage["name"]),
            count_key="predicted_n",
            duration_key="predicted_ms",
            run_dir=run_dir,
        ),
    )
    prefill = stage_number(
        "prefill",
        "prefill_",
        lambda stage: _timing_rate(
            completed(stage),
            workload=str(stage["name"]),
            count_key="prompt_n",
            duration_key="prompt_ms",
            run_dir=run_dir,
        ),
    )

    start, end = window
    junction = _in_window(_card_points(record, card, "junction_c"), start, end)
    memory = _in_window(_card_points(record, card, "mem_c"), start, end)
    peak_junction = max((point.value for point in junction), default=None)
    peak_mem = max((point.value for point in memory), default=None)
    if peak_junction is None:
        notes.append("no telemetry for this card inside the overlap window")

    band_power: float | None = None
    band_slope: float | None = None
    soak = _stage_named(record, "soak_")
    if soak is None:
        notes.append("band: no soak stage")
    else:
        soak_start = max(float(soak["started_mono"]), start)
        soak_end = min(float(soak["ended_mono"]), end)
        if soak_end <= soak_start:
            notes.append("band: the soak stage lies outside the overlap window")
        else:
            soak_temp = _in_window(_card_points(record, card, "junction_c"), soak_start, soak_end)
            soak_power = {point.mono: point.value for point in _card_points(record, card, "power_w")}
            selected = band_samples(soak_temp, Episode(start_mono=soak_start, end_mono=soak_end), band_c=BAND_C)
            if not selected:
                notes.append("band: the soak did not enter the 50-70 C band inside the overlap window")
            else:
                band_power = mean([soak_power[point.mono] for point in selected if point.mono in soak_power])
                try:
                    band_slope = ols_slope(
                        [point.mono for point in selected], [point.value for point in selected], band_c=BAND_C
                    ).slope_c_per_s
                except EstimatorError as exc:
                    notes.append(f"band slope: {exc}")

    time_to_bound: float | None = None
    state = str(outcome.get("state"))
    if state == "bounded":
        pair_info = stages_payload.get("pair")
        stop_mono = pair_info.get("stop_mono") if isinstance(pair_info, Mapping) else None
        if stop_mono is None:
            stop_mono = entry.get("stop_mono")
        if stop_mono is None:
            notes.append("time to bound: the run recorded no stop time")
        else:
            time_to_bound = float(stop_mono) - release_mono

    return CardMetrics(
        card=card,
        run_dir=run_dir,
        state=state,
        failure_boundary=outcome.get("failure_boundary"),
        lane_sweep_tokens_per_second=lane_sweep,
        decode_tokens_per_second=decode,
        prefill_tokens_per_second=prefill,
        peak_junction_c=peak_junction,
        peak_mem_c=peak_mem,
        band_power_w=band_power,
        band_slope_c_per_s=band_slope,
        time_to_bound_s=time_to_bound,
        run_end_mono=_run_end_mono(record),
        notes=tuple(notes),
    )


_DELTA_FIELDS = (
    "lane_sweep_tokens_per_second",
    "decode_tokens_per_second",
    "prefill_tokens_per_second",
    "peak_junction_c",
    "peak_mem_c",
    "band_power_w",
    "band_slope_c_per_s",
    "time_to_bound_s",
)


def analyze_pair(record_path: Path) -> PairAnalysis:
    """Analyze one `<idx>-pair.json` record and its two run directories."""
    payload = _read_record(record_path)
    cards = payload.get("cards")
    if not isinstance(cards, list) or len(cards) != 2:
        raise PairedAnalysisError(f"{record_path}: cards must list two card indices")
    release = payload.get("release_mono")
    if not isinstance(release, (int, float)) or isinstance(release, bool) or not math.isfinite(release):
        raise PairedAnalysisError(
            f"{record_path}: the pair has no release time; it never left the start gate, so there is no overlap window"
        )
    runs = _mapping(payload.get("runs"), "runs", record_path)
    records: dict[int, RunRecord] = {}
    entries: dict[int, Mapping[str, Any]] = {}
    try:
        for card in cards:
            entry = _mapping(runs.get(str(card)), f"runs.{card}", record_path)
            entries[card] = entry
            records[card] = load_run(_run_directory(record_path, card, entry))
    except LoadError as exc:
        raise PairedAnalysisError(str(exc)) from exc

    ends = {card: _run_end_mono(record) for card, record in records.items()}
    window = (float(release), min(ends.values()))
    if window[1] <= window[0]:
        raise PairedAnalysisError(f"{record_path}: the overlap window is empty ({window[0]} to {window[1]})")
    metrics = {
        card: _card_metrics(records[card], entries[card], card, release_mono=float(release), window=window)
        for card in cards
    }

    notes: list[str] = []
    lane_values = [item.lane_sweep_tokens_per_second for item in metrics.values()]
    present_lanes = [value for value in lane_values if value is not None]
    summed_lane = sum(present_lanes) if len(present_lanes) == len(lane_values) else None
    if summed_lane is None:
        notes.append("summed lane_sweep: not every card completed the stage")

    # Telemetry of one run holds both cards. Take the first listed card's run.
    source = records[cards[0]]
    overlap_sums: list[float] = []
    soak_sums: list[float] = []
    soak = _stage_named(source, "soak_")
    for observation in source.observations:
        if not window[0] <= observation.mono <= window[1]:
            continue
        if not all(card in observation.cards for card in cards):
            continue
        total = sum(observation.cards[card].power_w for card in cards)
        overlap_sums.append(total)
        if soak is not None and float(soak["started_mono"]) <= observation.mono <= float(soak["ended_mono"]):
            soak_sums.append(total)

    first, second = cards[0], cards[1]
    deltas: dict[str, float | None] = {}
    for name in _DELTA_FIELDS:
        a, b = getattr(metrics[first], name), getattr(metrics[second], name)
        deltas[name] = None if a is None or b is None else a - b

    return PairAnalysis(
        record_path=record_path,
        experiment=str(payload.get("experiment")),
        condition=str(payload.get("condition")),
        replicate=int(payload.get("replicate", 0)),
        release_mono=float(release),
        overlap_window=window,
        cards=metrics,
        summed_lane_sweep_tokens_per_second=summed_lane,
        summed_power_label=POWER_LABEL,
        summed_power_overlap_mean_w=mean(overlap_sums) if overlap_sums else None,
        summed_power_overlap_peak_w=max(overlap_sums) if overlap_sums else None,
        summed_power_soak_mean_w=mean(soak_sums) if soak_sums else None,
        summed_power_ticks=len(overlap_sums),
        delta_card0_minus_card1=deltas,
        notes=tuple(notes),
    )


def analyze_pair_root(path: Path) -> dict[str, Any]:
    """Analyze every pair record at `path`, grouped by condition id.

    Returns a mapping with the power label and one list of analyses per
    condition, in replicate order. The per-record deltas are card0 minus card1.
    """
    grouped: dict[str, list[PairAnalysis]] = defaultdict(list)
    for record_path in load_pair_records(path):
        analysis = analyze_pair(record_path)
        grouped[analysis.condition].append(analysis)
    return {
        "summed_power_label": POWER_LABEL,
        "summed_power_note": POWER_NOTE,
        "conditions": {
            condition: sorted(items, key=lambda item: item.replicate) for condition, items in sorted(grouped.items())
        },
    }
