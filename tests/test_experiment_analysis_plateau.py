"""Tests for `inferference.experiment.analysis.plateau`: synthetic runs, no files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferference.experiment.analysis.load import CardObservation, Observation, RunRecord
from inferference.experiment.analysis.plateau import plateau_test

_ARM_BOUND_C = 90.0


def _card(junction_c: float) -> CardObservation:
    return CardObservation(
        junction_c=junction_c,
        edge_c=junction_c - 5.0,
        mem_c=junction_c - 2.0,
        power_w=90.0,
        gpu_busy_percent=None,
        throttled=None,
        gtt_used_mib=None,
    )


def _observations(values: dict[float, float]) -> tuple[Observation, ...]:
    """One observation per `mono -> junction_c` entry, on card 0 only."""
    return tuple(
        Observation(utc=f"t{mono}", mono=mono, clock="monotonic", cards={0: _card(junction_c)})
        for mono, junction_c in sorted(values.items())
    )


def _contiguous_requests(start: float, end: float, *, lane: int = 0) -> list[dict[str, Any]]:
    """One request per second, covering `[start, end)` with no gap."""
    return [
        {"request_id": f"r{i}", "lane": lane, "dispatch_mono": float(i), "end_mono": float(i) + 1.0}
        for i in range(int(start), int(end))
    ]


def _record(
    *,
    requests: list[dict[str, Any]],
    started_mono: float,
    ended_mono: float,
    observations: tuple[Observation, ...],
    concurrency: int = 1,
) -> RunRecord:
    stage = {
        "name": "soak",
        "workload": "soak_300s",
        "concurrency": concurrency,
        "started_mono": started_mono,
        "ended_mono": ended_mono,
        "completed": True,
        "stop_reason": None,
        "requests": requests,
    }
    return RunRecord(
        path=Path("."),
        layout="030-v1",
        run=None,
        declaration=None,
        stages={"stages": [stage]},
        outcome=None,
        observations=observations,
    )


def _flat_temps(value: float, *, start: float, end: float, step: float = 1.0) -> tuple[Observation, ...]:
    count = int((end - start) / step) + 1
    return _observations({start + i * step: value for i in range(count)})


def _ramp_temps(
    start_value: float, slope: float, *, start: float, end: float, step: float = 1.0
) -> tuple[Observation, ...]:
    count = int((end - start) / step) + 1
    return _observations({start + i * step: start_value + slope * (i * step) for i in range(count)})


def test_a_flat_window_passes() -> None:
    record = _record(
        requests=_contiguous_requests(0, 120),
        started_mono=0.0,
        ended_mono=120.0,
        observations=_flat_temps(70.0, start=0.0, end=120.0),
    )

    result = plateau_test(record, stage_name="soak", card=0, arm_bound_c=_ARM_BOUND_C, window_s=120.0)

    assert result.passed is True
    assert result.reasons == ()
    assert result.upper_bound_c_per_s is not None
    assert result.upper_bound_c_per_s < 0.02
    assert result.mean_junction_c == 70.0
    assert result.load_ratio == 1.0


def test_a_ramp_of_0_05_c_per_s_fails_criterion_1() -> None:
    record = _record(
        requests=_contiguous_requests(0, 120),
        started_mono=0.0,
        ended_mono=120.0,
        observations=_ramp_temps(70.0, 0.05, start=0.0, end=120.0),
    )

    result = plateau_test(record, stage_name="soak", card=0, arm_bound_c=_ARM_BOUND_C, window_s=120.0)

    assert result.passed is False
    assert len(result.reasons) == 1
    assert "criterion 1" in result.reasons[0]
    assert result.upper_bound_c_per_s is not None
    assert result.upper_bound_c_per_s >= 0.02


def test_a_flat_window_at_arm_bound_minus_2_fails_criterion_2() -> None:
    flat_value = _ARM_BOUND_C - 2.0
    record = _record(
        requests=_contiguous_requests(0, 120),
        started_mono=0.0,
        ended_mono=120.0,
        observations=_flat_temps(flat_value, start=0.0, end=120.0),
    )

    result = plateau_test(record, stage_name="soak", card=0, arm_bound_c=_ARM_BOUND_C, window_s=120.0)

    assert result.passed is False
    assert len(result.reasons) == 1
    assert "criterion 2" in result.reasons[0]
    assert result.mean_junction_c == flat_value


def test_a_starved_queue_fails_criterion_3() -> None:
    # Busy at one request/second for the first 180s, then one request every
    # 20s for the final 120s (the plateau window) — each lane stays
    # technically "busy" within a hugely relaxed max_lane_gap_s, so only the
    # offered-load check (criterion 3) can fail.
    busy_requests = _contiguous_requests(0, 180)
    starved_requests = [
        {"request_id": f"s{i}", "lane": 0, "dispatch_mono": float(mono), "end_mono": float(mono) + 1.0}
        for i, mono in enumerate(range(180, 300, 20))
    ]
    record = _record(
        requests=busy_requests + starved_requests,
        started_mono=0.0,
        ended_mono=300.0,
        observations=_flat_temps(70.0, start=0.0, end=300.0, step=5.0),
    )

    result = plateau_test(
        record,
        stage_name="soak",
        card=0,
        arm_bound_c=_ARM_BOUND_C,
        window_s=120.0,
        max_lane_gap_s=1_000_000.0,
    )

    assert result.passed is False
    assert len(result.reasons) == 1
    assert "criterion 3" in result.reasons[0]
    assert result.load_ratio is not None
    assert result.load_ratio < 0.9


def test_a_lane_gap_of_5s_fails_criterion_4() -> None:
    requests = _contiguous_requests(0, 50) + [{"request_id": "gap", "lane": 0, "dispatch_mono": 55.0, "end_mono": 56.0}]
    requests += _contiguous_requests(56, 120)
    record = _record(
        requests=requests,
        started_mono=0.0,
        ended_mono=120.0,
        observations=_flat_temps(70.0, start=0.0, end=120.0),
    )

    result = plateau_test(record, stage_name="soak", card=0, arm_bound_c=_ARM_BOUND_C, window_s=120.0)

    assert result.passed is False
    assert len(result.reasons) == 1
    assert "criterion 4" in result.reasons[0]
    assert "lane 0" in result.reasons[0]
    assert "5.00s" in result.reasons[0]


def test_a_stage_shorter_than_the_window_fails_with_a_single_named_reason() -> None:
    record = _record(
        requests=_contiguous_requests(0, 60),
        started_mono=0.0,
        ended_mono=60.0,
        observations=_flat_temps(70.0, start=0.0, end=60.0),
    )

    result = plateau_test(record, stage_name="soak", card=0, arm_bound_c=_ARM_BOUND_C, window_s=120.0)

    assert result.passed is False
    assert result.reasons == ("stage shorter than window",)
    assert result.slope is None
    assert result.mean_junction_c is None
    assert result.load_ratio is None
