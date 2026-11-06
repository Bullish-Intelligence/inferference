"""Tests for the dual-card pair analysis, on fabricated run directories.

The fabricated pair has a known shape, so every number below can be derived by
hand. Both runs share one monotonic clock. The gate releases at t=100. Card 0
heats at 0.2 C/s and runs the whole soak. Card 1 heats at 0.3 C/s and stops at
the arm bound at t=250, in the soak. The overlap window is t=100 to t=250.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inferference.experiment import cli
from inferference.experiment.analysis.paired import (
    POWER_LABEL,
    PairedAnalysisError,
    analyze_pair,
    analyze_pair_root,
    load_pair_records,
)

_RELEASE = 100.0
_BOUND_AT = 250.0
_SOAK_END_CARD0 = 421.0
_SLOPES = {0: 0.2, 1: 0.3}
_POWER = {0: 200.0, 1: 180.0}
_ARM_CONDITION = "v620-cooling-v2-dual-cap-180w-lanes-8-abcdef123456"


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps({"schema_version": 1, **payload}, sort_keys=True) + "\n")


def _request(name: str, tokens: int, **timings: Any) -> dict[str, Any]:
    return {
        "request_id": name,
        "lane": 0,
        "seq": 0,
        "valid": True,
        "completion_tokens": tokens,
        "server_timings": timings,
    }


def _stage(name: str, start: float, end: float, requests: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "workload": name,
        "concurrency": 2,
        "started_mono": start,
        "ended_mono": end,
        "completed": True,
        "stop_reason": None,
        "requests": requests,
        **extra,
    }


def _telemetry(path: Path, *, last_tick: float, bound: bool) -> None:
    lines = []
    for tick in range(90, int(last_tick) + 1):
        cards = []
        for card in (0, 1):
            junction = 30.0 + _SLOPES[card] * max(0.0, tick - _RELEASE)
            cards.append(
                {
                    "card": card,
                    "temperature_hotspot_c": junction,
                    "temperature_edge_c": junction - 2,
                    "temperature_mem_c": junction + 5,
                    "power_w": _POWER[card],
                }
            )
        lines.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "utc": f"2026-10-28T00:00:{tick % 60:02d}Z",
                    "monotonic": float(tick),
                    "cards": cards,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _make_run(
    root: Path, card: int, *, lane_tokens: int, decode: tuple[int, int], prefill: tuple[int, int], bounded: bool
) -> Path:
    run_dir = root / f"card{card}" / "20261028T000000Z-v620-cooling-v2-dual-cap-180w-abcdef123456-r1"
    run_dir.mkdir(parents=True)
    _write(
        run_dir / "run.json",
        {"experiment": "v620-cooling-v2-dual-cap-180w", "condition": _ARM_CONDITION, "replicate": 1},
    )
    stages = [
        _stage("prefill_7799", 100.0, 101.0, [_request("p", 1, prompt_n=prefill[0], prompt_ms=prefill[1])]),
        _stage("decode_single_stream", 101.0, 111.0, [_request("d", 1, predicted_n=decode[0], predicted_ms=decode[1])]),
        _stage(
            "lane_sweep",
            111.0,
            121.0,
            [_request("a", lane_tokens), _request("b", lane_tokens)],
        ),
    ]
    if bounded:
        soak = _stage("soak_300s", 121.0, _BOUND_AT, [])
        soak["completed"] = False
        soak["stop_reason"] = "safety:arm_bound"
        stages.append(soak)
    else:
        stages.append(_stage("soak_300s", 121.0, _SOAK_END_CARD0, []))
    pair_info = {"peer_card": 1 - card, "release_mono": _RELEASE, "stop_mono": _BOUND_AT if bounded else None}
    _write(run_dir / "stages.json", {"start_gate": {"passed": True}, "stages": stages, "pair": pair_info})
    _write(
        run_dir / "outcome.json",
        {
            "state": "bounded" if bounded else "completed",
            "failure_boundary": "safety:arm_bound" if bounded else None,
            "detail": None,
        },
    )
    _telemetry(run_dir / "telemetry.jsonl", last_tick=_BOUND_AT if bounded else _SOAK_END_CARD0, bound=bounded)
    return run_dir


def _make_pair(root: Path, index: int = 0) -> Path:
    dir0 = _make_run(root, 0, lane_tokens=100, decode=(100, 4000), prefill=(7800, 2000), bounded=False)
    dir1 = _make_run(root, 1, lane_tokens=150, decode=(100, 3500), prefill=(7800, 2500), bounded=True)
    pair_dir = root / "pair"
    pair_dir.mkdir(exist_ok=True)
    record = pair_dir / f"{index:02d}-pair.json"
    _write(
        record,
        {
            "experiment": "v620-cooling-v2-dual-cap-180w",
            "condition": _ARM_CONDITION,
            "replicate": 1,
            "cards": [0, 1],
            "release_mono": _RELEASE,
            "first_bound_mono": _BOUND_AT,
            "runs": {
                "0": {"card": 0, "run_dir": str(dir0), "state": "completed", "stop_mono": None},
                "1": {"card": 1, "run_dir": str(dir1), "state": "bounded", "stop_mono": _BOUND_AT},
            },
        },
    )
    return record


def test_per_card_numbers_follow_the_overlap_window(tmp_path: Path) -> None:
    analysis = analyze_pair(_make_pair(tmp_path))

    assert analysis.release_mono == _RELEASE
    # The earlier run end is the bounded card's end.
    assert analysis.overlap_window == (_RELEASE, _BOUND_AT)
    card0, card1 = analysis.cards[0], analysis.cards[1]
    assert (card0.state, card1.state) == ("completed", "bounded")
    # lane_sweep: two requests of 100 or 150 tokens over a 10 s stage.
    assert card0.lane_sweep_tokens_per_second == pytest.approx(20.0)
    assert card1.lane_sweep_tokens_per_second == pytest.approx(30.0)
    assert card0.decode_tokens_per_second == pytest.approx(25.0)
    assert card1.decode_tokens_per_second == pytest.approx(100 * 1000 / 3500)
    assert card0.prefill_tokens_per_second == pytest.approx(3900.0)
    assert card1.prefill_tokens_per_second == pytest.approx(7800 * 1000 / 2500)
    # Peaks come from inside the window only: card 0 keeps heating after t=250.
    assert card0.peak_junction_c == pytest.approx(30.0 + 0.2 * 150)
    assert card1.peak_junction_c == pytest.approx(75.0)
    assert card0.peak_mem_c == pytest.approx(30.0 + 0.2 * 150 + 5)
    # Band samples of the soak inside the window.
    assert card0.band_power_w == pytest.approx(200.0)
    assert card1.band_power_w == pytest.approx(180.0)
    assert card0.band_slope_c_per_s == pytest.approx(0.2)
    assert card1.band_slope_c_per_s == pytest.approx(0.3)
    # Time to bound is measured from the pair release.
    assert card0.time_to_bound_s is None
    assert card1.time_to_bound_s == pytest.approx(_BOUND_AT - _RELEASE)


def test_combined_numbers_sum_both_cards(tmp_path: Path) -> None:
    analysis = analyze_pair(_make_pair(tmp_path))

    assert analysis.summed_lane_sweep_tokens_per_second == pytest.approx(50.0)
    assert analysis.summed_power_label == POWER_LABEL == "sum of power1_average"
    assert analysis.summed_power_overlap_mean_w == pytest.approx(380.0)
    assert analysis.summed_power_overlap_peak_w == pytest.approx(380.0)
    assert analysis.summed_power_soak_mean_w == pytest.approx(380.0)
    # One tick per second from t=100 to t=250.
    assert analysis.summed_power_ticks == 151


def test_the_replicate_delta_is_card0_minus_card1(tmp_path: Path) -> None:
    analysis = analyze_pair(_make_pair(tmp_path))

    delta = analysis.delta_card0_minus_card1
    assert delta["lane_sweep_tokens_per_second"] == pytest.approx(-10.0)
    assert delta["band_power_w"] == pytest.approx(20.0)
    assert delta["band_slope_c_per_s"] == pytest.approx(-0.1)
    assert delta["peak_junction_c"] == pytest.approx(60.0 - 75.0)
    # Only card 1 has a time to bound, so no difference exists.
    assert delta["time_to_bound_s"] is None


def test_a_card_that_did_not_finish_lane_sweep_blocks_the_sum(tmp_path: Path) -> None:
    record = _make_pair(tmp_path)
    run_dir = Path(json.loads(record.read_text())["runs"]["1"]["run_dir"])
    stages = json.loads((run_dir / "stages.json").read_text())
    sweep = next(item for item in stages["stages"] if item["name"] == "lane_sweep")
    sweep["completed"] = False
    sweep["stop_reason"] = "safety:arm_bound"
    (run_dir / "stages.json").write_text(json.dumps(stages))

    analysis = analyze_pair(record)

    assert analysis.cards[1].lane_sweep_tokens_per_second is None
    assert analysis.summed_lane_sweep_tokens_per_second is None
    assert any("lane_sweep" in note for note in analysis.notes)
    assert any("lane_sweep" in note for note in analysis.cards[1].notes)
    assert analysis.delta_card0_minus_card1["lane_sweep_tokens_per_second"] is None


def test_a_pair_without_a_release_time_has_no_window(tmp_path: Path) -> None:
    record = _make_pair(tmp_path)
    payload = json.loads(record.read_text())
    payload["release_mono"] = None
    record.write_text(json.dumps(payload))

    with pytest.raises(PairedAnalysisError, match="no release time"):
        analyze_pair(record)


def test_a_moved_artifact_root_still_resolves_the_run_directories(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.mkdir()
    _make_pair(original)
    moved = tmp_path / "moved"
    original.rename(moved)

    analysis = analyze_pair(moved / "pair" / "00-pair.json")

    assert analysis.cards[0].run_dir.parent == moved / "card0"


def test_records_load_from_a_file_a_pair_directory_or_a_root(tmp_path: Path) -> None:
    record = _make_pair(tmp_path)

    assert load_pair_records(record) == (record,)
    assert load_pair_records(tmp_path / "pair") == (record,)
    assert load_pair_records(tmp_path) == (record,)
    with pytest.raises(PairedAnalysisError, match="no \\*-pair.json"):
        load_pair_records(tmp_path / "card0")
    with pytest.raises(PairedAnalysisError, match="does not exist"):
        load_pair_records(tmp_path / "missing")


def test_analyze_pair_root_groups_records_by_condition_in_replicate_order(tmp_path: Path) -> None:
    _make_pair(tmp_path)

    result = analyze_pair_root(tmp_path)

    assert result["summed_power_label"] == "sum of power1_average"
    assert "not wall power" in result["summed_power_note"]
    assert list(result["conditions"]) == [_ARM_CONDITION]
    assert [item.replicate for item in result["conditions"][_ARM_CONDITION]] == [1]


def test_analyze_pair_prints_json_through_the_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _make_pair(tmp_path)

    assert cli.main(["analyze-pair", str(tmp_path)]) == 0

    result = json.loads(capsys.readouterr().out)
    entry = result["conditions"][_ARM_CONDITION][0]
    assert entry["overlap_window"] == [_RELEASE, _BOUND_AT]
    assert entry["summed_lane_sweep_tokens_per_second"] == pytest.approx(50.0)
    assert entry["cards"]["1"]["time_to_bound_s"] == pytest.approx(150.0)
    assert entry["delta_card0_minus_card1"]["lane_sweep_tokens_per_second"] == pytest.approx(-10.0)


def test_analyze_pair_reports_a_missing_record_through_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["analyze-pair", str(tmp_path / "missing")]) == 1
    assert "does not exist" in capsys.readouterr().err
