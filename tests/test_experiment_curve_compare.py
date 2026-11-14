"""Offline tests for the fan-curve comparison tool."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from inferference.experiment.analysis import curve_compare, run_summary
from inferference.experiment.analysis.curve_compare import (
    ROW_LABELS,
    CurveMetrics,
    Stats,
    direction_changes,
    render_table,
    settle_time_s,
    soak_duct_series,
)
from inferference.experiment.analysis.run_summary import DUCT_CHANNELS_ENV, RunSummaryError


@pytest.fixture(autouse=True)
def _declared_duct_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare channel 1 as the duct, as a machine does in `rig-local.sh`."""
    monkeypatch.setenv(DUCT_CHANNELS_ENV, "1")


def test_direction_changes_counts_turns_and_ignores_repeats() -> None:
    assert direction_changes([]) == 0
    assert direction_changes([5, 5, 5]) == 0
    assert direction_changes([1, 2, 3, 4]) == 0
    assert direction_changes([1, 2, 2, 3, 2, 2, 1, 2]) == 2
    assert direction_changes([1, 3, 1, 3, 1]) == 3


def _line(mono: float, pwm: float, rpm: float) -> dict[str, object]:
    ducts = [{"channel": ch, "pwm": pwm, "tach_rpm": rpm} for ch in (1, 2, 3, 4, 5)]
    ducts.append({"channel": 6, "pwm": 255, "tach_rpm": 3000})
    return {"monotonic": mono, "fan_ducts": ducts}


def test_soak_duct_series_keeps_only_duct_channels_inside_the_window() -> None:
    lines = [_line(1.0, 25, 1500), _line(10.0, 100, 8000), _line(11.0, 120, 9000), _line(30.0, 255, 12000)]
    pwm, rpm = soak_duct_series(lines, [(10.0, 20.0)], [1, 2, 3, 4, 5])
    assert pwm == [100.0, 120.0]
    assert rpm == [8000.0] * 5 + [9000.0] * 5
    all_pwm, all_rpm = soak_duct_series(lines, [(10.0, 20.0)], [1, 2, 3, 4, 5, 6])
    assert all_pwm == [125.83333333333333, 142.5]
    assert len(all_rpm) == 12
    one_pwm, one_rpm = soak_duct_series(lines, [(10.0, 20.0)], [1])
    assert one_pwm == [100.0, 120.0]
    assert one_rpm == [8000.0, 9000.0]


def _metrics(turns: int) -> CurveMetrics:
    return CurveMetrics(
        run_dir="/r",
        experiment="x",
        cap_w=140,
        state="completed",
        failure_boundary=None,
        peak_junction_c=63.0,
        peak_mem_c=68.0,
        peak_edge_c=57.0,
        decode_tok_s=18.7,
        prefill_tok_s=259.1,
        lane_sweep_tok_s=49.6,
        soak_tok_s=54.1,
        soak_s=300.0,
        duct_pwm=Stats(150, 175, 173),
        duct_rpm=Stats(9600, 11600, 11200),
        pwm_stdev=6.3,
        direction_changes=turns,
        direction_changes_per_min=turns / 5.0,
        hottest_slope_c_per_min=2.9,
        hottest_sensor="card0 mem",
        settle_time_s=240.0,
    )


def test_render_table_has_one_row_per_label_and_shows_errors() -> None:
    text = render_table([("old", _metrics(5)), ("bad", "no outcome.json")])
    lines = text.splitlines()
    assert len(lines) == 2 + len(ROW_LABELS)
    assert "1.00" in text
    assert "error: no outcome.json" in text


def test_settle_time_finds_the_start_of_the_final_flat_stretch() -> None:
    rising = [(float(t), 50.0 + 0.1 * t) for t in range(0, 200)]
    flat = [(float(t), 70.0) for t in range(200, 500)]
    settled = settle_time_s(rising + flat)
    assert settled is not None
    assert 200.0 <= settled <= 270.0


def test_settle_time_is_none_for_a_rising_or_short_series() -> None:
    assert settle_time_s([(float(t), 50.0 + 0.1 * t) for t in range(300)]) is None
    assert settle_time_s([(float(t), 70.0) for t in range(30)]) is None
    assert settle_time_s([]) is None


_EXHAUST = {"channel": 10, "pwm": 255, "tach_rpm": 3000}


def _patched_run(lines: list[dict[str, object]]) -> tuple[AbstractContextManager[object], ...]:
    """Patch the run readers so `curve_metrics` sees one soak stage and `lines` of telemetry."""
    summary = run_summary.RunSummary(
        run_dir="/r",
        experiment="x",
        cap_w=140,
        lanes="8",
        state="completed",
        failure_boundary=None,
        detail=None,
        peaks=(),
        prefill_tok_s=None,
        decode_tok_s=None,
        lane_sweep_tok_s=None,
        soak_tok_s=None,
        soak_stages=("soak_600s",),
        fan_pwm={},
        fan_rpm={},
        watchdog_target_pwm=None,
    )
    stage = {"name": "soak_600s", "started_mono": 0.0, "ended_mono": 5.0}
    record = SimpleNamespace(observations=[])
    return (
        patch.object(curve_compare, "summarize_run", return_value=summary),
        patch.object(curve_compare, "resolve_run_dir", return_value=Path("/r")),
        patch.object(curve_compare, "load_run", return_value=record),
        patch.object(curve_compare, "_stages", return_value=[stage]),
        patch.object(curve_compare, "_slope_over", return_value=None),
        patch.object(curve_compare, "load_jsonl", return_value=lines),
    )


def _duct_lines() -> list[dict[str, object]]:
    duct = {"channel": 1, "pwm": 140, "tach_rpm": 9000}
    return [{"monotonic": 1.0, "fan_ducts": [duct, _EXHAUST]}, {"monotonic": 2.0, "fan_ducts": [duct, _EXHAUST]}]


def test_curve_metrics_default_reports_the_duct_channel_only() -> None:
    with ExitStack() as stack:
        for item in _patched_run(_duct_lines()):
            stack.enter_context(item)
        metrics = curve_compare.curve_metrics(Path("/r"))
    assert metrics.duct_pwm == Stats(140, 140, 140)
    assert metrics.duct_rpm == Stats(9000, 9000, 9000)


def test_curve_metrics_fan_channel_overrides_the_default() -> None:
    with ExitStack() as stack:
        for item in _patched_run(_duct_lines()):
            stack.enter_context(item)
        metrics = curve_compare.curve_metrics(Path("/r"), fan_channels=[10])
    assert metrics.duct_pwm == Stats(255, 255, 255)


def test_curve_metrics_refuses_without_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DUCT_CHANNELS_ENV)
    with ExitStack() as stack:
        for item in _patched_run(_duct_lines()):
            stack.enter_context(item)
        with pytest.raises(RunSummaryError, match="--fan-channel"):
            curve_compare.curve_metrics(Path("/r"))


def test_curve_metrics_reads_the_declared_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DUCT_CHANNELS_ENV, "10")
    with ExitStack() as stack:
        for item in _patched_run(_duct_lines()):
            stack.enter_context(item)
        metrics = curve_compare.curve_metrics(Path("/r"))
    assert metrics.duct_pwm == Stats(255, 255, 255)


def test_cli_default_and_fan_channel(capsys: pytest.CaptureFixture[str]) -> None:
    with ExitStack() as stack:
        for item in _patched_run(_duct_lines()):
            stack.enter_context(item)
        assert curve_compare.main(["/r", "--json"]) == 0
        default = json.loads(capsys.readouterr().out)["runs"]["r"]
        assert curve_compare.main(["/r", "--json", "--fan-channel", "10"]) == 0
        chosen = json.loads(capsys.readouterr().out)["runs"]["r"]
    assert default["duct_pwm"]["maximum"] == 140
    assert chosen["duct_pwm"]["maximum"] == 255
