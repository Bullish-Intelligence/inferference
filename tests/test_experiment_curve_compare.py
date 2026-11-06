"""Offline tests for the fan-curve comparison tool."""

from __future__ import annotations

from inferference.experiment.analysis.curve_compare import (
    ROW_LABELS,
    CurveMetrics,
    Stats,
    direction_changes,
    render_table,
    settle_time_s,
    soak_duct_series,
)


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
    all_pwm, all_rpm = soak_duct_series(lines, [(10.0, 20.0)])
    assert all_pwm == [125.83333333333333, 142.5]
    assert len(all_rpm) == 12


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
