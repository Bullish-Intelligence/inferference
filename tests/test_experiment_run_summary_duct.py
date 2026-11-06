"""Offline tests for the duct PWM range in the run summary."""

from __future__ import annotations

from inferference.experiment.analysis.run_summary import (
    Range,
    RunSummary,
    _fan_ranges,
    render_oneline,
    render_text,
)


def _line(mono: float, duct_pwm: float) -> dict[str, object]:
    ducts = [{"channel": ch, "pwm": duct_pwm, "tach_rpm": 9000} for ch in (1, 2, 3, 4, 5)]
    ducts += [{"channel": ch, "pwm": 255, "tach_rpm": 3000} for ch in (6, 7, 8, 9, 10)]
    return {"monotonic": mono, "fan_ducts": ducts}


def _summary() -> RunSummary:
    lines = [_line(1.0, 100), _line(2.0, 180)]
    pwm, rpm, target = _fan_ranges(lines, [(0.0, 5.0)])
    return RunSummary(
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
        fan_pwm=pwm,
        fan_rpm=rpm,
        watchdog_target_pwm=target,
    )


def test_oneline_duct_range_uses_selected_channels() -> None:
    summary = _summary()
    assert summary.fan_pwm[6] == Range(255, 255)
    assert "duct_pwm=100-255" in render_oneline(summary).split()
    assert "duct_pwm=100-180" in render_oneline(summary, fan_channels=[1, 2, 3, 4, 5]).split()


def test_text_reports_channels_without_machine_labels() -> None:
    text = render_text(_summary())
    assert "channel 6:" in text
    assert "channel 10:" in text
    assert "channel 8:" in text
    assert "channel 3:" in text
