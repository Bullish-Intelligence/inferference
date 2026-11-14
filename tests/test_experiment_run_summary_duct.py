"""Offline tests for the duct PWM range in the run summary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from inferference.experiment.analysis import run_summary
from inferference.experiment.analysis.run_summary import (
    DUCT_CHANNELS_ENV,
    Range,
    RunSummary,
    RunSummaryError,
    _fan_ranges,
    render_oneline,
    render_text,
    resolve_duct_channels,
)


@pytest.fixture(autouse=True)
def _declared_duct_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare channel 1 as the duct, as a machine does in `rig-local.sh`."""
    monkeypatch.setenv(DUCT_CHANNELS_ENV, "1")


_EXHAUST = [{"channel": 10, "pwm": 255, "tach_rpm": 3000}]


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
    assert "duct_pwm=100-255" in render_oneline(summary, fan_channels=[1, 6]).split()
    assert "duct_pwm=100-180" in render_oneline(summary, fan_channels=[1, 2, 3, 4, 5]).split()


def test_oneline_default_reports_the_duct_channel_only() -> None:
    lines = [
        {"monotonic": 1.0, "fan_ducts": [{"channel": 1, "pwm": 140, "tach_rpm": 9000}] + _EXHAUST},
        {"monotonic": 2.0, "fan_ducts": [{"channel": 1, "pwm": 140, "tach_rpm": 9000}] + _EXHAUST},
    ]
    pwm, rpm, target = _fan_ranges(lines, [(0.0, 5.0)])
    summary = replace(_summary(), fan_pwm=pwm, fan_rpm=rpm, watchdog_target_pwm=target)
    assert summary.fan_pwm[10] == Range(255, 255)
    assert "duct_pwm=140-140" in render_oneline(summary).split()


def test_oneline_default_without_a_duct_reading_is_na() -> None:
    summary = replace(_summary(), fan_pwm={10: Range(255, 255)})
    assert "duct_pwm=n/a" in render_oneline(summary).split()


def test_cli_fan_channel_overrides_the_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary = replace(_summary(), fan_pwm={1: Range(140, 140), 10: Range(255, 255)})
    with patch.object(run_summary, "summarize_run", return_value=summary):
        assert run_summary.main([str(tmp_path), "--oneline"]) == 0
        assert "duct_pwm=140-140" in capsys.readouterr().out.split()
        assert run_summary.main([str(tmp_path), "--oneline", "--fan-channel", "10"]) == 0
        assert "duct_pwm=255-255" in capsys.readouterr().out.split()


def test_text_reports_channels_without_machine_labels() -> None:
    text = render_text(_summary())
    assert "channel 6:" in text
    assert "channel 10:" in text
    assert "channel 8:" in text
    assert "channel 3:" in text


def _duct_and_exhaust_summary() -> RunSummary:
    lines = [
        {"monotonic": 1.0, "fan_ducts": [{"channel": 1, "pwm": 140, "tach_rpm": 9000}] + _EXHAUST},
        {"monotonic": 2.0, "fan_ducts": [{"channel": 1, "pwm": 140, "tach_rpm": 9000}] + _EXHAUST},
    ]
    pwm, rpm, target = _fan_ranges(lines, [(0.0, 5.0)])
    return replace(_summary(), fan_pwm=pwm, fan_rpm=rpm, watchdog_target_pwm=target)


def test_text_default_lists_the_duct_channel_and_labels_the_exhaust_apart() -> None:
    text = render_text(_duct_and_exhaust_summary())
    duct_part, _, other_part = text.partition("not duct")
    assert "channel 1: pwm 140-140" in duct_part
    assert "channel 10" not in duct_part
    assert "channel 10: pwm 255-255" in other_part
    assert "channel 1:" not in other_part


def test_text_fan_channel_selects_the_duct_channels() -> None:
    text = render_text(_duct_and_exhaust_summary(), fan_channels=[10])
    duct_part, _, other_part = text.partition("not duct")
    assert "channel 10: pwm 255-255" in duct_part
    assert "channel 1: pwm 140-140" in other_part


def test_cli_text_honors_fan_channel(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(run_summary, "summarize_run", return_value=_duct_and_exhaust_summary()):
        assert run_summary.main([str(tmp_path)]) == 0
        default_duct = capsys.readouterr().out.partition("not duct")[0]
        assert "channel 1: pwm 140-140" in default_duct
        assert "channel 10" not in default_duct
        assert run_summary.main([str(tmp_path), "--fan-channel", "10"]) == 0
        assert "channel 10: pwm 255-255" in capsys.readouterr().out.partition("not duct")[0]


def test_resolve_duct_channels_prefers_the_explicit_list_over_the_declared_one() -> None:
    assert resolve_duct_channels([4, 5]) == (4, 5)
    assert resolve_duct_channels(None, {DUCT_CHANNELS_ENV: "2, 3"}) == (2, 3)


@pytest.mark.parametrize("declared", [None, "", "  ", ","])
def test_resolve_duct_channels_refuses_when_nothing_names_a_channel(declared: str | None) -> None:
    environ = {} if declared is None else {DUCT_CHANNELS_ENV: declared}
    with pytest.raises(RunSummaryError, match="--fan-channel"):
        resolve_duct_channels(None, environ)


def test_resolve_duct_channels_refuses_a_bad_declaration() -> None:
    with pytest.raises(RunSummaryError, match="comma list"):
        resolve_duct_channels(None, {DUCT_CHANNELS_ENV: "one"})


def test_the_library_holds_no_default_duct_channel() -> None:
    assert not hasattr(run_summary, "DUCT_CHANNELS")


def test_renderers_refuse_without_declared_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DUCT_CHANNELS_ENV)
    with pytest.raises(RunSummaryError, match=DUCT_CHANNELS_ENV):
        render_oneline(_summary())
    with pytest.raises(RunSummaryError, match=DUCT_CHANNELS_ENV):
        render_text(_summary())
    assert "duct_pwm=100-180" in render_oneline(_summary(), fan_channels=[1, 2, 3, 4, 5]).split()


def test_cli_refuses_without_channels_but_json_needs_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DUCT_CHANNELS_ENV)
    with patch.object(run_summary, "summarize_run", return_value=_duct_and_exhaust_summary()):
        assert run_summary.main([str(tmp_path), "--oneline"]) == 1
        assert DUCT_CHANNELS_ENV in capsys.readouterr().err
        assert run_summary.main([str(tmp_path), "--json"]) == 0
        assert '"fan_pwm"' in capsys.readouterr().out
        assert run_summary.main([str(tmp_path), "--oneline", "--fan-channel", "10"]) == 0
