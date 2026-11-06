"""Tests for the experiment safety monitor, against fabricated telemetry and sysfs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from inferference.experiment.safety import (
    GUARD_DEFAULT_PATTERN,
    ArmBound,
    SafetyError,
    SafetyMonitor,
    effective_machine_limit_c,
    guard_default_limit_c,
    sweep_amdgpu_temperatures,
)
from inferference.telemetry import CardSample, TelemetrySample, WatchdogSample

ROOT = Path(__file__).resolve().parents[1]
REAL_GUARD_SCRIPT = ROOT / "ci/runner/gpu-thermal-guard.sh"


def _card(
    card: int,
    *,
    junction_c: float = 50.0,
    edge_c: float = 45.0,
    mem_c: float = 48.0,
    gtt_used_mib: float = 0.0,
) -> CardSample:
    """A `CardSample` with every field filled, varying only what a test cares about."""
    return CardSample(
        card=card,
        pci=f"0000:{card:02x}:00.0",
        temperature_edge_c=edge_c,
        temperature_hotspot_c=junction_c,
        temperature_mem_c=mem_c,
        power_w=100.0,
        power_cap_w=150.0,
        vddgfx_mv=800,
        sclk_mhz=1500.0,
        mclk_mhz=945.0,
        gpu_busy_percent=50,
        mem_busy_percent=20,
        vram_used_mib=4096.0,
        vram_total_mib=30720.0,
        gtt_used_mib=gtt_used_mib,
        gtt_total_mib=60000.0,
        pp_dpm_sclk=(),
        pp_dpm_mclk=(),
        power_dpm_force_performance_level="manual",
        pp_power_profile_mode="POWER_SAVING",
        throttle_status_raw=0,
        throttled=False,
    )


def _sample(
    mono: float,
    cards: tuple[CardSample, ...],
    *,
    watchdog: WatchdogSample | None = None,
) -> TelemetrySample:
    """A `TelemetrySample` with a fixed monotonic clock, for a monitor tick."""
    return TelemetrySample(
        utc="2026-10-07T00:00:00+00:00",
        monotonic=mono,
        cards=cards,
        fan_ducts=(),
        watchdog=watchdog,
    )


def _ok_watchdog() -> WatchdogSample:
    return WatchdogSample(state="OK", target_pwm=255, max_gpu_junction_m_c=90000)


# ---- machine limit and arm bound, on a simulated two-card ramp --------------


def test_arm_bound_fires_for_the_ramping_card_only_while_the_other_stays_ok() -> None:
    arms = (
        ArmBound(card=0, bound_c=85.0, gtt_baseline_mib=0.0),
        ArmBound(card=1, bound_c=85.0, gtt_baseline_mib=0.0),
    )
    monitor = SafetyMonitor(
        machine_limit_c=95.0,
        arms=arms,
        unit_active=lambda: True,
    )

    # Ramp card 0 from 60 C to 90 C in 5 C steps; card 1 stays flat at 50 C.
    fired_for_card0 = False
    for step, junction in enumerate((60.0, 70.0, 80.0, 86.0, 90.0)):
        sample = _sample(
            mono=float(step),
            cards=(_card(0, junction_c=junction), _card(1, junction_c=50.0)),
            watchdog=_ok_watchdog(),
        )
        events = monitor.check(sample)
        arm_events = [e for e in events if e.verdict == "arm_bound"]
        assert all(e.card == 0 for e in arm_events)
        if junction >= 85.0:
            assert any(e.card == 0 for e in arm_events)
            fired_for_card0 = True
        else:
            assert arm_events == []
    assert fired_for_card0

    # Card 1 never crossed its own bound, in any tick above.
    final_sample = _sample(
        mono=10.0,
        cards=(_card(0, junction_c=90.0), _card(1, junction_c=50.0)),
        watchdog=_ok_watchdog(),
    )
    events = monitor.check(final_sample)
    assert not any(e.card == 1 and e.verdict == "arm_bound" for e in events)


def test_machine_limit_fires_when_any_card_reaches_it() -> None:
    arms = (ArmBound(card=0, bound_c=85.0, gtt_baseline_mib=0.0),)
    monitor = SafetyMonitor(
        machine_limit_c=95.0,
        arms=arms,
        unit_active=lambda: True,
    )

    sample = _sample(
        mono=0.0,
        cards=(_card(0, junction_c=96.0), _card(1, junction_c=50.0)),
        watchdog=_ok_watchdog(),
    )
    events = monitor.check(sample)

    assert any(e.verdict == "machine_limit" for e in events)


def test_machine_limit_comparison_is_at_or_above() -> None:
    arms = (ArmBound(card=0, bound_c=85.0, gtt_baseline_mib=0.0),)
    monitor = SafetyMonitor(machine_limit_c=95.0, arms=arms, unit_active=lambda: True)

    below = _sample(mono=0.0, cards=(_card(0, junction_c=94.9),), watchdog=_ok_watchdog())
    at_limit = _sample(mono=1.0, cards=(_card(0, junction_c=95.0),), watchdog=_ok_watchdog())

    assert not any(e.verdict == "machine_limit" for e in monitor.check(below))
    assert any(e.verdict == "machine_limit" for e in monitor.check(at_limit))


def test_machine_limit_precedes_equal_arm_bound_on_the_same_tick() -> None:
    monitor = SafetyMonitor(
        machine_limit_c=80.0,
        arms=(ArmBound(card=0, bound_c=80.0, gtt_baseline_mib=0.0),),
        unit_active=lambda: True,
    )
    sample = _sample(mono=0.0, cards=(_card(0, junction_c=80.0),), watchdog=_ok_watchdog())

    events = monitor.check(sample)

    assert [event.verdict for event in events[:2]] == ["machine_limit", "arm_bound"]


# ---- fan watchdog -------------------------------------------------------------


def test_degraded_fan_state_faults_at_once() -> None:
    monitor = SafetyMonitor(machine_limit_c=95.0, arms=(), unit_active=lambda: True)
    sample = _sample(
        mono=0.0,
        cards=(_card(0),),
        watchdog=WatchdogSample(state="DEGRADED", target_pwm=255, max_gpu_junction_m_c=90000),
    )

    events = monitor.check(sample)

    assert [e.verdict for e in events] == ["fan_fault"]
    assert events[0].card is None


def test_missing_watchdog_survives_the_grace_then_faults() -> None:
    monitor = SafetyMonitor(
        machine_limit_c=95.0,
        arms=(),
        fan_grace_s=5.0,
        unit_active=lambda: True,
    )

    tick0 = monitor.check(_sample(mono=0.0, cards=(_card(0),), watchdog=None))
    tick1 = monitor.check(_sample(mono=3.0, cards=(_card(0),), watchdog=None))
    tick2 = monitor.check(_sample(mono=6.0, cards=(_card(0),), watchdog=None))

    assert not any(e.verdict == "fan_fault" for e in tick0)
    assert not any(e.verdict == "fan_fault" for e in tick1)
    assert any(e.verdict == "fan_fault" for e in tick2)


def test_watchdog_returning_resets_the_grace_clock() -> None:
    monitor = SafetyMonitor(
        machine_limit_c=95.0,
        arms=(),
        fan_grace_s=5.0,
        unit_active=lambda: True,
    )

    monitor.check(_sample(mono=0.0, cards=(_card(0),), watchdog=None))
    monitor.check(_sample(mono=3.0, cards=(_card(0),), watchdog=_ok_watchdog()))
    # The grace clock reset at mono=3, so mono=7 (4s later) must not fault yet.
    tick = monitor.check(_sample(mono=7.0, cards=(_card(0),), watchdog=None))

    assert not any(e.verdict == "fan_fault" for e in tick)


def test_unit_inactive_is_graced_like_a_missing_watchdog() -> None:
    calls = 0

    def unit_active() -> bool:
        nonlocal calls
        calls += 1
        return False

    monitor = SafetyMonitor(
        machine_limit_c=95.0,
        arms=(),
        fan_grace_s=5.0,
        unit_active=unit_active,
        unit_check_interval_s=1000.0,
    )

    tick0 = monitor.check(_sample(mono=0.0, cards=(_card(0),), watchdog=_ok_watchdog()))
    tick1 = monitor.check(_sample(mono=6.0, cards=(_card(0),), watchdog=_ok_watchdog()))

    assert not any(e.verdict == "fan_fault" for e in tick0)
    assert any(e.verdict == "fan_fault" for e in tick1)
    # Rate-limited: one call establishes the baseline, the second tick is
    # still inside the (very long) check interval, so it reuses the cached
    # result instead of calling again.
    assert calls == 1


# ---- GTT spill -----------------------------------------------------------------


def test_gtt_spill_fires_on_any_increase() -> None:
    arms = (ArmBound(card=0, bound_c=200.0, gtt_baseline_mib=1000.0),)
    monitor = SafetyMonitor(
        machine_limit_c=200.0,
        arms=arms,
        unit_active=lambda: True,
    )

    under = _sample(mono=0.0, cards=(_card(0, gtt_used_mib=1000.0),), watchdog=_ok_watchdog())
    over = _sample(mono=1.0, cards=(_card(0, gtt_used_mib=1000.000001),), watchdog=_ok_watchdog())

    assert not any(e.verdict == "gtt_spill" for e in monitor.check(under))
    over_events = [e for e in monitor.check(over) if e.verdict == "gtt_spill"]
    assert len(over_events) == 1
    assert over_events[0].card == 0


# ---- fan profile drift ---------------------------------------------------------


def test_profile_drift_reports_a_mismatch_only_when_ok_and_requested() -> None:
    monitor = SafetyMonitor(
        machine_limit_c=200.0,
        arms=(),
        unit_active=lambda: True,
        requested_fan_profile="flat-max",
    )

    matching = WatchdogSample(state="OK", target_pwm=255, max_gpu_junction_m_c=90000, profile="flat-max")
    drifted = WatchdogSample(state="OK", target_pwm=255, max_gpu_junction_m_c=90000, profile="safe-high")
    faulted = WatchdogSample(state="DEGRADED", target_pwm=255, max_gpu_junction_m_c=90000, profile="safe-high")

    assert monitor.profile_drift(_sample(mono=0.0, cards=(_card(0),), watchdog=matching)) is None
    assert monitor.profile_drift(_sample(mono=1.0, cards=(_card(0),), watchdog=drifted)) is not None
    # A fan fault is check()'s business, not drift's.
    assert monitor.profile_drift(_sample(mono=2.0, cards=(_card(0),), watchdog=faulted)) is None
    assert monitor.profile_drift(_sample(mono=3.0, cards=(_card(0),), watchdog=None)) is None


def test_profile_drift_is_none_when_no_profile_was_requested() -> None:
    monitor = SafetyMonitor(machine_limit_c=200.0, arms=(), unit_active=lambda: True)
    sample = _sample(
        mono=0.0,
        cards=(_card(0),),
        watchdog=WatchdogSample(state="OK", target_pwm=255, max_gpu_junction_m_c=90000, profile="anything"),
    )

    assert monitor.profile_drift(sample) is None


# ---- sweep_amdgpu_temperatures, against a fabricated sysfs tree ----------------


def _write_amdgpu_sensor(drm_root: Path, card: int, sensor: int, value: str, label: str | None) -> Path:
    hwmon = drm_root / f"card{card}" / "device" / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True, exist_ok=True)
    (hwmon / "name").write_text("amdgpu\n")
    temp_file = hwmon / f"temp{sensor}_input"
    temp_file.write_text(f"{value}\n")
    if label is not None:
        (hwmon / f"temp{sensor}_label").write_text(f"{label}\n")
    return temp_file


def test_sweep_reads_labelled_sensors_across_cards(tmp_path: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    _write_amdgpu_sensor(drm_root, 0, 1, "45000", "edge")
    _write_amdgpu_sensor(drm_root, 0, 2, "52000", "junction")
    _write_amdgpu_sensor(drm_root, 1, 1, "40000", "edge")

    readings = sweep_amdgpu_temperatures(drm_root)

    assert readings == {
        "card0/edge": 45.0,
        "card0/junction": 52.0,
        "card1/edge": 40.0,
    }


def test_sweep_falls_back_to_the_filename_with_no_label(tmp_path: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    _write_amdgpu_sensor(drm_root, 0, 2, "52000", label=None)

    readings = sweep_amdgpu_temperatures(drm_root)

    assert readings == {"card0/temp2_input": 52.0}


def test_sweep_raises_on_an_unreadable_sensor_value(tmp_path: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    _write_amdgpu_sensor(drm_root, 0, 1, "ERROR: 2", "edge")

    with pytest.raises(SafetyError, match="invalid temperature sensor value"):
        sweep_amdgpu_temperatures(drm_root)


def test_sweep_raises_on_an_unreadable_sensor_file(tmp_path: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    sensor = _write_amdgpu_sensor(drm_root, 0, 1, "45000", "edge")
    sensor.unlink()
    sensor.mkdir()

    with pytest.raises(SafetyError, match="could not read temperature sensor"):
        sweep_amdgpu_temperatures(drm_root)


def test_sweep_raises_when_no_sensor_exists_at_all(tmp_path: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    drm_root.mkdir(parents=True)

    with pytest.raises(SafetyError, match="no amdgpu temperature sensor found"):
        sweep_amdgpu_temperatures(drm_root)


def test_sweep_raises_when_a_card_has_no_amdgpu_hwmon_chip(tmp_path: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    hwmon = drm_root / "card0" / "device" / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "name").write_text("not-amdgpu\n")
    (hwmon / "temp1_input").write_text("45000\n")

    with pytest.raises(SafetyError, match="no amdgpu temperature sensor found"):
        sweep_amdgpu_temperatures(drm_root)


# ---- effective_machine_limit_c, against the real guard script ------------------


def test_guard_default_limit_reads_the_real_script() -> None:
    assert guard_default_limit_c(REAL_GUARD_SCRIPT) == 80


def test_effective_machine_limit_falls_back_to_the_real_default_with_no_env_override() -> None:
    assert effective_machine_limit_c(REAL_GUARD_SCRIPT, env={}) == 80


def test_effective_machine_limit_honours_an_env_override() -> None:
    assert effective_machine_limit_c(REAL_GUARD_SCRIPT, env={"GPU_THERMAL_MAX_C": "95"}) == 95


def test_guard_default_limit_raises_without_the_expected_pattern(tmp_path: Path) -> None:
    fake_guard = tmp_path / "gpu-thermal-guard.sh"
    fake_guard.write_text("#!/usr/bin/env bash\necho hello\n")

    with pytest.raises(SafetyError, match="MAX_C"):
        guard_default_limit_c(fake_guard)

    with pytest.raises(SafetyError, match="MAX_C"):
        effective_machine_limit_c(fake_guard, env={})


def test_guard_default_pattern_matches_the_real_script_line() -> None:
    text = REAL_GUARD_SCRIPT.read_text()

    assert re.search(GUARD_DEFAULT_PATTERN, text, re.MULTILINE) is not None
