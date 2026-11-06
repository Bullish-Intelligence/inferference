"""Thermal and fan safety monitor for an experiment run.

This module evaluates :class:`inferference.telemetry.TelemetrySample` objects
against three limits: the machine's hard thermal ceiling, an arm's own bound,
and a GTT spill ceiling. It also watches the ARCTIC fan watchdog's state. It
holds no lease and writes no actuator: it only reads sysfs (through
:func:`sweep_amdgpu_temperatures`, for the machine-limit default) and
evaluates samples the caller already collected.

`guard_default_limit_c`, `_hottest_sensor` and the fan-grace logic in
`SafetyMonitor.check` port `ci/runner/gpu-thermal-guard.sh`'s `_read_temps`,
`_hottest`, and `_check_fan_state_watch`, so the runner's 0.25 s safety net
agrees with the independent 5 s watchdog on what "hottest" and "indeterminate"
mean.

"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..telemetry import CardSample, TelemetrySample

GUARD_DEFAULT_PATTERN = r'^MAX_C="\$\{GPU_THERMAL_MAX_C:-(\d+)\}"'

Verdict = Literal["ok", "arm_bound", "machine_limit", "fan_fault", "gtt_spill"]

_TEMP_VALUE = re.compile(r"^[0-9]+$")


class SafetyError(RuntimeError):
    """A sensor, a limit source, or a monitor input could not be trusted."""


def guard_default_limit_c(guard_script: Path) -> int:
    """Read the guard script's own compiled-in `MAX_C` default.

    Parses the line `MAX_C="${GPU_THERMAL_MAX_C:-<N>}"` out of
    `guard_script`'s text. Refuses with `SafetyError` when the script does not
    carry that exact pattern: a guard script that changed its own default
    shape must not be read by guesswork.
    """
    text = guard_script.read_text()
    match = re.search(GUARD_DEFAULT_PATTERN, text, re.MULTILINE)
    if match is None:
        raise SafetyError(f"{guard_script} does not carry the expected MAX_C default pattern")
    return int(match.group(1))


def effective_machine_limit_c(guard_script: Path, env: Mapping[str, str] | None = None) -> int:
    """The machine's thermal ceiling: `GPU_THERMAL_MAX_C` from `env`, else the guard's default.

    This never sets `GPU_THERMAL_MAX_C`: it only reads it. `env` defaults to
    `os.environ` when not given.
    """
    if env is None:
        env = os.environ
    raw = env.get("GPU_THERMAL_MAX_C")
    if raw is None:
        return guard_default_limit_c(guard_script)
    try:
        return int(raw)
    except ValueError as exc:
        raise SafetyError(f"GPU_THERMAL_MAX_C={raw!r} is not an integer") from exc


def sweep_amdgpu_temperatures(drm_root: Path = Path("/sys/class/drm")) -> dict[str, float]:
    """Every amdgpu hwmon temperature sensor under `drm_root`, in degrees C.

    A port of `gpu-thermal-guard.sh`'s `_read_temps`. Each key is
    `"card<N>/<label>"`: `<label>` is the sensor's own `tempN_label` file when
    readable, else the raw `tempN_input` filename, exactly like the shell
    function's `basename` fallback. An unreadable sensor value or a value that
    is not a plain non-negative integer raises `SafetyError`. Finding no
    sensor at all raises `SafetyError`: no sensor is not "cool enough", it is
    no thermal protection at all.
    """
    readings: dict[str, float] = {}
    for card_dir in sorted(drm_root.glob("card[0-9]*")):
        device_dir = card_dir / "device"
        if not device_dir.is_dir():
            continue
        for hwmon_dir in sorted((device_dir / "hwmon").glob("hwmon*")):
            name_file = hwmon_dir / "name"
            try:
                name = name_file.read_text().strip()
            except OSError:
                continue
            if name != "amdgpu":
                continue
            for temp_file in sorted(hwmon_dir.glob("temp*_input")):
                try:
                    raw_text = temp_file.read_text().strip()
                except OSError as exc:
                    raise SafetyError(f"could not read temperature sensor {temp_file}: {exc}") from exc
                if _TEMP_VALUE.match(raw_text) is None:
                    raise SafetyError(f"invalid temperature sensor value at {temp_file}: {raw_text!r}")
                label_file = temp_file.with_name(temp_file.name.removesuffix("_input") + "_label")
                try:
                    label = label_file.read_text().strip()
                except OSError:
                    label = temp_file.name
                readings[f"{card_dir.name}/{label}"] = int(raw_text) / 1000
    if not readings:
        raise SafetyError(f"no amdgpu temperature sensor found under {drm_root}; refusing to evaluate safety unguarded")
    return readings


def _hottest_sensor(card: CardSample) -> tuple[str, float]:
    """The hottest of a card's own edge, junction and memory sensors."""
    readings = {
        "edge": card.temperature_edge_c,
        "junction": card.temperature_hotspot_c,
        "mem": card.temperature_mem_c,
    }
    sensor = max(readings, key=lambda name: readings[name])
    return sensor, readings[sensor]


def _hottest_overall(sample: TelemetrySample) -> tuple[int, str, float] | None:
    """The hottest sensor over every card in `sample`, or `None` with no cards."""
    best: tuple[int, str, float] | None = None
    for card in sample.cards:
        sensor, value = _hottest_sensor(card)
        if best is None or value > best[2]:
            best = (card.card, sensor, value)
    return best


def _find_card(sample: TelemetrySample, card: int) -> CardSample | None:
    for candidate in sample.cards:
        if candidate.card == card:
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class ArmBound:
    """One arm's own thermal ceiling and GTT baseline, on its own card."""

    card: int
    bound_c: float
    gtt_baseline_mib: float | None


@dataclass(frozen=True, slots=True)
class SafetyEvent:
    """One safety verdict `SafetyMonitor.check` raised on one telemetry tick."""

    verdict: Verdict
    card: int | None
    observed: str
    mono: float


class SafetyMonitor:
    """Evaluates one telemetry tick against the machine limit, arm bounds, the fan, and GTT.

    This monitor never writes an actuator and never touches sysfs itself; it
    only evaluates the `TelemetrySample` objects its caller passes to
    `check`. `arms` may name several cards: an `arm_bound` or `gtt_spill`
    event names the one arm it fired for, and every other arm keeps running.
    A `machine_limit` or `fan_fault` event is global and ends the whole
    schedule; the runner, not this class, decides what a verdict stops.
    """

    def __init__(
        self,
        *,
        machine_limit_c: float,
        arms: Sequence[ArmBound],
        fan_grace_s: float = 10.0,
        unit_active: Callable[[], bool],
        unit_check_interval_s: float = 5.0,
        requested_fan_profile: str | None = None,
    ) -> None:
        self._machine_limit_c = machine_limit_c
        self._arms = tuple(arms)
        self._fan_grace_s = fan_grace_s
        self._unit_active = unit_active
        self._unit_check_interval_s = unit_check_interval_s
        self._requested_fan_profile = requested_fan_profile
        self._fan_indeterminate_since: float | None = None
        self._last_unit_check_mono: float | None = None
        self._last_unit_active: bool | None = None

    def _card_sample(self, sample: TelemetrySample, card: int) -> CardSample:
        found = _find_card(sample, card)
        if found is None:
            raise SafetyError(f"telemetry sample carries no card {card}, needed by a declared arm bound")
        return found

    def set_gtt_baseline(self, card: int, gtt_baseline_mib: float) -> None:
        """Set one arm's measured pre-load GTT baseline.

        The server records this value after backend initialization and
        before it loads the model. The runner refuses a missing baseline.
        """
        if not any(arm.card == card for arm in self._arms):
            raise SafetyError(f"card {card} has no arm bound")
        self._arms = tuple(
            ArmBound(card=arm.card, bound_c=arm.bound_c, gtt_baseline_mib=gtt_baseline_mib) if arm.card == card else arm
            for arm in self._arms
        )

    def _check_machine_limit(self, sample: TelemetrySample) -> SafetyEvent | None:
        hottest = _hottest_overall(sample)
        if hottest is None:
            return None
        card, sensor, value = hottest
        if value >= self._machine_limit_c:
            return SafetyEvent(
                verdict="machine_limit",
                card=card,
                observed=f"card{card}/{sensor} at {value} C >= machine limit {self._machine_limit_c} C",
                mono=sample.monotonic,
            )
        return None

    def _check_arm_bounds(self, sample: TelemetrySample) -> tuple[SafetyEvent, ...]:
        events: list[SafetyEvent] = []
        for arm in self._arms:
            card_sample = self._card_sample(sample, arm.card)
            sensor, value = _hottest_sensor(card_sample)
            if value >= arm.bound_c:
                events.append(
                    SafetyEvent(
                        verdict="arm_bound",
                        card=arm.card,
                        observed=f"card{arm.card}/{sensor} at {value} C >= arm bound {arm.bound_c} C",
                        mono=sample.monotonic,
                    )
                )
        return tuple(events)

    def _check_gtt_spill(self, sample: TelemetrySample) -> tuple[SafetyEvent, ...]:
        events: list[SafetyEvent] = []
        for arm in self._arms:
            card_sample = self._card_sample(sample, arm.card)
            if arm.gtt_baseline_mib is None:
                continue
            delta = card_sample.gtt_used_mib - arm.gtt_baseline_mib
            if delta > 0:
                events.append(
                    SafetyEvent(
                        verdict="gtt_spill",
                        card=arm.card,
                        observed=(
                            f"card{arm.card} GTT +{delta} MiB over its {arm.gtt_baseline_mib} MiB baseline, limit 0 MiB"
                        ),
                        mono=sample.monotonic,
                    )
                )
        return tuple(events)

    def _check_fan(self, sample: TelemetrySample) -> SafetyEvent | None:
        """Fan fault, with the guard's grace rule for an indeterminate reading.

        A watchdog reporting a state other than `OK` is a definite fault and
        fires at once. A missing watchdog (`sample.watchdog is None`), or an
        `OK` watchdog whose owning systemd unit `unit_active()` reports
        inactive, is indeterminate: it is tolerated for `fan_grace_s`, timed
        on `sample.monotonic`, before it also faults
        (`gpu-thermal-guard.sh`'s `_check_fan_state_watch`). `unit_active()`
        itself is called at most once per `unit_check_interval_s`; its cached
        result carries forward between calls.
        """
        watchdog = sample.watchdog

        if watchdog is not None and watchdog.state != "OK":
            self._fan_indeterminate_since = None
            return SafetyEvent(
                verdict="fan_fault",
                card=None,
                observed=f"fan watchdog state is {watchdog.state}",
                mono=sample.monotonic,
            )

        determinate_ok = False
        if watchdog is not None:
            due = (
                self._last_unit_check_mono is None
                or sample.monotonic - self._last_unit_check_mono >= self._unit_check_interval_s
            )
            if due:
                self._last_unit_active = self._unit_active()
                self._last_unit_check_mono = sample.monotonic
            determinate_ok = bool(self._last_unit_active)

        if determinate_ok:
            self._fan_indeterminate_since = None
            return None

        if self._fan_indeterminate_since is None:
            self._fan_indeterminate_since = sample.monotonic
            return None
        if sample.monotonic - self._fan_indeterminate_since < self._fan_grace_s:
            return None
        return SafetyEvent(
            verdict="fan_fault",
            card=None,
            observed=f"fan watchdog stayed indeterminate for {self._fan_grace_s}s",
            mono=sample.monotonic,
        )

    def check(self, sample: TelemetrySample) -> tuple[SafetyEvent, ...]:
        """Evaluate one tick. Returns every event this tick raised, in order.

        Order: machine limit, then each arm's bound, then the fan, then each
        arm's GTT spill. A caller that only cares whether anything fired can
        test `bool(monitor.check(sample))`.
        """
        events: list[SafetyEvent] = []

        machine_event = self._check_machine_limit(sample)
        if machine_event is not None:
            events.append(machine_event)

        events.extend(self._check_arm_bounds(sample))

        fan_event = self._check_fan(sample)
        if fan_event is not None:
            events.append(fan_event)

        events.extend(self._check_gtt_spill(sample))

        return tuple(events)

    def profile_drift(self, sample: TelemetrySample) -> str | None:
        """A fan-profile mismatch, if any: a condition change, not a safety event.

        Returns `None` when no fan profile was requested, the watchdog is
        unreadable or not `OK` (covered by `check`'s fan fault instead), or
        its reported profile matches what this run requested. Otherwise
        returns a message naming the drift; the runner maps a non-`None`
        result to an `invalid` outcome, never an abort.
        """
        if self._requested_fan_profile is None:
            return None
        watchdog = sample.watchdog
        if watchdog is None or watchdog.state != "OK":
            return None
        if watchdog.profile == self._requested_fan_profile:
            return None
        return f"fan profile drifted to {watchdog.profile!r}, requested {self._requested_fan_profile!r}"
