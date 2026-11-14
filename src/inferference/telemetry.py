"""A 0.25-second per-tick telemetry sampler for every amdgpu card on the host.

Interpreting a thermal arm needs a time series: a self-throttling card and a
card that is simply cooling-impaired look alike from the outside unless the
card's own throttler state is on the record. `ci/runner/gpu-thermal-guard.sh`
polls every 5 s and logs only a line, which is a failsafe, not a dataset. A
sampler that reads no `in0_input`, no `gpu_busy_percent`, no `power1_cap` and
no throttler state is not a dataset either. This module reads all of them.

Each :class:`TelemetrySample` is one tick: every amdgpu sensor on the cards,
the shared fan duct channels, the fan watchdog's own state, and the case fans,
carried together so a reader never has to align records by timestamp across
files.
Every tick carries both a UTC and a monotonic clock.

**Throttler status.** `gpu_metrics`'s `throttle_status` field (`uint32`, byte
offset 68 in the `gpu_metrics_v1_3` header used by this ASIC — verified against
the kernel's `struct gpu_metrics_v1_3` in `kgd_pp_interface.h` and cross-checked
live: it reads 0 on an idle card, matching `amd-smi metric --json`'s
`power.throttle_status: UNTHROTTLED`) is read raw, as a 32-bit bitmask. This
module reports only whether it is nonzero (`throttled`); it does not decode
individual bits to names like `THROTTLER_TEMP_EDGE` or `THROTTLER_PPT0`,
because the bit-to-reason mapping for this ASIC was not verified against
primary source in this project. A nonzero bitmask already answers the thermal
question (did some throttler fire?). It is recorded verbatim for a later
session to decode properly rather than guess now. `amd-smi` itself was
rejected as the live per-sample source: one call across both cards measured
0.56 s, more than double the 0.25 s sample budget.

This module performs read-only sysfs I/O. It writes no control file and holds
no lease.
"""

from __future__ import annotations

import json
import re
import struct
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SAMPLE_INTERVAL_S = 0.25
TELEMETRY_SCHEMA_VERSION = 1

DEFAULT_DRM_ROOT = Path("/sys/class/drm")
DEFAULT_HWMON_ROOT = Path("/sys/class/hwmon")
DEFAULT_WATCHDOG_STATUS_FILE = Path("/run/arctic-fan/status")
DEFAULT_FAN_CHIP_NAME = "arctic_fan"
DEFAULT_CASE_FAN_CHIP_NAME = "dell_smm"
# Kept for callers that still want the historical fixed pair. The reader's own
# default is discovery (see `_discover_fan_channels`), because a fixed pair
# silently misses channels that exist on the chip.
DEFAULT_FAN_CHANNELS: tuple[int, ...] = (2, 3)

# The watchdog rewrites its status file with a shell `>` redirect. The shell
# truncates the file first and writes after. A reader can see an empty file
# for a moment. A reader retries a short, bounded number of times. The total
# wait stays far below `SAMPLE_INTERVAL_S`.
WATCHDOG_READ_ATTEMPTS = 5
WATCHDOG_READ_RETRY_DELAY_S = 0.02

GPU_METRICS_THROTTLE_STATUS_OFFSET = 68

_PP_DPM_LINE = re.compile(r"^\s*(\d+):\s*(\d+)Mhz\s*(\*)?\s*$")
_PP_PROFILE_LINE = re.compile(r"^\s*(\d+)\s+([A-Z0-9_]+)(\*)?\s*:")
_WATCHDOG_LINE = re.compile(r"state=(\S+)\s+target_pwm=(\d+)\s+max_gpu_junction_mC=(\d+)")
_WATCHDOG_PROFILE = re.compile(r"profile=(\S+)")
_PWM_FILENAME = re.compile(r"^pwm(\d+)$")
_FAN_INPUT_FILENAME = re.compile(r"^fan(\d+)_input$")


class TelemetryError(RuntimeError):
    """A sensor could not be read, or its format did not parse."""


@dataclass(frozen=True, slots=True)
class DpmLevel:
    """One row of a `pp_dpm_sclk` / `pp_dpm_mclk` level table."""

    index: int
    mhz: int
    active: bool


@dataclass(frozen=True, slots=True)
class CardSample:
    """Every amdgpu sensor this module reads for one card, for one tick."""

    card: int
    pci: str
    temperature_edge_c: float
    temperature_hotspot_c: float
    temperature_mem_c: float
    power_w: float
    power_cap_w: float
    vddgfx_mv: int
    sclk_mhz: float
    mclk_mhz: float
    gpu_busy_percent: int
    mem_busy_percent: int
    vram_used_mib: float
    vram_total_mib: float
    gtt_used_mib: float
    gtt_total_mib: float
    pp_dpm_sclk: tuple[DpmLevel, ...]
    pp_dpm_mclk: tuple[DpmLevel, ...]
    power_dpm_force_performance_level: str
    pp_power_profile_mode: str
    throttle_status_raw: int
    throttled: bool


@dataclass(frozen=True, slots=True)
class FanDuctSample:
    """One ARCTIC fan controller channel: its commanded PWM and tach reading."""

    channel: int
    pwm: int
    tach_rpm: int


@dataclass(frozen=True, slots=True)
class CaseFanSample:
    """One Dell `dell_smm` case fan: its tach reading and, if offered, its control state.

    `fan` is the channel number N of `fanN_input`. Fan 1 blows air over the
    outside of the GPUs. `rpm` is the only required reading. The other fields
    are `None` when the driver does not offer the file or refuses the read.
    On the rig the driver refuses `fanN_label` (`EINVAL`) and `pwmN`
    (`ENODATA`), and it reports `pwmN_enable`.
    """

    fan: int
    rpm: int
    label: str | None = None
    pwm: int | None = None
    pwm_enable: int | None = None


@dataclass(frozen=True, slots=True)
class WatchdogSample:
    """The ARCTIC fan watchdog's own reported state, from its status file.

    `profile` is `None` until the fan watchdog writes a `profile=<name>` key
    onto the status line. Its absence is read-only information here, not a
    read failure.
    """

    state: str
    target_pwm: int
    max_gpu_junction_m_c: int
    profile: str | None = None


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """One tick: both clocks, both cards, every fan duct, the watchdog, and the case fans.

    `schema_version` names the shape of the JSON line this tick renders to,
    so a later reader never has to guess. It comes
    before `case_fans`, so an existing positional construction of this
    dataclass still works.

    `case_fans` is an optional, observation-only field. It is empty when the
    host has no Dell fan chip. Its addition does not change the meaning of
    any other field, so the schema version stays 1. A line from an older run
    has no `case_fans` key. A reader must treat a missing key as empty.
    """

    utc: str
    monotonic: float
    cards: tuple[CardSample, ...]
    fan_ducts: tuple[FanDuctSample, ...]
    watchdog: WatchdogSample | None
    schema_version: int = TELEMETRY_SCHEMA_VERSION
    case_fans: tuple[CaseFanSample, ...] = ()

    def to_json_line(self) -> str:
        """Render this tick as one JSON object, for a `telemetry.jsonl` line."""
        return json.dumps(asdict(self), separators=(",", ":"))


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise TelemetryError(f"could not read {path}: {exc}") from exc


def _read_int(path: Path) -> int:
    return int(_read_text(path).strip())


def _find_amdgpu_hwmon(device_dir: Path) -> Path:
    hwmon_parents = sorted((device_dir / "hwmon").glob("hwmon*"))
    for candidate in hwmon_parents:
        name_file = candidate / "name"
        if name_file.is_file() and name_file.read_text().strip() == "amdgpu":
            return candidate
    raise TelemetryError(f"no amdgpu hwmon chip found under {device_dir / 'hwmon'}")


def _pci_address(device_dir: Path) -> str:
    return device_dir.resolve().name


def _parse_pp_dpm_levels(text: str) -> tuple[DpmLevel, ...]:
    levels = []
    for line in text.splitlines():
        match = _PP_DPM_LINE.match(line)
        if match is None:
            continue
        index, mhz, star = match.groups()
        levels.append(DpmLevel(index=int(index), mhz=int(mhz), active=star is not None))
    if not levels:
        raise TelemetryError(f"no dpm level parsed from {text!r}")
    return tuple(levels)


def _parse_active_power_profile(text: str) -> str:
    for line in text.splitlines():
        match = _PP_PROFILE_LINE.match(line)
        if match is not None and match.group(3) == "*":
            return match.group(2)
    raise TelemetryError(f"no active ('*') row found in pp_power_profile_mode: {text!r}")


def _parse_watchdog_status(text: str) -> WatchdogSample:
    match = _WATCHDOG_LINE.search(text)
    if match is None:
        raise TelemetryError(f"could not parse fan watchdog status: {text!r}")
    state, target_pwm, max_junction = match.groups()
    profile_match = _WATCHDOG_PROFILE.search(text)
    profile = profile_match.group(1) if profile_match is not None else None
    return WatchdogSample(
        state=state,
        target_pwm=int(target_pwm),
        max_gpu_junction_m_c=int(max_junction),
        profile=profile,
    )


def read_watchdog_status(
    path: Path,
    *,
    attempts: int = WATCHDOG_READ_ATTEMPTS,
    retry_delay_s: float = WATCHDOG_READ_RETRY_DELAY_S,
    sleep: Callable[[float], None] = time.sleep,
) -> WatchdogSample:
    """Read and parse the fan watchdog status file, and retry a torn read.

    The watchdog rewrites the file with a shell `>` redirect, which truncates
    before it writes. A reader can see an empty file or half a line for a
    moment. A complete status is one parsable line that ends with a newline.
    This function reads again after `retry_delay_s` when the text is not
    complete. It also reads again after an `OSError`, because a file can be
    briefly absent while something replaces it.

    After `attempts` reads it raises `TelemetryError`, so a file that stays
    empty, malformed or missing still fails closed. `sleep` is injectable so
    a test does not wait.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    failure = ""
    for attempt in range(1, attempts + 1):
        try:
            text = path.read_text()
        except OSError as exc:
            failure = f"could not read {path}: {exc}"
        else:
            try:
                if not text.endswith("\n"):
                    raise TelemetryError(f"could not parse fan watchdog status: {text!r}")
                return _parse_watchdog_status(text)
            except TelemetryError as exc:
                failure = str(exc)
        if attempt < attempts:
            sleep(retry_delay_s)
    raise TelemetryError(f"{failure} (after {attempts} attempts)")


def _discover_fan_channels(hwmon_dir: Path) -> tuple[int, ...]:
    """Every channel N under `hwmon_dir` with both a `pwmN` and a `fanN_input`.

    A channel with `pwmN` but no matching tach file is skipped: it cannot
    report a reading, so including it would make a read fail where discovery
    could simply not have offered it.
    """
    channels = []
    for entry in hwmon_dir.iterdir():
        match = _PWM_FILENAME.match(entry.name)
        if match is None:
            continue
        channel = int(match.group(1))
        if (hwmon_dir / f"fan{channel}_input").is_file():
            channels.append(channel)
    return tuple(sorted(channels))


def _read_throttle_status(device_dir: Path) -> int:
    blob = (device_dir / "gpu_metrics").read_bytes()
    if len(blob) < GPU_METRICS_THROTTLE_STATUS_OFFSET + 4:
        raise TelemetryError(f"gpu_metrics blob is {len(blob)} bytes, too short to hold throttle_status")
    return struct.unpack_from("<I", blob, GPU_METRICS_THROTTLE_STATUS_OFFSET)[0]


class CardReader:
    """Reads every sensor this module tracks for one GPU card."""

    def __init__(self, card: int, drm_root: Path = DEFAULT_DRM_ROOT) -> None:
        self.card = card
        self.device_dir = drm_root / f"card{card}" / "device"
        self.hwmon_dir = _find_amdgpu_hwmon(self.device_dir)

    def read(self) -> CardSample:
        hwmon = self.hwmon_dir
        device = self.device_dir
        throttle_status = _read_throttle_status(device)
        return CardSample(
            card=self.card,
            pci=_pci_address(device),
            temperature_edge_c=_read_int(hwmon / "temp1_input") / 1000,
            temperature_hotspot_c=_read_int(hwmon / "temp2_input") / 1000,
            temperature_mem_c=_read_int(hwmon / "temp3_input") / 1000,
            power_w=_read_int(hwmon / "power1_average") / 1_000_000,
            power_cap_w=_read_int(hwmon / "power1_cap") / 1_000_000,
            vddgfx_mv=_read_int(hwmon / "in0_input"),
            sclk_mhz=_read_int(hwmon / "freq1_input") / 1_000_000,
            mclk_mhz=_read_int(hwmon / "freq2_input") / 1_000_000,
            gpu_busy_percent=_read_int(device / "gpu_busy_percent"),
            mem_busy_percent=_read_int(device / "mem_busy_percent"),
            vram_used_mib=_read_int(device / "mem_info_vram_used") / 1_048_576,
            vram_total_mib=_read_int(device / "mem_info_vram_total") / 1_048_576,
            gtt_used_mib=_read_int(device / "mem_info_gtt_used") / 1_048_576,
            gtt_total_mib=_read_int(device / "mem_info_gtt_total") / 1_048_576,
            pp_dpm_sclk=_parse_pp_dpm_levels(_read_text(device / "pp_dpm_sclk")),
            pp_dpm_mclk=_parse_pp_dpm_levels(_read_text(device / "pp_dpm_mclk")),
            power_dpm_force_performance_level=_read_text(device / "power_dpm_force_performance_level").strip(),
            pp_power_profile_mode=_parse_active_power_profile(_read_text(device / "pp_power_profile_mode")),
            throttle_status_raw=throttle_status,
            throttled=throttle_status != 0,
        )


class FanDuctReader:
    """Reads the ARCTIC fan controller's GPU duct channels and watchdog state.

    `channels=None` (the default) discovers every channel the chip actually
    offers, instead of assuming the historical pair `(2, 3)`: a fixed default
    silently misses channels that exist on the chip. Pass an explicit tuple
    to pin specific channels, for example in a test.
    """

    def __init__(
        self,
        hwmon_root: Path = DEFAULT_HWMON_ROOT,
        chip_name: str = DEFAULT_FAN_CHIP_NAME,
        channels: tuple[int, ...] | None = None,
        watchdog_status_file: Path = DEFAULT_WATCHDOG_STATUS_FILE,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.watchdog_status_file = watchdog_status_file
        self._sleep = sleep
        self.hwmon_dir = self._find_chip(hwmon_root, chip_name)
        self.channels = channels if channels is not None else _discover_fan_channels(self.hwmon_dir)

    @staticmethod
    def _find_chip(hwmon_root: Path, chip_name: str) -> Path:
        for candidate in sorted(hwmon_root.glob("hwmon*")):
            name_file = candidate / "name"
            if name_file.is_file() and name_file.read_text().strip() == chip_name:
                return candidate
        raise TelemetryError(f"no hwmon chip named {chip_name!r} found under {hwmon_root}")

    def read_ducts(self) -> tuple[FanDuctSample, ...]:
        samples = []
        for channel in self.channels:
            pwm = _read_int(self.hwmon_dir / f"pwm{channel}")
            tach = _read_int(self.hwmon_dir / f"fan{channel}_input")
            samples.append(FanDuctSample(channel=channel, pwm=pwm, tach_rpm=tach))
        return tuple(samples)

    def read_watchdog(self) -> WatchdogSample | None:
        if not self.watchdog_status_file.is_file():
            return None
        return read_watchdog_status(self.watchdog_status_file, sleep=self._sleep)


def _try_read_int(path: Path) -> int | None:
    """Read one integer, or return `None` when the file is absent, refused or not a number."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _try_read_label(path: Path) -> str | None:
    """Read one text label, or return `None` when the file is absent, refused or empty."""
    try:
        label = path.read_text().strip()
    except OSError:
        return None
    return label or None


class CaseFanReader:
    """Reads the Dell `dell_smm` case fans: rpm, label, `pwmN` and `pwmN_enable`.

    The case fans are an observation-only field. A missing chip must not stop
    a run, so this reader never raises. If the chip is absent, `read` returns
    an empty tuple. If one file is unreadable, `read` leaves out that field,
    or that fan when its rpm is unreadable.

    The reader finds the chip by its `name` file. It never relies on the
    hwmon index, which can change between boots. It lists the fans once, at
    construction, in the style of `FanDuctReader`.
    """

    def __init__(
        self,
        hwmon_root: Path = DEFAULT_HWMON_ROOT,
        chip_name: str = DEFAULT_CASE_FAN_CHIP_NAME,
    ) -> None:
        self.hwmon_dir = self._find_chip(hwmon_root, chip_name)
        self.fans = () if self.hwmon_dir is None else self._discover_fans(self.hwmon_dir)

    @staticmethod
    def _find_chip(hwmon_root: Path, chip_name: str) -> Path | None:
        try:
            candidates = sorted(hwmon_root.glob("hwmon*"))
        except OSError:
            return None
        for candidate in candidates:
            try:
                if (candidate / "name").read_text().strip() == chip_name:
                    return candidate
            except OSError:
                continue
        return None

    @staticmethod
    def _discover_fans(hwmon_dir: Path) -> tuple[int, ...]:
        try:
            entries = list(hwmon_dir.iterdir())
        except OSError:
            return ()
        fans = []
        for entry in entries:
            match = _FAN_INPUT_FILENAME.match(entry.name)
            if match is not None:
                fans.append(int(match.group(1)))
        return tuple(sorted(fans))

    def read(self) -> tuple[CaseFanSample, ...]:
        """Read every fan once. Never raises."""
        hwmon_dir = self.hwmon_dir
        if hwmon_dir is None:
            return ()
        samples = []
        for fan in self.fans:
            rpm = _try_read_int(hwmon_dir / f"fan{fan}_input")
            if rpm is None:
                continue
            samples.append(
                CaseFanSample(
                    fan=fan,
                    rpm=rpm,
                    label=_try_read_label(hwmon_dir / f"fan{fan}_label"),
                    pwm=_try_read_int(hwmon_dir / f"pwm{fan}"),
                    pwm_enable=_try_read_int(hwmon_dir / f"pwm{fan}_enable"),
                )
            )
        return tuple(samples)


class TelemetrySampler:
    """Samples both cards, the fan controller, and the Dell case fans at a fixed interval."""

    def __init__(
        self,
        cards: tuple[int, ...] = (0, 1),
        drm_root: Path = DEFAULT_DRM_ROOT,
        hwmon_root: Path = DEFAULT_HWMON_ROOT,
        fan_chip_name: str = DEFAULT_FAN_CHIP_NAME,
        fan_channels: tuple[int, ...] | None = None,
        watchdog_status_file: Path = DEFAULT_WATCHDOG_STATUS_FILE,
        sleep: Callable[[float], None] = time.sleep,
        case_fan_chip_name: str = DEFAULT_CASE_FAN_CHIP_NAME,
    ) -> None:
        self._card_readers = tuple(CardReader(card, drm_root=drm_root) for card in cards)
        self._fan_reader = FanDuctReader(
            hwmon_root=hwmon_root,
            chip_name=fan_chip_name,
            channels=fan_channels,
            watchdog_status_file=watchdog_status_file,
            sleep=sleep,
        )
        self._case_fan_reader = CaseFanReader(hwmon_root=hwmon_root, chip_name=case_fan_chip_name)

    def sample(self) -> TelemetrySample:
        """Read one tick: both clocks, every card, every fan duct, the watchdog, the case fans."""
        return TelemetrySample(
            utc=datetime.now(UTC).isoformat(),
            monotonic=time.monotonic(),
            cards=tuple(reader.read() for reader in self._card_readers),
            fan_ducts=self._fan_reader.read_ducts(),
            watchdog=self._fan_reader.read_watchdog(),
            case_fans=self._case_fan_reader.read(),
        )

    def run(self, duration_s: float, out_path: Path, interval_s: float = SAMPLE_INTERVAL_S) -> int:
        """Sample at `interval_s` for `duration_s`, appending one JSON line per tick.

        Returns the number of ticks written. A slow tick (a sample that itself
        took longer than `interval_s`) is written anyway; this never sleeps a
        negative amount or drops a sample to catch up.
        """
        deadline = time.monotonic() + duration_s
        written = 0
        with out_path.open("a") as handle:
            while time.monotonic() < deadline:
                tick_start = time.monotonic()
                sample = self.sample()
                handle.write(sample.to_json_line())
                handle.write("\n")
                handle.flush()
                written += 1
                elapsed = time.monotonic() - tick_start
                remaining = interval_s - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        return written


def load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """Read a `telemetry.jsonl` file back into plain dicts, one per tick."""
    lines = path.read_text().splitlines()
    return tuple(json.loads(line) for line in lines if line)
