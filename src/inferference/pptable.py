"""Codec for the Sienna Cichlid ``pp_table`` control surface.

The V620's board power cap, PPT time constants, TDC current limits and
per-sensor thermal limits all live inside one 2470-byte binary blob exposed at
``/sys/class/drm/card<N>/device/pp_table`` (0444, no checksum, no signature).
This module decodes that blob into a
typed, immutable structure, and encodes it back byte-for-byte.

Every offset below is byte-verified against a stock blob read from
``/sys/class/drm/card0/device/pp_table`` on an AMD Radeon Pro V620
(sha256 ``a6fc019fdada096422629293bee778e8857af3330fd2dc2de42dd9d9d921b1c8``).
That blob is vendor board-power firmware with no redistribution grant, so it is
not shipped with this package: the evidence lives in the private project log and
the tests take it from ``INFERFERENCE_PP_TABLE_DIR``.
The driver-portion layout is ``struct smu_11_0_7_powerplay_table``
(``inc/smu_v11_0_7_pptable.h``); the embedded SMU table is ``PPTable_t``
(``inc/pmfw_if/smu11_driver_if_sienna_cichlid.h``). The ``TdcLimit`` /
``TdcLimitTau`` offsets below are 846/850 (a commonly circulated table prints
them as 844/848), confirmed by parsing the stock blob and matching
its decoded values against live sysfs.

Every setter returns a new :class:`PPTable` with exactly the bytes of one
field patched; nothing else changes. This module performs no I/O — reading
and writing ``pp_table`` itself is a root operation outside its scope.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

STRUCT_SIZE = 2470
FORMAT_REVISION = 15
TABLE_REVISION = 2

PPT_THROTTLER_COUNT = 4
TDC_THROTTLER_COUNT = 2
TEMP_COUNT = 10
MAX_ODFEATURE = 32
MAX_ODSETTING = 64

_HEADER_SIZE = 0
_HEADER_FORMAT_REVISION = 2
_HEADER_CONTENT_REVISION = 3
_HEADER_TABLE_REVISION = 4

_SMALL_POWER_LIMIT1 = 22
_SMALL_POWER_LIMIT2 = 24
_BOOST_POWER_LIMIT = 26
_SOFTWARE_SHUTDOWN_TEMP = 28

_OD_TABLE = 182
_OD_REVISION = _OD_TABLE
_OD_FEATURE_COUNT = _OD_TABLE + 4
_OD_SETTING_COUNT = _OD_TABLE + 8
_OD_CAP = _OD_TABLE + 12
_OD_MAX = _OD_TABLE + 44
_OD_MIN = _OD_TABLE + 300

_SMC = 802
_SMC_VERSION = _SMC
_SMC_FEATURES_TO_RUN = _SMC + 4
_SMC_POWER_LIMIT_AC = _SMC + 12
_SMC_POWER_LIMIT_AC_TAU = _SMC + 20
_SMC_POWER_LIMIT_DC = _SMC + 28
_SMC_POWER_LIMIT_DC_TAU = _SMC + 36
_SMC_TDC_LIMIT = _SMC + 44
_SMC_TDC_LIMIT_TAU = _SMC + 48
_SMC_TEMPERATURE_LIMIT = _SMC + 52
_SMC_FIT_LIMIT = _SMC + 72

_FORMAT_RANGE: dict[str, tuple[int, int]] = {
    "<B": (0, 255),
    "<H": (0, 65535),
    "<I": (0, 4294967295),
    "<h": (-32768, 32767),
}


class PPTableError(ValueError):
    """A ``pp_table`` blob, or a value offered to a setter, is malformed."""


class PptThrottler(IntEnum):
    """Index into the four PPT power-limit slots. Only PPT0 is live."""

    PPT0 = 0
    PPT1 = 1
    PPT2 = 2
    PPT3 = 3


class TdcThrottler(IntEnum):
    """Index into the two TDC current-limit slots."""

    GFX = 0
    SOC = 1


class TempSensor(IntEnum):
    """Index into the ten thermal-limit slots, in SMU firmware order."""

    EDGE = 0
    HOTSPOT = 1
    MEM = 2
    VR_GFX = 3
    VR_MEM0 = 4
    VR_MEM1 = 5
    VR_SOC = 6
    LIQUID0 = 7
    LIQUID1 = 8
    PLX = 9


class OverdriveCapability(IntEnum):
    """Index into ``overdrive_table.cap[]``. All 16 are 0 on the stock VBIOS."""

    GFXCLK_LIMITS = 0
    GFXCLK_CURVE = 1
    UCLK_LIMITS = 2
    POWER_LIMIT = 3
    FAN_ACOUSTIC_LIMIT = 4
    FAN_SPEED_MIN = 5
    TEMPERATURE_FAN = 6
    TEMPERATURE_SYSTEM = 7
    MEMORY_TIMING_TUNE = 8
    FAN_ZERO_RPM_CONTROL = 9
    AUTO_UV_ENGINE = 10
    AUTO_OC_ENGINE = 11
    AUTO_OC_MEMORY = 12
    FAN_CURVE = 13
    AUTO_FAN_ACOUSTIC_LIMIT = 14
    POWER_MODE = 15


class OverdriveSetting(IntEnum):
    """Index into ``overdrive_table.max[]`` / ``.min[]``."""

    GFXCLKFMAX = 0
    GFXCLKFMIN = 1
    CUSTOM_GFX_VF_CURVE_A = 2
    CUSTOM_GFX_VF_CURVE_B = 3
    CUSTOM_GFX_VF_CURVE_C = 4
    CUSTOM_CURVE_VFT_FMIN = 5
    UCLKFMIN = 6
    UCLKFMAX = 7
    POWERPERCENTAGE = 8
    FANRPMMIN = 9
    FANRPMACOUSTICLIMIT = 10
    FANTARGETTEMPERATURE = 11
    OPERATINGTEMPMAX = 12
    ACTIMING = 13
    FAN_ZERO_RPM_CONTROL = 14
    AUTOUVENGINE = 15
    AUTOOCENGINE = 16
    AUTOOCMEMORY = 17


@dataclass(frozen=True, slots=True)
class FieldDelta:
    """One field that differs between two tables, named for a run log."""

    name: str
    old: object
    new: object


def _unpack_array(data: bytes, fmt: str, offset: int, count: int) -> tuple[int, ...]:
    size = struct.calcsize(fmt)
    return tuple(struct.unpack_from(fmt, data, offset + i * size)[0] for i in range(count))


@dataclass(frozen=True, slots=True)
class PPTable:
    """A decoded Sienna Cichlid ``pp_table`` blob.

    ``data`` is the full 2470-byte blob. Every read is a view onto it; every
    setter patches a copy and returns a new ``PPTable``, leaving every other
    byte identical to the original.
    """

    data: bytes

    def __post_init__(self) -> None:
        if len(self.data) != STRUCT_SIZE:
            raise PPTableError(f"pp_table blob is {len(self.data)} bytes; the Sienna Cichlid table is {STRUCT_SIZE}")
        size = struct.unpack_from("<H", self.data, _HEADER_SIZE)[0]
        if size != STRUCT_SIZE:
            raise PPTableError(f"pp_table header declares structuresize={size}, not {STRUCT_SIZE}")
        format_revision = self.data[_HEADER_FORMAT_REVISION]
        if format_revision != FORMAT_REVISION:
            raise PPTableError(f"pp_table format_revision={format_revision}, expected {FORMAT_REVISION}")
        table_revision = self.data[_HEADER_TABLE_REVISION]
        if table_revision != TABLE_REVISION:
            raise PPTableError(f"pp_table table_revision={table_revision}, expected {TABLE_REVISION}")

    def encode(self) -> bytes:
        """Return the raw blob, byte-identical to what was decoded."""
        return self.data

    # ---- header -----------------------------------------------------------

    @property
    def structure_size(self) -> int:
        return struct.unpack_from("<H", self.data, _HEADER_SIZE)[0]

    @property
    def format_revision(self) -> int:
        return self.data[_HEADER_FORMAT_REVISION]

    @property
    def content_revision(self) -> int:
        return self.data[_HEADER_CONTENT_REVISION]

    @property
    def table_revision(self) -> int:
        return self.data[_HEADER_TABLE_REVISION]

    # ---- driver-portion scalars --------------------------------------------

    @property
    def small_power_limit1(self) -> int:
        return struct.unpack_from("<H", self.data, _SMALL_POWER_LIMIT1)[0]

    @property
    def small_power_limit2(self) -> int:
        return struct.unpack_from("<H", self.data, _SMALL_POWER_LIMIT2)[0]

    @property
    def boost_power_limit(self) -> int:
        return struct.unpack_from("<H", self.data, _BOOST_POWER_LIMIT)[0]

    @property
    def software_shutdown_temp(self) -> int:
        return struct.unpack_from("<H", self.data, _SOFTWARE_SHUTDOWN_TEMP)[0]

    # ---- overdrive table ----------------------------------------------------

    @property
    def overdrive_revision(self) -> int:
        return self.data[_OD_REVISION]

    @property
    def overdrive_feature_count(self) -> int:
        return struct.unpack_from("<I", self.data, _OD_FEATURE_COUNT)[0]

    @property
    def overdrive_setting_count(self) -> int:
        return struct.unpack_from("<I", self.data, _OD_SETTING_COUNT)[0]

    @property
    def overdrive_cap(self) -> tuple[int, ...]:
        return tuple(self.data[_OD_CAP : _OD_CAP + MAX_ODFEATURE])

    @property
    def overdrive_max(self) -> tuple[int, ...]:
        return _unpack_array(self.data, "<I", _OD_MAX, MAX_ODSETTING)

    @property
    def overdrive_min(self) -> tuple[int, ...]:
        return _unpack_array(self.data, "<I", _OD_MIN, MAX_ODSETTING)

    # ---- smc_pptable --------------------------------------------------------

    @property
    def smc_version(self) -> int:
        return struct.unpack_from("<I", self.data, _SMC_VERSION)[0]

    @property
    def features_to_run(self) -> int:
        lo, hi = struct.unpack_from("<II", self.data, _SMC_FEATURES_TO_RUN)
        return lo | (hi << 32)

    @property
    def socket_power_limit_ac(self) -> tuple[int, ...]:
        return _unpack_array(self.data, "<H", _SMC_POWER_LIMIT_AC, PPT_THROTTLER_COUNT)

    @property
    def socket_power_limit_ac_tau(self) -> tuple[int, ...]:
        return _unpack_array(self.data, "<H", _SMC_POWER_LIMIT_AC_TAU, PPT_THROTTLER_COUNT)

    @property
    def socket_power_limit_dc(self) -> tuple[int, ...]:
        return _unpack_array(self.data, "<H", _SMC_POWER_LIMIT_DC, PPT_THROTTLER_COUNT)

    @property
    def socket_power_limit_dc_tau(self) -> tuple[int, ...]:
        return _unpack_array(self.data, "<H", _SMC_POWER_LIMIT_DC_TAU, PPT_THROTTLER_COUNT)

    @property
    def tdc_limit(self) -> tuple[int, ...]:
        return _unpack_array(self.data, "<H", _SMC_TDC_LIMIT, TDC_THROTTLER_COUNT)

    @property
    def tdc_limit_tau(self) -> tuple[int, ...]:
        return _unpack_array(self.data, "<H", _SMC_TDC_LIMIT_TAU, TDC_THROTTLER_COUNT)

    @property
    def temperature_limit(self) -> tuple[int, ...]:
        return _unpack_array(self.data, "<H", _SMC_TEMPERATURE_LIMIT, TEMP_COUNT)

    @property
    def fit_limit(self) -> int:
        return struct.unpack_from("<I", self.data, _SMC_FIT_LIMIT)[0]

    # ---- setters: each patches exactly one field's bytes -------------------

    def _patched(self, offset: int, fmt: str, value: int) -> PPTable:
        lo, hi = _FORMAT_RANGE[fmt]
        if not lo <= value <= hi:
            raise PPTableError(f"value {value} does not fit a {fmt!r} field (range {lo}..{hi})")
        buf = bytearray(self.data)
        struct.pack_into(fmt, buf, offset, value)
        return PPTable(bytes(buf))

    def set_socket_power_limit_ac(self, ppt: PptThrottler, watts: int) -> PPTable:
        return self._patched(_SMC_POWER_LIMIT_AC + 2 * int(ppt), "<H", watts)

    def set_socket_power_limit_ac_tau(self, ppt: PptThrottler, ms: int) -> PPTable:
        return self._patched(_SMC_POWER_LIMIT_AC_TAU + 2 * int(ppt), "<H", ms)

    def set_tdc_limit(self, throttler: TdcThrottler, amps: int) -> PPTable:
        return self._patched(_SMC_TDC_LIMIT + 2 * int(throttler), "<H", amps)

    def set_temperature_limit(self, sensor: TempSensor, celsius: int) -> PPTable:
        return self._patched(_SMC_TEMPERATURE_LIMIT + 2 * int(sensor), "<H", celsius)

    def set_overdrive_cap(self, feature: OverdriveCapability | int, enabled: bool) -> PPTable:
        index = int(feature)
        if not 0 <= index < MAX_ODFEATURE:
            raise PPTableError(f"overdrive capability index {index} is outside 0..{MAX_ODFEATURE - 1}")
        return self._patched(_OD_CAP + index, "<B", 1 if enabled else 0)

    def set_overdrive_max(self, setting: OverdriveSetting | int, value: int) -> PPTable:
        index = int(setting)
        if not 0 <= index < MAX_ODSETTING:
            raise PPTableError(f"overdrive setting index {index} is outside 0..{MAX_ODSETTING - 1}")
        return self._patched(_OD_MAX + 4 * index, "<I", value)

    def set_overdrive_min(self, setting: OverdriveSetting | int, value: int) -> PPTable:
        index = int(setting)
        if not 0 <= index < MAX_ODSETTING:
            raise PPTableError(f"overdrive setting index {index} is outside 0..{MAX_ODSETTING - 1}")
        return self._patched(_OD_MIN + 4 * index, "<I", value)

    def set_features_to_run_bit(self, bit: int, enabled: bool) -> PPTable:
        if not 0 <= bit < 64:
            raise PPTableError(f"features_to_run bit {bit} is outside 0..63")
        offset = _SMC_FEATURES_TO_RUN + (4 if bit >= 32 else 0)
        local_bit = bit % 32
        current = struct.unpack_from("<I", self.data, offset)[0]
        new_value = (current | (1 << local_bit)) if enabled else (current & ~(1 << local_bit) & 0xFFFFFFFF)
        return self._patched(offset, "<I", new_value)


def decode(blob: bytes) -> PPTable:
    """Decode a raw ``pp_table`` blob into a validated :class:`PPTable`."""
    return PPTable(bytes(blob))


_DIFF_FIELDS: tuple[str, ...] = (
    "small_power_limit1",
    "small_power_limit2",
    "boost_power_limit",
    "software_shutdown_temp",
    "overdrive_revision",
    "overdrive_feature_count",
    "overdrive_setting_count",
    "overdrive_cap",
    "overdrive_max",
    "overdrive_min",
    "smc_version",
    "features_to_run",
    "socket_power_limit_ac",
    "socket_power_limit_ac_tau",
    "socket_power_limit_dc",
    "socket_power_limit_dc_tau",
    "tdc_limit",
    "tdc_limit_tau",
    "temperature_limit",
    "fit_limit",
)


def diff(a: PPTable, b: PPTable) -> tuple[FieldDelta, ...]:
    """Render every field that differs between two tables, for run provenance."""
    deltas = []
    for name in _DIFF_FIELDS:
        old = getattr(a, name)
        new = getattr(b, name)
        if old != new:
            deltas.append(FieldDelta(name, old, new))
    return tuple(deltas)
