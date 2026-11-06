"""Codec tests for the V620 pp_table control surface."""

from __future__ import annotations

import pytest

from inferference.pptable import (
    FieldDelta,
    OverdriveCapability,
    OverdriveSetting,
    PPTable,
    PPTableError,
    PptThrottler,
    TdcThrottler,
    TempSensor,
    decode,
    diff,
)
from tests.pp_tables import STOCK_SHA256, real_stock_blob, synthetic_stock_blob


@pytest.fixture
def stock_blob() -> bytes:
    """A structurally valid table. The vendor blob is not redistributed here."""
    return synthetic_stock_blob()


@pytest.fixture
def stock_table(stock_blob: bytes) -> PPTable:
    return decode(stock_blob)


@pytest.fixture
def real_stock_table() -> PPTable:
    """The genuine stock table. Skips when the vendor blob is not supplied."""
    return decode(real_stock_blob())


def test_real_stock_blob_hash_is_the_pinned_value() -> None:
    import hashlib

    assert hashlib.sha256(real_stock_blob()).hexdigest() == STOCK_SHA256


# ---- round-trip and header validation ---------------------------------------


def test_round_trip_is_byte_identical_on_a_decoded_blob(stock_blob: bytes) -> None:
    table = decode(stock_blob)
    assert table.encode() == stock_blob


def test_decode_rejects_a_truncated_blob(stock_blob: bytes) -> None:
    with pytest.raises(PPTableError, match="2470"):
        decode(stock_blob[:-1])


def test_decode_rejects_a_wrong_structuresize(stock_blob: bytes) -> None:
    buf = bytearray(stock_blob)
    buf[0] = 0xFF
    with pytest.raises(PPTableError, match="structuresize"):
        decode(bytes(buf))


def test_decode_rejects_a_wrong_format_revision(stock_blob: bytes) -> None:
    buf = bytearray(stock_blob)
    buf[2] = 14
    with pytest.raises(PPTableError, match="format_revision"):
        decode(bytes(buf))


def test_decode_rejects_a_wrong_table_revision(stock_blob: bytes) -> None:
    buf = bytearray(stock_blob)
    buf[4] = 1
    with pytest.raises(PPTableError, match="table_revision"):
        decode(bytes(buf))


# ---- decoded values match the golden literals ------------------------------


def test_header_literals(stock_table: PPTable) -> None:
    assert stock_table.structure_size == 2470
    assert stock_table.format_revision == 15
    assert stock_table.content_revision == 0
    assert stock_table.table_revision == 2


def test_driver_portion_literals(real_stock_table: PPTable) -> None:
    assert real_stock_table.small_power_limit1 == 0
    assert real_stock_table.small_power_limit2 == 0
    assert real_stock_table.boost_power_limit == 0
    assert real_stock_table.software_shutdown_temp == 115


def test_overdrive_table_literals(real_stock_table: PPTable) -> None:
    assert real_stock_table.overdrive_revision == 129
    assert real_stock_table.overdrive_feature_count == 16
    assert real_stock_table.overdrive_setting_count == 30
    assert real_stock_table.overdrive_cap == (0,) * 32
    assert real_stock_table.overdrive_max[OverdriveSetting.GFXCLKFMAX] == 2650
    assert real_stock_table.overdrive_min[OverdriveSetting.GFXCLKFMAX] == 500
    assert real_stock_table.overdrive_max[OverdriveSetting.UCLKFMIN] == 1075
    assert real_stock_table.overdrive_min[OverdriveSetting.UCLKFMIN] == 674
    assert real_stock_table.overdrive_max[OverdriveSetting.POWERPERCENTAGE] == 10
    assert real_stock_table.overdrive_min[OverdriveSetting.POWERPERCENTAGE] == 7


def test_golden_power_and_temperature_literals(real_stock_table: PPTable) -> None:
    """The one test that must break loudly if a VBIOS change moves these."""
    assert real_stock_table.socket_power_limit_ac[PptThrottler.PPT0] == 250
    assert real_stock_table.temperature_limit[TempSensor.HOTSPOT] == 100


def test_smc_pptable_literals(real_stock_table: PPTable) -> None:
    assert real_stock_table.smc_version == 0x00000006
    assert real_stock_table.socket_power_limit_ac == (250, 0, 0, 0)
    assert real_stock_table.socket_power_limit_ac_tau == (0, 0, 0, 0)
    assert real_stock_table.socket_power_limit_dc == (250, 0, 0, 0)
    assert real_stock_table.socket_power_limit_dc_tau == (0, 0, 0, 0)
    assert real_stock_table.tdc_limit == (300, 55)
    assert real_stock_table.tdc_limit_tau == (0, 0)
    assert real_stock_table.fit_limit == 0
    assert real_stock_table.temperature_limit == (100, 100, 98, 115, 115, 115, 115, 0, 0, 0)
    assert real_stock_table.features_to_run == 0x00003772A37FFDFF


# ---- setters: exactly the targeted bytes change, and nothing else ----------


def _only_bytes_changed(before: bytes, after: bytes) -> set[int]:
    """The set of byte indices that actually differ.

    A changed field can still leave a byte unchanged (e.g. a u16 write whose
    high byte stays 0 on both sides), so callers assert this is a SUBSET of
    the field's byte range, never equality to it.
    """
    assert len(before) == len(after)
    return {i for i in range(len(before)) if before[i] != after[i]}


def test_set_socket_power_limit_ac_changes_only_its_two_bytes(stock_table: PPTable) -> None:
    patched = stock_table.set_socket_power_limit_ac(PptThrottler.PPT0, 220)
    changed = _only_bytes_changed(stock_table.data, patched.data)
    assert changed <= {814, 815}
    assert patched.socket_power_limit_ac[PptThrottler.PPT0] == 220
    assert patched.socket_power_limit_ac[1:] == (0, 0, 0)


def test_set_socket_power_limit_ac_tau(stock_table: PPTable) -> None:
    patched = stock_table.set_socket_power_limit_ac_tau(PptThrottler.PPT0, 1000)
    assert patched.socket_power_limit_ac_tau[PptThrottler.PPT0] == 1000
    assert _only_bytes_changed(stock_table.data, patched.data) <= {822, 823}


def test_set_tdc_limit_changes_only_its_slot(real_stock_table: PPTable) -> None:
    patched = real_stock_table.set_tdc_limit(TdcThrottler.GFX, 250)
    assert patched.tdc_limit == (250, 55)
    assert _only_bytes_changed(real_stock_table.data, patched.data) <= {846, 847}


def test_set_temperature_limit_hotspot(stock_table: PPTable) -> None:
    patched = stock_table.set_temperature_limit(TempSensor.HOTSPOT, 85)
    assert patched.temperature_limit[TempSensor.HOTSPOT] == 85
    assert patched.temperature_limit[TempSensor.EDGE] == 100
    assert _only_bytes_changed(stock_table.data, patched.data) <= {856, 857}


def test_set_overdrive_cap_flips_one_byte(stock_table: PPTable) -> None:
    patched = stock_table.set_overdrive_cap(OverdriveCapability.POWER_LIMIT, True)
    assert patched.overdrive_cap[OverdriveCapability.POWER_LIMIT] == 1
    assert _only_bytes_changed(stock_table.data, patched.data) == {194 + OverdriveCapability.POWER_LIMIT}


def test_set_overdrive_min_changes_only_its_four_bytes(stock_table: PPTable) -> None:
    patched = stock_table.set_overdrive_min(OverdriveSetting.POWERPERCENTAGE, 20)
    assert patched.overdrive_min[OverdriveSetting.POWERPERCENTAGE] == 20
    base = 482 + 4 * int(OverdriveSetting.POWERPERCENTAGE)
    assert _only_bytes_changed(stock_table.data, patched.data) <= {base, base + 1, base + 2, base + 3}


def test_set_overdrive_max_changes_only_its_four_bytes(stock_table: PPTable) -> None:
    patched = stock_table.set_overdrive_max(OverdriveSetting.GFXCLKFMAX, 2700)
    assert patched.overdrive_max[OverdriveSetting.GFXCLKFMAX] == 2700
    base = 226 + 4 * int(OverdriveSetting.GFXCLKFMAX)
    assert _only_bytes_changed(stock_table.data, patched.data) <= {base, base + 1, base + 2, base + 3}


def test_set_features_to_run_bit_sets_and_clears(real_stock_table: PPTable) -> None:
    cleared = real_stock_table.set_features_to_run_bit(1, False)  # DPM_GFXCLK, currently 1
    assert cleared.features_to_run == real_stock_table.features_to_run & ~0b10

    restored = cleared.set_features_to_run_bit(1, True)
    assert restored.features_to_run == real_stock_table.features_to_run
    assert restored.data == real_stock_table.data


def test_setter_rejects_an_out_of_range_value(stock_table: PPTable) -> None:
    with pytest.raises(PPTableError, match="range"):
        stock_table.set_socket_power_limit_ac(PptThrottler.PPT0, 70000)
    with pytest.raises(PPTableError, match="range"):
        stock_table.set_temperature_limit(TempSensor.EDGE, -1)


def test_setter_rejects_an_out_of_range_index(stock_table: PPTable) -> None:
    with pytest.raises(PPTableError, match="overdrive capability index"):
        stock_table.set_overdrive_cap(32, True)
    with pytest.raises(PPTableError, match="overdrive setting index"):
        stock_table.set_overdrive_max(64, 1)


def test_features_to_run_bit_rejects_an_out_of_range_bit(stock_table: PPTable) -> None:
    with pytest.raises(PPTableError, match="0..63"):
        stock_table.set_features_to_run_bit(64, True)


# ---- diff ---------------------------------------------------------------


def test_diff_of_a_table_against_itself_is_empty(stock_table: PPTable) -> None:
    assert diff(stock_table, stock_table) == ()


def test_diff_names_every_changed_field(real_stock_table: PPTable) -> None:
    patched = real_stock_table.set_temperature_limit(TempSensor.HOTSPOT, 85).set_tdc_limit(TdcThrottler.GFX, 250)

    deltas = diff(real_stock_table, patched)

    names = {delta.name for delta in deltas}
    assert names == {"temperature_limit", "tdc_limit"}
    assert FieldDelta(name="tdc_limit", old=(300, 55), new=(250, 55)) in deltas
