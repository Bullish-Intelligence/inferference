"""Tests for the per-tick V620 telemetry sampler, against a fabricated sysfs tree."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from inferference.telemetry import (
    GPU_METRICS_THROTTLE_STATUS_OFFSET,
    TELEMETRY_SCHEMA_VERSION,
    CardReader,
    CaseFanReader,
    FanDuctReader,
    TelemetryError,
    TelemetrySampler,
    load_jsonl,
    read_watchdog_status,
)


def _gpu_metrics_blob(throttle_status: int = 0) -> bytes:
    buf = bytearray(120)
    struct.pack_into("<HBB", buf, 0, 120, 1, 3)
    struct.pack_into("<I", buf, GPU_METRICS_THROTTLE_STATUS_OFFSET, throttle_status)
    return bytes(buf)


def _make_card(drm_root: Path, card: int, pci: str, *, throttle_status: int = 0) -> Path:
    real_device = drm_root / "pci-devices" / pci
    real_device.mkdir(parents=True)
    card_dir = drm_root / f"card{card}"
    card_dir.mkdir(parents=True)
    device = card_dir / "device"
    device.symlink_to(real_device, target_is_directory=True)

    hwmon = device / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "name").write_text("amdgpu\n")

    (hwmon / "temp1_input").write_text("45000\n")
    (hwmon / "temp2_input").write_text("52000\n")
    (hwmon / "temp3_input").write_text("48000\n")
    (hwmon / "power1_average").write_text("210000000\n")
    (hwmon / "power1_cap").write_text("250000000\n")
    (hwmon / "in0_input").write_text("800\n")
    (hwmon / "freq1_input").write_text("1500000000\n")
    (hwmon / "freq2_input").write_text("1000000000\n")

    (device / "gpu_busy_percent").write_text("37\n")
    (device / "mem_busy_percent").write_text("12\n")
    (device / "mem_info_vram_used").write_text(str(1024 * 1024 * 100) + "\n")
    (device / "mem_info_vram_total").write_text(str(1024 * 1024 * 1024 * 30) + "\n")
    (device / "mem_info_gtt_used").write_text(str(1024 * 1024 * 5) + "\n")
    (device / "mem_info_gtt_total").write_text(str(1024 * 1024 * 1024 * 60) + "\n")
    (device / "pp_dpm_sclk").write_text("0: 500Mhz\n1: 1500Mhz *\n")
    (device / "pp_dpm_mclk").write_text("0: 96Mhz *\n1: 1000Mhz\n")
    (device / "power_dpm_force_performance_level").write_text("manual\n")
    (device / "pp_power_profile_mode").write_text(
        "PROFILE_INDEX(NAME) CLOCK_TYPE(NAME)\n 0 BOOTUP_DEFAULT :\n 2   POWER_SAVING*:\n"
    )
    (device / "gpu_metrics").write_bytes(_gpu_metrics_blob(throttle_status))
    return device


@pytest.fixture
def drm_root(tmp_path: Path) -> Path:
    return tmp_path / "sys-class-drm"


def test_card_reader_reads_every_field(drm_root: Path) -> None:
    device = _make_card(drm_root, 0, "0000:04:00.0")
    device.resolve()  # sanity: path exists

    reader = CardReader(0, drm_root=drm_root)
    sample = reader.read()

    assert sample.card == 0
    assert sample.pci == "0000:04:00.0"
    assert sample.temperature_edge_c == 45.0
    assert sample.temperature_hotspot_c == 52.0
    assert sample.temperature_mem_c == 48.0
    assert sample.power_w == 210.0
    assert sample.power_cap_w == 250.0
    assert sample.vddgfx_mv == 800
    assert sample.sclk_mhz == 1500.0
    assert sample.mclk_mhz == 1000.0
    assert sample.gpu_busy_percent == 37
    assert sample.mem_busy_percent == 12
    assert sample.vram_used_mib == pytest.approx(100.0)
    assert sample.gtt_used_mib == pytest.approx(5.0)
    assert sample.power_dpm_force_performance_level == "manual"
    assert sample.pp_power_profile_mode == "POWER_SAVING"
    assert sample.throttle_status_raw == 0
    assert sample.throttled is False


def test_card_reader_reports_throttled_when_the_bitmask_is_nonzero(drm_root: Path) -> None:
    _make_card(drm_root, 0, "0000:04:00.0", throttle_status=0x1)
    reader = CardReader(0, drm_root=drm_root)

    sample = reader.read()

    assert sample.throttle_status_raw == 0x1
    assert sample.throttled is True


def test_pp_dpm_levels_parse_index_mhz_and_active_flag(drm_root: Path) -> None:
    _make_card(drm_root, 0, "0000:04:00.0")
    reader = CardReader(0, drm_root=drm_root)

    sample = reader.read()

    assert [(level.index, level.mhz, level.active) for level in sample.pp_dpm_sclk] == [
        (0, 500, False),
        (1, 1500, True),
    ]
    assert [(level.index, level.mhz, level.active) for level in sample.pp_dpm_mclk] == [
        (0, 96, True),
        (1, 1000, False),
    ]


def test_card_reader_refuses_a_device_dir_with_no_amdgpu_hwmon(tmp_path: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    device = drm_root / "card0" / "device"
    (device / "hwmon" / "hwmon0").mkdir(parents=True)
    (device / "hwmon" / "hwmon0" / "name").write_text("not-amdgpu\n")

    with pytest.raises(TelemetryError, match="amdgpu"):
        CardReader(0, drm_root=drm_root)


# ---- fan ducts and watchdog --------------------------------------------------


@pytest.fixture
def hwmon_root(tmp_path: Path) -> Path:
    root = tmp_path / "sys-class-hwmon"
    chip = root / "hwmon2"
    chip.mkdir(parents=True)
    (chip / "name").write_text("arctic_fan\n")
    (chip / "pwm2").write_text("25\n")
    (chip / "fan2_input").write_text("1529\n")
    (chip / "pwm3").write_text("30\n")
    (chip / "fan3_input").write_text("1647\n")
    return root


def test_fan_duct_reader_reads_both_channels(hwmon_root: Path) -> None:
    reader = FanDuctReader(hwmon_root=hwmon_root, channels=(2, 3))

    ducts = reader.read_ducts()

    assert [(duct.channel, duct.pwm, duct.tach_rpm) for duct in ducts] == [
        (2, 25, 1529),
        (3, 30, 1647),
    ]


def test_watchdog_status_is_parsed(tmp_path: Path, hwmon_root: Path) -> None:
    status_file = tmp_path / "fan-status"
    status_file.write_text("state=OK target_pwm=25 max_gpu_junction_mC=28000\n")
    reader = FanDuctReader(hwmon_root=hwmon_root, watchdog_status_file=status_file)

    watchdog = reader.read_watchdog()

    assert watchdog is not None
    assert watchdog.state == "OK"
    assert watchdog.target_pwm == 25
    assert watchdog.max_gpu_junction_m_c == 28000


def test_missing_watchdog_status_file_is_none_not_an_error(tmp_path: Path, hwmon_root: Path) -> None:
    reader = FanDuctReader(hwmon_root=hwmon_root, watchdog_status_file=tmp_path / "absent")

    assert reader.read_watchdog() is None


_VALID_STATUS = "state=OK target_pwm=25 max_gpu_junction_mC=28000\n"


class _Rewriter:
    """A fake `sleep` that writes the next scripted text, as a watchdog rewrite would."""

    def __init__(self, path: Path, later_texts: list[str]) -> None:
        self.path = path
        self.later_texts = list(later_texts)
        self.sleeps: list[float] = []

    def __call__(self, delay_s: float) -> None:
        self.sleeps.append(delay_s)
        if self.later_texts:
            self.path.write_text(self.later_texts.pop(0))


def test_watchdog_read_retries_an_empty_file_then_succeeds(tmp_path: Path) -> None:
    status_file = tmp_path / "fan-status"
    status_file.write_text("")
    rewriter = _Rewriter(status_file, ["  \n", _VALID_STATUS])

    sample = read_watchdog_status(status_file, sleep=rewriter)

    assert sample.state == "OK"
    assert len(rewriter.sleeps) == 2


def test_watchdog_read_retries_a_partial_line_then_succeeds(tmp_path: Path) -> None:
    status_file = tmp_path / "fan-status"
    status_file.write_text("state=OK target_pwm=25 max_gpu_junction_mC=28")
    rewriter = _Rewriter(status_file, ["state=OK target_pwm=25", _VALID_STATUS])

    sample = read_watchdog_status(status_file, sleep=rewriter)

    assert sample.max_gpu_junction_m_c == 28000
    assert len(rewriter.sleeps) == 2


def test_watchdog_read_fails_closed_after_the_last_attempt(tmp_path: Path) -> None:
    status_file = tmp_path / "fan-status"
    status_file.write_text("")
    sleeps: list[float] = []

    with pytest.raises(TelemetryError, match=r"could not parse fan watchdog status: ''.*after 5 attempts"):
        read_watchdog_status(status_file, sleep=sleeps.append)

    assert len(sleeps) == 4


def test_watchdog_read_fails_closed_on_a_missing_file(tmp_path: Path) -> None:
    sleeps: list[float] = []

    with pytest.raises(TelemetryError, match=r"could not read .*after 3 attempts"):
        read_watchdog_status(tmp_path / "absent", attempts=3, sleep=sleeps.append)

    assert len(sleeps) == 2


def test_sampler_reader_survives_one_empty_read(tmp_path: Path, hwmon_root: Path) -> None:
    status_file = tmp_path / "fan-status"
    status_file.write_text("")
    sleeps: list[float] = []

    def rewrite_complete(delay_s: float) -> None:
        sleeps.append(delay_s)
        status_file.write_text(_VALID_STATUS)

    reader = FanDuctReader(hwmon_root=hwmon_root, watchdog_status_file=status_file, sleep=rewrite_complete)

    watchdog = reader.read_watchdog()

    assert watchdog is not None and watchdog.state == "OK"
    assert len(sleeps) == 1


def test_watchdog_profile_is_parsed_when_present(tmp_path: Path, hwmon_root: Path) -> None:
    status_file = tmp_path / "fan-status"
    status_file.write_text("state=OK target_pwm=25 max_gpu_junction_mC=28000 profile=flat-max\n")
    reader = FanDuctReader(hwmon_root=hwmon_root, watchdog_status_file=status_file)

    watchdog = reader.read_watchdog()

    assert watchdog is not None
    assert watchdog.profile == "flat-max"


def test_watchdog_profile_is_none_when_absent(tmp_path: Path, hwmon_root: Path) -> None:
    status_file = tmp_path / "fan-status"
    status_file.write_text("state=OK target_pwm=25 max_gpu_junction_mC=28000\n")
    reader = FanDuctReader(hwmon_root=hwmon_root, watchdog_status_file=status_file)

    watchdog = reader.read_watchdog()

    assert watchdog is not None
    assert watchdog.profile is None


# ---- fan channel auto-discovery ----------------------------------------------


def test_fan_duct_reader_discovers_every_channel_with_a_matching_pair(tmp_path: Path) -> None:
    root = tmp_path / "sys-class-hwmon"
    chip = root / "hwmon2"
    chip.mkdir(parents=True)
    (chip / "name").write_text("arctic_fan\n")
    for channel in (1, 2, 3, 5):
        (chip / f"pwm{channel}").write_text("10\n")
        (chip / f"fan{channel}_input").write_text("1500\n")
    # Channel 4 has a pwm file but no tach file: discovery must skip it.
    (chip / "pwm4").write_text("10\n")

    reader = FanDuctReader(hwmon_root=root)

    assert reader.channels == (1, 2, 3, 5)


def test_fan_duct_reader_honours_an_explicit_channel_tuple(tmp_path: Path) -> None:
    root = tmp_path / "sys-class-hwmon"
    chip = root / "hwmon2"
    chip.mkdir(parents=True)
    (chip / "name").write_text("arctic_fan\n")
    for channel in (1, 2, 3):
        (chip / f"pwm{channel}").write_text("10\n")
        (chip / f"fan{channel}_input").write_text("1500\n")

    reader = FanDuctReader(hwmon_root=root, channels=(2, 3))

    assert reader.channels == (2, 3)


# ---- the sampler: one tick, both clocks, jsonl round-trip --------------------


def test_sampler_tick_carries_both_clocks_and_both_cards(tmp_path: Path, hwmon_root: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    _make_card(drm_root, 0, "0000:04:00.0")
    _make_card(drm_root, 1, "0000:03:00.0")
    status_file = tmp_path / "fan-status"
    status_file.write_text("state=OK target_pwm=25 max_gpu_junction_mC=28000\n")

    sampler = TelemetrySampler(
        cards=(0, 1),
        drm_root=drm_root,
        hwmon_root=hwmon_root,
        watchdog_status_file=status_file,
    )

    sample = sampler.sample()

    assert sample.utc
    assert sample.monotonic > 0
    assert [card.card for card in sample.cards] == [0, 1]
    assert len(sample.fan_ducts) == 2
    assert sample.watchdog is not None and sample.watchdog.state == "OK"
    assert sample.schema_version == TELEMETRY_SCHEMA_VERSION


def test_to_json_line_carries_the_schema_version(tmp_path: Path, hwmon_root: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    _make_card(drm_root, 0, "0000:04:00.0")
    status_file = tmp_path / "fan-status"
    status_file.write_text("state=OK target_pwm=25 max_gpu_junction_mC=28000\n")
    sampler = TelemetrySampler(cards=(0,), drm_root=drm_root, hwmon_root=hwmon_root, watchdog_status_file=status_file)

    line = sampler.sample().to_json_line()

    assert json.loads(line)["schema_version"] == TELEMETRY_SCHEMA_VERSION == 1


def test_run_writes_one_jsonl_line_per_tick_and_is_loadable(tmp_path: Path, hwmon_root: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    _make_card(drm_root, 0, "0000:04:00.0")
    status_file = tmp_path / "fan-status"
    status_file.write_text("state=OK target_pwm=25 max_gpu_junction_mC=28000\n")
    out_path = tmp_path / "telemetry.jsonl"

    sampler = TelemetrySampler(cards=(0,), drm_root=drm_root, hwmon_root=hwmon_root, watchdog_status_file=status_file)
    written = sampler.run(duration_s=0.05, out_path=out_path, interval_s=0.01)

    assert written >= 1
    records = load_jsonl(out_path)
    assert len(records) == written
    assert all("utc" in record and "monotonic" in record for record in records)
    assert all(len(record["cards"]) == 1 for record in records)

    # Every line is valid, independent JSON -- append-only, one object per tick.
    for line in out_path.read_text().splitlines():
        json.loads(line)


# ---- the Dell case fans: observation-only, fail-soft --------------------------


def _make_dell_chip(root: Path, name: str = "dell_smm", index: int = 3) -> Path:
    chip = root / f"hwmon{index}"
    chip.mkdir(parents=True)
    (chip / "name").write_text(f"{name}\n")
    return chip


def test_case_fan_reader_reads_rpm_label_and_pwm(tmp_path: Path) -> None:
    root = tmp_path / "hwmon"
    chip = _make_dell_chip(root)
    for fan, rpm in ((1, 1073), (2, 655)):
        (chip / f"fan{fan}_input").write_text(f"{rpm}\n")
        (chip / f"fan{fan}_label").write_text(f"Fan {fan}\n")
        (chip / f"pwm{fan}").write_text("128\n")
        (chip / f"pwm{fan}_enable").write_text("2\n")

    fans = CaseFanReader(hwmon_root=root).read()

    assert [(f.fan, f.rpm, f.label, f.pwm, f.pwm_enable) for f in fans] == [
        (1, 1073, "Fan 1", 128, 2),
        (2, 655, "Fan 2", 128, 2),
    ]


def test_case_fan_reader_finds_the_chip_by_name_not_by_index(tmp_path: Path) -> None:
    root = tmp_path / "hwmon"
    other = _make_dell_chip(root, name="coretemp", index=3)
    (other / "fan1_input").write_text("9999\n")
    chip = _make_dell_chip(root, index=7)
    (chip / "fan1_input").write_text("1073\n")

    fans = CaseFanReader(hwmon_root=root).read()

    assert [(f.fan, f.rpm) for f in fans] == [(1, 1073)]


def test_case_fan_reader_with_no_chip_returns_empty_and_never_raises(tmp_path: Path) -> None:
    root = tmp_path / "hwmon"
    _make_dell_chip(root, name="nvme", index=0)

    assert CaseFanReader(hwmon_root=root).read() == ()
    assert CaseFanReader(hwmon_root=tmp_path / "no-such-root").read() == ()


def test_case_fan_reader_without_pwm_files_keeps_the_rpm(tmp_path: Path) -> None:
    root = tmp_path / "hwmon"
    chip = _make_dell_chip(root)
    (chip / "fan1_input").write_text("1073\n")
    (chip / "fan4_input").write_text("904\n")

    fans = CaseFanReader(hwmon_root=root).read()

    assert [(f.fan, f.rpm, f.label, f.pwm, f.pwm_enable) for f in fans] == [
        (1, 1073, None, None, None),
        (4, 904, None, None, None),
    ]


def test_case_fan_reader_survives_an_unreadable_file(tmp_path: Path) -> None:
    root = tmp_path / "hwmon"
    chip = _make_dell_chip(root)
    (chip / "fan1_input").write_text("1073\n")
    # A directory in place of a file makes `read_text` raise `OSError`.
    (chip / "fan1_label").mkdir()
    (chip / "pwm1").mkdir()
    (chip / "pwm1_enable").write_text("2\n")
    (chip / "fan2_input").mkdir()
    (chip / "fan3_input").write_text("not a number\n")

    fans = CaseFanReader(hwmon_root=root).read()

    assert [(f.fan, f.rpm, f.label, f.pwm, f.pwm_enable) for f in fans] == [(1, 1073, None, None, 2)]


def test_case_fan_reader_treats_an_empty_label_as_absent(tmp_path: Path) -> None:
    root = tmp_path / "hwmon"
    chip = _make_dell_chip(root)
    (chip / "fan1_input").write_text("1073\n")
    (chip / "fan1_label").write_text("\n")

    assert CaseFanReader(hwmon_root=root).read()[0].label is None


def test_sampler_tick_carries_the_case_fans_in_json(tmp_path: Path, hwmon_root: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    _make_card(drm_root, 0, "0000:04:00.0")
    chip = _make_dell_chip(hwmon_root)
    (chip / "fan1_input").write_text("1073\n")
    (chip / "fan1_label").write_text("Fan 1\n")
    (chip / "pwm1_enable").write_text("2\n")
    status_file = tmp_path / "fan-status"
    status_file.write_text("state=OK target_pwm=25 max_gpu_junction_mC=28000\n")
    sampler = TelemetrySampler(cards=(0,), drm_root=drm_root, hwmon_root=hwmon_root, watchdog_status_file=status_file)

    record = json.loads(sampler.sample().to_json_line())

    assert record["schema_version"] == 1
    assert record["case_fans"] == [{"fan": 1, "rpm": 1073, "label": "Fan 1", "pwm": None, "pwm_enable": 2}]
    assert len(record["fan_ducts"]) == 2


def test_sampler_tick_without_a_dell_chip_has_empty_case_fans(tmp_path: Path, hwmon_root: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    _make_card(drm_root, 0, "0000:04:00.0")
    status_file = tmp_path / "fan-status"
    status_file.write_text("state=OK target_pwm=25 max_gpu_junction_mC=28000\n")
    sampler = TelemetrySampler(cards=(0,), drm_root=drm_root, hwmon_root=hwmon_root, watchdog_status_file=status_file)

    sample = sampler.sample()

    assert sample.case_fans == ()
    assert json.loads(sample.to_json_line())["case_fans"] == []
