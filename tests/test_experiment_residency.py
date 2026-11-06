"""Tests for the residency proof, against a fabricated server log and sysfs tree.

The base case is a fabricated `tests/fixtures/experiment/server-log-resident.log`: the
`using device` line carrying the PCI address, 66 `assigned to device Vulkan0`
lines, the `offloaded 66/66` line, the two server-reported flash-attention
lines, and a wrapper echo line (`>>`) that must never count as a flash-
attention report. Every other test starts from this base and edits exactly
one dimension, so each proves that `prove_residency` fails the ONE check that
dimension maps to and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferference.experiment.residency import (
    MemorySnapshot,
    ResidencyError,
    ResidencyThresholds,
    extract_used_pci,
    first_failure,
    flash_attention_lines,
    layer_assignments,
    list_devices_names,
    memory_snapshot,
    offload_counts,
    prove_residency,
    server_baseline,
    sysfs_pci_address,
)

_FIXTURE_LOG = Path(__file__).parent / "fixtures" / "experiment" / "server-log-resident.log"
_EXPECTED_PCI = "0000:03:00.0"
_MEASURED_CARD = 1
_OTHER_CARD = 0

_LIST_DEVICES_TEXT = "Available devices:\n  Vulkan0: AMD Radeon Pro V620 (RADV NAVI21) (30720 MiB, 30700 MiB free)\n"


def test_server_baseline_requires_one_matching_card() -> None:
    line = "INFERFERENCE_BASELINE card=1 pci=0000:03:00.0 gtt=1048576 vram=2097152\n"
    assert server_baseline(line, card=1, pci=_EXPECTED_PCI) == MemorySnapshot(1, 2.0, 1.0)
    for invalid in ("", line + line, line.replace("card=1", "card=0"), line.replace("03:00", "04:00")):
        with pytest.raises(ResidencyError):
            server_baseline(invalid, card=1, pci=_EXPECTED_PCI)


def _base_log() -> str:
    return _FIXTURE_LOG.read_text()


def _base_before() -> dict[int, MemorySnapshot]:
    return {
        _OTHER_CARD: MemorySnapshot(card=_OTHER_CARD, vram_used_mib=50.0, gtt_used_mib=5.0),
        _MEASURED_CARD: MemorySnapshot(card=_MEASURED_CARD, vram_used_mib=100.0, gtt_used_mib=5.0),
    }


def _base_after() -> dict[int, MemorySnapshot]:
    return {
        _OTHER_CARD: MemorySnapshot(card=_OTHER_CARD, vram_used_mib=50.0, gtt_used_mib=5.0),
        _MEASURED_CARD: MemorySnapshot(card=_MEASURED_CARD, vram_used_mib=5000.0, gtt_used_mib=5.0),
    }


def _base_thresholds() -> ResidencyThresholds:
    return ResidencyThresholds(min_vram_delta_mib=2000.0, gtt_limit_mib=512.0)


def _run(
    *,
    log: str | None = None,
    list_devices: str = _LIST_DEVICES_TEXT,
    before: dict[int, MemorySnapshot] | None = None,
    after: dict[int, MemorySnapshot] | None = None,
    expected_pci: str = _EXPECTED_PCI,
    live_pci: str | None = None,
    thresholds: ResidencyThresholds | None = None,
    output_valid: bool | None = True,
):
    return prove_residency(
        expected_pci=expected_pci,
        live_pci=live_pci if live_pci is not None else expected_pci,
        list_devices=list_devices,
        log=log if log is not None else _base_log(),
        before=before if before is not None else _base_before(),
        after=after if after is not None else _base_after(),
        measured_card=_MEASURED_CARD,
        thresholds=thresholds if thresholds is not None else _base_thresholds(),
        output_valid=output_valid,
    )


def _check(checks, number: int):
    return next(check for check in checks if check.number == number)


def _others_pass(checks, except_number: int) -> bool:
    return all(check.passed for check in checks if check.number != except_number)


# ---- the base fixture: everything passes ------------------------------------


def test_resident_fixture_passes_all_ten_checks() -> None:
    checks = _run()

    assert len(checks) == 10
    assert [check.number for check in checks] == list(range(1, 11))
    assert all(check.passed for check in checks)
    assert first_failure(checks) is None


# ---- one test per check, each broken alone ----------------------------------


def test_check1_malformed_expected_pci_fails_only_check_one() -> None:
    # A second, space-separated PCI group inside the "using device" line, with
    # expected_pci set to the SAME joined value extract_used_pci would return.
    # Check 2 (used_pci == expected_pci) therefore still passes; check 1
    # (expected_pci is one well-formed PCI address) fails because the joined,
    # two-address string does not match the single-address shape.
    log = _base_log().replace(
        "(0000:03:00.0) - 30700 MiB free",
        "(0000:03:00.0) (0000:04:00.0) - 30700 MiB free",
    )
    joined = "0000:03:00.0\n0000:04:00.0"
    assert extract_used_pci(log) == joined

    checks = _run(log=log, expected_pci=joined)

    assert _check(checks, 1).passed is False
    assert _others_pass(checks, except_number=1)


def test_check1_live_sysfs_pci_mismatch_fails_only_check_one() -> None:
    checks = _run(live_pci="0000:04:00.0")

    assert _check(checks, 1).passed is False
    assert _others_pass(checks, except_number=1)


def test_check2_second_pci_in_using_device_line_fails_only_check_two() -> None:
    log = _base_log().replace(
        "(0000:03:00.0) - 30700 MiB free",
        "(0000:03:00.0) (0000:04:00.0) - 30700 MiB free",
    )

    checks = _run(log=log)

    assert _check(checks, 2).passed is False
    assert _others_pass(checks, except_number=2)


def test_check3_card_not_idle_before_load_fails_only_check_three() -> None:
    before = _base_before()
    before[_MEASURED_CARD] = MemorySnapshot(card=_MEASURED_CARD, vram_used_mib=2000.0, gtt_used_mib=5.0)

    checks = _run(before=before)

    assert _check(checks, 3).passed is False
    assert _others_pass(checks, except_number=3)


def test_check4_incomplete_offload_fails_only_check_four() -> None:
    log = _base_log().replace("offloaded 66/66 layers to GPU", "offloaded 65/66 layers to GPU")

    checks = _run(log=log)

    assert offload_counts(log) == (65, 66)
    assert _check(checks, 4).passed is False
    assert _others_pass(checks, except_number=4)


def test_check5_layer_assigned_elsewhere_fails_only_check_five() -> None:
    log = _base_log().replace(
        "layer  65 assigned to device Vulkan0, is_swa = 0",
        "layer  65 assigned to device Vulkan1, is_swa = 0",
        1,
    )

    checks = _run(log=log)

    assert layer_assignments(log) == {"Vulkan0": 65, "Vulkan1": 1}
    assert _check(checks, 5).passed is False
    assert _others_pass(checks, except_number=5)


def test_check6_vram_delta_too_small_fails_only_check_six() -> None:
    after = _base_after()
    after[_MEASURED_CARD] = MemorySnapshot(card=_MEASURED_CARD, vram_used_mib=150.0, gtt_used_mib=5.0)

    checks = _run(after=after)

    assert _check(checks, 6).passed is False
    assert _others_pass(checks, except_number=6)


def test_check7_gtt_spill_fails_only_check_seven() -> None:
    after = _base_after()
    after[_MEASURED_CARD] = MemorySnapshot(card=_MEASURED_CARD, vram_used_mib=5000.0, gtt_used_mib=5.000001)

    checks = _run(after=after)

    assert _check(checks, 7).passed is False
    assert _others_pass(checks, except_number=7)


def test_check8_other_card_disturbed_fails_only_check_eight() -> None:
    after = _base_after()
    after[_OTHER_CARD] = MemorySnapshot(card=_OTHER_CARD, vram_used_mib=650.0, gtt_used_mib=5.0)

    checks = _run(after=after)

    assert _check(checks, 8).passed is False
    assert _others_pass(checks, except_number=8)


@pytest.mark.parametrize("missing_from", ("before", "after"))
def test_check8_missing_other_card_snapshot_fails_only_check_eight(missing_from: str) -> None:
    before = _base_before()
    after = _base_after()
    if missing_from == "before":
        before.pop(_OTHER_CARD)
    else:
        after.pop(_OTHER_CARD)

    checks = _run(before=before, after=after)

    assert _check(checks, 8).passed is False
    assert _others_pass(checks, except_number=8)


def test_check9_flash_attention_only_in_wrapper_echo_fails_check_nine() -> None:
    log = _base_log()
    for real_line in (
        "llama_init_from_model: enabling flash_attn since it is required for quantized V cache\n",
        "llama_context: flash_attn    = enabled\n",
    ):
        log = log.replace(real_line, "")

    checks = _run(log=log)

    assert _check(checks, 9).passed is False
    assert _others_pass(checks, except_number=9)


def test_check9_flash_attention_reported_disabled_fails_check_nine() -> None:
    log = _base_log().replace(
        "llama_context: flash_attn    = enabled",
        "llama_context: flash_attn    = disabled",
    )

    checks = _run(log=log)

    assert _check(checks, 9).passed is False
    assert _others_pass(checks, except_number=9)


def test_check10_invalid_output_fails_only_check_ten() -> None:
    checks = _run(output_valid=False)

    assert _check(checks, 10).passed is False
    assert _others_pass(checks, except_number=10)


def test_check10_missing_output_fails_only_check_ten() -> None:
    checks = _run(output_valid=None)

    assert _check(checks, 10).passed is False
    assert _others_pass(checks, except_number=10)


# ---- first_failure -----------------------------------------------------------


def test_first_failure_returns_the_earliest_failed_check() -> None:
    log = _base_log().replace("offloaded 66/66 layers to GPU", "offloaded 65/66 layers to GPU")
    checks = _run(log=log, output_valid=False)

    failure = first_failure(checks)

    assert failure is not None
    assert failure.number == 4


def test_first_failure_returns_none_when_every_check_passes() -> None:
    assert first_failure(_run()) is None


# ---- the wrapper-echo exclusion rule, directly (fact 4) ----------------------


def test_wrapper_echo_line_is_never_a_flash_attention_report() -> None:
    log = _base_log()
    assert any(line.startswith(">>") and "--flash-attn" in line for line in log.splitlines())

    lines = flash_attention_lines(log)

    assert all(not line.startswith(">>") for line in lines)
    assert all("--flash-attn" not in line for line in lines)
    assert any("enabled" in line.lower() for line in lines)


# ---- the parsing helpers, directly -------------------------------------------


def test_extract_used_pci_ports_the_gate_regex_on_the_base_log() -> None:
    assert extract_used_pci(_base_log()) == _EXPECTED_PCI


def test_extract_used_pci_returns_none_when_the_device_is_not_mentioned() -> None:
    assert extract_used_pci("nothing relevant here") is None


def test_list_devices_names_reports_no_pci_fact_one() -> None:
    names = list_devices_names(_LIST_DEVICES_TEXT)

    assert names == ("Vulkan0",)
    assert extract_used_pci(_LIST_DEVICES_TEXT) is None  # fact 1: no PCI in --list-devices output


def test_offload_counts_is_case_insensitive() -> None:
    assert offload_counts("I load_tensors: OFFLOADED 12/12 LAYERS TO gpu") == (12, 12)


def test_offload_counts_returns_none_without_a_match() -> None:
    assert offload_counts("no offload line here") is None


# ---- sysfs_pci_address and memory_snapshot: the _make_card pattern ----------


def _make_device_dir(drm_root: Path, card: int, pci: str) -> Path:
    """The `tests/test_telemetry.py::_make_card` shape: a real PCI dir, symlinked."""
    real_device = drm_root / "pci-devices" / pci
    real_device.mkdir(parents=True)
    card_dir = drm_root / f"card{card}"
    card_dir.mkdir(parents=True)
    device = card_dir / "device"
    device.symlink_to(real_device, target_is_directory=True)
    return device


def test_sysfs_pci_address_reads_the_symlinked_device_dir(tmp_path: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    _make_device_dir(drm_root, 1, "0000:03:00.0")

    assert sysfs_pci_address(1, drm_root=drm_root) == "0000:03:00.0"


def test_memory_snapshot_reads_vram_and_gtt_from_sysfs(tmp_path: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    device = _make_device_dir(drm_root, 1, "0000:03:00.0")
    (device / "mem_info_vram_used").write_text(str(1024 * 1024 * 200) + "\n")
    (device / "mem_info_gtt_used").write_text(str(1024 * 1024 * 7) + "\n")

    snapshot = memory_snapshot(1, drm_root=drm_root)

    assert snapshot.card == 1
    assert snapshot.vram_used_mib == 200.0
    assert snapshot.gtt_used_mib == 7.0
