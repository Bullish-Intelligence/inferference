"""CPU-only tests for the backend-neutral GPU isolation checks in ``benchkit.env``.

A fake DRM tree stands in for amdgpu sysfs, and ``nvidia-smi`` is absent, as on
a Vulkan rig. Every check must still fail closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inferference.benchkit import env as benchkit_env
from inferference.gpu_env import VISIBLE_DEVICE_VARS

MIB = 1048576
PCI = ("0000:19:00.0", "0000:67:00.0", "0000:68:00.0")


def _make_card(root: Path, number: int, *, used_mib: int, gtt_bytes: int, total_mib: int = 32768) -> None:
    pci_dir = root / "pci" / PCI[number]
    pci_dir.mkdir(parents=True, exist_ok=True)
    (pci_dir / "mem_info_vram_total").write_text(f"{total_mib * MIB}\n")
    (pci_dir / "mem_info_vram_used").write_text(f"{used_mib * MIB}\n")
    (pci_dir / "mem_info_gtt_used").write_text(f"{gtt_bytes}\n")
    card = root / "drm" / f"card{number}"
    card.mkdir(parents=True, exist_ok=True)
    (card / "device").symlink_to(pci_dir)


def _fake_drm(tmp_path: Path, cards: list[tuple[int, int]]) -> Path:
    for number, (used, gtt) in enumerate(cards):
        _make_card(tmp_path, number, used_mib=used, gtt_bytes=gtt)
    return tmp_path / "drm"


@pytest.fixture(autouse=True)
def _no_gpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no device variable and no nvidia-smi."""
    for var in VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLAMA_CI_GPU", "0")
    monkeypatch.delenv("LLAMA_SERVER_GPU", raising=False)
    monkeypatch.delenv("GPU_LEASE_CARD_UUID", raising=False)
    monkeypatch.setattr(benchkit_env, "_nvidia_rows", lambda: (127, "", "nvidia-smi: not found"))


def _snap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cards: list[tuple[int, int]]) -> dict[str, Any]:
    """Snapshot of a fake DRM tree. Each card is (used MiB, GTT bytes). Each call needs a new tmp dir."""
    monkeypatch.setattr(benchkit_env, "DRM_ROOT", _fake_drm(tmp_path, cards))
    return benchkit_env.gpu_snapshot()


def _snapshot_dirs(tmp_path: Path) -> tuple[Path, Path]:
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    return before, after


# --- the snapshot -------------------------------------------------------------------------------


def test_snapshot_reads_amdgpu_sysfs_when_nvidia_smi_is_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snap = _snap(monkeypatch, tmp_path, [(9000, 15 * MIB), (300, 15 * MIB)])
    assert snap["returncode"] == 0
    assert snap["source"] == "amdgpu-sysfs"
    assert benchkit_env.used_mib(snap) == [9000, 300]
    assert snap["gtt_bytes"] == [15 * MIB, 15 * MIB]
    assert snap["stdout"].splitlines()[0].split(", ")[2] == PCI[0]
    assert benchkit_env.driver_version(snap) is None


def test_snapshot_orders_cards_by_drm_number(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    drm = _fake_drm(tmp_path, [(100, 0), (200, 0), (300, 0)])
    # card10 must sort after card2, not before it.
    extra = drm / "card10"
    extra.mkdir()
    (extra / "device").symlink_to(tmp_path / "pci" / PCI[0])
    monkeypatch.setattr(benchkit_env, "DRM_ROOT", drm)
    assert benchkit_env.used_mib(benchkit_env.gpu_snapshot()) == [100, 200, 300, 100]


def test_snapshot_with_no_gpu_source_is_an_error_row_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "drm").mkdir()
    monkeypatch.setattr(benchkit_env, "DRM_ROOT", tmp_path / "drm")
    snap = benchkit_env.gpu_snapshot()
    assert snap["returncode"] != 0
    assert snap["stdout"] == ""
    assert snap["source"] == "none"
    with pytest.raises(RuntimeError, match="expected a GPU"):
        benchkit_env.require_gpu0_only(snap)


def test_snapshot_ignores_a_card_without_vram_counters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    drm = _fake_drm(tmp_path, [(100, 0)])
    stray = drm / "card1"
    stray.mkdir()
    (stray / "device").mkdir()
    monkeypatch.setattr(benchkit_env, "DRM_ROOT", drm)
    assert benchkit_env.used_mib(benchkit_env.gpu_snapshot()) == [100]


def test_snapshot_prefers_real_nvidia_smi_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = (
        "0, NVIDIA Example GPU, GPU-a, 595.84, 12288 MiB, 5000 MiB\n"
        "1, NVIDIA Example GPU, GPU-b, 595.84, 12288 MiB, 100 MiB"
    )
    monkeypatch.setattr(benchkit_env, "_nvidia_rows", lambda: (0, rows, ""))
    monkeypatch.setattr(benchkit_env, "DRM_ROOT", _fake_drm(tmp_path, [(1, 0), (1, 0)]))
    snap = benchkit_env.gpu_snapshot()
    assert snap["source"] == "nvidia-smi"
    assert benchkit_env.used_mib(snap) == [5000, 100]
    assert benchkit_env.driver_version(snap) == "595.84"


def test_a_driver_error_on_stdout_is_not_a_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """nvidia-smi wrappers print their error on stdout with exit 0. That must not read as a card."""
    monkeypatch.setattr(benchkit_env, "_nvidia_rows", lambda: (0, "NVIDIA-SMI has failed", ""))
    (tmp_path / "drm").mkdir()
    monkeypatch.setattr(benchkit_env, "DRM_ROOT", tmp_path / "drm")
    snap = benchkit_env.gpu_snapshot()
    assert snap["source"] == "none"
    assert snap["stdout"] == ""


# --- the isolation check ------------------------------------------------------------------------


def test_absolute_check_accepts_the_model_on_the_ci_card_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    benchkit_env.require_gpu0_only(_snap(monkeypatch, tmp_path, [(9000, 0), (100, 0)]))


@pytest.mark.parametrize(
    "cards",
    [
        [(9000, 0), (8000, 0)],  # the other card is busy
        [(100, 0), (100, 0)],  # the model is on no card
        [(100, 0), (9000, 0)],  # the model is on the wrong card
    ],
)
def test_absolute_check_rejects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cards: list[tuple[int, int]]) -> None:
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(_snap(monkeypatch, tmp_path, cards))


def test_a_single_card_passes_when_the_model_fills_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With one card the other-card check is vacuous. The model and GTT checks still apply."""
    benchkit_env.require_gpu0_only(_snap(monkeypatch, tmp_path, [(9000, 15 * MIB)]))


def test_a_single_card_without_the_model_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(_snap(monkeypatch, tmp_path, [(100, 0)]))


def test_a_single_card_delta_check_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    d1, d2 = _snapshot_dirs(tmp_path)
    before = _snap(monkeypatch, d1, [(200, 15 * MIB)])
    after = _snap(monkeypatch, d2, [(9200, 16 * MIB)])
    benchkit_env.require_gpu0_only(after, before=before)


def test_min_model_mib_is_a_parameter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A 2.6 GB model fills less than the default floor. The caller lowers the floor."""
    d1, d2, d3 = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (d1, d2, d3):
        d.mkdir()
    small = _snap(monkeypatch, d1, [(2700, 0), (100, 0)])
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(small)
    benchkit_env.require_gpu0_only(small, min_model_mib=2000)
    before = _snap(monkeypatch, d2, [(100, 0), (100, 0)])
    after = _snap(monkeypatch, d3, [(2800, 0), (100, 0)])
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(after, before=before)
    benchkit_env.require_gpu0_only(after, before=before, min_model_mib=2000)
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(small, min_model_mib=3000)


def test_absolute_check_rejects_a_gtt_spill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The absolute mode (benchkit/ab.py) checks GTT too: AGENTS.md rule 3."""
    snap = _snap(monkeypatch, tmp_path, [(9000, 900 * MIB), (100, 15 * MIB)])
    with pytest.raises(RuntimeError, match="spilled into host memory"):
        benchkit_env.require_gpu0_only(snap)


def test_absolute_check_accepts_a_small_gtt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    benchkit_env.require_gpu0_only(_snap(monkeypatch, tmp_path, [(9000, 16 * MIB), (100, 900 * MIB)]))


def test_absolute_check_fails_closed_on_an_unreadable_ci_gtt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snap = _snap(monkeypatch, tmp_path, [(9000, 0), (100, 0)])
    snap["gtt_bytes"] = [None, 0]
    with pytest.raises(RuntimeError, match="cannot read GTT"):
        benchkit_env.require_gpu0_only(snap)


def test_absolute_check_has_no_gtt_rule_on_a_source_without_a_counter() -> None:
    rows = "0, NVIDIA, GPU-a, 595.84, 12288 MiB, 5000 MiB\n1, NVIDIA, GPU-b, 595.84, 12288 MiB, 100 MiB"
    benchkit_env.require_gpu0_only({"stdout": rows, "gtt_bytes": None})


def test_the_card_is_matched_by_pci_address_not_drm_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The model sits on the card at PCI[1], which is DRM card1. LLAMA_CI_GPU says 0, and is wrong."""
    snap = _snap(monkeypatch, tmp_path, [(100, 0), (9000, 0)])
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(snap)  # by index: card 0 is cold
    benchkit_env.require_gpu0_only(snap, ci_pci=PCI[1])
    assert benchkit_env.ci_card_index(snap, PCI[1].upper()) == 1


def test_the_lease_variable_names_the_card(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snap = _snap(monkeypatch, tmp_path, [(100, 0), (9000, 0)])
    monkeypatch.setenv("GPU_LEASE_CARD_UUID", PCI[1])
    benchkit_env.require_gpu0_only(snap)
    # The other card is busy: matching by address must still check it.
    busy = _snap(monkeypatch, tmp_path / "busy", [(8000, 0), (9000, 0)])
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(busy)


def test_a_named_card_that_is_not_in_the_snapshot_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snap = _snap(monkeypatch, tmp_path, [(9000, 0), (100, 0)])
    with pytest.raises(RuntimeError, match="not in the GPU snapshot"):
        benchkit_env.require_gpu0_only(snap, ci_pci="0000:ff:00.0")


def test_ci_card_outside_the_snapshot_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLAMA_CI_GPU", "5")
    with pytest.raises(RuntimeError, match="not in the GPU snapshot"):
        benchkit_env.require_gpu0_only(_snap(monkeypatch, tmp_path, [(9000, 0), (100, 0)]))


def test_a_third_busy_card_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(_snap(monkeypatch, tmp_path, [(9000, 0), (100, 0), (5000, 0)]))


def test_delta_check_allows_a_card_that_serves_another_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The other card holds a resident model before and after. The load rose only the CI card."""
    d1, d2 = _snapshot_dirs(tmp_path)
    before = _snap(monkeypatch, d1, [(200, 15 * MIB), (20000, 15 * MIB)])
    after = _snap(monkeypatch, d2, [(9200, 31 * MIB), (20010, 15 * MIB)])
    benchkit_env.require_gpu0_only(after, before=before)


def test_delta_check_rejects_the_model_landing_on_the_other_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    d1, d2 = _snapshot_dirs(tmp_path)
    before = _snap(monkeypatch, d1, [(200, 0), (200, 0)])
    after = _snap(monkeypatch, d2, [(300, 0), (9000, 0)])
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(after, before=before)


def test_delta_check_rejects_a_model_that_split_over_both_cards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    d1, d2 = _snapshot_dirs(tmp_path)
    before = _snap(monkeypatch, d1, [(200, 0), (200, 0)])
    after = _snap(monkeypatch, d2, [(5000, 0), (4000, 0)])
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(after, before=before)


def test_delta_check_rejects_a_card_that_was_already_full(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A stale occupant must not pass for the model: the rise is what counts."""
    d1, d2 = _snapshot_dirs(tmp_path)
    before = _snap(monkeypatch, d1, [(9000, 0), (100, 0)])
    after = _snap(monkeypatch, d2, [(9000, 0), (100, 0)])
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(after, before=before)


def test_delta_check_rejects_a_gtt_spill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """VRAM rose as expected, but the model also filled host memory over PCIe."""
    d1, d2 = _snapshot_dirs(tmp_path)
    before = _snap(monkeypatch, d1, [(200, 15 * MIB), (100, 15 * MIB)])
    after = _snap(monkeypatch, d2, [(9200, 15 * MIB + 900 * MIB), (100, 15 * MIB)])
    with pytest.raises(RuntimeError, match="spilled into host memory"):
        benchkit_env.require_gpu0_only(after, before=before)


def test_delta_check_fails_closed_when_the_before_snapshot_has_no_gtt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    d1, d2 = _snapshot_dirs(tmp_path)
    before = _snap(monkeypatch, d1, [(200, 0), (100, 0)])
    before["gtt_bytes"] = None
    after = _snap(monkeypatch, d2, [(9200, 0), (100, 0)])
    with pytest.raises(RuntimeError, match="cannot compare GTT"):
        benchkit_env.require_gpu0_only(after, before=before)


def test_delta_check_rejects_an_empty_before_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    after = _snap(monkeypatch, tmp_path, [(9200, 0), (100, 0)])
    empty = {"returncode": 127, "stdout": "", "stderr": "", "source": "none", "gtt_bytes": None}
    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        benchkit_env.require_gpu0_only(after, before=empty)


def test_an_unreadable_row_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="unreadable GPU snapshot row"):
        benchkit_env.used_mib({"stdout": "0, a, b, c, lots, many"})


# --- the pin ------------------------------------------------------------------------------------


def test_require_ci_device_accepts_the_vulkan_variable_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "0")
    benchkit_env.require_ci_device()


def test_require_ci_device_refuses_a_run_with_no_pin() -> None:
    with pytest.raises(RuntimeError, match="not pinned") as info:
        benchkit_env.require_ci_device()
    assert "GGML_VK_VISIBLE_DEVICES" in str(info.value)


def test_require_ci_device_refuses_one_wrong_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA pinned right, Vulkan pinned wrong: a Vulkan build would use the wrong card."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="GGML_VK_VISIBLE_DEVICES='1'"):
        benchkit_env.require_ci_device()


@pytest.mark.parametrize("value", ["", "0,1", "1"])
def test_require_ci_device_refuses_a_pin_that_is_not_exactly_the_ci_device(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", value)
    with pytest.raises(RuntimeError, match="CI device mismatch"):
        benchkit_env.require_ci_device()


# --- the card identity --------------------------------------------------------------------------


def test_visible_gpu_uuid_is_the_pci_address_of_the_pinned_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(benchkit_env, "DRM_ROOT", _fake_drm(tmp_path, [(100, 0), (100, 0)]))
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1")
    assert benchkit_env.visible_gpu_uuid() == PCI[1]


def test_visible_gpu_uuid_is_none_with_no_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(benchkit_env, "DRM_ROOT", _fake_drm(tmp_path, [(100, 0), (100, 0)]))
    assert benchkit_env.visible_gpu_uuid() is None


def test_visible_gpu_uuid_is_none_for_a_card_that_does_not_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(benchkit_env, "DRM_ROOT", _fake_drm(tmp_path, [(100, 0)]))
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "3")
    assert benchkit_env.visible_gpu_uuid() is None


def test_visible_gpu_uuid_is_none_with_no_gpu_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "drm").mkdir()
    monkeypatch.setattr(benchkit_env, "DRM_ROOT", tmp_path / "drm")
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "0")
    assert benchkit_env.visible_gpu_uuid() is None
