"""GPU environment facts for the bench harness.

The isolation rule is load-bearing: a job is assigned one card, and a
benchmark must never decode on a card the serving router owns. A benchmark also
records a snapshot of the cards it can see, so a result carries the device
identity it was measured on.

The snapshot is backend-neutral. It reads ``nvidia-smi`` when that tool answers
with real rows. Otherwise it reads the amdgpu sysfs files, which is what a
Vulkan rig without NVIDIA cards has. Both give the same line format:
``index, name, id, driver, total MiB, used MiB``. Here ``id`` is the NVIDIA UUID
or the PCI address of the card. The index is the NVIDIA index, or the position
in sorted DRM card order, the same order that ``ci/runner/gpu-lease.sh`` uses.

Every check here fails closed. A snapshot that cannot tell is an error, never a
pass.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from inferference.gpu_env import VISIBLE_DEVICE_VARS, visible_devices

_QUERY = "index,name,uuid,driver_version,memory.total,memory.used"

# Where amdgpu cards show up. Tests point this at a fake tree.
DRM_ROOT = Path("/sys/class/drm")

# The model must fill at least this much VRAM on the CI card.
MIN_MODEL_MIB = 4000
# Another card may hold at most this much, or gain at most this much.
OTHER_CARD_MAX_MIB = 200
# Host memory the GPU reaches over PCIe (GTT) may grow by at most this much on
# the CI card while the model loads. The read staging buffer is 16 MiB. A real
# spill is hundreds of MiB or more.
GTT_RISE_LIMIT_MIB = 128

_ROW = re.compile(r"^\s*\d+,[^,]*,[^,]*,[^,]*,\s*\d+ MiB,\s*\d+ MiB\s*$")


def ci_device() -> str:
    """The CI device index (LLAMA_CI_GPU), defaulting to 0.

    Named-role rule: asserts compare against this variable, never the
    literal index and never a presence check.
    """
    return os.environ.get("LLAMA_CI_GPU", "0")


def require_ci_device() -> None:
    """Fail closed unless every device-visibility variable that is set names the CI device.

    The variables are the ones in ``VISIBLE_DEVICE_VARS``. A CUDA build reads
    one, a Vulkan build reads another, and each ignores the rest. So every
    variable that is set must name the CI device, and at least one must be set.
    The message names the expected and the actual values so a mis-routed run is
    refused loudly, not silently.
    """
    expected = ci_device()
    present = {var: os.environ[var] for var in VISIBLE_DEVICE_VARS if var in os.environ}
    if not present:
        names = ", ".join(VISIBLE_DEVICE_VARS)
        raise RuntimeError(f"CI device not pinned: none of {names} is set, expected {expected!r} (LLAMA_CI_GPU)")
    wrong = {var: value for var, value in present.items() if value != expected}
    if wrong:
        shown = ", ".join(f"{var}={value!r}" for var, value in wrong.items())
        raise RuntimeError(f"CI device mismatch: {shown}, expected {expected!r} (LLAMA_CI_GPU)")


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _drm_cards(root: Path) -> list[dict[str, Any]]:
    """The amdgpu cards under ``root``, in sorted DRM card order.

    Same rule as ``ci/runner/gpu-lease.sh``: a card counts when it reports its
    VRAM total and used bytes. The index is the position in that order.
    """

    def card_number(path: Path) -> int:
        return int(path.name.removeprefix("card"))

    cards: list[dict[str, Any]] = []
    candidates = [p for p in root.glob("card[0-9]*") if p.name.removeprefix("card").isdigit()]
    for entry in sorted(candidates, key=card_number):
        device = entry / "device"
        total = _read_int(device / "mem_info_vram_total")
        used = _read_int(device / "mem_info_vram_used")
        if total is None or used is None:
            continue
        gtt = _read_int(device / "mem_info_gtt_used")
        pci = device.resolve().name
        name = (_read_text(device / "product_name") or "amdgpu").replace(",", " ")
        cards.append(
            {
                "index": len(cards),
                "name": name,
                "id": pci,
                "total_mib": total // 1048576,
                "used_mib": used // 1048576,
                "gtt_bytes": gtt,
            }
        )
    return cards


def _nvidia_rows() -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader"], capture_output=True, text=True
        )
    except OSError as exc:
        return 127, "", str(exc)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def gpu_snapshot() -> dict[str, Any]:
    """Snapshot of every GPU (index, name, id, driver, memory).

    Keys: ``returncode``, ``stdout`` (one line per card), ``stderr``, ``source``
    (``nvidia-smi``, ``amdgpu-sysfs`` or ``none``) and ``gtt_bytes`` (per card,
    or ``None`` when the source has no GTT counter).

    ``nvidia-smi`` counts only when it prints real rows. A wrapper that prints a
    driver error on stdout is not a GPU. Then the amdgpu sysfs files answer. A
    host with neither reads as returncode 127 with empty stdout, so
    ``require_gpu0_only`` still fails closed on it.
    """
    rc, out, err = _nvidia_rows()
    if rc == 0 and out and all(_ROW.match(line) for line in out.splitlines()):
        return {"returncode": 0, "stdout": out, "stderr": err, "source": "nvidia-smi", "gtt_bytes": None}
    cards = _drm_cards(DRM_ROOT)
    if cards:
        lines = [
            f"{c['index']}, {c['name']}, {c['id']}, unknown, {c['total_mib']} MiB, {c['used_mib']} MiB" for c in cards
        ]
        return {
            "returncode": 0,
            "stdout": "\n".join(lines),
            "stderr": "",
            "source": "amdgpu-sysfs",
            "gtt_bytes": [c["gtt_bytes"] for c in cards],
        }
    note = f"nvidia-smi gave no GPU rows (rc {rc}: {err or out[:80]!r}) and {DRM_ROOT} holds no amdgpu card"
    return {"returncode": rc or 127, "stdout": "", "stderr": note, "source": "none", "gtt_bytes": None}


def visible_gpu_uuid() -> str | None:
    """The identity (UUID, or PCI address on amdgpu) of the GPU that the environment pins, else None.

    The index is a function of the device-visibility variable and reads 0 for
    whichever card was exposed; only the UUID or the PCI address names a card.
    No variable set, an unreadable index, or no GPU source all read as None.
    """
    visible = visible_devices()
    if not visible:
        return None
    try:
        phys = int(visible.split(",")[0])
    except ValueError:
        return None
    snapshot = gpu_snapshot()
    if snapshot["returncode"] != 0:
        return None
    lines = snapshot["stdout"].splitlines()
    if phys >= len(lines):
        return None
    return lines[phys].split(", ")[2]


def _cards(snapshot: dict[str, Any]) -> list[tuple[int, int]]:
    """(total MiB, used MiB) per card, in snapshot order. Raise on a row that does not parse."""
    cards: list[tuple[int, int]] = []
    for line in snapshot["stdout"].splitlines():
        fields = line.split(", ")
        try:
            cards.append((int(fields[-2].split()[0]), int(fields[-1].split()[0])))
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"unreadable GPU snapshot row: {line!r}") from exc
    return cards


def used_mib(snapshot: dict[str, Any]) -> list[int]:
    """Used MiB per card, in snapshot order."""
    return [used for _total, used in _cards(snapshot)]


def ci_card_index(snapshot: dict[str, Any], pci: str | None = None) -> int:
    """The snapshot position of the CI card.

    The card is named by its identity when the caller has one: ``pci`` (the PCI
    address on amdgpu, the UUID on NVIDIA), or ``GPU_LEASE_CARD_UUID`` that
    ``ci/runner/gpu-lease.sh`` exports. The snapshot order is the sorted DRM
    card order, and a Vulkan or driver enumeration can differ from it. A match
    by identity does not depend on any order. A named card that the snapshot
    lacks is an error, never a fallback to the index.

    With no identity, the card is ``LLAMA_CI_GPU``, an index into the snapshot.
    """
    count = len(snapshot["stdout"].splitlines())
    wanted = (pci if pci is not None else os.environ.get("GPU_LEASE_CARD_UUID", "")).strip()
    if wanted:
        ids = [line.split(", ")[2].strip().lower() for line in snapshot["stdout"].splitlines()]
        if wanted.lower() not in ids:
            raise RuntimeError(f"CI card {wanted!r} is not in the GPU snapshot (ids {ids})")
        return ids.index(wanted.lower())
    ci = int(ci_device())
    if not 0 <= ci < count:
        raise RuntimeError(f"CI device {ci} is not in the GPU snapshot ({count} cards)")
    return ci


def require_gpu0_only(
    snapshot: dict[str, Any],
    before: dict[str, Any] | None = None,
    *,
    min_model_mib: int = MIN_MODEL_MIB,
    ci_pci: str | None = None,
) -> None:
    """Fail-closed GPU isolation: the model is on the CI card and on no other.

    Needs at least one card in the snapshot. With one card the other-card check
    has nothing to compare and holds vacuously. The CI card is found by
    ``ci_card_index``: by identity when ``ci_pci`` or ``GPU_LEASE_CARD_UUID``
    names it, else ``LLAMA_CI_GPU`` as an index. ``min_model_mib`` is the least
    VRAM that the model must fill (``MIN_MODEL_MIB`` by default).

    Without ``before``, the snapshot is absolute: the CI card holds at least
    ``min_model_mib`` and every other card holds at most ``OTHER_CARD_MAX_MIB``.
    That rule needs the other cards idle. On a source with a GTT counter, the
    CI card must also hold at most ``GTT_RISE_LIMIT_MIB`` of GTT, because RADV
    puts an overflow in host memory and still reports success.

    With ``before`` (a snapshot taken before the model loaded), the check is on
    the change: the CI card rose by at least ``min_model_mib``, no other card
    rose by more than ``OTHER_CARD_MAX_MIB``, and on a source with a GTT
    counter the CI card's GTT rose by at most ``GTT_RISE_LIMIT_MIB``. A card
    that serves another model during the run is then allowed. The GTT rule
    catches a model that spilled into host memory while the VRAM looks right.
    """
    after_cards = used_mib(snapshot)
    if not after_cards:
        raise RuntimeError(f"expected a GPU in the GPU snapshot (source {snapshot.get('source', 'unknown')!r})")
    ci = ci_card_index(snapshot, ci_pci)
    others = [i for i in range(len(after_cards)) if i != ci]
    gtt_after = snapshot.get("gtt_bytes")

    if before is None:
        if after_cards[ci] < min_model_mib or any(after_cards[i] > OTHER_CARD_MAX_MIB for i in others):
            shown = ", ".join(f"card {i}={after_cards[i]} MiB" for i in others) or "none"
            raise RuntimeError(f"GPU isolation rejected: CI device={after_cards[ci]} MiB, others: {shown}")
        if gtt_after is not None:
            if len(gtt_after) != len(after_cards) or gtt_after[ci] is None:
                raise RuntimeError("GPU isolation rejected: cannot read GTT of the CI device")
            if gtt_after[ci] > GTT_RISE_LIMIT_MIB * 1048576:
                raise RuntimeError(
                    f"GPU isolation rejected: GTT on the CI device holds {gtt_after[ci]} bytes"
                    f" (limit {GTT_RISE_LIMIT_MIB} MiB): the model spilled into host memory"
                )
        return

    before_cards = used_mib(before)
    if len(before_cards) != len(after_cards):
        raise RuntimeError(
            f"GPU isolation rejected: {len(before_cards)} cards before the load, {len(after_cards)} after"
        )
    rise = [a - b for a, b in zip(after_cards, before_cards, strict=True)]
    if rise[ci] < min_model_mib or any(rise[i] > OTHER_CARD_MAX_MIB for i in others):
        shown = ", ".join(f"card {i} {rise[i]:+d} MiB" for i in others) or "none"
        raise RuntimeError(f"GPU isolation rejected: CI device {rise[ci]:+d} MiB, others: {shown}")
    gtt_before = before.get("gtt_bytes")
    if gtt_after is not None:
        if gtt_before is None or len(gtt_before) != len(gtt_after) or None in (gtt_after[ci], gtt_before[ci]):
            raise RuntimeError("GPU isolation rejected: cannot compare GTT before and after the load")
        gtt_rise = gtt_after[ci] - gtt_before[ci]
        if gtt_rise > GTT_RISE_LIMIT_MIB * 1048576:
            raise RuntimeError(
                f"GPU isolation rejected: GTT on the CI device rose {gtt_rise} bytes (limit {GTT_RISE_LIMIT_MIB} MiB):"
                " the model spilled into host memory"
            )


def driver_version(snapshot: dict[str, Any]) -> str | None:
    """The driver version from a snapshot (column 4), or None when the source does not give one.

    ``nvidia-smi`` gives it. The amdgpu sysfs source writes ``unknown``.
    """
    lines = snapshot["stdout"].splitlines()
    if not lines:
        return None
    fields = lines[0].split(", ")
    if len(fields) <= 3 or fields[3] in ("", "unknown"):
        return None
    return fields[3]
