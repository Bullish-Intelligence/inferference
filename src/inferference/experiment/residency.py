"""The placement proof: did a model load land on the card it was told to.

llama.cpp never refuses an overcommit (`AGENTS.md`, "NO CPU FALLBACK. EVER").
A spilled load returns `{"success": true}`, reaches status `loaded`, and still
names the right Vulkan device in every layer-assignment line. The server log
and the `--list-devices` text are not proof on their own; sysfs VRAM and GTT
accounting are. This module runs the ten checks that together ARE proof, ports
the device-to-PCI extraction byte-for-byte from `ci/gates/vulkan_mi25_smoke.sh`
lines 147-149, and corrects three facts an earlier harness got wrong.

Corrected facts (verified against a real stored run artifact):

1. `llama-server --list-devices` prints no PCI address. It names a logical
   device only (`Vulkan0: AMD Radeon Pro V620 (RADV NAVI21) (...)`).
2. The PCI address lives on one server-log line instead:
   `... using device Vulkan0 (AMD Radeon Pro V620 (RADV NAVI21)) (0000:03:00.0) ...`.
3. The second log line a naive reader calls "a flash-attention mention" is the
   wrapper's own argv echo (`>> ... --flash-attn auto ...`), not the server
   reporting its own state. This module excludes every line that starts with
   `>>` and every line that contains `--flash-attn`.

This module never reads live hardware except through `sysfs_pci_address` and
`memory_snapshot`, and both take `drm_root` as a constructor-style parameter
with the real `/sys/class/drm` as the default, so a test fabricates a tree
under `tmp_path` instead. Every other function parses log TEXT or sysfs
snapshots a caller already collected; none of them touch a filesystem, start
a process, or read a clock. Every one of the ten checks is a pass/fail
boolean with its observed value recorded: there is no "probably fine" result.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_PCI_CORE = r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]"
_PCI_FULL_RE = re.compile(rf"^{_PCI_CORE}$")
_PCI_IN_PARENS_RE = re.compile(rf"\({_PCI_CORE}\)")
_DEVICE_LINE_RE = re.compile(r"^\s*(Vulkan\d+):", re.MULTILINE)
_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers?\s+to GPU", re.IGNORECASE)
_LAYER_ASSIGNED_RE = re.compile(r"assigned to device (\w+)")
_FLASH_ATTENTION_RE = re.compile(r"flash[_ -]?att(?:n|ention)", re.IGNORECASE)

_MEASURED_LOGICAL_DEVICE = "Vulkan0"


class ResidencyError(RuntimeError):
    """A caller gave `prove_residency` input it cannot evaluate, or a sysfs read failed."""


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """One card's VRAM and GTT accounting, at one instant, from its own sysfs."""

    card: int
    vram_used_mib: float
    gtt_used_mib: float


def server_baseline(log: str, *, card: int, pci: str) -> MemorySnapshot:
    """Read the server's post-initialization, pre-load memory counters."""
    pattern = re.compile(rf"^INFERFERENCE_BASELINE card=(\d+) pci=({_PCI_CORE}) gtt=(\d+) vram=(\d+)$", re.MULTILINE)
    matches = pattern.findall(log)
    if len(matches) != 1:
        raise ResidencyError("server must report exactly one memory baseline")
    actual_card, actual_pci, gtt, vram = matches[0]
    if int(actual_card) != card or actual_pci.lower() != pci.lower():
        raise ResidencyError("server baseline names the wrong card or PCI address")
    gtt_bytes, vram_bytes = int(gtt), int(vram)
    if gtt_bytes >= 2**64 or vram_bytes >= 2**64:
        raise ResidencyError("server baseline counters exceed 64 bits")
    return MemorySnapshot(card=card, vram_used_mib=vram_bytes / 1048576, gtt_used_mib=gtt_bytes / 1048576)


@dataclass(frozen=True, slots=True)
class ResidencyCheck:
    """One of the ten checks: its number, name, pass/fail, and what was observed."""

    number: int
    name: str
    passed: bool
    observed: str


# A card with this much VRAM in use (MiB) is not idle. Preflight reuses it for the peer card.
IDLE_VRAM_MAX_MIB = 1024.0


@dataclass(frozen=True, slots=True, kw_only=True)
class ResidencyThresholds:
    """The three numbers the ten checks compare against.

    `min_vram_delta_mib` and `gtt_limit_mib` have no default: they come from
    the run, not from this module. `min_vram_delta_mib` is the resident
    model's footprint `base_mib` (`inferference.placement.Footprint`) — a
    resident model commits at least its lane-independent cost, so a smaller
    rise is evidence of a partial or missing load. `gtt_limit_mib` limits
    changes on other cards. The measured card permits no GTT increase.
    Fields are keyword-only so the one
    field with a default (`idle_vram_max_mib`) can sit first without forcing
    the other two to also carry one.
    """

    idle_vram_max_mib: float = IDLE_VRAM_MAX_MIB
    min_vram_delta_mib: float
    gtt_limit_mib: float


def sysfs_pci_address(card: int, drm_root: Path = Path("/sys/class/drm")) -> str:
    """The PCI address DRM reports for `card`: `basename(readlink -f <drm_root>/card<N>/device)`."""
    device_dir = drm_root / f"card{card}" / "device"
    try:
        return device_dir.resolve(strict=True).name
    except OSError as exc:
        raise ResidencyError(f"could not resolve {device_dir}: {exc}") from exc


def memory_snapshot(card: int, drm_root: Path = Path("/sys/class/drm")) -> MemorySnapshot:
    """Read `card`'s current VRAM and GTT use from its own sysfs accounting.

    Reads `mem_info_vram_used` and `mem_info_gtt_used` directly, never
    `CPU_Mapped model buffer size` from a log: that line is normal for
    non-offloaded embedding weights on a perfectly resident model
    (`AGENTS.md`, "Do NOT gate on `CPU_Mapped`").
    """
    device_dir = drm_root / f"card{card}" / "device"
    try:
        vram_bytes = int((device_dir / "mem_info_vram_used").read_text().strip())
        gtt_bytes = int((device_dir / "mem_info_gtt_used").read_text().strip())
    except (OSError, ValueError) as exc:
        raise ResidencyError(f"could not read memory accounting for card {card} under {device_dir}: {exc}") from exc
    return MemorySnapshot(card=card, vram_used_mib=vram_bytes / 1_048_576, gtt_used_mib=gtt_bytes / 1_048_576)


def extract_used_pci(log: str, logical_device: str = "Vulkan0") -> str | None:
    """Port of `ci/gates/vulkan_mi25_smoke.sh:147-149`'s PCI extraction, exactly.

    Finds the first line matching
    `using device <logical_device> \\(.*\\) \\([0-9a-fA-F:.]+\\)`, takes ONLY the
    matched substring (as `grep -o` does, not the whole line), then finds
    every `(XXXX:XX:XX.X)`-shaped match inside that substring and joins them
    with newlines, parentheses stripped. More than one match therefore never
    equals a single expected PCI address — this is deliberate, never the
    weaker substring `or` test `run_dual_card_arms.py:331` used.
    """
    using_device_re = re.compile(rf"using device {re.escape(logical_device)} \(.*\) \([0-9a-fA-F:.]+\)")
    matched_text: str | None = None
    for line in log.splitlines():
        found = using_device_re.search(line)
        if found is not None:
            matched_text = found.group(0)
            break
    if matched_text is None:
        return None
    pcis = _PCI_IN_PARENS_RE.findall(matched_text)
    if not pcis:
        return None
    return "\n".join(pci.strip("()") for pci in pcis)


def list_devices_names(text: str) -> tuple[str, ...]:
    """Every logical device name `--list-devices` printed, in order (fact 1: no PCI here)."""
    return tuple(_DEVICE_LINE_RE.findall(text))


def offload_counts(log: str) -> tuple[int, int] | None:
    """`(offloaded, total)` from an `offloaded N/M layers to GPU` line, or `None`."""
    match = _OFFLOAD_RE.search(log)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def layer_assignments(log: str) -> dict[str, int]:
    """How many `assigned to device <X>` lines named each logical device."""
    counts: dict[str, int] = {}
    for device in _LAYER_ASSIGNED_RE.findall(log):
        counts[device] = counts.get(device, 0) + 1
    return counts


def flash_attention_lines(log: str) -> tuple[str, ...]:
    """Lines that report the server's OWN flash-attention state.

    Excludes every line that starts with `>>` (the wrapper's argv echo, fact
    4 — not the server speaking) and every line that contains `--flash-attn`
    (the flag text, not a state report), even outside the `>>` prefix.
    """
    lines = []
    for line in log.splitlines():
        if line.startswith(">>"):
            continue
        if "--flash-attn" in line:
            continue
        if _FLASH_ATTENTION_RE.search(line):
            lines.append(line)
    return tuple(lines)


def _card_delta(
    before: Mapping[int, MemorySnapshot],
    after: Mapping[int, MemorySnapshot],
    card: int,
    field: str,
) -> float:
    if card not in before or card not in after:
        raise ResidencyError(f"card {card} is missing from the before/after snapshot mapping")
    return getattr(after[card], field) - getattr(before[card], field)


def prove_residency(
    *,
    expected_pci: str,
    live_pci: str,
    list_devices: str,
    log: str,
    before: Mapping[int, MemorySnapshot],
    after: Mapping[int, MemorySnapshot],
    measured_card: int,
    thresholds: ResidencyThresholds,
    output_valid: bool | None,
) -> tuple[ResidencyCheck, ...]:
    """Run the ten residency checks, in order, and return every one of them.

    Every check runs regardless of an earlier failure, so `residency.json`
    records all ten. The caller aborts on `first_failure`, never on the
    first `False` it happens to see.

    The caller reads `live_pci` from `sysfs_pci_address(measured_card, drm_root)`.
    Check 1 compares that observed identity with the declared identity. This
    keeps filesystem access injectable while checking the actual card.
    """
    checks: list[ResidencyCheck] = []

    # 1. The declared PCI address must match the measured card's live sysfs identity.
    pci_matches = (
        bool(_PCI_FULL_RE.match(expected_pci)) and bool(_PCI_FULL_RE.match(live_pci)) and live_pci == expected_pci
    )
    checks.append(
        ResidencyCheck(
            number=1,
            name="card_identity",
            passed=pci_matches,
            observed=f"live_pci={live_pci!r} expected_pci={expected_pci!r}",
        )
    )

    # 2. --list-devices shows exactly one scoped device, and the server log's
    #    own "using device" line names the same PCI address (fact 1 + fact 2:
    #    the PCI never appears in --list-devices, only in the log).
    device_names = list_devices_names(list_devices)
    used_pci = extract_used_pci(log, _MEASURED_LOGICAL_DEVICE)
    check2_passed = device_names == (_MEASURED_LOGICAL_DEVICE,) and used_pci == expected_pci
    checks.append(
        ResidencyCheck(
            number=2,
            name="device_scope_and_log_pci",
            passed=check2_passed,
            observed=f"list_devices={device_names!r} used_pci={used_pci!r} expected_pci={expected_pci!r}",
        )
    )

    # 3. The measured card was idle, not already carrying another load, before
    #    this one started.
    if measured_card not in before:
        raise ResidencyError(f"measured_card {measured_card} is missing from the 'before' snapshot mapping")
    before_vram = before[measured_card].vram_used_mib
    check3_passed = before_vram < thresholds.idle_vram_max_mib
    checks.append(
        ResidencyCheck(
            number=3,
            name="idle_before_load",
            passed=check3_passed,
            observed=f"before_vram_used_mib={before_vram:.2f} idle_vram_max_mib={thresholds.idle_vram_max_mib:.2f}",
        )
    )

    # 4. Every layer offloaded; none left on the CPU backend.
    counts = offload_counts(log)
    check4_passed = counts is not None and counts[0] == counts[1]
    checks.append(
        ResidencyCheck(
            number=4,
            name="layer_offload_complete",
            passed=check4_passed,
            observed=f"offload_counts={counts!r}",
        )
    )

    # 5. Every per-layer assignment line names the measured logical device,
    #    and none names any other one.
    assignments = layer_assignments(log)
    on_measured = assignments.get(_MEASURED_LOGICAL_DEVICE, 0)
    elsewhere = sum(count for device, count in assignments.items() if device != _MEASURED_LOGICAL_DEVICE)
    check5_passed = on_measured > 0 and elsewhere == 0
    checks.append(
        ResidencyCheck(
            number=5,
            name="layer_placement",
            passed=check5_passed,
            observed=f"assignments={assignments!r}",
        )
    )

    # 6. The measured card's own VRAM accounting rose by at least the
    #    resident model's lane-independent footprint.
    if measured_card not in after:
        raise ResidencyError(f"measured_card {measured_card} is missing from the 'after' snapshot mapping")
    vram_delta = _card_delta(before, after, measured_card, "vram_used_mib")
    check6_passed = vram_delta >= thresholds.min_vram_delta_mib
    checks.append(
        ResidencyCheck(
            number=6,
            name="vram_committed",
            passed=check6_passed,
            observed=f"vram_delta_mib={vram_delta:.2f} min_vram_delta_mib={thresholds.min_vram_delta_mib:.2f}",
        )
    )

    # 7. The measured card's GTT did NOT rise (no host-RAM
    #    spill). Read from sysfs only, never from a "CPU_Mapped" log line.
    gtt_delta = _card_delta(before, after, measured_card, "gtt_used_mib")
    check7_passed = gtt_delta <= 0
    checks.append(
        ResidencyCheck(
            number=7,
            name="measured_card_gtt",
            passed=check7_passed,
            observed=f"gtt_delta_mib={gtt_delta:.6f} gtt_limit_mib=0",
        )
    )

    # 8. Every OTHER card was left alone: neither its VRAM nor its GTT moved
    #    past the same ceiling.
    other_cards = sorted((set(before) | set(after)) - {measured_card})
    other_deltas: dict[int, tuple[float, float]] = {}
    missing_cards = sorted(card for card in other_cards if card not in before or card not in after)
    check8_passed = not missing_cards
    for card in other_cards:
        if card in missing_cards:
            continue
        vram_other = _card_delta(before, after, card, "vram_used_mib")
        gtt_other = _card_delta(before, after, card, "gtt_used_mib")
        other_deltas[card] = (vram_other, gtt_other)
        if vram_other > thresholds.gtt_limit_mib or gtt_other > thresholds.gtt_limit_mib:
            check8_passed = False
    checks.append(
        ResidencyCheck(
            number=8,
            name="other_cards_undisturbed",
            passed=check8_passed,
            observed=(
                f"other_card_deltas_mib={other_deltas!r} missing_cards={missing_cards!r} "
                f"gtt_limit_mib={thresholds.gtt_limit_mib:.2f}"
            ),
        )
    )

    # 9. Flash attention is enabled, reported by the server itself (never the
    #    wrapper's argv echo).
    flash_lines = flash_attention_lines(log)
    check9_passed = len(flash_lines) > 0 and any("enabled" in line.lower() for line in flash_lines)
    checks.append(
        ResidencyCheck(
            number=9,
            name="flash_attention_enabled",
            passed=check9_passed,
            observed=f"flash_attention_lines={flash_lines!r}",
        )
    )

    # 10. The probe request actually produced a valid response. Never a
    #     substitute for checks 1-8.
    check10_passed = output_valid is True
    checks.append(
        ResidencyCheck(
            number=10,
            name="output_valid",
            passed=check10_passed,
            observed=f"output_valid={output_valid!r}",
        )
    )

    return tuple(checks)


def first_failure(checks: Sequence[ResidencyCheck]) -> ResidencyCheck | None:
    """The first failed check, in the order `prove_residency` returned them, or `None`."""
    for check in checks:
        if not check.passed:
            return check
    return None
