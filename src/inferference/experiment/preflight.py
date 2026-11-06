"""Preflight: the contract gate a run must pass before a server ever starts.

This module runs the fourteen checks, in order, and refuses on the first
failure (`PreflightRefusal`), naming every check that already passed. Check 10 additionally carries a derived readback:
the measured card's `power1_cap` and `temp*_crit` must equal the live table's
own `SocketPowerLimitAc[PPT0]` and `TemperatureLimit[EDGE, HOTSPOT, MEM]`, so
"the table's bytes are on the card" is never mistaken for "the SMU applied
them". There is no warning level and no third outcome:
a check either passes or the whole preflight stops. Preflight never starts a
server and never writes an actuator; it only reads sysfs, the config
directory, the build manifest, the repo's own `.git`, and a `pp_table` blob,
and compares what it reads against the spec.

**Dual-card specs.** A spec with `fixed.placement` adds three checks after
`vram_fit`: `placement_pair` (each listed card has exactly one resident model,
and both models are the same GGUF with the same footprint), `peer_cap` (the peer
card carries the stock table and the declared cap), and `peer_idle` (the peer
card holds less than the idle VRAM ceiling). `preflight_pair` also checks
`ports_distinct` and runs the full preflight once for each card.

**Duplication, by design.** `inferference.placement.sysfs_free_mib` already
reads a card's VRAM, but it hard-codes `/sys/class/drm` and this module must
not edit `inferference.placement`. So `vram_fit` (check 14) reads VRAM
through `inferference.experiment.residency.memory_snapshot` (for
`mem_info_vram_used`) plus a direct read of `mem_info_vram_total`, both
through the overridable `HostPaths.drm_root`. The duplication is accepted
rather than changing code outside this module's safety boundary.

**Every root-owned path is a constructor field with a real default**
(`HostPaths`), so a test fabricates its own tree under `tmp_path` and never
touches the live system. Fan profile readback is deliberately NOT a
preflight check: it needs the fan lease held, so it belongs to
`inferference.experiment.control` and is called by the runner later, not
here. Preflight only checks that the fan watchdog itself reports `state=OK`
and that its systemd unit is active.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from ..placement import Device, Footprint, ModelSpec, PlacementError
from ..pptable import FieldDelta, PPTableError, PptThrottler, TempSensor
from ..pptable import decode as pptable_decode
from ..pptable import diff as pptable_diff
from ..reconcile import read_config_dir
from ..telemetry import FanDuctReader, TelemetryError, read_watchdog_status
from .control import ControlError, normalize_power_profile, read_power_state, verify_power_profile
from .residency import IDLE_VRAM_MAX_MIB, ResidencyError, memory_snapshot, sysfs_pci_address
from .safety import SafetyError, effective_machine_limit_c
from .spec import ExperimentSpec
from .workload import WorkloadError, parse_workload

_CHECK_ORDER: tuple[str, ...] = (
    "workloads",
    "host_load",
    "machine_limit",
    "wrappers",
    "model",
    "model_sha256",
    "build",
    "identity",
    "card_identity",
    "pp_table",
    "power_profile",
    "fan_watchdog",
    "fan_stall_floors",
    "vram_fit",
)


class PreflightRefusal(RuntimeError):
    """One check failed. Carries every check that already passed."""

    def __init__(self, check: str, detail: str, passed: Sequence[PreflightCheck]) -> None:
        self.check = check
        self.detail = detail
        self.passed = tuple(passed)
        super().__init__(f"preflight check {check!r} refused: {detail}")


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One check's name and what it observed, once it has passed."""

    name: str
    observed: Any


@dataclass(frozen=True, slots=True)
class HostPaths:
    """Every root-owned path preflight reads, overridable for a test's `tmp_path`."""

    drm_root: Path = Path("/sys/class/drm")
    hwmon_root: Path = Path("/sys/class/hwmon")
    loadavg: Path = Path("/proc/loadavg")
    current_system: Path = Path("/run/current-system")
    fan_status_file: Path = Path("/run/arctic-fan/status")
    repo_root: Path = Path(".")


_DEFAULT_HOST_PATHS = HostPaths()


@dataclass(frozen=True, slots=True)
class RunTarget:
    """What this one run targets: the card, the server, and the config to read."""

    card: int
    port: int
    server_bin: Path
    config_root: Path
    stock_pp_table: Path
    thermal_guard: Path
    gpu_lease: Path
    expect_pci: str | None = None
    stall_floors_rpm: Mapping[int, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreflightSnapshot:
    """Every check's result, the resolved model and device, and identity facts.

    `pp_table_bytes` is the LIVE table read during preflight (never the stock
    reference): the runner writes it verbatim to `pp_table.bin`.
    `snapshot_sha256` is a sha256 of this snapshot's own canonical JSON
    (`to_json()`, `sort_keys=True`, `separators=(",", ":")`), so a later
    reader can tell a tampered record from the one that was written.
    """

    checks: tuple[PreflightCheck, ...]
    resolved_model: ModelSpec
    footprint: Footprint
    device: Device
    identity: Mapping[str, Any]
    pp_table_bytes: bytes
    snapshot_sha256: str

    def to_json(self) -> dict[str, Any]:
        """This snapshot as a JSON-ready dict, for `preflight.json`.

        Excludes `pp_table_bytes`: the raw blob is written separately, as
        `pp_table.bin`, and its hash already lives in `identity["pp_table"]`.
        """
        return {
            "checks": [{"name": check.name, "observed": check.observed} for check in self.checks],
            "resolved_model": self.resolved_model.model_dump(mode="json"),
            "footprint": self.footprint.model_dump(mode="json"),
            "device": self.device.model_dump(mode="json"),
            "identity": dict(self.identity),
            "snapshot_sha256": self.snapshot_sha256,
        }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sha256_file(path: Path) -> str:  # shared with the runner for model re-hashing
    """Hash a file's bytes in chunks, so a ~16 GiB model does not need to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_DERIVED_INT = re.compile(r"^[0-9]+$")


def _find_amdgpu_hwmon_dir(drm_root: Path, card: int) -> Path:
    """One card's own amdgpu hwmon directory, found by its `name` file.

    Mirrors how `inferference.telemetry.CardReader` and
    `inferference.experiment.safety.sweep_amdgpu_temperatures` locate a card's
    hwmon: glob the card's `device/hwmon/hwmon*` and require the chip to name
    itself `amdgpu`. The first directory is never blindly trusted. Raises
    `OSError` when no amdgpu hwmon directory answers.
    """
    device_dir = drm_root / f"card{card}" / "device"
    for hwmon_dir in sorted(device_dir.glob("hwmon/hwmon*")):
        try:
            name = (hwmon_dir / "name").read_text().strip()
        except OSError:
            continue
        if name == "amdgpu":
            return hwmon_dir
    raise OSError(f"no amdgpu hwmon directory under {device_dir}")


def _read_hwmon_int(path: Path) -> int:
    """One hwmon value, strictly: a plain non-negative integer and nothing else.

    Readable sysfs values are never anything else; a value that is not makes
    the comparison below unsound, so it raises (`OSError`) rather than
    guessing, exactly like `sweep_amdgpu_temperatures`.
    """
    text = path.read_text().strip()
    if _DERIVED_INT.match(text) is None:
        raise OSError(f"{path} carries {text!r}, not a plain non-negative integer")
    return int(text)


# The derived attributes check 10 compares per sensor: `temp1/2/3` map to
# EDGE/HOTSPOT/MEM on this hardware (verified against live `temp*_crit`), and
# `power1_cap` is derived from `SocketPowerLimitAc[PPT0]`
# (`sienna_cichlid_get_power_limit`).
_DERIVED_TEMPS: tuple[tuple[TempSensor, str], ...] = (
    (TempSensor.EDGE, "temp1_crit"),
    (TempSensor.HOTSPOT, "temp2_crit"),
    (TempSensor.MEM, "temp3_crit"),
)


def _read_card_power_cap(drm_root: Path, card: int, *, with_min: bool, subject: str) -> tuple[Path, int, int | None]:
    """One card's hwmon directory, `power1_cap`, and (if asked) `power1_cap_min`.

    `subject` names the card in an error ("the measured card's", "peer card 1's").
    Raises `OSError` with a message that names what could not be read.
    """
    try:
        hwmon_dir = _find_amdgpu_hwmon_dir(drm_root, card)
        cap_uw = _read_hwmon_int(hwmon_dir / "power1_cap")
    except OSError as exc:
        raise OSError(f"could not read {subject} derived power cap: {exc}") from exc
    cap_min_uw: int | None = None
    if with_min:
        try:
            cap_min_uw = _read_hwmon_int(hwmon_dir / "power1_cap_min")
        except OSError as exc:
            raise OSError(f"could not read {subject} power1_cap_min: {exc}") from exc
    return hwmon_dir, cap_uw, cap_min_uw


def _power_cap_refusal(
    *,
    declared_cap_w: int | None,
    cap_uw: int,
    cap_min_uw: int | None,
    ppt0_w: int,
    prefix: str = "",
) -> str | None:
    """Compare one card's `power1_cap` readback with the spec. Return a refusal text or `None`.

    Without a declared cap, `power1_cap` must equal the live table's own
    `SocketPowerLimitAc[PPT0]`. With a declared cap, `power1_cap_min` must not
    exceed it and `power1_cap` must equal it. `prefix` names a peer card.
    """
    if declared_cap_w is None:
        if cap_uw != ppt0_w * 1_000_000:
            return (
                f"{prefix}hwmon power1_cap reads {cap_uw / 1_000_000} W, but the live table's "
                f"SocketPowerLimitAc[PPT0] is {ppt0_w} W -- the SMU has not taken this table "
                "into effect; re-run the upload's readback and do not measure"
            )
        return None
    # Check the device's minimum before comparing its current cap.
    declared_cap_uw = declared_cap_w * 1_000_000
    if cap_min_uw is not None and cap_min_uw > declared_cap_uw:
        return (
            f"{prefix}hwmon power1_cap_min reads {cap_min_uw / 1_000_000} W, above the declared "
            f"power_cap_w {declared_cap_w} W; the device cannot use this cap"
        )
    if cap_uw != declared_cap_uw:
        return (
            f"{prefix}hwmon power1_cap reads {cap_uw / 1_000_000} W, but the spec declares "
            f"power_cap_w {declared_cap_w} W -- write power1_cap as root before the run; "
            "a GPU reset restores the stock cap"
        )
    return None


def _read_git_revision(repo_root: Path) -> str:
    """The repo's current revision, by file reads only — never a `git` command.

    Reads `.git/HEAD`. A detached HEAD holds the sha directly; a branch HEAD
    holds `ref: <ref>`, resolved first against the loose ref file, then
    against `packed-refs`.
    """
    git_dir = repo_root / ".git"
    head_text = (git_dir / "HEAD").read_text().strip()
    if not head_text.startswith("ref: "):
        return head_text
    ref = head_text.removeprefix("ref: ").strip()
    ref_path = git_dir / ref
    if ref_path.is_file():
        return ref_path.read_text().strip()
    packed_path = git_dir / "packed-refs"
    if packed_path.is_file():
        for line in packed_path.read_text().splitlines():
            if line.endswith(f" {ref}"):
                return line.split()[0]
    raise OSError(f"ref {ref!r} named by {git_dir / 'HEAD'} was found in neither a loose ref nor packed-refs")


def _peer_of(cards: Sequence[int], card: int) -> int | None:
    """The other card of a two-card declaration, or `None` when `card` is not in it."""
    if card not in cards:
        return None
    return next(other for other in cards if other != card)


def preflight(
    spec: ExperimentSpec,
    target: RunTarget,
    *,
    paths: HostPaths = _DEFAULT_HOST_PATHS,
    env: Mapping[str, str] | None = None,
    unit_active: Callable[[], bool],
    model_sha256: Callable[[Path], str] = _sha256_file,
    sleep: Callable[[float], None] = time.sleep,
) -> PreflightSnapshot:
    """Run every check in order; refuse on the first failure.

    The `pp_table` check verifies the live table hash and its field deltas
    against the spec, then the derived readback (`power1_cap`, `temp*_crit`).
    With `power_cap_w` declared, the live table must be stock, `power1_cap`
    must equal the declared watts, and `power1_cap_min` must not exceed them.
    Returns a `PreflightSnapshot` only when all fourteen checks pass. Raises
    `PreflightRefusal` otherwise, naming the first check that failed and
    every check that passed before it.
    """
    passed: list[PreflightCheck] = []
    identity: dict[str, Any] = {}

    def _refuse(name: str, detail: str) -> NoReturn:
        raise PreflightRefusal(name, detail, tuple(passed))

    def _record(name: str, observed: Any) -> None:
        passed.append(PreflightCheck(name=name, observed=observed))

    # 1. workloads -------------------------------------------------------
    parsed_workloads: list[dict[str, Any]] = []
    for name in spec.workloads:
        try:
            plan = parse_workload(name)
        except WorkloadError as exc:
            _refuse("workloads", f"workload {name!r} does not parse: {exc}")
        parsed_workloads.append({"name": plan.name, "kind": plan.kind})
    _record("workloads", parsed_workloads)

    # 2. host_load ---------------------------------------------------------
    try:
        loadavg_text = paths.loadavg.read_text()
    except OSError as exc:
        _refuse("host_load", f"could not read {paths.loadavg}: {exc}")
    try:
        loadavg_1m = float(loadavg_text.split()[0])
    except (IndexError, ValueError) as exc:
        _refuse("host_load", f"could not parse first field of {paths.loadavg} ({loadavg_text!r}): {exc}")
    if not math.isfinite(loadavg_1m) or loadavg_1m < 0:
        _refuse("host_load", f"loadavg_1m must be a finite non-negative number, got {loadavg_1m!r}")
    limit = spec.fixed.host.loadavg_1m_max
    if loadavg_1m > limit:
        _refuse("host_load", f"loadavg_1m {loadavg_1m} exceeds the declared limit {limit}")
    _record("host_load", {"loadavg_1m": loadavg_1m, "limit": limit})

    # 3. machine_limit -----------------------------------------------------
    try:
        effective_limit = effective_machine_limit_c(target.thermal_guard, env)
    except (SafetyError, OSError) as exc:
        _refuse("machine_limit", str(exc))
    declared_limit = spec.fixed.safety.machine_limit_c
    if declared_limit != effective_limit:
        _refuse(
            "machine_limit",
            f"spec declares machine_limit_c={declared_limit}, but the effective limit "
            f"(env GPU_THERMAL_MAX_C, else the guard's own default) is {effective_limit}",
        )
    _record("machine_limit", {"declared": declared_limit, "effective": effective_limit})

    # 4. wrappers ------------------------------------------------------------
    for label, path in (("thermal_guard", target.thermal_guard), ("gpu_lease", target.gpu_lease)):
        if not path.is_file():
            _refuse("wrappers", f"{label} script not found: {path}")
        if not os.access(path, os.X_OK):
            _refuse("wrappers", f"{label} script is not executable: {path}")
    _record("wrappers", {"thermal_guard": str(target.thermal_guard), "gpu_lease": str(target.gpu_lease)})

    # 5. model ---------------------------------------------------------------
    try:
        placement, specs = read_config_dir(target.config_root)
    except PlacementError as exc:
        _refuse("model", str(exc))
    device = next((d for d in placement.devices if d.card == target.card), None)
    if device is None:
        _refuse("model", f"no device is declared for card {target.card} in {target.config_root}/placement.yaml")
    try:
        resident_here = [r for r in placement.resident if device.id in r.devices]
    except PlacementError as exc:
        _refuse("model", str(exc))
    if len(resident_here) != 1:
        _refuse(
            "model",
            f"expected exactly one resident model on device {device.id}, found "
            f"{len(resident_here)}: {[r.model for r in resident_here]}",
        )
    resolved_model = specs[resident_here[0].model]
    if resolved_model.loras:
        _refuse("model", f"model {resolved_model.name!r} declares lora adapter(s); not supported by preflight")
    try:
        footprint = resolved_model.require_footprint()
    except PlacementError as exc:
        _refuse("model", str(exc))
    tokens_per_slot = spec.fixed.context.tokens_per_slot
    if footprint.ctx_per_lane != tokens_per_slot:
        _refuse(
            "model",
            f"model {resolved_model.name!r}'s footprint was measured at ctx_per_lane="
            f"{footprint.ctx_per_lane}, but the spec declares tokens_per_slot={tokens_per_slot}. "
            f"Re-measure the footprint; it is never extrapolated.",
        )
    _record("model", {"device": device.id, "model": resolved_model.name, "ctx_per_lane": footprint.ctx_per_lane})

    # 6. model_sha256 ----------------------------------------------------
    if spec.fixed.model.sha256_must_match:
        declared_sha256 = spec.fixed.model.sha256
        if declared_sha256 is None:
            _refuse("model_sha256", "fixed.model.sha256_must_match is true, but fixed.model.sha256 is not declared")
        try:
            computed_sha256 = model_sha256(resolved_model.model)
        except OSError as exc:
            _refuse("model_sha256", f"could not hash model {resolved_model.model}: {exc}")
        if computed_sha256 != declared_sha256:
            _refuse(
                "model_sha256",
                f"model sha256 mismatch: declared {declared_sha256}, computed {computed_sha256}",
            )
        identity["model_sha256"] = computed_sha256
        _record("model_sha256", {"declared": declared_sha256, "computed": computed_sha256})
    else:
        identity["model_sha256"] = None
        _record("model_sha256", "sha256_must_match is false; not checked")

    # 7. build -----------------------------------------------------------
    manifest_path = target.server_bin.parent.parent / "build-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        _refuse("build", f"could not read build manifest {manifest_path}: {exc}")
    if not isinstance(manifest, dict):
        _refuse("build", f"build manifest {manifest_path} must contain a JSON object")
    manifest_build_id = manifest.get("build_id")
    manifest_features = manifest.get("features")
    if spec.fixed.build.manifest_must_match:
        declared_build_id = spec.fixed.build.build_id
        if declared_build_id is None:
            _refuse("build", "fixed.build.manifest_must_match is true, but fixed.build.build_id is not declared")
        if not isinstance(manifest_build_id, str) or not manifest_build_id:
            _refuse("build", f"build manifest {manifest_path} has no non-empty string build_id")
        if declared_build_id != manifest_build_id:
            _refuse("build", f"build_id mismatch: declared {declared_build_id}, manifest {manifest_build_id}")
        if spec.fixed.build.features is not None:
            if not isinstance(manifest_features, list) or not all(isinstance(item, str) for item in manifest_features):
                _refuse("build", f"build manifest {manifest_path} has invalid features; expected a list of strings")
            if set(spec.fixed.build.features) != set(manifest_features):
                _refuse(
                    "build",
                    f"build features mismatch: declared {spec.fixed.build.features}, manifest {manifest_features}",
                )
    identity["build"] = {"path": str(manifest_path), "build_id": manifest_build_id, "features": manifest_features}
    _record("build", identity["build"])

    # 8. identity --------------------------------------------------------
    try:
        repo_revision = _read_git_revision(paths.repo_root)
    except OSError as exc:
        _refuse("identity", f"could not read repo revision under {paths.repo_root}: {exc}")
    try:
        nixos_generation = os.readlink(paths.current_system)
    except OSError as exc:
        _refuse("identity", f"could not read {paths.current_system}: {exc}")
    identity["repo_revision"] = repo_revision
    identity["nixos_generation"] = nixos_generation
    _record("identity", {"repo_revision": repo_revision, "nixos_generation": nixos_generation})

    # 9. card_identity -----------------------------------------------------
    try:
        card_pci = sysfs_pci_address(target.card, drm_root=paths.drm_root)
    except ResidencyError as exc:
        _refuse("card_identity", str(exc))
    if target.expect_pci is not None and card_pci != target.expect_pci:
        _refuse(
            "card_identity",
            f"card {target.card}'s sysfs PCI address {card_pci} does not match --expect-pci {target.expect_pci}",
        )
    identity["card_pci"] = card_pci
    _record("card_identity", {"card": target.card, "pci": card_pci, "expect_pci": target.expect_pci})

    # 10. pp_table -----------------------------------------------------------
    try:
        stock_bytes = target.stock_pp_table.read_bytes()
    except OSError as exc:
        _refuse("pp_table", f"could not read the stock reference pp_table {target.stock_pp_table}: {exc}")
    try:
        stock_table = pptable_decode(stock_bytes)
    except PPTableError as exc:
        _refuse("pp_table", f"stock reference pp_table {target.stock_pp_table} is malformed: {exc}")
    live_pp_table_path = paths.drm_root / f"card{target.card}" / "device" / "pp_table"
    try:
        live_bytes = live_pp_table_path.read_bytes()
    except OSError as exc:
        _refuse("pp_table", f"could not read the live pp_table {live_pp_table_path}: {exc}")
    try:
        live_table = pptable_decode(live_bytes)
    except PPTableError as exc:
        _refuse("pp_table", f"live pp_table {live_pp_table_path} is malformed: {exc}")
    stock_sha256 = hashlib.sha256(stock_bytes).hexdigest()
    live_sha256 = hashlib.sha256(live_bytes).hexdigest()
    arm = spec.fixed.control.pp_table
    declared_cap_w = spec.fixed.control.power_cap_w
    declared_name: str | None = None
    declared_pp_sha256: str | None = None
    if declared_cap_w is not None:
        # The `power_cap_w` actuator is a hwmon `power1_cap` write on a patched
        # kernel. The pp_table must stay stock: a modified table means an
        # upload happened, and an upload can wedge the SMU.
        control_mode = "power_cap"
        if live_sha256 != stock_sha256:
            _refuse(
                "pp_table",
                f"power_cap_w {declared_cap_w} requires the stock pp_table, but live pp_table sha256 "
                f"{live_sha256} does not match the stock reference {stock_sha256}",
            )
    elif arm is not None:
        control_mode = "pp_table"
        declared_name = arm.name
        declared_pp_sha256 = arm.sha256
        if declared_pp_sha256 is None:
            _refuse("pp_table", f"pp_table arm {arm.name!r} has no sha256 declared yet")
        if live_sha256 != declared_pp_sha256:
            _refuse(
                "pp_table",
                f"live pp_table sha256 {live_sha256} does not match arm {arm.name!r}'s declared sha256 "
                f"{declared_pp_sha256}",
            )
        computed_diff = sorted(pptable_diff(stock_table, live_table), key=lambda d: d.name)
        expected_diff = sorted(
            (FieldDelta(name=d.field, old=d.old, new=d.new) for d in arm.deltas_from_stock),
            key=lambda d: d.name,
        )
        if computed_diff != expected_diff:
            _refuse(
                "pp_table",
                f"pp_table deltas for arm {arm.name!r} do not match its declared deltas_from_stock: "
                f"computed={computed_diff!r} declared={expected_diff!r}",
            )
    else:
        control_mode = "stock"
        if live_sha256 != stock_sha256:
            _refuse(
                "pp_table",
                f"live pp_table sha256 {live_sha256} does not match the stock reference {stock_sha256}, "
                f"and no pp_table arm is declared for this spec",
            )
    # 10b. derived readback -------------------------------------------------
    # The hash proves the driver holds this table's bytes; it does not prove
    # the SMU recomputed its limits from them. `power1_cap` and `temp*_crit`
    # are the derived values the driver exposes from the uploaded table — the
    # same attributes an upload's readback compares. A mismatch means the table is accepted by sysfs but
    # not in effect on the SMU, so the run refuses: a spec cap that "did not
    # take" must never be measured as if it had.
    live_ppt0_w = live_table.socket_power_limit_ac[PptThrottler.PPT0]
    try:
        hwmon_dir, live_power_cap_uw, live_power_cap_min_uw = _read_card_power_cap(
            paths.drm_root, target.card, with_min=declared_cap_w is not None, subject="the measured card's"
        )
    except OSError as exc:
        _refuse("pp_table", str(exc))
    cap_refusal = _power_cap_refusal(
        declared_cap_w=declared_cap_w, cap_uw=live_power_cap_uw, cap_min_uw=live_power_cap_min_uw, ppt0_w=live_ppt0_w
    )
    if cap_refusal is not None:
        _refuse("pp_table", cap_refusal)
    for sensor, crit_file in _DERIVED_TEMPS:
        expected_mc = live_table.temperature_limit[sensor] * 1000
        try:
            observed_mc = _read_hwmon_int(hwmon_dir / crit_file)
        except OSError as exc:
            _refuse("pp_table", f"could not read the measured card's {crit_file}: {exc}")
        if observed_mc != expected_mc:
            _refuse(
                "pp_table",
                f"hwmon {crit_file} reads {observed_mc / 1000} C, but the live table's "
                f"TemperatureLimit[{sensor.name}] is {expected_mc / 1000} C -- the SMU has not "
                "taken this table into effect; re-run the upload's readback and do not measure",
            )
    pp_table_info = {
        "declared_name": declared_name,
        "declared_sha256": declared_pp_sha256,
        "live_sha256": live_sha256,
        "stock_sha256": stock_sha256,
        "deltas_verified": True,
        "control_mode": control_mode,
        "declared_power_cap_w": declared_cap_w,
        "live_power1_cap_uw": live_power_cap_uw,
        "live_power1_cap_min_uw": live_power_cap_min_uw,
    }
    identity["pp_table"] = pp_table_info
    _record("pp_table", pp_table_info)

    # 11. power_profile --------------------------------------------------
    cards = tuple(
        sorted(
            int(card_dir.name[4:])
            for card_dir in paths.drm_root.glob("card[0-9]*")
            if card_dir.name.startswith("card") and card_dir.name[4:].isdigit() and (card_dir / "device").exists()
        )
    )
    if target.card not in cards:
        _refuse("power_profile", f"target card {target.card} was not found under {paths.drm_root}")
    try:
        power_state = read_power_state(cards, drm_root=paths.drm_root)
    except ControlError as exc:
        _refuse("power_profile", str(exc))
    try:
        verify_power_profile(spec.fixed.control.power_profile, power_state)
    except ControlError as exc:
        _refuse("power_profile", str(exc))
    power_profile_info = {
        "declared": normalize_power_profile(spec.fixed.control.power_profile),
        "state": power_state,
    }
    identity["power_profile"] = power_profile_info
    _record("power_profile", power_profile_info)

    # 12. fan_watchdog -----------------------------------------------------
    # The watchdog rewrites this file with a shell redirect, so a read can see
    # an empty file for a moment. The shared reader retries, then fails closed.
    try:
        watchdog_state = read_watchdog_status(paths.fan_status_file, sleep=sleep).state
    except TelemetryError as exc:
        _refuse("fan_watchdog", f"could not read fan watchdog status {paths.fan_status_file}: {exc}")
    if watchdog_state != "OK":
        _refuse("fan_watchdog", f"fan watchdog state is {watchdog_state!r}, not OK")
    active = unit_active()
    if not active:
        _refuse("fan_watchdog", "fan watchdog systemd unit is not active")
    _record("fan_watchdog", {"state": watchdog_state, "unit_active": active})

    # 13. fan_stall_floors -------------------------------------------------
    try:
        fan_reader = FanDuctReader(hwmon_root=paths.hwmon_root)
        ducts = fan_reader.read_ducts()
    except TelemetryError as exc:
        _refuse("fan_stall_floors", str(exc))
    stall_report: dict[str, Any] = {}
    stall_failures: list[str] = []
    detected_channels = {duct.channel for duct in ducts}
    missing_channels = sorted(set(target.stall_floors_rpm) - detected_channels)
    if missing_channels:
        _refuse("fan_stall_floors", f"requested fan channel(s) were not detected: {missing_channels}")
    for duct in ducts:
        floor = target.stall_floors_rpm.get(duct.channel)
        if floor is None:
            stall_report[str(duct.channel)] = {"tach_rpm": duct.tach_rpm, "floor_rpm": "ungated"}
            continue
        stall_report[str(duct.channel)] = {"tach_rpm": duct.tach_rpm, "floor_rpm": floor}
        if duct.tach_rpm < floor:
            stall_failures.append(f"channel {duct.channel}: tach {duct.tach_rpm} rpm < floor {floor} rpm")
    if stall_failures:
        _refuse("fan_stall_floors", "fan stall floor breached: " + "; ".join(stall_failures))
    _record("fan_stall_floors", stall_report)

    # 14. vram_fit -----------------------------------------------------------
    try:
        snapshot = memory_snapshot(target.card, drm_root=paths.drm_root)
    except ResidencyError as exc:
        _refuse("vram_fit", str(exc))
    total_path = paths.drm_root / f"card{target.card}" / "device" / "mem_info_vram_total"
    try:
        total_mib = int(total_path.read_text().strip()) / 1_048_576
    except (OSError, ValueError) as exc:
        _refuse("vram_fit", f"could not read {total_path}: {exc}")
    free_mib = total_mib - snapshot.vram_used_mib
    needed_mib = footprint.total_mib(spec.fixed.context.slots) + device.reserve_mib
    if free_mib < needed_mib:
        _refuse(
            "vram_fit",
            f"card {target.card} has {free_mib:.1f} MiB free, needs {needed_mib:.1f} MiB "
            f"(footprint at {spec.fixed.context.slots} slots + {device.reserve_mib} MiB reserve)",
        )
    _record("vram_fit", {"free_mib": free_mib, "needed_mib": needed_mib})

    # 15-17. dual-card checks. A single-card spec runs none of them. --------
    pair = spec.fixed.placement
    if pair is not None:
        peer_card = _peer_of(pair.cards, target.card)
        if peer_card is None:
            _refuse("placement_pair", f"target card {target.card} is not in fixed.placement.cards {list(pair.cards)}")

        # 15. placement_pair: each listed card holds exactly one resident
        # model, and both are the same GGUF with the same footprint.
        pair_models: dict[int, ModelSpec] = {}
        for pair_card in pair.cards:
            pair_device = next((d for d in placement.devices if d.card == pair_card), None)
            if pair_device is None:
                _refuse(
                    "placement_pair",
                    f"no device is declared for card {pair_card} in {target.config_root}/placement.yaml",
                )
            try:
                resident_on_card = [r for r in placement.resident if pair_device.id in r.devices]
            except PlacementError as exc:
                _refuse("placement_pair", str(exc))
            if len(resident_on_card) != 1:
                _refuse(
                    "placement_pair",
                    f"expected exactly one resident model on device {pair_device.id} (card {pair_card}), found "
                    f"{len(resident_on_card)}: {[r.model for r in resident_on_card]}",
                )
            pair_models[pair_card] = specs[resident_on_card[0].model]
        peer_model = pair_models[peer_card]
        if peer_model.loras:
            _refuse("placement_pair", f"model {peer_model.name!r} declares lora adapter(s); not supported")
        try:
            peer_footprint = peer_model.require_footprint()
        except PlacementError as exc:
            _refuse("placement_pair", str(exc))
        if peer_model.model.resolve() != resolved_model.model.resolve():
            _refuse(
                "placement_pair",
                f"card {target.card} runs {resolved_model.model} but card {peer_card} runs {peer_model.model}; "
                "both cards must run the same GGUF",
            )
        if peer_footprint != footprint:
            _refuse(
                "placement_pair",
                f"card {target.card} and card {peer_card} declare different footprints ({footprint} and "
                f"{peer_footprint}); both cards must carry the same footprint",
            )
        _record(
            "placement_pair",
            {
                "cards": list(pair.cards),
                "models": {str(card): model.name for card, model in pair_models.items()},
                "model": str(resolved_model.model),
            },
        )

        # 16. peer_cap: the peer card carries the stock table and the same cap.
        peer_label = f"peer card {peer_card}"
        peer_table_path = paths.drm_root / f"card{peer_card}" / "device" / "pp_table"
        try:
            peer_bytes = peer_table_path.read_bytes()
        except OSError as exc:
            _refuse("peer_cap", f"could not read the {peer_label} live pp_table {peer_table_path}: {exc}")
        peer_sha256 = hashlib.sha256(peer_bytes).hexdigest()
        if peer_sha256 != stock_sha256:
            _refuse(
                "peer_cap",
                f"{peer_label} live pp_table sha256 {peer_sha256} does not match the stock reference {stock_sha256}",
            )
        try:
            _, peer_cap_uw, peer_cap_min_uw = _read_card_power_cap(
                paths.drm_root, peer_card, with_min=declared_cap_w is not None, subject=f"{peer_label}'s"
            )
        except OSError as exc:
            _refuse("peer_cap", str(exc))
        # The peer table is byte-identical to the stock table, so the stock
        # table's own PPT0 limit is the peer's.
        peer_refusal = _power_cap_refusal(
            declared_cap_w=declared_cap_w,
            cap_uw=peer_cap_uw,
            cap_min_uw=peer_cap_min_uw,
            ppt0_w=stock_table.socket_power_limit_ac[PptThrottler.PPT0],
            prefix=f"{peer_label}: ",
        )
        if peer_refusal is not None:
            _refuse("peer_cap", peer_refusal)
        _record(
            "peer_cap",
            {
                "card": peer_card,
                "live_sha256": peer_sha256,
                "declared_power_cap_w": declared_cap_w,
                "live_power1_cap_uw": peer_cap_uw,
                "live_power1_cap_min_uw": peer_cap_min_uw,
            },
        )

        # 17. peer_idle: the peer card holds no model yet. Both runs of a pair
        # finish preflight before either one loads a server.
        try:
            peer_snapshot = memory_snapshot(peer_card, drm_root=paths.drm_root)
        except ResidencyError as exc:
            _refuse("peer_idle", str(exc))
        idle_max_mib = IDLE_VRAM_MAX_MIB
        if peer_snapshot.vram_used_mib >= idle_max_mib:
            _refuse(
                "peer_idle",
                f"{peer_label} has {peer_snapshot.vram_used_mib:.1f} MiB of VRAM in use, "
                f"at or above the idle ceiling {idle_max_mib:.1f} MiB",
            )
        _record("peer_idle", {"card": peer_card, "vram_used_mib": peer_snapshot.vram_used_mib})

    snapshot_payload = {
        "checks": [{"name": check.name, "observed": check.observed} for check in passed],
        "resolved_model": resolved_model.model_dump(mode="json"),
        "footprint": footprint.model_dump(mode="json"),
        "device": device.model_dump(mode="json"),
        "identity": identity,
    }
    return PreflightSnapshot(
        checks=tuple(passed),
        resolved_model=resolved_model,
        footprint=footprint,
        device=device,
        identity=identity,
        pp_table_bytes=live_bytes,
        snapshot_sha256=_canonical_sha256(snapshot_payload),
    )


def preflight_pair(
    spec: ExperimentSpec,
    targets: Sequence[RunTarget],
    *,
    paths: HostPaths = _DEFAULT_HOST_PATHS,
    env: Mapping[str, str] | None = None,
    unit_active: Callable[[], bool],
    model_sha256: Callable[[Path], str] = _sha256_file,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[PreflightSnapshot, ...]:
    """Run `preflight` for both cards of a dual-card spec; refuse on the first failure.

    Two checks belong to the pair and run first. `placement_pair` needs one
    target for each card the spec lists. `ports_distinct` needs a different
    server port for each target. Each card then runs the full `preflight`,
    which adds the `placement_pair`, `peer_cap`, and `peer_idle` checks. The
    model hash is computed once and reused for the second card.
    """
    pair = spec.fixed.placement
    if pair is None:
        raise PreflightRefusal("placement_pair", "the spec declares no fixed.placement", ())
    target_cards = sorted(target.card for target in targets)
    if target_cards != sorted(pair.cards):
        raise PreflightRefusal(
            "placement_pair",
            f"targets name cards {target_cards}, but fixed.placement.cards is {sorted(pair.cards)}",
            (),
        )
    ports = [target.port for target in targets]
    if len(set(ports)) != len(ports):
        raise PreflightRefusal("ports_distinct", f"both cards use one server port: {ports}", ())
    hashes: dict[Path, str] = {}

    def cached_sha256(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in hashes:
            hashes[resolved] = model_sha256(path)
        return hashes[resolved]

    ordered = sorted(targets, key=lambda target: pair.cards.index(target.card))
    return tuple(
        preflight(spec, target, paths=paths, env=env, unit_active=unit_active, model_sha256=cached_sha256, sleep=sleep)
        for target in ordered
    )


def start_gate(
    spec: ExperimentSpec,
    *,
    read_junctions: Callable[[], Mapping[int, float]],
    timeout_s: float = 600,
    poll_s: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Hold until every card's junction sensor cools to the start band, or refuse.

    Every card `read_junctions()` reports must stay at or below
    `max_c + tolerance_c` continuously for `hold_s` seconds before this
    returns. A reading above the band at any point resets the hold — a dip
    below the band and a later rise above it starts the clock over, never
    silently. Refuses with `PreflightRefusal("start_gate", ...)` if the band
    is never held for `hold_s` seconds within `timeout_s` seconds total.
    """
    gate = spec.fixed.start_gate
    threshold = gate.max_c + gate.tolerance_c
    started_utc = datetime.now(UTC).isoformat()
    started_mono = clock()
    deadline_mono = started_mono + timeout_s
    hold_start_mono: float | None = None
    junctions: Mapping[int, float] = {}

    while True:
        now_mono = clock()
        if now_mono > deadline_mono:
            raise PreflightRefusal(
                "start_gate",
                f"cards did not hold <= {threshold} C for {gate.hold_s}s within {timeout_s}s "
                f"(last reading: {dict(junctions)!r})",
                (),
            )
        junctions = read_junctions()
        now_mono = clock()
        if now_mono > deadline_mono:
            raise PreflightRefusal(
                "start_gate",
                f"cards did not hold <= {threshold} C for {gate.hold_s}s within {timeout_s}s "
                f"(last reading: {dict(junctions)!r})",
                (),
            )
        if not junctions:
            raise PreflightRefusal("start_gate", "read_junctions() returned no cards", ())
        hottest = max(junctions.values())
        if hottest <= threshold:
            if hold_start_mono is None:
                hold_start_mono = now_mono
            elif now_mono - hold_start_mono >= gate.hold_s:
                return {
                    "started_utc": started_utc,
                    "started_mono": started_mono,
                    "passed_mono": now_mono,
                    "max_c": gate.max_c,
                    "tolerance_c": gate.tolerance_c,
                    "hold_s": gate.hold_s,
                    "passed": True,
                }
        else:
            hold_start_mono = None
        if now_mono >= deadline_mono:
            raise PreflightRefusal(
                "start_gate",
                f"cards did not hold <= {threshold} C for {gate.hold_s}s within {timeout_s}s "
                f"(last reading: {dict(junctions)!r})",
                (),
            )
        sleep(poll_s)
