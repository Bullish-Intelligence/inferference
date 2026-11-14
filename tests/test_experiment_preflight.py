"""Tests for the preflight contract gate, against a fully fabricated host.

The base fixture (`_build_host`) passes every one of the fourteen checks.
Every "breaks only this check" test starts from that base and edits exactly
one dimension — a sysfs file, a wrapper script, the spec, or a `RunTarget`
field — then asserts that `PreflightRefusal.check` names the one check that
dimension maps to, and nothing else. This mirrors
`tests/test_experiment_residency.py`'s own "one test per check" shape.

The stock `pp_table` reference is a SYNTHETIC 2470-byte table built from the
public header constants in `inferference.pptable` (see `tests/pp_tables.py`).
The vendor blob is AMD firmware data and is not redistributed with this
repository, and these checks only compare a live table against its own
baseline, so real firmware bytes would add nothing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from inferference.experiment.preflight import (
    HostPaths,
    PreflightRefusal,
    RunTarget,
    preflight,
    preflight_pair,
    start_gate,
)
from inferference.experiment.spec import (
    BatchFixed,
    BuildFixed,
    CacheFixed,
    ContextFixed,
    ControlFixed,
    Endpoints,
    ExperimentSpec,
    Factor,
    Fixed,
    HostFixed,
    ModelFixed,
    PlacementFixed,
    PPTableArm,
    PPTableFieldDelta,
    Replication,
    SafetyFixed,
    StartGateFixed,
)
from inferference.pptable import PptThrottler, TempSensor
from inferference.pptable import decode as pptable_decode
from tests.pp_tables import synthetic_stock_blob
from tests.test_experiment_control import make_cpu_root

_STOCK_PP_TABLE_SHA256 = hashlib.sha256(synthetic_stock_blob()).hexdigest()

_CARD = 0
_PCI = "0000:03:00.0"
_EXPECTED_CHECK_ORDER = (
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


def _stock_pp_table_bytes() -> bytes:
    return synthetic_stock_blob()


def _build_spec(
    *,
    model_sha256: str,
    build_id: str = "fake-build-123",
    build_features: tuple[str, ...] = ("vulkan",),
    workloads: tuple[str, ...] = ("prefill_100", "decode_single_stream"),
    machine_limit_c: int = 80,
    loadavg_1m_max: float = 5.0,
    power_profile: str = "power-saving",
    context_slots: int = 2,
    tokens_per_slot: int = 4096,
) -> ExperimentSpec:
    fixed = Fixed(
        model=ModelFixed(sha256_must_match=True, sha256=model_sha256),
        build=BuildFixed(manifest_must_match=True, build_id=build_id, features=build_features),
        context=ContextFixed(slots=context_slots, tokens_per_slot=tokens_per_slot),
        cache=CacheFixed(k="q8_0", v="q8_0"),
        batch=BatchFixed(size=1024, ubatch=1024),
        control=ControlFixed(fan_profile="flat-max", power_profile=power_profile, pp_table=None),
        safety=SafetyFixed(machine_limit_c=machine_limit_c, arm_bound_c=machine_limit_c - 5, gtt_limit_mib=512),
        start_gate=StartGateFixed(max_c=24, tolerance_c=1, hold_s=30),
        host=HostFixed(loadavg_1m_max=loadavg_1m_max),
    )
    return ExperimentSpec(
        experiment="wp11-preflight-test",
        schema_version=1,
        fixed=fixed,
        factor=Factor(name="lanes", values=("2",)),
        workloads=workloads,
        replication=Replication(n=1, order="counterbalanced", cooldown_between="start_gate"),
        endpoints=Endpoints(primary="band_slope", band_c=(50, 70), secondary=()),
    )


def _make_device_dir(drm_root: Path, card: int, pci: str) -> Path:
    """The `tests/test_telemetry.py::_make_card` shape: a real PCI dir, symlinked."""
    real_device = drm_root / "pci-devices" / pci
    real_device.mkdir(parents=True)
    card_dir = drm_root / f"card{card}"
    card_dir.mkdir(parents=True)
    device = card_dir / "device"
    device.symlink_to(real_device, target_is_directory=True)
    return device


def _build_host(tmp_path: Path) -> tuple[HostPaths, RunTarget, ExperimentSpec]:
    """A fully fabricated host and config that passes every one of the 14 checks."""
    drm_root = tmp_path / "sys-class-drm"
    device = _make_device_dir(drm_root, _CARD, _PCI)
    (device / "mem_info_vram_used").write_text(f"{100 * 1024 * 1024}\n")
    (device / "mem_info_vram_total").write_text(f"{4096 * 1024 * 1024}\n")
    (device / "mem_info_gtt_used").write_text("0\n")
    (device / "power_dpm_force_performance_level").write_text("manual\n")
    (device / "pp_power_profile_mode").write_text(
        "PROFILE_INDEX(NAME) CLOCK_TYPE(NAME)\n 0 BOOTUP_DEFAULT :\n 2   POWER_SAVING*:\n"
    )
    (device / "pp_table").write_bytes(_stock_pp_table_bytes())

    # The measured card's own amdgpu hwmon: the derived-readback half of
    # preflight check 10 compares `power1_cap` and `temp*_crit` against the
    # live table above (the stock blob: PPT0 250 W, EDGE/HOTSPOT 100 C,
    # MEM 98 C).
    card_hwmon = device / "hwmon" / "hwmon1"
    card_hwmon.mkdir(parents=True)
    for name, value in {
        "name": "amdgpu\n",
        "power1_cap": "250000000\n",
        "temp1_label": "edge\n",
        "temp2_label": "junction\n",
        "temp3_label": "mem\n",
        "temp1_crit": "100000\n",
        "temp2_crit": "100000\n",
        "temp3_crit": "98000\n",
    }.items():
        (card_hwmon / name).write_text(value)

    hwmon_root = tmp_path / "sys-class-hwmon"
    fan_chip = hwmon_root / "hwmon2"
    fan_chip.mkdir(parents=True)
    (fan_chip / "name").write_text("arctic_fan\n")
    (fan_chip / "pwm2").write_text("255\n")
    (fan_chip / "fan2_input").write_text("1500\n")
    (fan_chip / "pwm3").write_text("255\n")
    (fan_chip / "fan3_input").write_text("1400\n")

    loadavg = tmp_path / "loadavg"
    loadavg.write_text("1.23 1.10 1.05 2/300 12345\n")

    nix_target = tmp_path / "nix-store-system-42"
    nix_target.mkdir()
    current_system = tmp_path / "current-system"
    current_system.symlink_to(nix_target, target_is_directory=True)

    fan_status_file = tmp_path / "fan-status"
    fan_status_file.write_text("state=OK target_pwm=255 max_gpu_junction_mC=36000 profile=flat-max\n")

    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "refs" / "heads" / "main").write_text("deadbeefcafef00d\n")

    config_root = tmp_path / "config"
    (config_root / "models").mkdir(parents=True)
    model_path = config_root / "model.gguf"
    model_path.write_bytes(b"not a real gguf, just a dummy placeholder model file\n")
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    (config_root / "placement.yaml").write_text(
        yaml.safe_dump(
            {
                "devices": [{"id": "Vulkan0", "card": _CARD, "reserve_mib": 100}],
                "resident": [{"model": "m", "device": "Vulkan0"}],
                "router": {"host": "127.0.0.1", "port": 8100, "models_max": 1},
            }
        )
    )
    (config_root / "models" / "m.yaml").write_text(
        yaml.safe_dump(
            {
                "model": str(model_path),
                "ctx_per_lane": 4096,
                "lanes": 1,
                "footprint": {
                    "ctx_per_lane": 4096,
                    "base_mib": 1000.0,
                    "per_lane_mib": 200.0,
                    "measured_by": "sysfs-vram-delta",
                },
            }
        )
    )

    out_dir = tmp_path / "out"
    (out_dir / "bin").mkdir(parents=True)
    server_bin = out_dir / "bin" / "llama-server"
    server_bin.write_text("")
    (out_dir / "build-manifest.json").write_text(json.dumps({"build_id": "fake-build-123", "features": ["vulkan"]}))

    thermal_guard = tmp_path / "gpu-thermal-guard.sh"
    thermal_guard.write_text('#!/usr/bin/env bash\nMAX_C="${GPU_THERMAL_MAX_C:-80}"\n')
    thermal_guard.chmod(0o755)
    gpu_lease = tmp_path / "gpu-lease.sh"
    gpu_lease.write_text('#!/usr/bin/env bash\nexec "$@"\n')
    gpu_lease.chmod(0o755)

    stock_pp_table = tmp_path / "stock-card0.pp_table"
    stock_pp_table.write_bytes(_stock_pp_table_bytes())

    paths = HostPaths(
        drm_root=drm_root,
        hwmon_root=hwmon_root,
        loadavg=loadavg,
        current_system=current_system,
        fan_status_file=fan_status_file,
        repo_root=repo_root,
    )
    target = RunTarget(
        card=_CARD,
        port=18080,
        server_bin=server_bin,
        config_root=config_root,
        stock_pp_table=stock_pp_table,
        thermal_guard=thermal_guard,
        gpu_lease=gpu_lease,
        expect_pci=_PCI,
        stall_floors_rpm={2: 1000.0},
    )
    spec = _build_spec(model_sha256=model_sha256)
    return paths, target, spec


def _preflight(paths: HostPaths, target: RunTarget, spec: ExperimentSpec, *, unit_active=lambda: True, sleep=None):
    extra = {} if sleep is None else {"sleep": sleep}
    return preflight(spec, target, paths=paths, env={}, unit_active=unit_active, **extra)


def replace_fixed_pp_table(spec: ExperimentSpec, arm: PPTableArm) -> ExperimentSpec:
    """One spec identical to `spec`, but declaring `arm` as its pp_table variant."""
    control = spec.fixed.control.model_copy(update={"pp_table": arm})
    fixed = spec.fixed.model_copy(update={"control": control})
    return spec.model_copy(update={"fixed": fixed})


# ---- the all-pass case: a stable snapshot -----------------------------------


def test_all_checks_pass_and_the_snapshot_is_stable(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)

    snapshot_1 = _preflight(paths, target, spec)
    snapshot_2 = _preflight(paths, target, spec)

    assert [check.name for check in snapshot_1.checks] == list(_EXPECTED_CHECK_ORDER)
    assert snapshot_1.resolved_model.name == "m"
    assert snapshot_1.device.id == "Vulkan0"
    assert snapshot_1.pp_table_bytes == _stock_pp_table_bytes()
    assert snapshot_1.snapshot_sha256 == snapshot_2.snapshot_sha256
    assert snapshot_1.snapshot_sha256 == snapshot_1.to_json()["snapshot_sha256"]


# ---- one test per check, each broken alone ----------------------------------


def test_check_workloads_refuses_a_bare_workload_name(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    spec = spec.model_copy(update={"workloads": ("decode",)})

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "workloads"
    assert excinfo.value.passed == ()


def test_check_workloads_refuses_an_unknown_corpus_set(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    spec = spec.model_copy(update={"workloads": ("corpus_no-such-family",)})

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "workloads"


def test_check_workloads_accepts_a_corpus_set(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    spec = spec.model_copy(update={"workloads": ("corpus_all",)})

    snapshot = _preflight(paths, target, spec)

    assert snapshot.checks[0].observed == [{"name": "corpus_all", "kind": "corpus"}]


def test_check_workloads_refuses_agent_turns_that_overflow_a_slot(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    spec = spec.model_copy(update={"workloads": ("agent_turns_2x4",)})

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "workloads"
    assert "tokens_per_slot is 4096" in excinfo.value.detail
    assert excinfo.value.passed == ()


def test_check_workloads_lets_agent_turns_that_fit_a_slot_pass_the_check(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    context = spec.fixed.context.model_copy(update={"tokens_per_slot": 12000})
    spec = spec.model_copy(
        update={
            "workloads": ("agent_turns_2x4",),
            "fixed": spec.fixed.model_copy(update={"context": context}),
        }
    )

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    # The host fixture declares 4096 tokens per lane, so a later check refuses.
    assert excinfo.value.check != "workloads"


def test_check_host_load_refuses_above_the_declared_limit(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    paths.loadavg.write_text("9.50 8.00 7.00 3/400 99999\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "host_load"
    assert [check.name for check in excinfo.value.passed] == ["workloads"]


def test_check_host_load_refuses_nan(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    paths.loadavg.write_text("nan 0 0 1/1 1\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "host_load"


def test_check_machine_limit_refuses_on_a_guard_mismatch(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    target.thermal_guard.write_text('#!/usr/bin/env bash\nMAX_C="${GPU_THERMAL_MAX_C:-70}"\n')
    target.thermal_guard.chmod(0o755)

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "machine_limit"
    assert [check.name for check in excinfo.value.passed] == ["workloads", "host_load"]


def test_check_wrappers_refuses_a_non_executable_script(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    target.gpu_lease.chmod(0o644)

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "wrappers"


def test_check_model_refuses_a_footprint_measured_at_another_context(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    model_yaml = target.config_root / "models" / "m.yaml"
    data = yaml.safe_load(model_yaml.read_text())
    data["footprint"]["ctx_per_lane"] = 2048
    model_yaml.write_text(yaml.safe_dump(data))

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "model"


def test_check_model_sha256_refuses_a_tampered_model_file(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    model_path = target.config_root / "model.gguf"
    model_path.write_bytes(model_path.read_bytes() + b"tampered\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "model_sha256"
    assert [check.name for check in excinfo.value.passed] == [
        "workloads",
        "host_load",
        "machine_limit",
        "wrappers",
        "model",
    ]


def test_check_model_sha256_refuses_an_unreadable_model(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    (target.config_root / "model.gguf").unlink()

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "model"


def test_check_build_refuses_a_build_id_mismatch(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    manifest_path = target.server_bin.parent.parent / "build-manifest.json"
    manifest_path.write_text(json.dumps({"build_id": "some-other-build", "features": ["vulkan"]}))

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "build"


def test_check_build_refuses_a_non_object_manifest(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    manifest_path = target.server_bin.parent.parent / "build-manifest.json"
    manifest_path.write_text(json.dumps(["not", "an", "object"]))

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "build"


def test_check_identity_refuses_an_unreadable_git_ref(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    (paths.repo_root / ".git" / "refs" / "heads" / "main").unlink()

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "identity"


def test_check_card_identity_refuses_an_expect_pci_mismatch(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    target = dataclasses.replace(target, expect_pci="0000:04:00.0")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "card_identity"


def test_check_pp_table_refuses_a_live_table_that_drifted_from_stock(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    variant = pptable_decode(_stock_pp_table_bytes()).set_temperature_limit(TempSensor.HOTSPOT, 77).encode()
    assert variant != _stock_pp_table_bytes()
    (paths.drm_root / f"card{_CARD}" / "device" / "pp_table").write_bytes(variant)

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "pp_table"


# ---- check 10's derived readback: the SMU must agree with the table --------


def _card_hwmon(paths: HostPaths) -> Path:
    return paths.drm_root / f"card{_CARD}" / "device" / "hwmon" / "hwmon1"


def test_check_pp_table_refuses_a_power_cap_readback_that_disagrees_with_the_table(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    # The table still declares 250 W, but the SMU has not taken it into
    # effect: hwmon still reports a stale (or fallback) power cap.
    (_card_hwmon(paths) / "power1_cap").write_text("180000000\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "pp_table"
    assert "power1_cap" in excinfo.value.detail


def test_check_pp_table_refuses_a_temperature_crit_readback_that_disagrees(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    (_card_hwmon(paths) / "temp2_crit").write_text("85000\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "pp_table"
    assert "temp2_crit" in excinfo.value.detail


def test_check_pp_table_refuses_when_the_measured_card_has_no_amdgpu_hwmon(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    shutil.rmtree(_card_hwmon(paths))

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "pp_table"
    assert "derived power cap" in excinfo.value.detail and "no amdgpu hwmon" in excinfo.value.detail


def test_check_pp_table_accepts_a_variant_table_when_the_derived_readback_agrees(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    variant = pptable_decode(_stock_pp_table_bytes()).set_socket_power_limit_ac(PptThrottler.PPT0, 220).encode()
    variant_sha = hashlib.sha256(variant).hexdigest()
    (paths.drm_root / f"card{_CARD}" / "device" / "pp_table").write_bytes(variant)
    (_card_hwmon(paths) / "power1_cap").write_text("220000000\n")
    armed = replace_fixed_pp_table(
        spec,
        PPTableArm(
            name="cap-220w",
            sha256=variant_sha,
            deltas_from_stock=(
                PPTableFieldDelta(field="socket_power_limit_ac", old=(250, 0, 0, 0), new=(220, 0, 0, 0)),
            ),
        ),
    )

    snapshot = _preflight(paths, target, armed)

    assert snapshot.identity["pp_table"]["live_sha256"] == variant_sha
    assert snapshot.identity["pp_table"]["declared_name"] == "cap-220w"
    assert snapshot.pp_table_bytes == variant


# ---- check 10 with power_cap_w: a hwmon power1_cap write on a patched kernel -


def _power_cap_spec(spec: ExperimentSpec, watts: int) -> ExperimentSpec:
    control = spec.fixed.control.model_copy(update={"power_cap_w": watts})
    return spec.model_copy(update={"fixed": spec.fixed.model_copy(update={"control": control})})


def _patched_kernel_host(tmp_path: Path, watts: int = 220) -> tuple[HostPaths, RunTarget, ExperimentSpec]:
    """The base host, but with the patched kernel's floor and the operator's cap write."""
    paths, target, spec = _build_host(tmp_path)
    (_card_hwmon(paths) / "power1_cap_min").write_text("120000000\n")
    (_card_hwmon(paths) / "power1_cap").write_text(f"{watts * 1_000_000}\n")
    return paths, target, _power_cap_spec(spec, watts)


def test_check_pp_table_accepts_a_power_cap_on_a_patched_kernel(tmp_path: Path) -> None:
    paths, target, spec = _patched_kernel_host(tmp_path, 220)

    snapshot = _preflight(paths, target, spec)

    info = snapshot.identity["pp_table"]
    assert [check.name for check in snapshot.checks] == list(_EXPECTED_CHECK_ORDER)
    assert info["control_mode"] == "power_cap"
    assert info["declared_power_cap_w"] == 220
    assert info["live_power1_cap_uw"] == 220_000_000
    assert info["live_power1_cap_min_uw"] == 120_000_000
    assert info["live_sha256"] == _STOCK_PP_TABLE_SHA256
    assert snapshot.pp_table_bytes == _stock_pp_table_bytes()


def test_stock_spec_records_the_stock_control_mode(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)

    info = _preflight(paths, target, spec).identity["pp_table"]

    assert info["control_mode"] == "stock"
    assert info["declared_power_cap_w"] is None
    assert info["live_power1_cap_uw"] == 250_000_000
    assert info["live_power1_cap_min_uw"] is None


def test_check_pp_table_refuses_a_power_cap_that_differs_from_the_declared_one(tmp_path: Path) -> None:
    paths, target, spec = _patched_kernel_host(tmp_path, 220)
    (_card_hwmon(paths) / "power1_cap").write_text("250000000\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "pp_table"
    assert "power1_cap reads 250.0 W" in excinfo.value.detail
    assert "power_cap_w 220" in excinfo.value.detail


def test_check_pp_table_refuses_an_unpatched_kernel(tmp_path: Path) -> None:
    paths, target, spec = _patched_kernel_host(tmp_path, 220)
    (_card_hwmon(paths) / "power1_cap_min").write_text("250000000\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "pp_table"
    assert "power1_cap_min reads 250.0 W" in excinfo.value.detail
    assert "device cannot use this cap" in excinfo.value.detail


def test_check_pp_table_refuses_a_power_cap_when_the_floor_file_is_missing(tmp_path: Path) -> None:
    paths, target, spec = _patched_kernel_host(tmp_path, 220)
    (_card_hwmon(paths) / "power1_cap_min").unlink()

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "pp_table"
    assert "power1_cap_min" in excinfo.value.detail


def test_check_pp_table_refuses_a_power_cap_with_a_non_stock_table(tmp_path: Path) -> None:
    paths, target, spec = _patched_kernel_host(tmp_path, 220)
    variant = pptable_decode(_stock_pp_table_bytes()).set_socket_power_limit_ac(PptThrottler.PPT0, 220).encode()
    (paths.drm_root / f"card{_CARD}" / "device" / "pp_table").write_bytes(variant)

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "pp_table"
    assert "requires the stock pp_table" in excinfo.value.detail
    assert hashlib.sha256(variant).hexdigest() in excinfo.value.detail
    assert _STOCK_PP_TABLE_SHA256 in excinfo.value.detail


def test_check_pp_table_still_checks_temperature_crit_with_a_power_cap(tmp_path: Path) -> None:
    paths, target, spec = _patched_kernel_host(tmp_path, 220)
    (_card_hwmon(paths) / "temp2_crit").write_text("85000\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "pp_table"
    assert "temp2_crit" in excinfo.value.detail


def test_check_power_profile_refuses_a_profile_mismatch(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    (paths.drm_root / f"card{_CARD}" / "device" / "pp_power_profile_mode").write_text(
        "PROFILE_INDEX(NAME) CLOCK_TYPE(NAME)\n 0 BOOTUP_DEFAULT*:\n 2   POWER_SAVING :\n"
    )

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "power_profile"


def test_check_power_profile_checks_cards_outside_the_model_placement(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    other = _make_device_dir(paths.drm_root, 1, "0000:04:00.0")
    (other / "power_dpm_force_performance_level").write_text("manual\n")
    (other / "pp_power_profile_mode").write_text(" 2   COMPUTE*:\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "power_profile"


def _with_cpu_epp(paths: HostPaths, spec: ExperimentSpec, cpu_root: Path, cpu_epp: str | None):
    host = HostFixed(loadavg_1m_max=spec.fixed.host.loadavg_1m_max, cpu_epp=cpu_epp)
    fixed = spec.fixed.model_copy(update={"host": host})
    return dataclasses.replace(paths, cpu_root=cpu_root), spec.model_copy(update={"fixed": fixed})


def test_check_cpu_epp_passes_and_records_the_state_when_the_spec_declares_it(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    paths, spec = _with_cpu_epp(paths, spec, make_cpu_root(tmp_path / "cpu", epp="performance"), "performance")

    snapshot = _preflight(paths, target, spec)

    names = [check.name for check in snapshot.checks]
    assert names.index("cpu_epp") == names.index("power_profile") + 1
    observed = next(check.observed for check in snapshot.checks if check.name == "cpu_epp")
    assert observed["declared"] == "performance"
    assert observed["state"]["epp"] == {"0": "performance", "1": "performance"}


def test_check_cpu_epp_refuses_a_mismatch(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    paths, spec = _with_cpu_epp(paths, spec, make_cpu_root(tmp_path / "cpu", epp="balance_performance"), "performance")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "cpu_epp"
    assert "power_profile" in [check.name for check in excinfo.value.passed]


def test_check_cpu_epp_is_not_run_when_the_spec_does_not_declare_it(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    paths = dataclasses.replace(paths, cpu_root=tmp_path / "no-such-cpu-root")

    snapshot = _preflight(paths, target, spec)

    assert "cpu_epp" not in [check.name for check in snapshot.checks]


def test_check_fan_watchdog_refuses_when_its_unit_is_not_active(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec, unit_active=lambda: False)

    assert excinfo.value.check == "fan_watchdog"


def test_check_fan_watchdog_refuses_a_non_ok_state(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    paths.fan_status_file.write_text("state=DEGRADED target_pwm=255 max_gpu_junction_mC=36000\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "fan_watchdog"


def test_check_fan_watchdog_survives_an_empty_read_during_a_rewrite(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    good = paths.fan_status_file.read_text()
    paths.fan_status_file.write_text("")
    sleeps: list[float] = []

    def finish_rewrite(delay_s: float) -> None:
        sleeps.append(delay_s)
        paths.fan_status_file.write_text(good)

    snapshot = _preflight(paths, target, spec, sleep=finish_rewrite)

    assert "fan_watchdog" in [check.name for check in snapshot.checks]
    assert len(sleeps) == 1


def test_check_fan_watchdog_refuses_a_file_that_stays_empty(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    paths.fan_status_file.write_text("")
    sleeps: list[float] = []

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec, sleep=sleeps.append)

    assert excinfo.value.check == "fan_watchdog"
    assert "after 5 attempts" in str(excinfo.value)
    assert len(sleeps) == 4


def test_check_fan_stall_floors_refuses_a_channel_below_its_floor(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    target = dataclasses.replace(target, stall_floors_rpm={2: 2000.0})

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "fan_stall_floors"


def test_check_fan_stall_floors_refuses_an_undetected_requested_channel(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    target = dataclasses.replace(target, stall_floors_rpm={2: 1000.0, 4: 500.0})

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "fan_stall_floors"
    assert "4" in excinfo.value.detail


def test_check_vram_fit_refuses_an_undersized_card(tmp_path: Path) -> None:
    paths, target, spec = _build_host(tmp_path)
    (paths.drm_root / f"card{_CARD}" / "device" / "mem_info_vram_used").write_text(f"{3900 * 1024 * 1024}\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "vram_fit"
    expected_passed = [name for name in _EXPECTED_CHECK_ORDER if name != "vram_fit"]
    assert [check.name for check in excinfo.value.passed] == expected_passed


# ---- start_gate --------------------------------------------------------------


def _fake_clock(start: float = 0.0):
    state = {"now": start}

    def clock() -> float:
        return state["now"]

    def sleep(seconds: float) -> None:
        state["now"] += seconds

    return clock, sleep


def test_start_gate_passes_once_every_card_cools(tmp_path: Path) -> None:
    _, _, spec = _build_host(tmp_path)
    clock, sleep = _fake_clock()
    calls = {"n": 0}

    def read_junctions() -> dict[int, float]:
        n = calls["n"]
        calls["n"] += 1
        temperature = max(30.0 - n * 2.0, 20.0)
        return {0: temperature, 1: temperature}

    result = start_gate(spec, read_junctions=read_junctions, timeout_s=600, poll_s=1.0, clock=clock, sleep=sleep)

    assert result["passed"] is True
    assert result["max_c"] == 24
    assert result["tolerance_c"] == 1
    assert result["hold_s"] == 30
    assert result["passed_mono"] - result["started_mono"] >= 30.0


def test_start_gate_refuses_at_the_timeout_when_it_never_cools(tmp_path: Path) -> None:
    _, _, spec = _build_host(tmp_path)
    clock, sleep = _fake_clock()

    def read_junctions() -> dict[int, float]:
        return {0: 40.0}

    with pytest.raises(PreflightRefusal) as excinfo:
        start_gate(spec, read_junctions=read_junctions, timeout_s=5.0, poll_s=1.0, clock=clock, sleep=sleep)

    assert excinfo.value.check == "start_gate"


def test_start_gate_restarts_the_hold_after_a_dip_and_a_rise(tmp_path: Path) -> None:
    _, _, spec = _build_host(tmp_path)
    clock, sleep = _fake_clock()
    calls = {"n": 0}

    def read_junctions() -> dict[int, float]:
        n = calls["n"]
        calls["n"] += 1
        if n == 5:
            return {0: 40.0}  # one spike above the band, mid-hold
        return {0: 20.0}  # otherwise comfortably within the band

    result = start_gate(spec, read_junctions=read_junctions, timeout_s=100.0, poll_s=1.0, clock=clock, sleep=sleep)

    assert result["passed"] is True
    # Had the spike at n=5 NOT restarted the hold, this would have passed at
    # mono==30 (30s after the very first reading). A restarted hold cannot
    # pass before the spike plus a full fresh hold_s.
    assert result["passed_mono"] >= 35.0


def test_start_gate_does_not_pass_after_its_timeout(tmp_path: Path) -> None:
    _, _, spec = _build_host(tmp_path)
    spec = spec.model_copy(
        update={
            "fixed": spec.fixed.model_copy(
                update={"start_gate": spec.fixed.start_gate.model_copy(update={"hold_s": 1})}
            )
        }
    )
    clock, sleep = _fake_clock()

    def read_junctions() -> dict[int, float]:
        return {0: 20.0}

    with pytest.raises(PreflightRefusal) as excinfo:
        start_gate(spec, read_junctions=read_junctions, timeout_s=1.0, poll_s=2.0, clock=clock, sleep=sleep)

    assert excinfo.value.check == "start_gate"


# ---- dual-card checks: placement_pair, peer_cap, peer_idle, ports_distinct ----

_PEER = 1
_PEER_PCI = "0000:04:00.0"
_PAIR_CHECK_ORDER = (*_EXPECTED_CHECK_ORDER, "placement_pair", "peer_cap", "peer_idle")


def _build_pair_host(
    tmp_path: Path, *, power_cap_w: int | None = None
) -> tuple[HostPaths, RunTarget, RunTarget, ExperimentSpec]:
    """The single-card host plus a second card, a second resident model, and a pair spec."""
    paths, target, spec = _build_host(tmp_path)
    drm_root = paths.drm_root
    peer_device = _make_device_dir(drm_root, _PEER, _PEER_PCI)
    (peer_device / "mem_info_vram_used").write_text(f"{100 * 1024 * 1024}\n")
    (peer_device / "mem_info_vram_total").write_text(f"{4096 * 1024 * 1024}\n")
    (peer_device / "mem_info_gtt_used").write_text("0\n")
    (peer_device / "power_dpm_force_performance_level").write_text("manual\n")
    (peer_device / "pp_power_profile_mode").write_text(
        "PROFILE_INDEX(NAME) CLOCK_TYPE(NAME)\n 0 BOOTUP_DEFAULT :\n 2   POWER_SAVING*:\n"
    )
    (peer_device / "pp_table").write_bytes(_stock_pp_table_bytes())
    peer_hwmon = peer_device / "hwmon" / "hwmon3"
    peer_hwmon.mkdir(parents=True)
    for name, value in {
        "name": "amdgpu\n",
        "power1_cap": "250000000\n",
        "temp1_crit": "100000\n",
        "temp2_crit": "100000\n",
        "temp3_crit": "98000\n",
    }.items():
        (peer_hwmon / name).write_text(value)
    if power_cap_w is not None:
        for device in (drm_root / f"card{_CARD}" / "device", peer_device):
            hwmon_dir = next((device / "hwmon").iterdir())
            (hwmon_dir / "power1_cap").write_text(f"{power_cap_w * 1_000_000}\n")
            (hwmon_dir / "power1_cap_min").write_text("120000000\n")

    config_root = target.config_root
    (config_root / "placement.yaml").write_text(
        yaml.safe_dump(
            {
                "devices": [
                    {"id": "Vulkan0", "card": _CARD, "reserve_mib": 100},
                    {"id": "Vulkan1", "card": _PEER, "reserve_mib": 100},
                ],
                "resident": [{"model": "m", "device": "Vulkan0"}, {"model": "m1", "device": "Vulkan1"}],
                "router": {"host": "127.0.0.1", "port": 8100, "models_max": 2},
            }
        )
    )
    shutil.copyfile(config_root / "models" / "m.yaml", config_root / "models" / "m1.yaml")

    control = spec.fixed.control.model_copy(update={"power_cap_w": power_cap_w})
    fixed = spec.fixed.model_copy(
        update={
            "control": control,
            "placement": PlacementFixed(mode="per-card", cards=(_CARD, _PEER), on_arm_bound="continue_peer"),
        }
    )
    pair_spec = spec.model_copy(update={"fixed": fixed})
    peer_target = dataclasses.replace(target, card=_PEER, port=target.port + 1, expect_pci=_PEER_PCI)
    return paths, target, peer_target, pair_spec


def _peer_hwmon(paths: HostPaths) -> Path:
    return next((paths.drm_root / f"card{_PEER}" / "device" / "hwmon").iterdir())


def test_pair_checks_pass_for_either_card_and_extend_the_check_list(tmp_path: Path) -> None:
    paths, target, peer_target, spec = _build_pair_host(tmp_path)

    first = _preflight(paths, target, spec)
    second = _preflight(paths, peer_target, spec)

    assert [check.name for check in first.checks] == list(_PAIR_CHECK_ORDER)
    assert [check.name for check in second.checks] == list(_PAIR_CHECK_ORDER)
    pair_info = next(check.observed for check in first.checks if check.name == "placement_pair")
    assert pair_info["models"] == {"0": "m", "1": "m1"}
    assert first.device.id == "Vulkan0"
    assert second.device.id == "Vulkan1"
    assert second.resolved_model.name == "m1"


def test_pair_checks_pass_with_a_declared_cap_on_both_cards(tmp_path: Path) -> None:
    paths, target, peer_target, spec = _build_pair_host(tmp_path, power_cap_w=180)

    snapshot = _preflight(paths, peer_target, spec)

    peer_cap = next(check.observed for check in snapshot.checks if check.name == "peer_cap")
    assert peer_cap["card"] == _CARD
    assert peer_cap["live_power1_cap_uw"] == 180_000_000
    assert _preflight(paths, target, spec).identity["pp_table"]["live_power1_cap_uw"] == 180_000_000


def test_a_single_card_spec_runs_no_pair_check(tmp_path: Path) -> None:
    paths, target, _peer_target, spec = _build_pair_host(tmp_path)
    plain = spec.model_copy(update={"fixed": spec.fixed.model_copy(update={"placement": None})})

    snapshot = _preflight(paths, target, plain)

    assert [check.name for check in snapshot.checks] == list(_EXPECTED_CHECK_ORDER)


def test_peer_cap_refuses_a_peer_with_the_wrong_cap(tmp_path: Path) -> None:
    paths, target, _peer_target, spec = _build_pair_host(tmp_path, power_cap_w=180)
    (_peer_hwmon(paths) / "power1_cap").write_text("250000000\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "peer_cap"
    assert "peer card 1" in excinfo.value.detail
    assert "power_cap_w 180" in excinfo.value.detail
    assert excinfo.value.passed[-1].name == "placement_pair"


def test_peer_cap_refuses_an_unpatched_peer_kernel_floor(tmp_path: Path) -> None:
    paths, target, _peer_target, spec = _build_pair_host(tmp_path, power_cap_w=180)
    (_peer_hwmon(paths) / "power1_cap_min").write_text("200000000\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "peer_cap"
    assert "power1_cap_min" in excinfo.value.detail


def test_peer_cap_refuses_a_peer_with_a_modified_table(tmp_path: Path) -> None:
    paths, target, _peer_target, spec = _build_pair_host(tmp_path)
    peer_table = paths.drm_root / f"card{_PEER}" / "device" / "pp_table"
    peer_table.write_bytes(peer_table.read_bytes() + b"\x00")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "peer_cap"
    assert "peer card 1" in excinfo.value.detail
    assert "stock" in excinfo.value.detail


def test_peer_idle_refuses_a_busy_peer_card(tmp_path: Path) -> None:
    paths, target, _peer_target, spec = _build_pair_host(tmp_path)
    (paths.drm_root / f"card{_PEER}" / "device" / "mem_info_vram_used").write_text(f"{2048 * 1024 * 1024}\n")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "peer_idle"
    assert "peer card 1" in excinfo.value.detail
    assert [check.name for check in excinfo.value.passed][-1] == "peer_cap"


def _rewrite_peer_model(target: RunTarget, **changes: object) -> None:
    path = target.config_root / "models" / "m1.yaml"
    raw = yaml.safe_load(path.read_text())
    for key, value in changes.items():
        if key == "footprint":
            raw["footprint"].update(value)
        else:
            raw[key] = value
    path.write_text(yaml.safe_dump(raw))


def test_placement_pair_refuses_two_different_ggufs(tmp_path: Path) -> None:
    paths, target, _peer_target, spec = _build_pair_host(tmp_path)
    other = target.config_root / "other.gguf"
    other.write_bytes(b"another model\n")
    _rewrite_peer_model(target, model=str(other))

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "placement_pair"
    assert "same GGUF" in excinfo.value.detail


def test_placement_pair_refuses_two_different_footprints(tmp_path: Path) -> None:
    paths, target, _peer_target, spec = _build_pair_host(tmp_path)
    _rewrite_peer_model(target, footprint={"base_mib": 1200.0})

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "placement_pair"
    assert "footprint" in excinfo.value.detail


def test_placement_pair_refuses_a_peer_with_two_resident_models(tmp_path: Path) -> None:
    paths, target, _peer_target, spec = _build_pair_host(tmp_path)
    placement_path = target.config_root / "placement.yaml"
    raw = yaml.safe_load(placement_path.read_text())
    raw["resident"].append({"model": "m2", "device": "Vulkan1"})
    raw["router"]["models_max"] = 3
    placement_path.write_text(yaml.safe_dump(raw))
    shutil.copyfile(target.config_root / "models" / "m.yaml", target.config_root / "models" / "m2.yaml")

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "placement_pair"
    assert "exactly one resident model" in excinfo.value.detail


def test_placement_pair_refuses_a_target_outside_the_declared_cards(tmp_path: Path) -> None:
    paths, target, _peer_target, spec = _build_pair_host(tmp_path)
    placement = PlacementFixed(mode="per-card", cards=(1, 2), on_arm_bound="continue_peer")
    spec = spec.model_copy(update={"fixed": spec.fixed.model_copy(update={"placement": placement})})

    with pytest.raises(PreflightRefusal) as excinfo:
        _preflight(paths, target, spec)

    assert excinfo.value.check == "placement_pair"
    assert "not in fixed.placement.cards" in excinfo.value.detail


def test_preflight_pair_returns_one_snapshot_per_card_and_hashes_the_model_once(tmp_path: Path) -> None:
    paths, target, peer_target, spec = _build_pair_host(tmp_path)
    hashed: list[Path] = []

    def counting_sha256(path: Path) -> str:
        hashed.append(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    snapshots = preflight_pair(
        spec, [peer_target, target], paths=paths, env={}, unit_active=lambda: True, model_sha256=counting_sha256
    )

    assert [snapshot.device.id for snapshot in snapshots] == ["Vulkan0", "Vulkan1"]
    assert len(hashed) == 1


def test_preflight_pair_refuses_a_shared_port(tmp_path: Path) -> None:
    paths, target, peer_target, spec = _build_pair_host(tmp_path)
    peer_target = dataclasses.replace(peer_target, port=target.port)

    with pytest.raises(PreflightRefusal) as excinfo:
        preflight_pair(spec, [target, peer_target], paths=paths, env={}, unit_active=lambda: True)

    assert excinfo.value.check == "ports_distinct"


def test_preflight_pair_refuses_targets_that_miss_a_declared_card(tmp_path: Path) -> None:
    paths, target, _peer_target, spec = _build_pair_host(tmp_path)

    with pytest.raises(PreflightRefusal) as excinfo:
        preflight_pair(spec, [target], paths=paths, env={}, unit_active=lambda: True)
    assert excinfo.value.check == "placement_pair"

    plain = spec.model_copy(update={"fixed": spec.fixed.model_copy(update={"placement": None})})
    with pytest.raises(PreflightRefusal) as excinfo:
        preflight_pair(plain, [target], paths=paths, env={}, unit_active=lambda: True)
    assert "no fixed.placement" in excinfo.value.detail
