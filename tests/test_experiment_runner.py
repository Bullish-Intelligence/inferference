"""Runner integration tests on a fabricated host and the WP10 fake server."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from inferference.experiment.control import FanLease, arbitrate, parse_request
from inferference.experiment.preflight import HostPaths, RunTarget
from inferference.experiment.runner import run_experiment
from inferference.experiment.server import ServerProcess
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
    Replication,
    SafetyFixed,
    StartGateFixed,
)
from tests.pp_tables import synthetic_stock_blob

_STOCK_TABLE = synthetic_stock_blob()
_FAKE_SERVER = Path(__file__).resolve().parent / "fixtures" / "experiment" / "fake_llama_server.py"
_PCI = {0: "0000:03:00.0", 1: "0000:04:00.0"}
_MIB = 1024 * 1024


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace `path` atomically for a reader thread in this process.

    There is no `fsync`: the readers share this host, and `os.replace` already
    hides a partial file from them. An `fsync` only adds disk-flush stalls,
    which exceed 200 ms under disk load and race the 250 ms readback window and
    the fake loader's wait for a server to become healthy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode())


def _wait_for(predicate: Any, *, timeout_s: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


@dataclass
class _Host:
    paths: HostPaths
    target: RunTarget
    spec_path: Path
    artifact_root: Path
    request_file: Path
    status_file: Path
    sigint_file: Path
    lease_budget_file: Path
    card0_device: Path
    watchdog: _FakeWatchdog

    def env(self, **extra: str) -> dict[str, str]:
        return {
            **os.environ,
            "FAKE_SERVER_SIGINT_COUNT_FILE": str(self.sigint_file),
            "FAKE_SERVER_DRM_ROOT": str(self.paths.drm_root),
            "GPU_LEASE_BUDGET_FILE": str(self.lease_budget_file),
            **extra,
        }

    def lease_factory(self, profile: str) -> FanLease:
        return FanLease(
            profile,
            request_file=self.request_file,
            status_file=self.status_file,
            readback_timeout_s=1.0,
            poll_s=0.002,
        )


class _FakeWatchdog:
    """Drive the real lease arbitration rule against temporary files."""

    def __init__(self, request_file: Path, status_file: Path, *, honor: bool) -> None:
        self.request_file = request_file
        self.status_file = status_file
        self.honor = honor
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="fake-fan-watchdog", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            request = None
            if self.honor and self.request_file.is_file():
                try:
                    request = parse_request(self.request_file.read_text())
                except (OSError, ValueError):
                    request = None
            profile, _reason = arbitrate(
                request,
                now=datetime.now(UTC),
                owner_alive=True,
                allowlist=frozenset({"flat-max"}),
                fault=False,
                default_profile="default",
            )
            _atomic_text(self.status_file, f"state=OK target_pwm=255 max_gpu_junction_mC=36000 profile={profile}\n")
            self.stop_event.wait(0.002)


class _FakeResidentLoader:
    """Raise fake card-0 VRAM after the fake server starts, then restore it."""

    def __init__(self, host: _Host) -> None:
        self.host = host
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="fake-resident-loader", daemon=True)
        self.loaded: set[Path] = set()

    def __enter__(self) -> _FakeResidentLoader:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1)

    def _run(self) -> None:
        usage_file = self.host.card0_device / "mem_info_vram_used"
        baseline = 100 * _MIB
        while not self.stop_event.is_set():
            if self.host.artifact_root.exists():
                for run_dir in self.host.artifact_root.iterdir():
                    output = run_dir / ".server-output"
                    if output.exists():
                        if run_dir not in self.loaded:
                            _atomic_text(usage_file, f"{baseline + 60 * _MIB}\n")
                            self.loaded.add(run_dir)
                    elif run_dir in self.loaded and (run_dir / "server.log").exists():
                        _atomic_text(usage_file, f"{baseline}\n")
                        self.loaded.remove(run_dir)
            self.stop_event.wait(0.002)


@contextmanager
def _make_host(
    tmp_path: Path,
    *,
    factor_values: tuple[str, ...] = ("1",),
    workloads: tuple[str, ...] = ("soak_1s",),
    honor_lease: bool = True,
    power_cap_w: int | None = None,
) -> Iterator[_Host]:
    drm_root = tmp_path / "sys-class-drm"
    hwmon_root = tmp_path / "sys-class-hwmon"
    device_roots: dict[int, Path] = {}
    for card, pci in _PCI.items():
        card_root = drm_root / "pci-devices" / pci
        device_roots[card] = card_root
        (card_root / "hwmon" / f"hwmon{card + 1}").mkdir(parents=True)
        (drm_root / f"card{card}").mkdir(parents=True)
        (drm_root / f"card{card}" / "device").symlink_to(card_root, target_is_directory=True)
        values = {
            "pp_table": _STOCK_TABLE,
            "mem_info_vram_used": f"{100 * _MIB}\n",
            "mem_info_vram_total": f"{8192 * _MIB}\n",
            "mem_info_gtt_used": f"{10 * _MIB}\n",
            "mem_info_gtt_total": f"{8192 * _MIB}\n",
            "power_dpm_force_performance_level": "manual\n",
            "pp_power_profile_mode": "PROFILE_INDEX(NAME) CLOCK_TYPE(NAME)\n 2   POWER_SAVING*:\n",
            "gpu_busy_percent": "1\n",
            "mem_busy_percent": "1\n",
            "pp_dpm_sclk": "0: 500Mhz *\n1: 800Mhz\n",
            "pp_dpm_mclk": "0: 500Mhz *\n1: 945Mhz\n",
            "gpu_metrics": "",
        }
        for name, value in values.items():
            path = card_root / name
            if name == "gpu_metrics":
                path.write_bytes(b"\x00" * 72)
            elif name == "pp_table":
                assert isinstance(value, bytes)
                path.write_bytes(value)
            else:
                assert isinstance(value, str)
                path.write_text(value)
        device_hwmon = card_root / "hwmon" / f"hwmon{card + 1}"
        # `power1_cap` and `temp*_crit` must agree with the stock pp_table the
        # host serves: preflight's derived readback (check 10) compares them
        # (PPT0 250 W, EDGE/HOTSPOT 100 C, MEM 98 C).
        hwmon_values = {
            "name": "amdgpu\n",
            "temp1_input": "20000\n",
            "temp2_input": "20000\n",
            "temp3_input": "20000\n",
            "power1_average": "35000000\n",
            "power1_cap": "250000000\n",
            "in0_input": "900\n",
            "freq1_input": "500000000\n",
            "freq2_input": "945000000\n",
            "temp1_label": "edge\n",
            "temp2_label": "junction\n",
            "temp3_label": "mem\n",
            "temp1_crit": "100000\n",
            "temp2_crit": "100000\n",
            "temp3_crit": "98000\n",
        }
        if power_cap_w is not None:
            # A patched kernel: the floor is 120 W and the owner wrote the cap.
            hwmon_values["power1_cap"] = f"{power_cap_w * 1_000_000}\n"
            hwmon_values["power1_cap_min"] = "120000000\n"
        for name, value in hwmon_values.items():
            (device_hwmon / name).write_text(value)

    fan_chip = hwmon_root / "hwmon5"
    fan_chip.mkdir(parents=True)
    for name, value in {
        "name": "arctic_fan\n",
        "pwm2": "255\n",
        "fan2_input": "1500\n",
        "pwm3": "255\n",
        "fan3_input": "1400\n",
    }.items():
        (fan_chip / name).write_text(value)

    loadavg = tmp_path / "loadavg"
    loadavg.write_text("0.10 0.10 0.10 1/100 123\n")
    system_target = tmp_path / "nix-store-system"
    system_target.mkdir()
    current_system = tmp_path / "current-system"
    current_system.symlink_to(system_target, target_is_directory=True)

    request_file = tmp_path / "arctic-fan" / "request"
    status_file = tmp_path / "arctic-fan" / "status"
    request_file.parent.mkdir()
    status_file.write_text("state=OK target_pwm=255 max_gpu_junction_mC=36000 profile=default\n")
    watchdog = _FakeWatchdog(request_file, status_file, honor=honor_lease)
    watchdog.start()

    repo_root = tmp_path / "repo"
    (repo_root / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo_root / ".git" / "refs" / "heads" / "main").write_text("deadbeefcafef00d\n")

    config_root = tmp_path / "config"
    (config_root / "models").mkdir(parents=True)
    model_path = config_root / "model.gguf"
    model_path.write_bytes(b"fabricated model identity only\n")
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (config_root / "placement.yaml").write_text(
        yaml.safe_dump(
            {
                "devices": [{"id": "Vulkan0", "card": 0, "reserve_mib": 100}],
                "resident": [{"model": "m", "device": "Vulkan0"}],
                "router": {"host": "127.0.0.1", "port": 8100, "models_max": 1},
            }
        )
    )
    (config_root / "models" / "m.yaml").write_text(
        yaml.safe_dump(
            {
                "model": str(model_path),
                "ctx_per_lane": 128,
                "lanes": 1,
                "footprint": {
                    "ctx_per_lane": 128,
                    "base_mib": 50.0,
                    "per_lane_mib": 10.0,
                    "measured_by": "fabricated-sysfs-vram-delta",
                },
            }
        )
    )

    output = tmp_path / "out"
    (output / "bin").mkdir(parents=True)
    server_bin = output / "bin" / "llama-server"
    shutil.copyfile(_FAKE_SERVER, server_bin)
    server_bin.chmod(0o755)
    (output / "build-manifest.json").write_text(json.dumps({"build_id": "fake-build", "features": ["vulkan"]}))

    lease_budget_file = tmp_path / "lease-budget"
    gpu_lease = tmp_path / "gpu-lease.sh"
    gpu_lease.write_text('#!/bin/sh\nprintf \'%s\\n\' "$1" > "$GPU_LEASE_BUDGET_FILE"\nshift 2\nexec "$@"\n')
    gpu_lease.chmod(0o755)
    thermal_guard = tmp_path / "gpu-thermal-guard.sh"
    thermal_guard.write_text('#!/bin/sh\nMAX_C="${GPU_THERMAL_MAX_C:-80}"\nshift 2\nexec "$@"\n')
    thermal_guard.chmod(0o755)

    stock_path = tmp_path / "stock-card0.pp_table"
    stock_path.write_bytes(_STOCK_TABLE)
    target = RunTarget(
        card=0,
        port=_free_port(),
        server_bin=server_bin,
        config_root=config_root,
        stock_pp_table=stock_path,
        thermal_guard=thermal_guard,
        gpu_lease=gpu_lease,
        expect_pci=_PCI[0],
        stall_floors_rpm={2: 1000.0},
    )
    paths = HostPaths(
        drm_root=drm_root,
        hwmon_root=hwmon_root,
        loadavg=loadavg,
        current_system=current_system,
        fan_status_file=status_file,
        repo_root=repo_root,
    )
    spec = ExperimentSpec(
        experiment="wp12-runner-test",
        schema_version=1,
        fixed=Fixed(
            model=ModelFixed(sha256_must_match=True, sha256=model_sha),
            build=BuildFixed(manifest_must_match=True, build_id="fake-build", features=("vulkan",)),
            context=ContextFixed(slots=2, tokens_per_slot=128),
            cache=CacheFixed(k="q8_0", v="q8_0"),
            batch=BatchFixed(size=64, ubatch=64),
            control=ControlFixed(fan_profile="flat-max", power_profile="power-saving", power_cap_w=power_cap_w),
            safety=SafetyFixed(machine_limit_c=80, arm_bound_c=75, gtt_limit_mib=512),
            start_gate=StartGateFixed(max_c=24, tolerance_c=1, hold_s=0),
            host=HostFixed(loadavg_1m_max=5.0),
        ),
        factor=Factor(name="lanes", values=factor_values),
        workloads=workloads,
        replication=Replication(n=1, order="counterbalanced", cooldown_between="start_gate"),
        endpoints=Endpoints(primary="band_slope", band_c=(50, 70), secondary=()),
    )
    spec_path = tmp_path / "experiment.yaml"
    spec_path.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    artifact_root = tmp_path / "artifacts"
    host = _Host(
        paths=paths,
        target=target,
        spec_path=spec_path,
        artifact_root=artifact_root,
        request_file=request_file,
        status_file=status_file,
        sigint_file=tmp_path / "sigint-count",
        lease_budget_file=lease_budget_file,
        card0_device=device_roots[0],
        watchdog=watchdog,
    )
    try:
        yield host
    finally:
        watchdog.close()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _run(host: _Host, *, env: dict[str, str] | None = None):
    with _FakeResidentLoader(host):
        return run_experiment(
            host.spec_path,
            host.target,
            artifact_root=host.artifact_root,
            paths=host.paths,
            env=host.env() if env is None else env,
            unit_active=lambda: True,
            sample_interval_s=0.01,
            fan_lease_factory=host.lease_factory,
            http_timeout_s=3,
        )


def _watch(host: _Host, action: Any) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def run() -> None:
        action(stop)

    thread = threading.Thread(target=run, name="runner-test-mutation", daemon=True)
    thread.start()
    return stop, thread


def _wait_for_workload_request(host: _Host) -> bool:
    def dispatched() -> bool:
        return any(
            len(tuple((directory / "requests").glob("*.request.json"))) >= 2
            for directory in host.artifact_root.glob("*")
        )

    return _wait_for(dispatched)


def test_completed_run_writes_every_declared_artifact_and_one_sigint(tmp_path: Path) -> None:
    with _make_host(tmp_path, workloads=("lane_sweep",)) as host:
        results = _run(host)
        assert len(results) == 1
        result = results[0]
        assert result.state == "completed"
        assert result.failure_boundary is None
        run_dir = result.run_dir

        expected = {
            "run.json",
            "declaration.json",
            "preflight.json",
            "resolved-command.json",
            "identity.json",
            "residency.json",
            "telemetry.jsonl",
            "requests",
            "stages.json",
            "outcome.json",
            "control.json",
            "pp_table.bin",
            "server.log",
            "list-devices.txt",
        }
        assert expected <= {path.name for path in run_dir.iterdir()}
        run_json = json.loads((run_dir / "run.json").read_text())
        assert run_json["spec_sha256"] == hashlib.sha256(host.spec_path.read_bytes()).hexdigest()
        for name in (
            "declaration.json",
            "preflight.json",
            "resolved-command.json",
            "identity.json",
            "residency.json",
            "stages.json",
            "outcome.json",
            "control.json",
        ):
            payload = json.loads((run_dir / name).read_text())
            assert payload["schema_version"] == 1
            assert "written_utc" in payload
        assert len(json.loads((run_dir / "residency.json").read_text())["checks"]) == 10
        stages = json.loads((run_dir / "stages.json").read_text())
        assert stages["start_gate"]["passed"] is True
        assert stages["stages"][0]["concurrency"] == 1
        assert stages["stages"][0]["requests"][0]["valid"] is True
        telemetry = [json.loads(line) for line in (run_dir / "telemetry.jsonl").read_text().splitlines()]
        assert telemetry and all(sample["schema_version"] == 1 for sample in telemetry)
        requests = tuple((run_dir / "requests").iterdir())
        assert requests
        assert any(path.name.endswith(".request.json") for path in requests)
        assert any(path.name.endswith(".response.sse") for path in requests)
        assert json.loads((run_dir / "outcome.json").read_text())["state"] == "completed"
        control = json.loads((run_dir / "control.json").read_text())
        assert control["fan"]["restore_confirmed"] is True
        assert not host.request_file.exists()
        assert host.sigint_file.read_text() == "1"
        assert host.lease_budget_file.read_text().strip() == "70"
        assert "Vulkan0" in (run_dir / "list-devices.txt").read_text()


def test_arm_bound_run_is_bounded_and_next_scheduled_run_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _make_host(tmp_path, factor_values=("1", "2")) as host:
        monkeypatch.setattr("inferference.experiment.runner.wait_for_bindable_port", lambda _port: None)
        initial_sigint_mtime = host.sigint_file.stat().st_mtime_ns if host.sigint_file.exists() else None

        def ramp_then_cool(stop: threading.Event) -> None:
            if not _wait_for_workload_request(host):
                return
            for sensor in ("temp1_input", "temp2_input", "temp3_input"):
                _atomic_text(host.card0_device / "hwmon" / "hwmon1" / sensor, "76000\n")
            _wait_for(
                lambda: host.sigint_file.exists() and host.sigint_file.stat().st_mtime_ns != initial_sigint_mtime,
                timeout_s=5,
            )
            for sensor in ("temp1_input", "temp2_input", "temp3_input"):
                _atomic_text(host.card0_device / "hwmon" / "hwmon1" / sensor, "20000\n")
            stop.set()

        stop, thread = _watch(host, ramp_then_cool)
        try:
            results = _run(host)
        finally:
            stop.set()
            thread.join(timeout=1)

        assert [result.state for result in results] == ["bounded", "completed"], [
            (
                result.state,
                result.failure_boundary,
                json.loads((result.run_dir / "outcome.json").read_text())["detail"],
            )
            for result in results
        ]
        assert results[0].failure_boundary == "safety:arm_bound"
        assert results[1].run_dir.exists()
        assert all(not host.request_file.exists() for _ in results)
        assert all(
            json.loads((result.run_dir / "outcome.json").read_text())["state"] == state
            for result, state in zip(results, ("bounded", "completed"), strict=True)
        )


def test_arm_bound_stream_death_at_shutdown_stays_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A safety stop that kills an in-flight stream must stay `bounded`.

    Live regression, 2026-10-08: the real SIGINT closed a streaming response
    mid-body, httpx raised inside the lane thread, and the runner escalated an
    already-recorded `bounded/safety:arm_bound` outcome to
    `aborted/server:request`, which also ended the whole schedule
    (runs/arm-250w-20261008T185648Z, first scheduled run). A request failure
    caused by the runner's own stop is a consequence, not a root cause. The
    fake server's FAKE_SERVER_STREAM_DELAY_S holds request #2 open mid-body
    so the SIGINT lands under the client exactly like the live failure.
    """
    with _make_host(tmp_path, factor_values=("1", "2")) as host:
        monkeypatch.setattr("inferference.experiment.runner.wait_for_bindable_port", lambda _port: None)
        initial_sigint_mtime = host.sigint_file.stat().st_mtime_ns if host.sigint_file.exists() else None

        def ramp_then_cool(stop: threading.Event) -> None:
            if not _wait_for_workload_request(host):
                return
            for sensor in ("temp1_input", "temp2_input", "temp3_input"):
                _atomic_text(host.card0_device / "hwmon" / "hwmon1" / sensor, "76000\n")
            _wait_for(
                lambda: host.sigint_file.exists() and host.sigint_file.stat().st_mtime_ns != initial_sigint_mtime,
                timeout_s=5,
            )
            for sensor in ("temp1_input", "temp2_input", "temp3_input"):
                _atomic_text(host.card0_device / "hwmon" / "hwmon1" / sensor, "20000\n")
            stop.set()

        stop, thread = _watch(host, ramp_then_cool)
        try:
            results = _run(host, env=host.env(FAKE_SERVER_STREAM_DELAY_S="1.5"))
        finally:
            stop.set()
            thread.join(timeout=1)

        assert [result.state for result in results] == ["bounded", "completed"], [
            (
                result.state,
                result.failure_boundary,
                json.loads((result.run_dir / "outcome.json").read_text())["detail"],
            )
            for result in results
        ]
        assert results[0].failure_boundary == "safety:arm_bound"
        first_outcome = json.loads((results[0].run_dir / "outcome.json").read_text())
        assert "failed" not in str(first_outcome["detail"]).lower()
        assert results[1].run_dir.exists()


def test_table_drift_after_arm_bound_is_invalid_and_stops_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _make_host(tmp_path, factor_values=("1", "2")) as host:
        monkeypatch.setattr("inferference.experiment.runner.wait_for_bindable_port", lambda _port: None)
        table = host.paths.drm_root / "card0" / "device" / "pp_table"
        original_stop = ServerProcess.stop

        def change_table_after_stop(server: ServerProcess) -> None:
            original_stop(server)
            _atomic_write(table, table.read_bytes() + b"changed after stop\n")

        monkeypatch.setattr("inferference.experiment.runner.ServerProcess.stop", change_table_after_stop)

        def ramp_to_arm_bound(stop: threading.Event) -> None:
            if not _wait_for_workload_request(host):
                return
            for sensor in ("temp1_input", "temp2_input", "temp3_input"):
                _atomic_text(host.card0_device / "hwmon" / "hwmon1" / sensor, "76000\n")
            _wait_for(lambda: host.sigint_file.exists(), timeout_s=5)
            stop.set()

        watcher_stop, watcher = _watch(host, ramp_to_arm_bound)
        try:
            results = _run(host)
        finally:
            watcher_stop.set()
            watcher.join(timeout=1)

        assert len(results) == 1
        assert results[0].state == "invalid"
        assert results[0].failure_boundary == "drift:pp_table"
        outcome = json.loads((results[0].run_dir / "outcome.json").read_text())
        assert outcome["state"] == "invalid"
        assert outcome["failure_boundary"] == "drift:pp_table"
        assert not host.request_file.exists()


def test_machine_limit_during_server_shutdown_overrides_arm_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _make_host(tmp_path, factor_values=("1", "2")) as host:
        monkeypatch.setattr("inferference.experiment.runner.wait_for_bindable_port", lambda _port: None)
        original_stop = ServerProcess.stop

        def cross_machine_limit(server: ServerProcess) -> None:
            for sensor in ("temp1_input", "temp2_input", "temp3_input"):
                _atomic_text(host.card0_device / "hwmon" / "hwmon1" / sensor, "81000\n")
            time.sleep(0.08)
            original_stop(server)

        monkeypatch.setattr("inferference.experiment.runner.ServerProcess.stop", cross_machine_limit)

        def ramp_to_arm_bound(stop: threading.Event) -> None:
            if _wait_for_workload_request(host):
                for sensor in ("temp1_input", "temp2_input", "temp3_input"):
                    _atomic_text(host.card0_device / "hwmon" / "hwmon1" / sensor, "76000\n")
            stop.wait()

        watcher_stop, watcher = _watch(host, ramp_to_arm_bound)
        try:
            results = _run(host)
        finally:
            watcher_stop.set()
            watcher.join(timeout=1)

        assert len(results) == 1
        assert results[0].state == "aborted"
        assert results[0].failure_boundary == "safety:machine_limit"
        outcome = json.loads((results[0].run_dir / "outcome.json").read_text())
        assert outcome["state"] == "aborted"
        assert outcome["failure_boundary"] == "safety:machine_limit"
        assert not host.request_file.exists()


def test_machine_limit_ramp_aborts_and_stops_schedule(tmp_path: Path) -> None:
    with _make_host(tmp_path, factor_values=("1", "2")) as host:

        def ramp(stop: threading.Event) -> None:
            if _wait_for_workload_request(host):
                for sensor in ("temp1_input", "temp2_input", "temp3_input"):
                    _atomic_text(host.card0_device / "hwmon" / "hwmon1" / sensor, "81000\n")

        stop, thread = _watch(host, ramp)
        try:
            results = _run(host)
        finally:
            stop.set()
            thread.join(timeout=1)
        assert len(results) == 1
        assert results[0].state == "aborted"
        assert results[0].failure_boundary == "safety:machine_limit"
        assert not host.request_file.exists()
        assert host.sigint_file.read_text() == "1"


def test_pp_table_change_during_run_is_invalid(tmp_path: Path) -> None:
    with _make_host(tmp_path) as host:
        table = host.paths.drm_root / "card0" / "device" / "pp_table"

        def drift(_stop: threading.Event) -> None:
            if _wait_for_workload_request(host):
                _atomic_write(table, table.read_bytes() + b"changed\n")

        stop, thread = _watch(host, drift)
        try:
            result = _run(host)[0]
        finally:
            stop.set()
            thread.join(timeout=1)
        assert result.state == "invalid"
        assert result.failure_boundary == "drift:pp_table"
        assert not host.request_file.exists()


def test_power_cap_run_records_the_cap_in_the_control_artifact(tmp_path: Path) -> None:
    with _make_host(tmp_path, power_cap_w=220) as host:
        result = _run(host)[0]

        assert result.state == "completed"
        run_dir = next(host.artifact_root.iterdir())
        control = json.loads((run_dir / "control.json").read_text())["pp_table"]
        assert control["control_mode"] == "power_cap"
        assert control["declared_power_cap_w"] == 220
        assert control["live_power1_cap_uw"] == 220_000_000
        assert control["live_power1_cap_min_uw"] == 120_000_000
        assert control["final_power1_cap_uw"] == 220_000_000
        identity = json.loads((run_dir / "identity.json").read_text())["pp_table"]
        assert identity["control_mode"] == "power_cap"
        assert identity["live_power1_cap_uw"] == 220_000_000


def test_power_cap_reset_to_stock_during_run_is_invalid(tmp_path: Path) -> None:
    with _make_host(tmp_path, power_cap_w=220) as host:
        cap_file = host.card0_device / "hwmon" / "hwmon1" / "power1_cap"

        def reset(_stop: threading.Event) -> None:
            if _wait_for_workload_request(host):
                _atomic_text(cap_file, "250000000\n")

        stop, thread = _watch(host, reset)
        try:
            result = _run(host)[0]
        finally:
            stop.set()
            thread.join(timeout=1)
        assert result.state == "invalid"
        assert result.failure_boundary == "drift:power_cap"
        run_dir = next(host.artifact_root.iterdir())
        control = json.loads((run_dir / "control.json").read_text())["pp_table"]
        assert control["final_power1_cap_uw"] == 250_000_000
        assert not host.request_file.exists()


def test_preflight_refusal_records_passed_checks_and_removes_no_stale_lease(tmp_path: Path) -> None:
    with _make_host(tmp_path) as host:
        _atomic_text(host.paths.loadavg, "99.0 99 99 1/1 1\n")
        result = _run(host)[0]
        assert result.state == "aborted"
        assert result.failure_boundary == "preflight:host_load"
        refusal = json.loads((result.run_dir / "preflight.json").read_text())
        assert refusal["refused"] is True
        assert refusal["check"] == "host_load"
        assert refusal["passed"]
        assert not host.request_file.exists()


def test_fan_lease_control_failure_removes_request(tmp_path: Path) -> None:
    with _make_host(tmp_path, honor_lease=False) as host:
        result = _run(host)[0]
        assert result.state == "aborted"
        assert result.failure_boundary == "control:fan_lease"
        assert not host.request_file.exists()
        assert not host.sigint_file.exists()


def test_residency_failure_stops_fake_server_once_and_removes_lease(tmp_path: Path) -> None:
    with _make_host(tmp_path) as host:
        result = _run(host, env=host.env(FAKE_SERVER_PCI="0000:04:00.0"))[0]
        assert result.state == "aborted"
        assert result.failure_boundary.startswith("residency:")
        assert host.sigint_file.read_text() == "1"
        assert not host.request_file.exists()


def test_invalid_prefill_output_stops_dispatch(tmp_path: Path) -> None:
    with _make_host(tmp_path, workloads=("prefill_50",)) as host:
        result = _run(host)[0]
        assert result.state == "aborted"
        assert result.failure_boundary.startswith("output:")
        stages = json.loads((result.run_dir / "stages.json").read_text())
        assert stages["stages"][-1]["requests"][0]["valid"] is False
        assert not host.request_file.exists()


def test_telemetry_read_failure_aborts_and_releases_lease(tmp_path: Path) -> None:
    with _make_host(tmp_path) as host:

        def break_sensor(_stop: threading.Event) -> None:
            if _wait_for_workload_request(host):
                _atomic_text(host.card0_device / "hwmon" / "hwmon1" / "temp2_input", "not-a-temperature\n")

        stop, thread = _watch(host, break_sensor)
        try:
            result = _run(host)[0]
        finally:
            stop.set()
            thread.join(timeout=1)
        assert result.state == "aborted"
        assert result.failure_boundary == "telemetry"
        assert not host.request_file.exists()
        assert host.sigint_file.read_text() == "1"
