"""Fake-router tests for ``ci/runner/evict-and-run-eval.sh``.

The script chains two steps -- evict a card from the router via
``ci/runner/inferference-reconcile.py evict``, wait for the card to drain
below the runner's residency floor, then exec the experiment run CLI. These
tests replace every child process with a stub python (the same PATH-fake
pattern as ``tests/test_gpu_thermal_guard.py``) and fabricate the DRM sysfs
tree, so nothing here touches real hardware, the real router, or the real
server binary.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ci/runner/evict-and-run-eval.sh"

_STUB_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "-c" ]; then
    if [[ "$2" == *read_config_dir* ]]; then
        echo "Vulkan0"
        exit 0
    fi
    if [[ "$2" == *socket* ]]; then
        echo "31137"
        exit 0
    fi
    echo "stub-python: unmatched -c payload" >&2
    exit 9
fi
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "inferference.experiment" ]; then
    printf '%s\n' "$@" >> "$STUB_RUN_LOG"
    exit "${STUB_RUN_RC:-0}"
fi
case "$1" in
    *inferference-reconcile.py*)
        printf '%s\n' "$@" >> "$STUB_EVICT_LOG"
        echo ">> evicted (stub) from Vulkan0"
        exit "${STUB_EVICT_RC:-0}"
        ;;
esac
echo "stub-python: unexpected argv: $*" >&2
exit 9
"""

_MIB = 1024 * 1024


def _make_host(tmp_path: Path, *, vram_mib: int = 21_743) -> dict[str, str]:
    """Fabricate a DRM tree, config dir, server binary, and a stub python.

    Every card's ``device`` is a symlink into a directory named after its PCI
    address, so ``readlink -f`` -- exactly what the script does -- resolves it.
    """
    sysfs = tmp_path / "sysfs"
    sysfs.mkdir(parents=True)
    for card_index, pci in ((0, "0000:04:00.0"), (1, "0000:03:00.0")):
        pci_dir = tmp_path / "pci" / pci
        pci_dir.mkdir(parents=True)
        (pci_dir / "mem_info_vram_used").write_text(f"{vram_mib * _MIB}\n")
        (sysfs / f"card{card_index}").mkdir()
        (sysfs / f"card{card_index}" / "device").symlink_to(pci_dir, target_is_directory=True)

    config_root = tmp_path / "config"
    config_root.mkdir()
    spec = tmp_path / "arm-stub.yaml"
    spec.write_text("experiment: stub\nschema_version: 1\n")
    stock_blob = tmp_path / "stock.pp_table"
    stock_blob.write_bytes(b"stock-blob-bytes\n")
    server_bin = tmp_path / "server-bin"
    server_bin.write_text("#!/usr/bin/env bash\necho fake-server\n")
    server_bin.chmod(0o755)

    stub = tmp_path / "stub-python"
    stub.write_text(_STUB_PYTHON)
    stub.chmod(0o755)
    runs_dir = tmp_path / "runs"
    logs = tmp_path / "logs"
    logs.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(stub),
            "EVICT_SYSFS_ROOT": str(sysfs),
            "EVICT_CONFIG_ROOT": str(config_root),
            "EVICT_SERVER_BIN": str(server_bin),
            "EVICT_STOCK_PP_TABLE": str(stock_blob),
            "EVICT_RUNS_DIR": str(runs_dir),
            "STUB_RUN_LOG": str(logs / "run.log"),
            "STUB_EVICT_LOG": str(logs / "evict.log"),
            "STUB_EVICT_RC": "0",
            "STUB_RUN_RC": "0",
            "EVICT_DRAIN_POLL_S": "0.05",
        }
    )
    return env


def _run_script(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _drain_card_later(env: dict[str, str], delay_s: float = 0.3) -> None:
    """Rewrite card0's VRAM file below the floor after a short delay."""
    vram_file = Path(env["EVICT_SYSFS_ROOT"]) / "card0" / "device" / "mem_info_vram_used"

    def _drain() -> None:
        time.sleep(delay_s)
        temporary = vram_file.with_suffix(".tmp")
        temporary.write_text(f"{16 * _MIB}\n")
        os.replace(temporary, vram_file)

    threading.Thread(target=_drain).start()


def _run_argv(env: dict[str, str]) -> list[str]:
    log = Path(env["STUB_RUN_LOG"])
    if not log.exists():
        return []
    return log.read_text().splitlines()


def test_evicts_then_runs_with_the_documented_flags(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    _drain_card_later(env)
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec))

    assert result.returncode == 0, result.stderr
    evict_log = Path(env["STUB_EVICT_LOG"]).read_text().splitlines()
    assert evict_log[-2:] == ["evict", "Vulkan0"]
    assert ">> evicting card 0 (placement device: Vulkan0)" in result.stdout
    assert ">> card 0 drained: 16 MiB in use" in result.stdout

    argv = _run_argv(env)
    assert argv[:4] == ["-m", "inferference.experiment", "run", str(spec)]
    for flag, value in (
        ("--card", "0"),
        ("--port", "31137"),
        ("--server-bin", env["EVICT_SERVER_BIN"]),
        ("--config-root", env["EVICT_CONFIG_ROOT"]),
        ("--stock-pp-table", env["EVICT_STOCK_PP_TABLE"]),
        ("--expect-pci", "0000:04:00.0"),
    ):
        index = argv.index(flag)
        assert argv[index + 1] == value

    artifact_index = argv.index("--artifact-root") + 1
    artifact_root = Path(argv[artifact_index])
    expected = re.escape(Path(env["EVICT_RUNS_DIR"]).as_posix()) + r"/arm-stub-\d{8}T\d{6}Z$"
    assert re.fullmatch(expected, artifact_root.as_posix()), artifact_root
    assert not artifact_root.exists()  # the run CLI owns creation; write-once is enforced by the fresh name


def test_aborts_without_running_when_the_card_never_drains(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["EVICT_DRAIN_TIMEOUT_S"] = "1"
    spec = tmp_path / "arm-stub.yaml"
    (Path(env["EVICT_SYSFS_ROOT"]) / "card0" / "device" / "mem_info_vram_used").write_text(f"{21_743 * _MIB}\n")

    result = _run_script(env, str(spec))

    assert result.returncode != 0
    assert "did not drain below 1024 MiB" in result.stderr
    assert "the run is not started" in result.stderr
    assert _run_argv(env) == []


def test_aborts_when_eviction_fails_and_never_polls(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["STUB_EVICT_RC"] = "3"
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec))

    assert result.returncode != 0
    assert "eviction failed" in result.stderr
    assert _run_argv(env) == []


def test_honors_a_fixed_port_and_forwards_trailing_args(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["EVICT_PORT"] = "45678"
    _drain_card_later(env)
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec), "--", "--extra-key", "extra-value")

    assert result.returncode == 0, result.stderr
    argv = _run_argv(env)
    assert argv[argv.index("--port") + 1] == "45678"
    assert argv[-2:] == ["--extra-key", "extra-value"]


def test_card_one_maps_to_its_own_pci_address(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    vram = Path(env["EVICT_SYSFS_ROOT"]) / "card1" / "device" / "mem_info_vram_used"
    temporary = vram.with_suffix(".tmp")
    temporary.write_text(f"{16 * _MIB}\n")
    os.replace(temporary, vram)
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec), "--card", "1")

    assert result.returncode == 0, result.stderr
    argv = _run_argv(env)
    assert argv[argv.index("--card") + 1] == "1"
    assert argv[argv.index("--expect-pci") + 1] == "0000:03:00.0"


def test_refuses_a_missing_spec_and_a_dead_server_binary(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    result = _run_script(env, str(tmp_path / "not-a-spec.yaml"))

    assert result.returncode != 0
    assert "spec not found" in result.stderr

    env = _make_host(tmp_path / "second")
    env["EVICT_SERVER_BIN"] = str(tmp_path / "second" / "server-bin")
    (tmp_path / "second" / "server-bin").unlink()
    result = _run_script(env, str(tmp_path / "second" / "arm-stub.yaml"))
    assert result.returncode != 0
    assert "server binary not executable" in result.stderr


def test_no_evict_skips_the_router_but_still_gates_on_the_drain(tmp_path: Path) -> None:
    env = _make_host(tmp_path, vram_mib=16)
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec), "--no-evict")

    assert result.returncode == 0, result.stderr
    assert not Path(env["STUB_EVICT_LOG"]).exists()
    assert ">> --no-evict: skipping the router evict" in result.stdout
    assert ">> card 0 drained: 16 MiB in use" in result.stdout
    assert _run_argv(env)[:4] == ["-m", "inferference.experiment", "run", str(spec)]


def test_no_evict_still_refuses_a_card_that_holds_a_model(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["EVICT_DRAIN_TIMEOUT_S"] = "0"
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec), "--no-evict")

    assert result.returncode == 1
    assert "did not drain below 1024 MiB" in result.stderr
    assert not Path(env["STUB_EVICT_LOG"]).exists()
    assert _run_argv(env) == []


def test_artifact_root_carries_the_experiment_number(tmp_path: Path) -> None:
    env = _make_host(tmp_path, vram_mib=16)
    spec_dir = tmp_path / "101-power-cap"
    spec_dir.mkdir()
    spec = spec_dir / "arm-160w.yaml"
    spec.write_text("experiment: stub\nschema_version: 1\n")

    result = _run_script(env, str(spec), "--no-evict")

    assert result.returncode == 0, result.stderr
    argv = _run_argv(env)
    artifact_root = Path(argv[argv.index("--artifact-root") + 1])
    assert re.fullmatch(r"101-arm-160w-\d{8}T\d{6}Z", artifact_root.name), artifact_root
