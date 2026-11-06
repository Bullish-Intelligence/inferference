"""Fake-router tests for ``ci/runner/evict-and-run-pair.sh``.

The script chains the owner steps of ``evict-and-run-eval.sh`` for two cards:
evict each card from the router, wait for both to drain below the runner's
residency floor, then exec the ``run-pair`` command. These tests replace every
child process with a stub python (the same PATH-fake pattern as
``tests/test_evict_and_run_eval.py``) and fabricate the DRM sysfs tree, so
nothing here touches real hardware, the real router, or the real server binary.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ci/runner/evict-and-run-pair.sh"

_STUB_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "-c" ]; then
    if [[ "$2" == *read_config_dir* ]]; then
        echo "Vulkan${EVICT_CARD}"
        exit 0
    fi
    if [[ "$2" == *socket* ]]; then
        echo "31137,31138"
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
        echo ">> evicted (stub)"
        exit "${STUB_EVICT_RC:-0}"
        ;;
esac
echo "stub-python: unexpected argv: $*" >&2
exit 9
"""

_MIB = 1024 * 1024
_PCI = {0: "0000:04:00.0", 1: "0000:03:00.0"}


def _make_host(tmp_path: Path, *, vram_mib: int = 21_743) -> dict[str, str]:
    """Fabricate a DRM tree, config dir, server binary, and a stub python.

    Every card's ``device`` is a symlink into a directory named after its PCI
    address, so ``readlink -f`` -- exactly what the script does -- resolves it.
    """
    sysfs = tmp_path / "sysfs"
    sysfs.mkdir(parents=True)
    for card_index, pci in _PCI.items():
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


def _set_vram(env: dict[str, str], card: int, mib: int) -> None:
    vram_file = Path(env["EVICT_SYSFS_ROOT"]) / f"card{card}" / "device" / "mem_info_vram_used"
    temporary = vram_file.with_suffix(".tmp")
    temporary.write_text(f"{mib * _MIB}\n")
    os.replace(temporary, vram_file)


def _drain_later(env: dict[str, str], cards: tuple[int, ...], delay_s: float = 0.3) -> None:
    def _drain() -> None:
        time.sleep(delay_s)
        for card in cards:
            _set_vram(env, card, 16)

    threading.Thread(target=_drain).start()


def _run_argv(env: dict[str, str]) -> list[str]:
    log = Path(env["STUB_RUN_LOG"])
    if not log.exists():
        return []
    return log.read_text().splitlines()


def test_evicts_both_cards_then_runs_with_the_documented_flags(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    _drain_later(env, (0, 1))
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec))

    assert result.returncode == 0, result.stderr
    evict_log = Path(env["STUB_EVICT_LOG"]).read_text().splitlines()
    assert evict_log == [
        "ci/runner/inferference-reconcile.py",
        "evict",
        "Vulkan0",
        "ci/runner/inferference-reconcile.py",
        "evict",
        "Vulkan1",
    ]
    assert ">> evicting card 0 (placement device: Vulkan0)" in result.stdout
    assert ">> evicting card 1 (placement device: Vulkan1)" in result.stdout
    assert ">> card 0 drained: 16 MiB in use" in result.stdout
    assert ">> card 1 drained: 16 MiB in use" in result.stdout

    argv = _run_argv(env)
    assert argv[:4] == ["-m", "inferference.experiment", "run-pair", str(spec)]
    for flag, value in (
        ("--cards", "0,1"),
        ("--ports", "31137,31138"),
        ("--server-bin", env["EVICT_SERVER_BIN"]),
        ("--config-root", env["EVICT_CONFIG_ROOT"]),
        ("--stock-pp-table", env["EVICT_STOCK_PP_TABLE"]),
    ):
        index = argv.index(flag)
        assert argv[index + 1] == value
    pci_values = [argv[index + 1] for index, item in enumerate(argv) if item == "--expect-pci"]
    assert pci_values == ["0=0000:04:00.0", "1=0000:03:00.0"]

    artifact_root = Path(argv[argv.index("--artifact-root") + 1])
    expected = re.escape(Path(env["EVICT_RUNS_DIR"]).as_posix()) + r"/arm-stub-pair-\d{8}T\d{6}Z$"
    assert re.fullmatch(expected, artifact_root.as_posix()), artifact_root
    assert not artifact_root.exists()  # the run CLI owns creation


def test_no_evict_skips_the_router_but_still_waits_for_the_drain(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    _drain_later(env, (0, 1))
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec), "--no-evict")

    assert result.returncode == 0, result.stderr
    assert not Path(env["STUB_EVICT_LOG"]).exists()
    assert "--no-evict" in result.stdout
    assert ">> card 1 drained: 16 MiB in use" in result.stdout
    assert _run_argv(env)[2] == "run-pair"


def test_no_evict_still_refuses_a_card_that_never_drains(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["EVICT_DRAIN_TIMEOUT_S"] = "1"
    _set_vram(env, 0, 16)
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec), "--no-evict")

    assert result.returncode != 0
    assert "card 1 did not drain below 1024 MiB" in result.stderr
    assert "the run is not started" in result.stderr
    assert _run_argv(env) == []


def test_aborts_without_running_when_the_first_card_never_drains(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["EVICT_DRAIN_TIMEOUT_S"] = "1"
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec))

    assert result.returncode != 0
    assert "card 0 did not drain below 1024 MiB" in result.stderr
    assert _run_argv(env) == []


def test_aborts_when_eviction_fails_and_never_polls(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["STUB_EVICT_RC"] = "3"
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec))

    assert result.returncode != 0
    assert "eviction of card 0 failed" in result.stderr
    assert "waiting for cards" not in result.stdout
    assert _run_argv(env) == []


def test_honors_fixed_ports_and_forwards_trailing_args(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["EVICT_PORTS"] = "45678,45679"
    _drain_later(env, (0, 1))
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec), "--", "--extra-key", "extra-value")

    assert result.returncode == 0, result.stderr
    argv = _run_argv(env)
    assert argv[argv.index("--ports") + 1] == "45678,45679"
    assert argv[-2:] == ["--extra-key", "extra-value"]


def test_cards_option_selects_the_pair_and_its_pci_addresses(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    _set_vram(env, 0, 16)
    _set_vram(env, 1, 16)
    spec = tmp_path / "arm-stub.yaml"

    result = _run_script(env, str(spec), "--cards", "1,0", "--no-evict")

    assert result.returncode == 0, result.stderr
    argv = _run_argv(env)
    assert argv[argv.index("--cards") + 1] == "1,0"
    pci_values = [argv[index + 1] for index, item in enumerate(argv) if item == "--expect-pci"]
    assert pci_values == ["1=0000:03:00.0", "0=0000:04:00.0"]


def test_refuses_a_bad_cards_value(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    spec = tmp_path / "arm-stub.yaml"

    for bad in ("0", "0,0", "a,b", "0,1,2"):
        result = _run_script(env, str(spec), "--cards", bad)
        assert result.returncode != 0, bad
        assert "--cards must" in result.stderr


def test_refuses_a_missing_spec_a_dead_server_binary_and_a_missing_card(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    result = _run_script(env, str(tmp_path / "not-a-spec.yaml"))
    assert result.returncode != 0
    assert "spec not found" in result.stderr

    env = _make_host(tmp_path / "second")
    (tmp_path / "second" / "server-bin").unlink()
    result = _run_script(env, str(tmp_path / "second" / "arm-stub.yaml"))
    assert result.returncode != 0
    assert "server binary not executable" in result.stderr

    env = _make_host(tmp_path / "third")
    result = _run_script(env, str(tmp_path / "third" / "arm-stub.yaml"), "--cards", "0,5")
    assert result.returncode != 0
    assert "is DRM card 5 present" in result.stderr
