"""The evict launchers follow the reconcile CLI for the default config dir.

No GPU, no router. A stub python stands in for every child. The stub answers
`inferference-reconcile.py config-dir` the way the CLI does: it prints
$INFERFERENCE_CONFIG_DIR when set, else a fake system dir. The tests prove the
launchers set no config variable of their own, and never name a home dir.

The CLI side is tested against the real module with a fake system dir and a
fake home dir.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECONCILE = ROOT / "ci/runner/inferference-reconcile.py"
EVAL_SCRIPT = ROOT / "ci/runner/evict-and-run-eval.sh"
PAIR_SCRIPT = ROOT / "ci/runner/evict-and-run-pair.sh"

SPEC = importlib.util.spec_from_file_location("inferference_reconcile_cli_evict_default", RECONCILE)
assert SPEC and SPEC.loader
reconciler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reconciler)

_MIB = 1024 * 1024

_STUB_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "-c" ]; then
    if [[ "$2" == *read_config_dir* ]]; then
        echo "${EVICT_RESOLVED_CONFIG_DIR}" >> "$STUB_LOOKUP_LOG"
        echo "Vulkan${EVICT_CARD:-0}"
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
    env | grep '^INFERFERENCE_' >> "$STUB_RUN_ENV_LOG" || true
    exit 0
fi
case "$1" in
    *inferference-reconcile.py*)
        if [ "${2:-}" = "config-dir" ]; then
            echo config-dir >> "$STUB_CONFIG_DIR_LOG"
            [ "${STUB_CONFIG_DIR_RC:-0}" = "0" ] || exit "$STUB_CONFIG_DIR_RC"
            echo "${INFERFERENCE_CONFIG_DIR:-$STUB_SYSTEM_DIR}"
            exit 0
        fi
        printf '%s\n' "$@" >> "$STUB_EVICT_LOG"
        env | grep '^INFERFERENCE_' >> "$STUB_EVICT_ENV_LOG" || true
        exit 0
        ;;
esac
echo "stub-python: unexpected argv: $*" >&2
exit 9
"""


def _make_host(tmp_path: Path) -> dict[str, str]:
    sysfs = tmp_path / "sysfs"
    sysfs.mkdir()
    for card_index, pci in ((0, "0000:04:00.0"), (1, "0000:03:00.0")):
        pci_dir = tmp_path / "pci" / pci
        pci_dir.mkdir(parents=True)
        (pci_dir / "mem_info_vram_used").write_text(f"{16 * _MIB}\n")
        (sysfs / f"card{card_index}").mkdir()
        (sysfs / f"card{card_index}" / "device").symlink_to(pci_dir, target_is_directory=True)

    system_dir = tmp_path / "etc-inferference"
    system_dir.mkdir()
    fake_home = tmp_path / "home"
    (fake_home / ".config" / "inferference-dual").mkdir(parents=True)
    (fake_home / ".config" / "inferference").mkdir(parents=True)
    stock = tmp_path / "stock.pp_table"
    stock.write_bytes(b"stock\n")
    server_bin = tmp_path / "server-bin"
    server_bin.write_text("#!/usr/bin/env bash\n")
    server_bin.chmod(0o755)
    spec = tmp_path / "arm.yaml"
    spec.write_text("experiment: stub\n")
    stub = tmp_path / "stub-python"
    stub.write_text(_STUB_PYTHON)
    stub.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()

    env = {k: v for k, v in os.environ.items() if not k.startswith(("INFERFERENCE_", "EVICT_"))}
    env.update(
        {
            "HOME": str(fake_home),
            "PYTHON_BIN": str(stub),
            "EVICT_SYSFS_ROOT": str(sysfs),
            "EVICT_SERVER_BIN": str(server_bin),
            "EVICT_STOCK_PP_TABLE": str(stock),
            "EVICT_RUNS_DIR": str(tmp_path / "runs"),
            "EVICT_PORT": "31137",
            "EVICT_PORTS": "31137,31138",
            "EVICT_DRAIN_POLL_S": "0.05",
            "STUB_SYSTEM_DIR": str(system_dir),
            "STUB_LOOKUP_LOG": str(logs / "lookup.log"),
            "STUB_CONFIG_DIR_LOG": str(logs / "config-dir.log"),
            "STUB_EVICT_LOG": str(logs / "evict.log"),
            "STUB_EVICT_ENV_LOG": str(logs / "evict-env.log"),
            "STUB_RUN_LOG": str(logs / "run.log"),
            "STUB_RUN_ENV_LOG": str(logs / "run-env.log"),
            "SPEC": str(spec),
        }
    )
    return env


def _run(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), env["SPEC"]],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _lines(env: dict[str, str], key: str) -> list[str]:
    path = Path(env[key])
    return path.read_text().splitlines() if path.exists() else []


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


SCRIPTS = pytest.mark.parametrize("script", [EVAL_SCRIPT, PAIR_SCRIPT], ids=["eval", "pair"])


@SCRIPTS
def test_no_env_asks_the_reconcile_cli_and_sets_no_config_variable(tmp_path: Path, script: Path) -> None:
    env = _make_host(tmp_path)
    result = _run(script, env)
    assert result.returncode == 0, result.stderr
    system_dir = env["STUB_SYSTEM_DIR"]
    assert _lines(env, "STUB_CONFIG_DIR_LOG") == ["config-dir"]
    assert set(_lines(env, "STUB_LOOKUP_LOG")) == {system_dir}
    assert _lines(env, "STUB_EVICT_ENV_LOG") == []  # evict saw no INFERFERENCE_* variable
    assert _lines(env, "STUB_RUN_ENV_LOG") == []
    assert _flag(_lines(env, "STUB_RUN_LOG"), "--config-root") == system_dir


@SCRIPTS
def test_the_home_dirs_are_never_a_script_default(tmp_path: Path, script: Path) -> None:
    env = _make_host(tmp_path)
    result = _run(script, env)
    assert result.returncode == 0, result.stderr
    home_dirs = [str(Path(env["HOME"]) / ".config" / "inferference"), "inferference-dual"]
    for line in _lines(env, "STUB_LOOKUP_LOG") + _lines(env, "STUB_RUN_LOG"):
        assert not any(h in line for h in home_dirs)
    text = script.read_text()
    assert "$HOME/.config" not in text
    assert "~/.config/inferference-dual" not in text


@SCRIPTS
def test_env_config_dir_passes_through_unchanged(tmp_path: Path, script: Path) -> None:
    env = _make_host(tmp_path)
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    env["INFERFERENCE_CONFIG_DIR"] = str(explicit)
    result = _run(script, env)
    assert result.returncode == 0, result.stderr
    assert set(_lines(env, "STUB_LOOKUP_LOG")) == {str(explicit)}
    assert set(_lines(env, "STUB_EVICT_ENV_LOG")) == {f"INFERFERENCE_CONFIG_DIR={explicit}"}
    assert _flag(_lines(env, "STUB_RUN_LOG"), "--config-root") == str(explicit)


@SCRIPTS
def test_evict_config_root_wins_and_skips_the_cli_lookup(tmp_path: Path, script: Path) -> None:
    env = _make_host(tmp_path)
    forced = tmp_path / "forced"
    forced.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    env["EVICT_CONFIG_ROOT"] = str(forced)
    env["INFERFERENCE_CONFIG_DIR"] = str(other)
    result = _run(script, env)
    assert result.returncode == 0, result.stderr
    assert _lines(env, "STUB_CONFIG_DIR_LOG") == []
    assert set(_lines(env, "STUB_EVICT_ENV_LOG")) == {f"INFERFERENCE_CONFIG_DIR={forced}"}
    assert _flag(_lines(env, "STUB_RUN_LOG"), "--config-root") == str(forced)


@SCRIPTS
def test_the_preset_variable_is_never_set_by_the_script(tmp_path: Path, script: Path) -> None:
    env = _make_host(tmp_path)
    env["EVICT_CONFIG_ROOT"] = str(tmp_path / "etc-inferference")
    result = _run(script, env)
    assert result.returncode == 0, result.stderr
    assert not any("INFERFERENCE_PRESET_INI" in line for line in _lines(env, "STUB_EVICT_ENV_LOG"))
    assert "INFERFERENCE_PRESET_INI=" not in script.read_text().replace("never sets", "")


@SCRIPTS
def test_a_failed_cli_lookup_stops_before_any_evict(tmp_path: Path, script: Path) -> None:
    env = _make_host(tmp_path)
    env["STUB_CONFIG_DIR_RC"] = "78"
    result = _run(script, env)
    assert result.returncode == 1
    assert "ABORT" in result.stderr
    assert "config dir" in result.stderr
    assert _lines(env, "STUB_EVICT_LOG") == []
    assert _lines(env, "STUB_RUN_LOG") == []


# ------------------------------------------------------------ the CLI side --
@pytest.fixture
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERFERENCE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("INFERFERENCE_PRESET_INI", raising=False)
    monkeypatch.delenv("INFERFERENCE_ROUTER", raising=False)
    reconciler._ANNOUNCED.clear()


def _fake_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, system: bool) -> tuple[Path, Path]:
    system_dir = tmp_path / "etc-inferference"
    home_dir = tmp_path / "home-inferference"
    home_dir.mkdir()
    if system:
        system_dir.mkdir()
    monkeypatch.setattr(reconciler, "SYSTEM_CONFIG_DIR", system_dir)
    monkeypatch.setattr(reconciler, "HOME_CONFIG_DIR", str(home_dir))
    return system_dir, home_dir


@pytest.mark.usefixtures("_clean")
def test_config_dir_command_prints_the_system_dir_on_stdout_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    system_dir, _home = _fake_dirs(monkeypatch, tmp_path, system=True)
    assert reconciler.main(["config-dir"]) == 0
    out = capsys.readouterr()
    assert out.out == f"{system_dir}\n"
    assert "system default" in out.err


@pytest.mark.usefixtures("_clean")
def test_config_dir_command_follows_the_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_dirs(monkeypatch, tmp_path, system=True)
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("INFERFERENCE_CONFIG_DIR", str(explicit))
    assert reconciler.main(["config-dir"]) == 0
    assert capsys.readouterr().out == f"{explicit}\n"


@pytest.mark.usefixtures("_clean")
def test_config_dir_command_warns_on_the_home_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _system, home_dir = _fake_dirs(monkeypatch, tmp_path, system=False)
    assert reconciler.main(["config-dir"]) == 0
    out = capsys.readouterr()
    assert out.out == f"{home_dir}\n"
    assert "WARNING" in out.err
