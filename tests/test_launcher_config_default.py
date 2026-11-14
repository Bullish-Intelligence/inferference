"""The router and server launchers follow the reconcile CLI for the config dir.

No GPU, no router, no real config. A stub python stands in for the library
venv. It answers `inferference-reconcile.py config-dir` the way the CLI does:
it prints $INFERFERENCE_CONFIG_DIR when set, else a fake system dir. It answers
`preset-ini` with the real CLI, pointed at the fake system dir. The tests
prove these things.

1. With $INFERFERENCE_CONFIG_DIR and $INFERFERENCE_PRESET_INI set, as the
   system unit sets them, the launcher passes both through and never asks the
   CLI.
2. With no env, the launcher asks the CLI, and the preset never goes inside the
   system dir. It goes to the caller's state dir.
3. With only the config dir set, the preset sits beside a writable config dir,
   as before, and never inside /etc.
4. `--check` still resolves and renders.
5. The launcher takes the default preset path from the CLI, so the launcher and
   `apply` cannot disagree (a contract test runs the real CLI).
6. A literal `~` in $INFERFERENCE_CONFIG_DIR expands to $HOME, as python does.

The fake home and the fake system dir come from the environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "ci/runner/inferference-router-launch.sh"
SERVER = ROOT / "ci/runner/inferference-server-launch.sh"
PATCH = ROOT / "ci/build/patches/vulkan-residency.patch"
TEMPLATES = (
    ROOT / "ci/runner/inferference-router.service.in",
    ROOT / "ci/runner/inferference-server@.service.in",
)

_STUB_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "-c" ]; then
    if [[ "$2" == *read_config_dir* ]]; then
        echo "127.0.0.1 8100 4"
        exit 0
    fi
    exec "$STUB_REAL_PYTHON" -I "$@"
fi
case "${1:-}" in
    *inferference-reconcile.py)
        case "${2:-}" in
            config-dir)
                echo config-dir >> "$STUB_ASK_LOG"
                [ "${STUB_CONFIG_DIR_RC:-0}" = "0" ] || exit "$STUB_CONFIG_DIR_RC"
                echo "${INFERFERENCE_CONFIG_DIR:-$STUB_SYSTEM_DIR}"
                exit 0
                ;;
            preset-ini)
                echo preset-ini >> "$STUB_ASK_LOG"
                exec "$STUB_REAL_PYTHON" "$STUB_CLI_SHIM"
                ;;
            render)
                echo "render $*" >> "$STUB_ASK_LOG"
                mkdir -p "$(dirname "$4")"
                echo "[stub]" > "$4"
                echo "rendered preset to $4"
                exit 0
                ;;
        esac
        ;;
esac
echo "stub-python: unexpected argv: $*" >&2
exit 9
"""


# Runs the real CLI `preset-ini` with the fake system dir, and no system runtime dir.
_CLI_SHIM = """
import importlib.util
import os
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("cli_shim", os.environ["STUB_CLI"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
system = Path(os.environ["STUB_SYSTEM_DIR"])
module.SYSTEM_CONFIG_DIR = system
module.SYSTEM_PRESET_INI = system.parent / "no-run-dir" / "models-preset.ini"
sys.exit(module.main(["preset-ini"]))
"""


def _make_host(tmp_path: Path) -> dict[str, str]:
    """A fake home, a fake system dir, a fake artefact, and a clean env."""
    home = tmp_path / "home"
    home.mkdir()
    system = tmp_path / "system-config"
    (system / "models").mkdir(parents=True)
    (system / "placement.yaml").write_text("router: {}\n")
    server = tmp_path / "server"
    (server / "bin").mkdir(parents=True)
    (server / "lib").mkdir()
    (server / "bin" / "llama-server").write_text("#!/usr/bin/env bash\n")
    (server / "bin" / "llama-server").chmod(0o755)
    (server / "lib" / "libggml-vulkan.so").write_text("")
    (server / "build-manifest.json").write_text(
        json.dumps(
            {
                "build_id": "b-test",
                "vulkan_residency_patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
            }
        )
    )
    icd = tmp_path / "radeon_icd.json"
    icd.write_text("{}")
    stub = tmp_path / "stub-python"
    stub.write_text(_STUB_PYTHON)
    stub.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    shim = tmp_path / "cli-shim.py"
    shim.write_text(_CLI_SHIM)

    env = {k: v for k, v in os.environ.items() if not k.startswith(("INFERFERENCE_", "XDG_"))}
    env.update(
        {
            "HOME": str(home),
            "INFERFERENCE_PYTHON": str(stub),
            "INFERFERENCE_SERVER_DIR": str(server),
            "INFERFERENCE_VULKAN_ICD": str(icd),
            "STUB_REAL_PYTHON": sys.executable,
            "STUB_SYSTEM_DIR": str(system),
            "STUB_ASK_LOG": str(logs / "ask.log"),
            "STUB_CLI": str(ROOT / "ci/runner/inferference-reconcile.py"),
            "STUB_CLI_SHIM": str(shim),
        }
    )
    return env


def _run(script: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=60)


def _asks(env: dict[str, str]) -> list[str]:
    path = Path(env["STUB_ASK_LOG"])
    return path.read_text().splitlines() if path.exists() else []


def _printed(proc: subprocess.CompletedProcess[str]) -> dict[str, str]:
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


def _state_preset(env: dict[str, str]) -> str:
    return str(Path(env["HOME"]) / ".local/state/inferference/models-preset.ini")


# ----------------------------------------------------------------- the unit --
def test_router_with_unit_env_passes_both_through_and_asks_nothing(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    preset = tmp_path / "run" / "models-preset.ini"
    env["INFERFERENCE_CONFIG_DIR"] = env["STUB_SYSTEM_DIR"]
    env["INFERFERENCE_PRESET_INI"] = str(preset)
    proc = _run(ROUTER, env, "--check")
    assert proc.returncode == 0, proc.stderr
    assert "--check: resolved, not started." in proc.stdout
    assert f"preset          {preset}" in proc.stdout
    assert preset.read_text() == "[stub]\n"
    assert not any(a == "config-dir" for a in _asks(env))
    assert _asks(env) == [f"render {ROOT}/ci/runner/inferference-reconcile.py render --out {preset}"]
    assert not (Path(env["HOME"]) / ".local").exists()


def test_router_unit_env_print_config(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["INFERFERENCE_CONFIG_DIR"] = "/etc/inferference"
    env["INFERFERENCE_PRESET_INI"] = "/run/inferference/models-preset.ini"
    got = _printed(_run(ROUTER, env, "--print-config"))
    assert got == {"config_dir": "/etc/inferference", "preset_ini": "/run/inferference/models-preset.ini"}
    assert _asks(env) == []


# ----------------------------------------------------------------- no env ----
def test_router_no_env_asks_the_cli_and_uses_the_system_dir(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    got = _printed(_run(ROUTER, env, "--print-config"))
    assert got["config_dir"] == env["STUB_SYSTEM_DIR"]
    assert _asks(env) == ["config-dir", "preset-ini"]


def test_router_no_env_never_writes_the_preset_inside_the_system_dir(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    system = Path(env["STUB_SYSTEM_DIR"])
    system.chmod(0o555)  # like /etc: the caller cannot write it
    try:
        if os.access(system, os.W_OK):
            pytest.skip("the caller can write a 0555 dir (root): the read-only case cannot be set up")
        proc = _run(ROUTER, env, "--check")
    finally:
        system.chmod(0o755)
    assert proc.returncode == 0, proc.stderr
    assert f"preset          {_state_preset(env)}" in proc.stdout
    assert Path(_state_preset(env)).read_text() == "[stub]\n"
    assert not (system / "models-preset.ini").exists()
    assert "goes to the caller's state dir" in proc.stderr


def test_router_no_env_with_a_system_dir_inside_etc_uses_the_state_dir(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["STUB_SYSTEM_DIR"] = "/etc/inferference"  # print-config reads nothing
    got = _printed(_run(ROUTER, env, "--print-config"))
    assert got == {"config_dir": "/etc/inferference", "preset_ini": _state_preset(env)}


def test_router_state_dir_follows_xdg_state_home(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["STUB_SYSTEM_DIR"] = "/etc/inferference"
    env["XDG_STATE_HOME"] = str(tmp_path / "xdg")
    got = _printed(_run(ROUTER, env, "--print-config"))
    assert got["preset_ini"] == str(tmp_path / "xdg/inferference/models-preset.ini")


def test_router_home_fallback_dir_that_the_caller_can_write_keeps_the_preset_beside_it(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    home_cfg = Path(env["HOME"]) / ".config" / "inferference"
    home_cfg.mkdir(parents=True)
    env["STUB_SYSTEM_DIR"] = str(home_cfg)  # the CLI would answer the home dir
    got = _printed(_run(ROUTER, env, "--print-config"))
    assert got == {"config_dir": str(home_cfg), "preset_ini": str(home_cfg / "models-preset.ini")}


def test_router_stops_when_the_cli_cannot_name_the_dir(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["STUB_CONFIG_DIR_RC"] = "78"
    proc = _run(ROUTER, env, "--check")
    assert proc.returncode == 78
    assert "could not name the config dir" in proc.stderr


def test_router_stops_when_no_python_can_ask(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["INFERFERENCE_PYTHON"] = str(tmp_path / "missing-python")
    proc = _run(ROUTER, env, "--check")
    assert proc.returncode == 78
    assert "to ask the reconcile CLI for the config dir" in proc.stderr


# ------------------------------------------------------- config dir only ----
def test_router_config_env_only_keeps_the_preset_beside_a_writable_dir(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    cfg = tmp_path / "my-config"
    (cfg / "models").mkdir(parents=True)
    (cfg / "placement.yaml").write_text("router: {}\n")
    env["INFERFERENCE_CONFIG_DIR"] = str(cfg)
    proc = _run(ROUTER, env, "--check")
    assert proc.returncode == 0, proc.stderr
    assert (cfg / "models-preset.ini").read_text() == "[stub]\n"
    assert [a for a in _asks(env) if a == "config-dir"] == []  # the env dir passes through
    assert "preset-ini" in _asks(env)  # the CLI names the preset path


def test_router_config_env_inside_etc_never_gets_a_preset_there(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["INFERFERENCE_CONFIG_DIR"] = "/etc/inferference"
    got = _printed(_run(ROUTER, env, "--print-config"))
    assert got["preset_ini"] == _state_preset(env)


def test_router_check_still_refuses_a_missing_placement(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    env["INFERFERENCE_CONFIG_DIR"] = str(empty)
    proc = _run(ROUTER, env, "--check")
    assert proc.returncode == 78
    assert "no placement at" in proc.stderr


# ----------------------------------------------------------------- server ----
def test_server_with_env_passes_it_through(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["INFERFERENCE_CONFIG_DIR"] = "/etc/inferference"
    assert _printed(_run(SERVER, env, "web", "--print-config")) == {"config_dir": "/etc/inferference"}
    assert _asks(env) == []


def test_server_with_no_env_asks_the_cli(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    assert _printed(_run(SERVER, env, "web", "--print-config")) == {"config_dir": env["STUB_SYSTEM_DIR"]}
    assert _asks(env) == ["config-dir"]


def test_server_stops_when_the_cli_cannot_name_the_dir(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["STUB_CONFIG_DIR_RC"] = "78"
    proc = _run(SERVER, env, "web", "--print-config")
    assert proc.returncode == 78
    assert "could not name the config dir" in proc.stderr


# ------------------------------------------------------------- the files ----
@pytest.mark.parametrize("script", [ROUTER, SERVER], ids=["router", "server"])
def test_launchers_name_no_home_default(script: Path) -> None:
    code = [ln for ln in script.read_text().splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "HOME}/.config/inferference" in ln or "~/.config/inferference" in ln]


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_unit_templates_set_no_home_config_dir(template: Path) -> None:
    live = [ln for ln in template.read_text().splitlines() if ln.startswith("Environment=INFERFERENCE_CONFIG_DIR")]
    assert live == []


# ------------------------------------------------------- one source: the CLI ----
def _cli_module():  # noqa: ANN202
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cli_for_launcher_contract", ROOT / "ci/runner/inferference-reconcile.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case", ["writable-dir", "etc-dir", "unwritable-dir", "unset", "xdg-state"])
def test_launcher_and_apply_use_one_default_preset_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """The launcher's `--print-config` equals the `ini_path()` that `apply` reads.

    The real CLI answers in both. Before, the launcher had its own copy of the
    rule, and a default launch wrote a path that `apply` did not read.
    """
    env = _make_host(tmp_path)
    env["INFERFERENCE_PYTHON"] = sys.executable  # the real CLI, no stub
    home = Path(env["HOME"])
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    if case == "writable-dir":
        env["INFERFERENCE_CONFIG_DIR"] = str(cfg)
    elif case == "etc-dir":
        env["INFERFERENCE_CONFIG_DIR"] = "/etc/inferference"
    elif case == "unwritable-dir":
        cfg.chmod(0o555)
        env["INFERFERENCE_CONFIG_DIR"] = str(cfg)
        if os.access(cfg, os.W_OK):
            pytest.skip("the caller can write a 0555 dir (root)")
    elif case == "xdg-state":
        env["INFERFERENCE_CONFIG_DIR"] = "/etc/inferference"
        env["XDG_STATE_HOME"] = str(tmp_path / "xdg")
    for key in ("INFERFERENCE_CONFIG_DIR", "INFERFERENCE_PRESET_INI", "XDG_STATE_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(home))
    for key in ("INFERFERENCE_CONFIG_DIR", "XDG_STATE_HOME"):
        if key in env:
            monkeypatch.setenv(key, env[key])
    cli = _cli_module()
    try:
        got = _printed(_run(ROUTER, env, "--print-config"))
        expected = (str(cli.ini_path()), str(cli.config_dir()))
    finally:
        cfg.chmod(0o755)
    assert (got["preset_ini"], got["config_dir"]) == expected


# --------------------------------------------------------------------- tilde ----
def test_router_expands_a_literal_tilde_in_the_config_dir(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    home = Path(env["HOME"])
    (home / "cfg" / "models").mkdir(parents=True)
    (home / "cfg" / "placement.yaml").write_text("router: {}\n")
    env["INFERFERENCE_CONFIG_DIR"] = "~/cfg"
    got = _printed(_run(ROUTER, env, "--print-config"))
    assert got == {"config_dir": str(home / "cfg"), "preset_ini": str(home / "cfg" / "models-preset.ini")}
    # The check reads placement.yaml from the expanded dir, not from a literal `~` dir.
    proc = _run(ROUTER, env, "--check")
    assert proc.returncode == 0, proc.stderr
    assert (home / "cfg" / "models-preset.ini").read_text() == "[stub]\n"
    assert not (Path.cwd() / "~").exists()


def test_router_expands_a_bare_tilde_and_keeps_other_tildes(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["INFERFERENCE_PRESET_INI"] = str(tmp_path / "p.ini")
    env["INFERFERENCE_CONFIG_DIR"] = "~"
    assert _printed(_run(ROUTER, env, "--print-config"))["config_dir"] == env["HOME"]
    env["INFERFERENCE_CONFIG_DIR"] = "/data/~x/cfg"  # a tilde inside the path is literal
    assert _printed(_run(ROUTER, env, "--print-config"))["config_dir"] == "/data/~x/cfg"


def test_server_expands_a_literal_tilde_in_the_config_dir(tmp_path: Path) -> None:
    env = _make_host(tmp_path)
    env["INFERFERENCE_CONFIG_DIR"] = "~/cfg"
    assert _printed(_run(SERVER, env, "web", "--print-config")) == {"config_dir": f"{env['HOME']}/cfg"}
