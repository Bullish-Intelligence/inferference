"""The serving services run from the Nix package, not from a checkout.

No GPU, no router. The tests prove these things.

1. `inferference-reconcile` and `inferference-balancer` are console scripts, and
   the reconcile logic lives in the library (`inferference.reconcile_cli`).
2. The launchers copied to a bare tree (no checkout, no `src/`, no
   `ci/build/patches/`) still work: the residency patch comes from
   $INFERFERENCE_RESIDENCY_PATCH or from beside the script.
3. No launcher puts a checkout `src/` on the Python path.
4. Without $INFERFERENCE_PYTHON, a launcher uses the checkout venv only when it
   exists. Else it fails by naming the variable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tests.test_launcher_config_default import _make_host

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ci/runner"
ROUTER = RUNNER / "inferference-router-launch.sh"
BALANCER = RUNNER / "inferference-balancer-launch.sh"
RELOAD = RUNNER / "inferference-router-reload.sh"
SHIM = RUNNER / "inferference-reconcile.py"
PATCH = ROOT / "ci/build/patches/vulkan-residency.patch"
LAUNCHERS = (ROUTER, BALANCER, RELOAD)


def _code(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#")]


# ------------------------------------------------------- the entry points ----
def test_pyproject_declares_the_console_scripts() -> None:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["inferference-reconcile"] == "inferference.reconcile_cli:main"
    assert scripts["inferference-balancer"] == "inferference.balancer:main"


def test_the_reconcile_cli_is_a_library_module() -> None:
    from inferference import reconcile_cli

    assert callable(reconcile_cli.main)
    assert "sys.path" not in Path(reconcile_cli.__file__ or "").read_text()


def test_the_reconcile_cli_runs_as_a_module_and_through_the_shim(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("INFERFERENCE_")}
    env["INFERFERENCE_CONFIG_DIR"] = str(cfg)
    via_module = subprocess.run(
        [sys.executable, "-m", "inferference.reconcile_cli", "config-dir"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    via_shim = subprocess.run(
        [sys.executable, str(SHIM), "config-dir"], env=env, capture_output=True, text=True, timeout=60
    )
    assert via_module.returncode == 0, via_module.stderr
    assert via_module.stdout.strip() == str(cfg)
    assert (via_shim.returncode, via_shim.stdout) == (via_module.returncode, via_module.stdout)


def test_the_reconcile_cli_keeps_its_exit_code_for_a_bad_config(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("INFERFERENCE_")}
    env["INFERFERENCE_CONFIG_DIR"] = str(tmp_path / "missing")
    result = subprocess.run(
        [sys.executable, "-m", "inferference.reconcile_cli", "status"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 78
    assert "!! [reconcile]" in result.stderr


def test_the_shim_is_thin() -> None:
    text = SHIM.read_text()
    assert "sys.path" not in text
    assert len(text.splitlines()) < 20


# ------------------------------------------------------- no checkout path ----
@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: p.name)
def test_no_launcher_puts_a_checkout_src_on_the_python_path(script: Path) -> None:
    code = "\n".join(_code(script))
    assert "sys.path" not in code
    assert "PYTHONPATH" not in code
    assert "/src" not in code


# ------------------------------------------------- the residency patch ----
def _bare_runner(tmp_path: Path) -> Path:
    """Copy the router launcher alone to a tree with no checkout around it."""
    runner = tmp_path / "pkg" / "ci" / "runner"
    runner.mkdir(parents=True)
    shutil.copy(ROUTER, runner / ROUTER.name)
    return runner


def _router_env(tmp_path: Path) -> dict[str, str]:
    env = _make_host(tmp_path)
    env["INFERFERENCE_CONFIG_DIR"] = env["STUB_SYSTEM_DIR"]
    env["INFERFERENCE_PRESET_INI"] = str(tmp_path / "run" / "preset.ini")
    return env


def test_the_patch_next_to_the_script_is_used(tmp_path: Path) -> None:
    env = _router_env(tmp_path)
    runner = _bare_runner(tmp_path)
    (runner / "vulkan-residency.patch").write_bytes(PATCH.read_bytes())
    proc = subprocess.run(
        ["bash", str(runner / ROUTER.name), "--check"], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert "--check: resolved, not started." in proc.stdout


def test_the_patch_variable_wins_over_the_file_next_to_the_script(tmp_path: Path) -> None:
    env = _router_env(tmp_path)
    runner = _bare_runner(tmp_path)
    (runner / "vulkan-residency.patch").write_text("a different patch\n")
    other = tmp_path / "other.patch"
    other.write_text("the patch the server carries\n")
    manifest = Path(env["INFERFERENCE_SERVER_DIR"]) / "build-manifest.json"
    manifest.write_text(
        json.dumps(
            {"build_id": "b-test", "vulkan_residency_patch_sha256": hashlib.sha256(other.read_bytes()).hexdigest()}
        )
    )
    env["INFERFERENCE_RESIDENCY_PATCH"] = str(other)
    proc = subprocess.run(
        ["bash", str(runner / ROUTER.name), "--check"], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr


def test_a_server_without_the_current_patch_is_refused(tmp_path: Path) -> None:
    env = _router_env(tmp_path)
    runner = _bare_runner(tmp_path)
    (runner / "vulkan-residency.patch").write_text("a newer patch\n")
    proc = subprocess.run(
        ["bash", str(runner / ROUTER.name), "--check"], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 78
    assert "does not carry the current residency patch" in proc.stderr


def test_a_missing_patch_is_refused_by_name(tmp_path: Path) -> None:
    env = _router_env(tmp_path)
    runner = _bare_runner(tmp_path)
    proc = subprocess.run(
        ["bash", str(runner / ROUTER.name), "--check"], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 78
    assert "INFERFERENCE_RESIDENCY_PATCH" in proc.stderr


def test_the_server_comes_from_the_server_dir_variable(tmp_path: Path) -> None:
    env = _router_env(tmp_path)
    env["INFERFERENCE_SERVER_DIR"] = str(tmp_path / "no-server")
    proc = subprocess.run(["bash", str(ROUTER), "--check"], env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 78
    assert f"no Mode A server artefact at {tmp_path / 'no-server'}" in proc.stderr
    assert "INFERFERENCE_SERVER_DIR" in proc.stderr


# ------------------------------------------------- the interpreter ----
_PY_STUB = """#!/usr/bin/env bash
echo "PYTHONPATH=${PYTHONPATH-unset}"
echo "ARGS=$*"
"""


def _bare_launcher(tmp_path: Path, script: Path) -> Path:
    runner = tmp_path / "pkg" / "ci" / "runner"
    runner.mkdir(parents=True, exist_ok=True)
    shutil.copy(script, runner / script.name)
    (runner / "inferference-reconcile.py").write_text("")
    return runner / script.name


def _run_bare(launcher: Path, tmp_path: Path, **extra: str) -> subprocess.CompletedProcess[str]:
    cfg = tmp_path / "cfg"
    cfg.mkdir(exist_ok=True)
    (cfg / "placement.yaml").write_text("router: {}\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("INFERFERENCE_") and k != "PYTHONPATH"}
    env["INFERFERENCE_CONFIG_DIR"] = str(cfg)
    env.update(extra)
    return subprocess.run(["bash", str(launcher), "--check"], env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("script", [ROUTER, BALANCER], ids=lambda p: p.name)
def test_without_the_variable_and_without_a_checkout_venv_the_launcher_names_the_variable(
    tmp_path: Path, script: Path
) -> None:
    proc = _run_bare(_bare_launcher(tmp_path, script), tmp_path)
    assert proc.returncode == 78
    assert "INFERFERENCE_PYTHON" in proc.stderr


def test_the_balancer_falls_back_to_the_checkout_venv_only_when_it_exists(tmp_path: Path) -> None:
    launcher = _bare_launcher(tmp_path, BALANCER)
    venv_py = tmp_path / "pkg" / "ci" / "library" / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text(_PY_STUB)
    venv_py.chmod(0o755)
    proc = _run_bare(launcher, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "-m inferference.balancer" in proc.stdout


def test_the_balancer_sets_no_python_path(tmp_path: Path) -> None:
    launcher = _bare_launcher(tmp_path, BALANCER)
    stub = tmp_path / "py"
    stub.write_text(_PY_STUB)
    stub.chmod(0o755)
    proc = _run_bare(launcher, tmp_path, INFERFERENCE_PYTHON=str(stub))
    assert proc.returncode == 0, proc.stderr
    assert "PYTHONPATH=unset" in proc.stdout


def test_the_reload_script_uses_the_variable_python(tmp_path: Path) -> None:
    launcher = _bare_launcher(tmp_path, RELOAD)
    stub = tmp_path / "py"
    stub.write_text(_PY_STUB)
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("INFERFERENCE_")}
    env.update(INFERFERENCE_PYTHON=str(stub), INFERFERENCE_RELOAD_LOCK=str(tmp_path / "lock"))
    proc = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "inferference-reconcile.py --timeout 600 apply" in proc.stdout
