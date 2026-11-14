"""ci/runner/inferference-balancer-launch.sh: interpreter, config dir, and --check."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "ci/runner/inferference-balancer-launch.sh"

PLACEMENT = """\
devices: [{id: D0, card: 0}, {id: D1, card: 1}]
resident: [{model: a, device: D0}, {model: b, device: D1}]
router: {host: 127.0.0.1, port: 8101, models_max: 2}
balancer: {host: 127.0.0.1, port: 8100, groups: {pub: [a, b]}}
"""


def _run(tmp_path: Path, python: str | None, placement: str | None, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("INFERFERENCE_")}
    cfg = tmp_path / "cfg"
    cfg.mkdir(exist_ok=True)
    if placement is not None:
        (cfg / "placement.yaml").write_text(placement)
    env["INFERFERENCE_CONFIG_DIR"] = str(cfg)
    if python is not None:
        env["INFERFERENCE_PYTHON"] = python
    return subprocess.run(["bash", str(LAUNCH), *args], env=env, capture_output=True, text=True, timeout=60)


def test_check_validates_and_exits_zero(tmp_path: Path) -> None:
    import sys

    result = _run(tmp_path, sys.executable, PLACEMENT, "--check")
    assert result.returncode == 0, result.stderr
    assert "127.0.0.1:8100" in result.stdout and "127.0.0.1:8101" in result.stdout


def test_check_refuses_a_bad_placement(tmp_path: Path) -> None:
    import sys

    result = _run(tmp_path, sys.executable, PLACEMENT.replace("[a, b]", "[a, zz]"), "--check")
    assert result.returncode == 78
    assert "not a resident model" in result.stderr


def test_a_missing_python_is_refused_by_name(tmp_path: Path) -> None:
    result = _run(tmp_path, str(tmp_path / "no-python"), PLACEMENT, "--check")
    assert result.returncode == 78
    assert "no python at" in result.stderr


def test_a_missing_placement_is_refused(tmp_path: Path) -> None:
    import sys

    result = _run(tmp_path, sys.executable, None, "--check")
    assert result.returncode == 78
    assert "no placement" in result.stderr


@pytest.mark.skipif(not LAUNCH.exists(), reason="launch script missing")
def test_the_script_is_executable_bash() -> None:
    assert os.access(LAUNCH, os.X_OK)
    assert subprocess.run(["bash", "-n", str(LAUNCH)]).returncode == 0
