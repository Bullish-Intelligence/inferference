"""`ci/runner/run-suite.sh` classifies a suite fail-closed. No GPU and no native build.

Each test copies the script into a tmp tree, next to a tiny fake suite. The script
finds its suite from its own path, so it runs the fake suite with the real
classification code. The probe is a shell stub, and the tests are plain pytest files.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "ci" / "runner" / "run-suite.sh"

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="run-suite.sh asks gcc for its library paths")

_PASSING = "def test_ok():\n    assert True\n"
_SKIPPING = "import pytest\n\n\ndef test_gpu():\n    pytest.skip('no device')\n"


def _make(
    tmp_path: Path, env: dict, tests: dict[str, str], *, probe_ok: bool = True, features: list[str] | None = None
):
    """Build a tmp repo with one suite `demo`; return (script, out_dir)."""
    root = tmp_path / "repo"
    (root / "ci" / "runner").mkdir(parents=True)
    script = root / "ci" / "runner" / "run-suite.sh"
    shutil.copy(_SCRIPT, script)
    suite = root / "suites" / "demo"
    (suite / "probes").mkdir(parents=True)
    (suite / "tests").mkdir()
    probe = suite / "probes" / "capability.sh"
    probe.write_text(f"#!/usr/bin/env bash\nexit {0 if probe_ok else 1}\n")
    probe.chmod(0o755)
    for name, body in tests.items():
        (suite / "tests" / name).write_text(body)
    spec = {
        "suite": "demo",
        "capability": {"probe": "probes/capability.sh", "required": True},
        "env": env,
        "tests": [f"tests/{n}" for n in tests],
        "skip_on_stock": False,
    }
    (suite / "suite.json").write_text(json.dumps(spec))
    out = tmp_path / "out"
    (out / "lib").mkdir(parents=True)
    (out / "build-manifest.json").write_text(json.dumps({"features": features if features is not None else ["fork"]}))
    return script, out


def _run(script: Path, out: Path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MODEL_", "TOKENIZER_"))}
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(script), "--suite", "demo", "--out", str(out), "--library-python", sys.executable],
        capture_output=True,
        text=True,
        env=env,
    )


def _file(tmp_path: Path, name: str = "model.gguf") -> str:
    path = tmp_path / name
    path.write_text("x")
    return str(path)


def test_unset_role_variable_fails_and_names_it(tmp_path: Path) -> None:
    script, out = _make(tmp_path, {"MODEL_ROLE": "alpha"}, {"test_a.py": _PASSING})
    done = _run(script, out)
    assert done.returncode == 1
    assert "[suite-verdict] demo: FAIL" in done.stdout
    assert "MODEL_ALPHA is not set" in done.stderr


def test_unreadable_role_file_fails_and_names_the_path(tmp_path: Path) -> None:
    script, out = _make(tmp_path, {"MODEL_ROLE": "alpha"}, {"test_a.py": _PASSING})
    gone = str(tmp_path / "missing.gguf")
    done = _run(script, out, MODEL_ALPHA=gone)
    assert done.returncode == 1
    assert f"MODEL_ALPHA names no readable file: {gone}" in done.stderr


def test_every_listed_variable_is_enforced_and_every_gap_is_reported(tmp_path: Path) -> None:
    env = {"MODEL_ROLE": "alpha", "MODEL_VARS": ["MODEL_PROBE_DIR", "TOKENIZER_DEFAULT"]}
    script, out = _make(tmp_path, env, {"test_a.py": _PASSING})
    done = _run(script, out, MODEL_ALPHA=_file(tmp_path))
    assert done.returncode == 1
    assert "MODEL_PROBE_DIR is not set" in done.stderr
    assert "TOKENIZER_DEFAULT is not set" in done.stderr
    # A directory counts as readable.
    done = _run(
        script, out, MODEL_ALPHA=_file(tmp_path), MODEL_PROBE_DIR=str(tmp_path), TOKENIZER_DEFAULT=str(tmp_path)
    )
    assert done.returncode == 0, done.stderr
    assert "[suite-verdict] demo: PASS" in done.stdout


def test_inventory_gap_on_a_stock_build_is_a_skip(tmp_path: Path) -> None:
    script, out = _make(
        tmp_path, {"MODEL_ROLE": "alpha", "MODEL_VARS": ["MODEL_PROBE_DIR"]}, {"test_a.py": _PASSING}, features=[]
    )
    spec_path = tmp_path / "repo" / "suites" / "demo" / "suite.json"
    spec = json.loads(spec_path.read_text())
    spec["skip_on_stock"] = True
    spec_path.write_text(json.dumps(spec))
    done = _run(script, out, MODEL_ALPHA=_file(tmp_path))
    assert done.returncode == 0
    assert "[suite-verdict] demo: SKIP" in done.stdout


def test_a_role_name_with_an_underscore_is_refused(tmp_path: Path) -> None:
    script, out = _make(tmp_path, {"MODEL_ROLE": "default_model"}, {"test_a.py": _PASSING})
    done = _run(script, out)
    assert done.returncode == 2
    assert "is not a role name" in done.stderr


def test_all_gpu_tests_skipped_is_a_skip_not_a_pass(tmp_path: Path) -> None:
    env = {"MODEL_ROLE": "alpha", "GPU": True}
    script, out = _make(tmp_path, env, {"test_gpu.py": _SKIPPING})
    done = _run(script, out, MODEL_ALPHA=_file(tmp_path))
    assert done.returncode == 0
    assert "[suite-verdict] demo: SKIP" in done.stdout
    assert "every GPU test skipped" in done.stderr
    assert "PASS" not in done.stdout.split("[suite-verdict]")[-1]


def test_non_gpu_tests_passing_do_not_hide_skipped_gpu_tests(tmp_path: Path) -> None:
    env = {"MODEL_ROLE": "alpha", "GPU": True, "GPU_TESTS": ["tests/test_gpu.py"]}
    script, out = _make(tmp_path, env, {"test_plain.py": _PASSING, "test_gpu.py": _SKIPPING})
    done = _run(script, out, MODEL_ALPHA=_file(tmp_path))
    assert done.returncode == 0
    assert "[suite-verdict] demo: SKIP" in done.stdout


def test_a_passing_gpu_test_gives_pass(tmp_path: Path) -> None:
    env = {"MODEL_ROLE": "alpha", "GPU": True, "GPU_TESTS": ["tests/test_gpu.py"]}
    script, out = _make(tmp_path, env, {"test_plain.py": _SKIPPING, "test_gpu.py": _PASSING})
    done = _run(script, out, MODEL_ALPHA=_file(tmp_path))
    assert done.returncode == 0, done.stderr
    assert "[suite-verdict] demo: PASS" in done.stdout


def test_a_failing_gpu_test_still_fails(tmp_path: Path) -> None:
    env = {"MODEL_ROLE": "alpha", "GPU": True}
    script, out = _make(tmp_path, env, {"test_gpu.py": "def test_bad():\n    assert False\n"})
    done = _run(script, out, MODEL_ALPHA=_file(tmp_path))
    assert done.returncode == 1
    assert "[suite-verdict] demo: FAIL" in done.stdout


def test_all_skipped_non_gpu_suite_still_fails(tmp_path: Path) -> None:
    script, out = _make(tmp_path, {"MODEL_ROLE": "alpha", "GPU": False}, {"test_a.py": _SKIPPING})
    done = _run(script, out, MODEL_ALPHA=_file(tmp_path))
    assert done.returncode == 1
    assert "[suite-verdict] demo: FAIL" in done.stdout
