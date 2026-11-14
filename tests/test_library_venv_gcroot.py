"""The GC root of the library venv interpreter is named by its store entry.

No uv, no Nix. A fake `nix-store` records its arguments. The venv already
exists, so the script adds the root and stops. The fault guarded here: the root
had one fixed name, so a second venv on another interpreter dropped the first
root, and a garbage collection could delete the interpreter of the first venv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ci/runner/library-venv.sh"

_FAKE_NIX_STORE = """#!/usr/bin/env bash
echo "$*" >> "$STUB_NIX_LOG"
"""


def _store_executables() -> list[Path]:
    """Two world-runnable executables in different /nix/store entries."""
    found: dict[str, Path] = {}
    for name in ("bash", "env", "cat", "ls", "sh"):
        which = shutil.which(name)
        if not which:
            continue
        real = Path(os.path.realpath(which))
        if not str(real).startswith("/nix/store/"):
            continue
        entry = "/".join(str(real).split("/")[:4])
        if os.access(real, os.X_OK) and (real.stat().st_mode & 0o005) == 0o005:
            found.setdefault(entry, real)
    return list(found.values())


def _venv(tmp_path: Path, name: str) -> Path:
    venv = tmp_path / name
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text("#!/usr/bin/env bash\necho Python 3.13.0\n")
    python.chmod(0o755)
    (venv / ".library-scope").write_text("full\n")
    return venv


def _run(tmp_path: Path, venv: Path, base: Path, gcroots: Path) -> subprocess.CompletedProcess[str]:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "nix-store"
    fake.write_text(_FAKE_NIX_STORE)
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LIBRARY_", "STUB_"))}
    env.update(
        {
            "PATH": f"{bindir}:{env['PATH']}",
            "LIBRARY_VENV": str(venv),
            "LIBRARY_BASE_PYTHON": str(base),
            "LIBRARY_GCROOT_DIR": str(gcroots),
            "STUB_NIX_LOG": str(tmp_path / "nix.log"),
        }
    )
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)


def test_two_interpreters_get_two_gc_roots(tmp_path: Path) -> None:
    bases = _store_executables()
    if len(bases) < 2:
        pytest.skip("this host has fewer than two world-runnable executables in distinct store entries")
    gcroots = tmp_path / "gcroots"
    for index, base in enumerate(bases[:2]):
        proc = _run(tmp_path, _venv(tmp_path, f"venv{index}"), base, gcroots)
        assert proc.returncode == 0, proc.stderr
    calls = (tmp_path / "nix.log").read_text().splitlines()
    assert len(calls) == 2
    roots = [call.split()[1] for call in calls]
    assert roots[0] != roots[1]  # one root per interpreter
    for root, base in zip(roots, bases[:2], strict=True):
        entry = "/".join(str(base).split("/")[:4])
        assert root == f"{gcroots}/library-python-{os.path.basename(entry)}"
    assert not any(root.endswith("/library-python") for root in roots)  # the fixed name is gone


def test_the_same_interpreter_keeps_one_root(tmp_path: Path) -> None:
    bases = _store_executables()
    if not bases:
        pytest.skip("this host has no world-runnable executable in /nix/store")
    gcroots = tmp_path / "gcroots"
    for index in range(2):
        proc = _run(tmp_path, _venv(tmp_path, f"venv{index}"), bases[0], gcroots)
        assert proc.returncode == 0, proc.stderr
    roots = {call.split()[1] for call in (tmp_path / "nix.log").read_text().splitlines()}
    assert len(roots) == 1
