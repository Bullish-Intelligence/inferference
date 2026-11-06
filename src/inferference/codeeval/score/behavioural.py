"""Layer 3 — run the task's ORIGINAL tests against the model's artifact.

This is the strongest signal in the harness, and the reason to prefer tasks
derived from real commits: the tests already exist and were never written with
the model in mind.

The tests are copied verbatim. The harness never edits them to make an artifact
pass, and a task whose tests it had to change would not be a measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .. import sandbox
from ..task import Task
from .result import LayerResult

__all__ = ["grade_pytest"]


def grade_pytest(
    task: Task,
    artifact: str,
    workdir: Path,
    *,
    timeout_s: float = 120.0,
) -> LayerResult:
    """Materialize the task's files, overwrite the target with the artifact, test."""
    tests = task.test_files
    if not tests:
        return LayerResult(layer="pytest", status="skip", reason="not declared")
    if not sandbox.available():
        return LayerResult(
            layer="pytest",
            status="skip",
            reason="bwrap not on PATH; refusing to run model code unsandboxed",
        )

    workdir.mkdir(parents=True, exist_ok=True)

    # Every context file, then the tests, then the artifact over its target.
    for rel in task.context.files:
        _place(workdir / rel, task.read(rel))
    for rel in tests:
        _place(workdir / rel, task.read(rel))
    # The tests import the task's files as "files.<module>". A conftest at the
    # work root puts that root on sys.path. It is harness infrastructure and is
    # identical for every task, so it cannot encode a task-specific fix.
    _place(
        workdir / "conftest.py",
        "import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path(__file__).parent))\n",
    )

    target = task.output.path
    if not target:
        return LayerResult(layer="pytest", status="skip", reason="task declares no output.path")
    _place(workdir / target, artifact)

    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        *tests,
    ]
    result = sandbox.run(argv, workdir=workdir, timeout_s=timeout_s)

    tail = (result.stdout or result.stderr).strip().splitlines()[-12:]
    detail = {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "tail": tail,
        "tests": tests,
    }
    (workdir / "pytest-output.txt").write_text(result.stdout + "\n" + result.stderr)

    if result.timed_out:
        return LayerResult(layer="pytest", status="fail", reason="timed out", detail=detail)
    if result.returncode == 0:
        return LayerResult(layer="pytest", status="pass", detail=detail)
    return LayerResult(
        layer="pytest",
        status="fail",
        reason=f"pytest exit {result.returncode}",
        detail=detail,
    )


def _place(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
