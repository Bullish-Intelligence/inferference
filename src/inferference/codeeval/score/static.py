"""Layer 2 — ruff. Reported as violations by rule code, not as pass/fail.

A count keeps "clean but for import order" distinct from "twelve undefined
names". The layer still carries a status, because a run needs a headline: it
passes at zero violations and fails above zero, while the detail holds the
breakdown that a summary actually reads.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from ..task import Task
from .result import LayerResult

__all__ = ["grade_ruff"]


def grade_ruff(task: Task, artifact_path: Path, timeout_s: float = 60.0) -> LayerResult:
    rules = task.grading.ruff
    if not rules:
        return LayerResult(layer="ruff", status="skip", reason="not declared")
    if task.language != "python":
        return LayerResult(layer="ruff", status="skip", reason=f"language {task.language!r}")
    ruff = shutil.which("ruff")
    if ruff is None:
        # Fail-closed: a missing linter is a loud skip, never a silent pass.
        return LayerResult(layer="ruff", status="skip", reason="ruff not on PATH")

    cmd = [
        ruff,
        "check",
        "--no-cache",
        "--isolated",
        "--select",
        ",".join(rules),
        "--output-format",
        "json",
        str(artifact_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return LayerResult(layer="ruff", status="skip", reason="ruff timed out")

    try:
        findings = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return LayerResult(
            layer="ruff",
            status="skip",
            reason="ruff produced no parsable report",
            detail={"stderr": proc.stderr[:400]},
        )

    by_code = Counter(f.get("code") or "?" for f in findings)
    return LayerResult(
        layer="ruff",
        status="pass" if not findings else "fail",
        reason="" if not findings else f"{len(findings)} violation(s)",
        detail={"count": len(findings), "by_code": dict(by_code), "rules": rules},
    )
