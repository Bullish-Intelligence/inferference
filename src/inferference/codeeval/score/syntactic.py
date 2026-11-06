"""Layer 1 — does the artifact parse? Binary and near-free."""

from __future__ import annotations

import ast
import json

from ..task import Task
from .result import LayerResult

__all__ = ["grade_parse"]


def grade_parse(task: Task, artifact: str) -> LayerResult:
    if task.output.kind == "prose":
        return LayerResult(layer="parse", status="skip", reason="prose output")

    if task.output.kind == "json":
        try:
            json.loads(artifact)
        except json.JSONDecodeError as exc:
            return LayerResult(layer="parse", status="fail", reason=str(exc))
        return LayerResult(layer="parse", status="pass")

    if task.language != "python":
        return LayerResult(
            layer="parse",
            status="skip",
            reason=f"no parser wired for {task.language!r}",
        )

    try:
        ast.parse(artifact)
    except SyntaxError as exc:
        return LayerResult(
            layer="parse",
            status="fail",
            reason=f"{exc.msg} (line {exc.lineno})",
        )
    return LayerResult(layer="parse", status="pass")
