"""Layer 0 — pull the asked-for artifact out of the response.

Extraction failure is a TASK FAILURE and is counted on its own. "The model
wrote prose where a file was asked for" is a real capability defect, and it is
the first thing structured generation should fix. The extraction rate with and
without structured output is a headline number of this harness.
"""

from __future__ import annotations

import json
import re

from ..task import Task
from .result import LayerResult

__all__ = ["extract_artifact"]

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


def _blocks(text: str) -> list[tuple[str, str]]:
    return [(m.group(1).lower(), m.group(2)) for m in _FENCE.finditer(text)]


def extract_artifact(task: Task, text: str) -> tuple[str | None, LayerResult]:
    """Return ``(artifact, result)``. ``artifact`` is None when extraction fails."""
    kind = task.output.kind

    if kind == "prose":
        stripped = text.strip()
        if not stripped:
            return None, LayerResult(layer="extract", status="fail", reason="empty answer")
        return stripped, LayerResult(layer="extract", status="pass")

    blocks = _blocks(text)
    if not blocks:
        return None, LayerResult(
            layer="extract",
            status="fail",
            reason="no fenced code block in the response",
            detail={"chars": len(text)},
        )

    if kind == "json":
        for _, body in reversed(blocks):
            try:
                json.loads(body)
            except json.JSONDecodeError:
                continue
            return body, LayerResult(layer="extract", status="pass")
        return None, LayerResult(
            layer="extract",
            status="fail",
            reason="no fenced block parsed as JSON",
            detail={"blocks": len(blocks)},
        )

    # full_file / patch / snippet: prefer a block whose fence names the task's
    # language, else the largest block. Several blocks is itself worth recording,
    # because it usually means the model explained as well as answered.
    wanted = {task.language, _alias(task.language)}
    candidates = [b for lang, b in blocks if lang in wanted] or [b for _, b in blocks]
    artifact = max(candidates, key=len)
    return artifact, LayerResult(
        layer="extract",
        status="pass",
        detail={"blocks": len(blocks), "picked_chars": len(artifact)},
    )


def _alias(language: str) -> str:
    return {"python": "py", "javascript": "js", "typescript": "ts"}.get(language, language)
