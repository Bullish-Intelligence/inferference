"""Grading layers, cheapest first.

Every layer returns a :class:`LayerResult`. A layer that cannot run reports
``status="skip"`` with a reason. **A skipped layer is never a pass.** That is
the repo's fail-closed rule applied to scoring: a missing sandbox, a missing
linter or a missing judge must be visible in the summary, not absorbed.
"""

from __future__ import annotations

from .behavioural import grade_pytest
from .extract import extract_artifact
from .result import LayerResult, TaskScore
from .static import grade_ruff
from .syntactic import grade_parse

__all__ = [
    "LayerResult",
    "TaskScore",
    "extract_artifact",
    "grade_parse",
    "grade_pytest",
    "grade_ruff",
]
