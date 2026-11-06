"""codeeval — a serving-level coding evaluation harness.

The harness talks to an OpenAI-compatible endpoint over HTTP, exactly as a real
client does. It does not link ``llama_cpp``. That keeps the numbers equal to
what a user experiences, and lets one harness measure the local router and a
hosted baseline with the same code path.

This package is the library form: the corpus loader, the prompt renderer, the
timing client, the drive loop and the graders. All of it is testable on a CPU
with no GPU and no network. ``src/inferference/benchkit/runners/`` holds the orchestration
shell.

No model is named in this package, in the corpus, or in any default. A model
comes from a run's command line.
"""

from __future__ import annotations

from .client import ChatClient, Endpoint, RequestTimings, Response
from .drive import DriveResult, run_single
from .render import RenderedPrompt, render
from .report import RunReport, summarize
from .task import Task, TaskSet, load_task_set

__all__ = [
    "ChatClient",
    "DriveResult",
    "Endpoint",
    "RenderedPrompt",
    "RequestTimings",
    "Response",
    "RunReport",
    "Task",
    "TaskSet",
    "load_task_set",
    "render",
    "run_single",
    "summarize",
]
