"""Task -> prompt. Deterministic, verbatim, never padded and never truncated.

Two rules:

* **No padding.** Filler is not real context. A task's token count is a property
  of the task.
* **No truncation.** A prompt that does not fit the endpoint's per-lane context
  is SKIPPED and reported as skipped. Cutting it silently would produce a number
  for a task the model never saw.

The template is versioned. Changing it changes ``TEMPLATE_VERSION``, which lands
in the envelope, because a prompt change invalidates comparison with older runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .task import Task

__all__ = ["RenderedPrompt", "TEMPLATE_VERSION", "render", "estimate_tokens"]

TEMPLATE_VERSION = 1

SYSTEM = (
    "You are a precise software engineer. Follow the instruction exactly. "
    "Return only what is asked for, with no preamble and no closing commentary."
)

_KIND_CONTRACT = {
    "full_file": (
        "Return the COMPLETE new contents of {path} in a single fenced code "
        "block. Do not abbreviate any part of it and do not write "
        "placeholders such as '# unchanged'."
    ),
    "patch": ("Return a unified diff against {path} in a single fenced code block."),
    "json": ("Return one JSON object in a single fenced code block. It must satisfy the schema shown above."),
    "snippet": ("Return only the requested code in a single fenced code block."),
    "prose": ("Answer in prose. Do not return a code block unless you are quoting."),
}


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """One prompt, plus what it is made of."""

    task_id: str
    system: str
    user: str
    template_version: int
    prompt_sha256: str
    approx_tokens: int

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def estimate_tokens(text: str) -> int:
    """A rough, tokenizer-free size, used only to decide skip-vs-send.

    It is deliberately an ESTIMATE and is never reported as a token count. The
    server's own ``n_prompt`` is the truth, and it lands in the record.
    """
    return max(1, len(text) // 4)


def render(task: Task) -> RenderedPrompt:
    """Build the prompt for one task. Same task + same template = same bytes."""
    parts: list[str] = []

    if task.context.files:
        parts.append("Here are the relevant files.\n")
        for rel in task.context.files:  # declared order, never sorted
            lang = _fence_lang(rel, task.language)
            parts.append(f"--- {rel} ---\n```{lang}\n{task.read(rel)}\n```\n")

    if task.context.cursor:
        parts.append(f"The cursor is at {task.context.cursor}.\n")

    if task.output.kind == "json" and task.output.schema_file:
        parts.append(f"The answer must satisfy this JSON schema.\n```json\n{task.read(task.output.schema_file)}\n```\n")

    parts.append(f"# Task\n\n{task.instruction.strip()}\n")

    contract = _KIND_CONTRACT[task.output.kind].format(path=task.output.path or "")
    parts.append(f"# Output\n\n{contract}")

    user = "\n".join(parts)
    return RenderedPrompt(
        task_id=task.id,
        system=SYSTEM,
        user=user,
        template_version=TEMPLATE_VERSION,
        prompt_sha256=hashlib.sha256((SYSTEM + "\n" + user).encode()).hexdigest(),
        approx_tokens=estimate_tokens(SYSTEM + user),
    )


def _fence_lang(rel: str, default: str) -> str:
    suffix = rel.rsplit(".", 1)[-1] if "." in rel else ""
    return {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "rs": "rust",
        "go": "go",
        "sh": "bash",
        "json": "json",
        "md": "markdown",
        "yaml": "yaml",
        "yml": "yaml",
    }.get(suffix, default)
