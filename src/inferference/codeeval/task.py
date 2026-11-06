"""The corpus: one task per directory, loaded and content-hashed.

A task is data, not code. It names the context files it carries, the artifact
it wants back, and which grading layers apply to it. Every referenced file is
hashed at load time, so a drifted corpus fails closed instead of silently
changing what a recorded number means.

Layout::

    tasks/codeeval/<family>/<slug>/
        task.yaml        the declaration below
        files/           context files, included verbatim in the prompt
        tests/           the ORIGINAL tests, run against the model's artifact
        rubric.md        optional, for the judge layer
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CorpusError",
    "Grading",
    "OutputSpec",
    "Task",
    "TaskContext",
    "TaskSet",
    "load_task",
    "load_task_set",
]

FAMILIES = (
    "complete",
    "implement",
    "refactor",
    "debug",
    "test-write",
    "explain",
    "translate",
    "structured",
    "longctx",
    "cli-shape",
)

OutputKind = Literal["full_file", "patch", "json", "snippet", "prose"]


class CorpusError(RuntimeError):
    """The corpus is malformed, incomplete, or has drifted from its hashes."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskContext(_Strict):
    """The files the prompt carries, in declared order."""

    files: list[str] = Field(default_factory=list)
    cursor: str | None = None


class OutputSpec(_Strict):
    """What artifact the answer must yield."""

    kind: OutputKind
    path: str | None = None
    schema_file: str | None = None

    @model_validator(mode="after")
    def _needs_path(self) -> OutputSpec:
        if self.kind in ("full_file", "patch") and not self.path:
            raise ValueError(f"output.kind={self.kind} requires output.path")
        if self.kind == "json" and not self.schema_file:
            raise ValueError("output.kind=json requires output.schema_file")
        return self


class Grading(_Strict):
    """Which layers apply. An absent layer is not run and not scored."""

    parse: bool = True
    ruff: list[str] | None = None
    pytest: list[str] | None = None
    judge: str | None = None


class Task(_Strict):
    """One evaluation task, fully resolved against its directory."""

    id: str
    family: str
    version: int = 1
    language: str = "python"
    difficulty: int = Field(ge=1, le=5)
    origin: str = "hand-written"
    instruction: str
    context: TaskContext = Field(default_factory=TaskContext)
    output: OutputSpec
    grading: Grading = Field(default_factory=Grading)
    max_output_tokens: int = 1200
    reference: str | None = None
    """A known-good answer. Used ONLY by the corpus self-test, never in a prompt."""

    directory: Path
    file_hashes: dict[str, str] = Field(default_factory=dict)

    @field_validator("family")
    @classmethod
    def _known_family(cls, v: str) -> str:
        if v not in FAMILIES:
            raise ValueError(f"unknown family {v!r}; known: {', '.join(FAMILIES)}")
        return v

    def read(self, rel: str) -> str:
        return (self.directory / rel).read_text()

    @property
    def test_files(self) -> list[str]:
        return list(self.grading.pytest or [])


class TaskSet(_Strict):
    """A loaded, validated slice of the corpus."""

    root: Path
    tasks: list[Task]

    def __len__(self) -> int:
        return len(self.tasks)

    def slice(
        self,
        *,
        families: list[str] | None = None,
        ids: list[str] | None = None,
        limit: int | None = None,
    ) -> TaskSet:
        """Deterministic subset. Order is the corpus order, never shuffled."""
        picked = [t for t in self.tasks if (families is None or t.family in families) and (ids is None or t.id in ids)]
        if limit is not None:
            picked = picked[:limit]
        return TaskSet(root=self.root, tasks=picked)

    @property
    def corpus_hash(self) -> str:
        """One hash over every task's declaration and files. Names a corpus state."""
        h = hashlib.sha256()
        for t in self.tasks:
            h.update(t.id.encode())
            for rel in sorted(t.file_hashes):
                h.update(rel.encode())
                h.update(t.file_hashes[rel].encode())
        return h.hexdigest()


def load_task(directory: Path) -> Task:
    """Load and validate one task directory. Raises CorpusError on any defect."""
    decl = directory / "task.yaml"
    if not decl.is_file():
        raise CorpusError(f"no task.yaml in {directory}")
    try:
        raw: dict[str, Any] = yaml.safe_load(decl.read_text()) or {}
    except yaml.YAMLError as exc:
        raise CorpusError(f"{decl}: malformed YAML: {exc}") from exc

    # Referenced files must exist before the model sees them; hash each one so a
    # later edit to the corpus cannot silently change a recorded measurement.
    hashes: dict[str, str] = {"task.yaml": sha256_file(decl)}
    context = raw.get("context") or {}
    referenced = list(context.get("files") or [])
    referenced += list((raw.get("grading") or {}).get("pytest") or [])
    schema_file = (raw.get("output") or {}).get("schema_file")
    if schema_file:
        referenced.append(schema_file)
    judge = (raw.get("grading") or {}).get("judge")
    if judge:
        referenced.append(judge)
    if raw.get("reference"):
        referenced.append(raw["reference"])
    for rel in referenced:
        target = directory / rel
        if not target.is_file():
            raise CorpusError(f"{decl}: declares {rel!r}, which does not exist")
        hashes[rel] = sha256_file(target)

    try:
        return Task(directory=directory, file_hashes=hashes, **raw)
    except Exception as exc:  # pydantic ValidationError and ValueError alike
        raise CorpusError(f"{decl}: {exc}") from exc


def load_task_set(root: Path) -> TaskSet:
    """Load every task under ``root``, sorted by id for a stable order."""
    root = Path(root)
    if not root.is_dir():
        raise CorpusError(f"corpus root {root} does not exist")
    tasks = [load_task(d.parent) for d in sorted(root.glob("*/*/task.yaml"))]
    if not tasks:
        raise CorpusError(f"corpus root {root} holds no tasks")
    seen: set[str] = set()
    for t in tasks:
        if t.id in seen:
            raise CorpusError(f"duplicate task id {t.id!r}")
        seen.add(t.id)
    tasks.sort(key=lambda t: t.id)
    return TaskSet(root=root, tasks=tasks)
