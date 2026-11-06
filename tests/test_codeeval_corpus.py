"""The corpus self-test: it loads, and every reference answer scores clean.

This is the step-4 gate from the concept document. A grader that cannot pass a
known-good answer, or that passes a known-bad one, is not measuring anything.
It needs no GPU and no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferference.codeeval import load_task_set, sandbox
from inferference.codeeval.render import render
from inferference.codeeval.score import extract_artifact, grade_parse, grade_pytest

CORPUS = Path(__file__).resolve().parents[1] / "tasks" / "codeeval"
TASKS = load_task_set(CORPUS)
WITH_REFERENCE = [t for t in TASKS.tasks if t.reference]
WITH_TESTS = [t for t in WITH_REFERENCE if t.grading.pytest]


def test_corpus_loads():
    assert len(TASKS) >= 10
    assert len({t.family for t in TASKS.tasks}) >= 6


def test_corpus_hash_is_stable():
    assert load_task_set(CORPUS).corpus_hash == TASKS.corpus_hash


@pytest.mark.parametrize("task", TASKS.tasks, ids=lambda t: t.id)
def test_prompt_renders_deterministically(task):
    first, second = render(task), render(task)
    assert first.prompt_sha256 == second.prompt_sha256
    assert task.instruction.strip()[:20] in first.user
    # A reference answer must never leak into the prompt.
    if task.reference:
        assert task.read(task.reference) not in first.user


@pytest.mark.parametrize("task", WITH_REFERENCE, ids=lambda t: t.id)
def test_reference_extracts_and_parses(task):
    assert task.reference
    fenced = f"Here you go.\n\n```python\n{task.read(task.reference)}\n```\n"
    artifact, result = extract_artifact(task, fenced)
    assert result.status == "pass", result.reason
    assert artifact is not None
    assert grade_parse(task, artifact).status == "pass"


@pytest.mark.parametrize("task", TASKS.tasks, ids=lambda t: t.id)
def test_prose_only_answer_fails_extraction(task):
    if task.output.kind == "prose":
        pytest.skip("prose output has no fence to miss")
    artifact, result = extract_artifact(task, "I would approach this as follows.")
    assert artifact is None
    assert result.status == "fail"


@pytest.mark.skipif(not sandbox.available(), reason="bwrap not on PATH")
@pytest.mark.parametrize("task", WITH_TESTS, ids=lambda t: t.id)
def test_reference_passes_its_own_tests(task, tmp_path):
    assert task.reference
    result = grade_pytest(task, task.read(task.reference), tmp_path / "ref")
    assert result.status == "pass", result.detail.get("tail")


@pytest.mark.skipif(not sandbox.available(), reason="bwrap not on PATH")
@pytest.mark.parametrize("task", WITH_TESTS, ids=lambda t: t.id)
def test_unmodified_starting_file_fails_its_tests(task, tmp_path):
    """The starting file must NOT already pass, or the task measures nothing."""
    assert task.output.path
    starting = task.read(task.output.path)
    result = grade_pytest(task, starting, tmp_path / "start")
    assert result.status == "fail"


def test_missing_sandbox_is_a_skip_not_a_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "available", lambda: False)
    task = WITH_TESTS[0]
    assert task.reference
    result = grade_pytest(task, task.read(task.reference), tmp_path / "nosandbox")
    assert result.status == "skip"
    assert not result.passed
