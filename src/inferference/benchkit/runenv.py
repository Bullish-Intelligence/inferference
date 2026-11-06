"""Isolated run directory.

Every bench run lands in its own directory under an artifacts root carrying the
canonical run facts — the envelope v2 contract (``envelope.json``) plus the
reference's v1 artifacts (``command.txt``, ``runtime-environment.txt``,
``stdout-stderr.log``, ``git-status-before``). The name fixes the v1
``git-commit.txt`` vs ``git-status-before`` drift by recording BOTH.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .env import visible_gpu_uuid
from .envelope import RunEnvelope


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z").replace(":", "").replace("-", "")


def _git_status_before(repo: Path) -> dict[str, str | bool]:
    """HEAD sha + dirty state at run start (fail-closed: unknown is recorded, not assumed)."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False)
    head = result.stdout.strip() if result.returncode == 0 else "unknown"
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=False)
    return {"head": head, "dirty": dirty.returncode == 0 and bool(dirty.stdout.strip())}


class RunContext:
    """One isolated run directory with its envelope + v1 artifacts."""

    def __init__(self, artifacts_root: Path, *, name: str, repo: Path) -> None:
        run_id = uuid.uuid4().hex[:12]
        self.run_id = run_id
        self.dir = artifacts_root / f"{name}-{_utc_now()}Z-{run_id}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.repo = repo
        self.stdout_stderr_log = self.dir / "stdout-stderr.log"

    def write_command(self, argv: list[str]) -> None:
        (self.dir / "command.txt").write_text(" ".join(argv) + "\n")

    def write_runtime_environment(self, extra: dict[str, Any] | None = None) -> None:
        import platform

        facts = {
            "python": sys.version,
            "platform": platform.platform(),
            "env": {k: os.environ.get(k) for k in ("LLAMA_CPP_LIB_PATH", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")},
            # The device identity is the GPU UUID, not the index: the index is a
            # function of CUDA_VISIBLE_DEVICES and reads 0 for whichever card was
            # exposed.
            "gpu_uuid": visible_gpu_uuid(),
        }
        if extra:
            facts.update(extra)
        (self.dir / "runtime-environment.txt").write_text(json.dumps(facts, indent=2) + "\n")

    def write_git_status_before(self) -> None:
        status = _git_status_before(self.repo)
        (self.dir / "git-status-before.json").write_text(json.dumps(status, indent=2) + "\n")
        head = status["head"]
        assert isinstance(head, str)
        (self.dir / "git-commit.txt").write_text(head + "\n")

    def write_envelope(self, envelope: RunEnvelope) -> Path:
        target = self.dir / "envelope.json"
        target.write_text(envelope.model_dump_json(indent=2) + "\n")
        return target

    def log(self, line: str) -> None:
        with self.stdout_stderr_log.open("a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)
