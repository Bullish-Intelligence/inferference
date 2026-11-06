"""Validation and deterministic selection for the versioned JSON workload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_corpus(entries: list[dict[str, Any]], schemas: dict[str, Any]) -> None:
    seen: set[str] = set()
    for index, item in enumerate(entries, start=1):
        required = {"id", "category", "prompt", "schema_id", "schema_ref", "max_tokens", "input_metadata"}
        missing = required.difference(item)
        if missing:
            raise ValueError(f"entry {index} missing {sorted(missing)}")
        if item["id"] in seen or item["id"] != f"p17-{index:04d}":
            raise ValueError(f"entry {index} has unstable id")
        seen.add(item["id"])
        if not isinstance(item["prompt"], str) or not item["prompt"] or item["schema_id"] not in schemas:
            raise ValueError(f"entry {index} has invalid prompt or schema")
        if not isinstance(item["max_tokens"], int) or item["max_tokens"] <= 0:
            raise ValueError(f"entry {index} has invalid max_tokens")


def select(entries: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Return the stable prefix used for 10/100/1000 regression comparisons."""
    if count <= 0 or count > len(entries):
        raise ValueError("count must be within the corpus")
    return entries[:count]
