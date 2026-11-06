from __future__ import annotations

import importlib.util
import json
from pathlib import Path

# The json_workload corpus moves with the bench port (03-BENCHMARK-LIBRARY-
# the corpus files also stay byte-copied):
#   bench/workloads/json_workload/ — byte-copied from the reference
#   benchmarks/project17/ (sha256-verified against the manifest, Task 1).
ROOT = Path(__file__).parents[1] / "bench" / "workloads" / "json_workload"


def _module():
    spec = importlib.util.spec_from_file_location("project17_workload", ROOT / "workload.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corpus_prefix_manifest_and_validation() -> None:
    workload = _module()
    small = ROOT / "json_workload_100.jsonl"
    large = ROOT / "json_workload_1000.jsonl"
    manifest = json.loads((ROOT / "json_workload_manifest.json").read_text())
    assert large.read_bytes().splitlines(keepends=True)[:100] == small.read_bytes().splitlines(keepends=True)
    assert workload.sha256(small) == manifest["files"][small.name]["sha256"]
    assert workload.sha256(large) == manifest["files"][large.name]["sha256"]
    registry = json.loads((ROOT / "schema_registry_v1.json").read_text())
    workload.validate_corpus(workload.read_jsonl(large), registry["schemas"])


def test_selection_is_deterministic_prefix() -> None:
    workload = _module()
    entries = workload.read_jsonl(ROOT / "json_workload_1000.jsonl")
    assert [item["id"] for item in workload.select(entries, 10)] == [f"p17-{value:04d}" for value in range(1, 11)]
