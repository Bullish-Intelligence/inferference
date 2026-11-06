"""Workload registry — versioned declarative workloads.

The json_workload corpus ships with the package and is
sha256-verified against its manifest. Its corpus utilities (``read_jsonl`` /
``sha256`` / ``validate_corpus`` / ``select``) are a data-side module;
``load()`` here is the registry API that validates the manifest + corpus hashes
and exposes the stable-prefix selection used by ``bench run json_workload@v1``.

The registry is the single public entry point; the corpus module is imported
via importlib (it is a data dir, not a package) — the one documented coupling
point between registry and corpus, mirroring how the ported CPU test
(tests/test_project17_workload.py) loads it.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A source checkout holds the corpus under bench/. The wheel carries a copy
# beside this module. An explicit override takes precedence over both.
_PACKAGED_WORKLOADS_DIR = Path(__file__).resolve().parent / "workloads"
_DEFAULT_WORKLOADS_DIR = (
    _PACKAGED_WORKLOADS_DIR
    if _PACKAGED_WORKLOADS_DIR.is_dir()
    else Path(__file__).resolve().parents[3] / "bench" / "workloads"
)
_WORKLOADS_DIR = Path(os.environ.get("INFERFERENCE_WORKLOADS_DIR", _DEFAULT_WORKLOADS_DIR)).expanduser()


@dataclass(frozen=True, slots=True)
class Workload:
    """One loaded, validated workload corpus."""

    name: str
    version: str
    corpus_dir: Path
    manifest: dict[str, Any]
    schema_registry: dict[str, Any]
    entries: list[dict[str, Any]]

    def corpus_sha256(self, filename: str) -> str:
        return _corpus_module(self.corpus_dir).sha256(self.corpus_dir / filename)

    def select(self, count: int) -> list[dict[str, Any]]:
        """The deterministic stable prefix (10/100/1000 regression comparisons)."""
        return _corpus_module(self.corpus_dir).select(self.entries, count)


def _corpus_module(corpus_dir: Path) -> Any:
    spec = importlib.util.spec_from_file_location("workload", corpus_dir / "workload.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REGISTRY: dict[str, Workload] = {}


def load(name: str, version: str | None = None) -> Workload:
    """Load (once) and validate a named workload; ``version`` selects the corpus.

    Only ``json_workload@project17-json-workload-v1`` exists today. The manifest
    is validated (files + sha256 + prefix contract) exactly like the ported CPU
    test, so a corrupted or drifted corpus fails closed at load time.
    """
    key = name if version is None else f"{name}@{version}"
    if key in _REGISTRY:
        return _REGISTRY[key]
    corpus_dir = _WORKLOADS_DIR / name
    if not (corpus_dir / "json_workload_manifest.json").is_file():
        raise ValueError(f"unknown workload {name!r} (no manifest in {corpus_dir})")
    manifest = json.loads((corpus_dir / "json_workload_manifest.json").read_text())
    if version is not None and manifest.get("corpus_version") != version:
        raise ValueError(f"workload {name!r} version mismatch: {manifest.get('corpus_version')} != {version!r}")
    corpus_module = _corpus_module(corpus_dir)
    for filename, meta in manifest.get("files", {}).items():
        actual = corpus_module.sha256(corpus_dir / filename)
        if actual != meta["sha256"]:
            raise ValueError(f"corpus file {filename!r} sha256 mismatch: {actual} != {meta['sha256']}")
    registry = json.loads((corpus_dir / manifest.get("schema_registry", "schema_registry_v1.json")).read_text())
    entries = corpus_module.read_jsonl(corpus_dir / "json_workload_1000.jsonl")
    corpus_module.validate_corpus(entries, registry["schemas"])
    workload = Workload(
        name=name,
        version=str(manifest.get("corpus_version")),
        corpus_dir=corpus_dir,
        manifest=manifest,
        schema_registry=registry,
        entries=entries,
    )
    _REGISTRY[key] = workload
    return workload


def list_workloads() -> list[str]:
    """Names of corpora with a json_workload_manifest.json (auto-discovery)."""
    return sorted(p.name for p in _WORKLOADS_DIR.iterdir() if (p / "json_workload_manifest.json").exists())
