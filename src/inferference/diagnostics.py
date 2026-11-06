"""Best-effort runtime version diagnostics without importing heavy runtimes."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class RuntimeDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    python_version: str
    platform: str
    llama_cpp_python_version: str | None = None
    llama_cpp_library_path: str | None = None
    llama_cpp_commit: str | None = None
    llama_cpp_build_id: str | None = None
    # Whether the pointed-at lib is the P2 fork exporting the mixed-batch
    # per-sequence LoRA routing surface. Read from the build manifest (no native
    # import), so it reflects the lib on LLAMA_CPP_LIB_PATH even before load.
    seq_adapter_routing: bool | None = None
    ggml_version: str | None = None
    xgrammar_version: str | None = None
    torch_version: str | None = None


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _build_manifest(library_path: str | None) -> dict[str, object]:
    if not library_path:
        return {}
    library_dir = Path(library_path).expanduser()
    candidates = (library_dir.parent / "build-manifest.json", library_dir / "build-manifest.json")
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def _text(manifest: dict[str, object], *names: str) -> str | None:
    for name in names:
        value = manifest.get(name)
        if value is not None:
            return str(value)
    return None


def _flag(manifest: dict[str, object], name: str) -> bool | None:
    value = manifest.get(name)
    if value is None:
        return None
    return bool(value)


def collect_runtime_diagnostics() -> RuntimeDiagnostics:
    """Report installed/runtime facts, returning ``None`` for unavailable evidence."""
    library_path = os.environ.get("LLAMA_CPP_LIB_PATH")
    manifest = _build_manifest(library_path)
    return RuntimeDiagnostics(
        python_version=platform.python_version(),
        platform=platform.platform(),
        llama_cpp_python_version=_package_version("llama-cpp-python"),
        llama_cpp_library_path=library_path,
        llama_cpp_commit=_text(manifest, "llama_cpp_commit", "commit", "ref"),
        llama_cpp_build_id=_text(manifest, "build_id", "profile"),
        seq_adapter_routing=_flag(manifest, "seq_adapter_routing"),
        ggml_version=_text(manifest, "ggml_version"),
        xgrammar_version=_package_version("xgrammar"),
        torch_version=_package_version("torch"),
    )
