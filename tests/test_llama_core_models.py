from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from inferference import (
    ArtifactIdentity,
    EngineConfig,
    GenerationRequest,
    GenerationResult,
    LlamaEngineFingerprint,
    collect_runtime_diagnostics,
    register_artifact,
)


def test_boundary_models_validate_generation_and_benchmark_shapes() -> None:
    config = EngineConfig(model_path="model.gguf", n_ctx=1024, active_loras=("router-a",))
    request = GenerationRequest(prompt="Hello", max_tokens=4)
    result = GenerationResult(text="Hi", prompt_token_count=1, completion_token_count=1, finish_reason="stop")

    assert config.active_loras == ("router-a",)
    assert request.temperature == 0
    assert result.token_ids == ()


def test_artifact_registration_hashes_once_then_validates_by_stat(tmp_path: Path) -> None:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"small stand-in")

    identity = register_artifact(artifact, metadata={"architecture": "qwen35"})

    assert identity.sha256 == hashlib.sha256(b"small stand-in").hexdigest()
    assert identity.matches_path(artifact)
    assert identity.metadata == (("architecture", "qwen35"),)
    artifact.write_bytes(b"changed")
    os.utime(artifact, ns=(identity.mtime_ns + 1, identity.mtime_ns + 1))
    assert not identity.matches_path(artifact)


def test_fingerprint_is_frozen_and_includes_adapter_identity(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    tokenizer = tmp_path / "tokenizer.json"
    adapter = tmp_path / "router.lora"
    for path in (model, tokenizer, adapter):
        path.write_text(path.name)
    fingerprint = LlamaEngineFingerprint(
        model=register_artifact(model),
        tokenizer=register_artifact(tokenizer),
        llama_cpp_python_version="0.3.34",
        llama_cpp_commit="0ab9d6fed",
        backend="cuda",
        n_ctx=1024,
        kv_type_k="f16",
        kv_type_v="f16",
        rope_freq_base=1_000_000,
        active_loras=(register_artifact(adapter),),
    )

    assert len(fingerprint.cache_key()) == 64
    with pytest.raises(ValidationError):
        fingerprint.backend = "cpu"  # type: ignore[misc]


def test_runtime_diagnostics_reads_a_runtime_build_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    (tmp_path / "build-manifest.json").write_text(
        json.dumps({"commit": "0ab9d6fed", "profile": "cuda-3060", "ggml_version": "0.18.0"})
    )
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", str(library_dir))

    diagnostics = collect_runtime_diagnostics()

    assert diagnostics.llama_cpp_library_path == str(library_dir)
    assert diagnostics.llama_cpp_commit == "0ab9d6fed"
    assert diagnostics.llama_cpp_build_id == "cuda-3060"
    assert diagnostics.ggml_version == "0.18.0"


def test_artifact_identity_rejects_non_hex_digest() -> None:
    with pytest.raises(ValidationError, match="hexadecimal"):
        ArtifactIdentity(path="model.gguf", sha256="x" * 64, size_bytes=0, mtime_ns=0)
