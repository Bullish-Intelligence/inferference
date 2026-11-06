"""Strict, immutable compatibility keys for llama.cpp runtime artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactIdentity(_FrozenModel):
    """Content digest plus the inexpensive file identity used to validate it later."""

    path: str
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    inode: int | None = Field(default=None, ge=0)
    metadata: tuple[tuple[str, str], ...] = ()

    @field_validator("sha256")
    @classmethod
    def _sha256_is_hex(cls, value: str) -> str:
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError("sha256 must be hexadecimal") from exc
        return value.lower()

    def matches_path(self, path: str | Path) -> bool:
        """Check metadata only; never rehash a multi-GB model during startup."""
        candidate = file_identity(path)
        return (
            self.size_bytes == candidate["size_bytes"]
            and self.mtime_ns == candidate["mtime_ns"]
            and self.inode == candidate["inode"]
        )


def file_identity(path: str | Path) -> dict[str, int | None]:
    """Return inexpensive stat facts suitable for validating a registered digest."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": getattr(stat, "st_ino", None)}


def register_artifact(path: str | Path, *, metadata: dict[str, Any] | None = None) -> ArtifactIdentity:
    """Hash an artifact once at explicit registration time.

    Callers persist the returned object and later use :meth:`matches_path`, which
    only checks file metadata.  Selected GGUF metadata can be supplied as strings
    to make the registration fact more informative without parsing model weights.
    """
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    identity = file_identity(resolved)
    frozen_metadata = tuple(sorted((str(key), str(value)) for key, value in (metadata or {}).items()))
    return ArtifactIdentity(path=str(resolved), sha256=digest.hexdigest(), metadata=frozen_metadata, **identity)


class LlamaEngineFingerprint(_FrozenModel):
    """All facts that make tokenizer, grammar, KV, and LoRA state compatible."""

    model: ArtifactIdentity
    tokenizer: ArtifactIdentity
    llama_cpp_python_version: str
    llama_cpp_commit: str | None = None
    llama_cpp_build_id: str | None = None
    # True when the loaded libllama exports the mixed-batch per-sequence
    # LoRA routing surface (llama_set_seq_adapters / llama_set_seq_adapter) — i.e. the
    # private P2 fork rather than a stock lib. Part of the key so a fork-built engine
    # and a stock engine over the same model never share cache state.
    seq_adapter_routing: bool = False
    backend: str
    n_ctx: int = Field(gt=0)
    kv_type_k: str | None = None
    kv_type_v: str | None = None
    rope_scaling_type: str | None = None
    rope_freq_base: float | None = Field(default=None, gt=0)
    rope_freq_scale: float | None = Field(default=None, gt=0)
    swa_full: bool | None = None
    active_loras: tuple[ArtifactIdentity, ...] = ()

    @field_validator("backend", "llama_cpp_python_version")
    @classmethod
    def _required_text_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value

    def cache_key(self) -> str:
        """A deterministic digest suitable for grammar and prefix-cache keys."""
        payload = self.model_dump_json(by_alias=True, exclude_none=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
