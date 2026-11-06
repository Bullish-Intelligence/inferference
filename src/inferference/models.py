"""Pydantic models at the public inference and benchmark boundaries.

These models deliberately stop at the boundary.  The owned decode loop should
pass plain token IDs and numpy views after a request has been validated.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EngineConfig(_BoundaryModel):
    """Stable llama.cpp context settings that affect inference semantics."""

    model_path: str
    n_ctx: int = Field(gt=0)
    n_batch: int = Field(default=512, gt=0)
    n_ubatch: int | None = Field(default=None, gt=0)
    n_gpu_layers: int = Field(default=0, ge=-1)
    seed: int | None = None
    backend: str = "cpu"
    rope_scaling_type: str | None = None
    rope_freq_base: float | None = Field(default=None, gt=0)
    rope_freq_scale: float | None = Field(default=None, gt=0)
    yarn_ext_factor: float | None = Field(default=None, gt=0)
    swa_full: bool | None = None
    kv_type_k: str | None = None
    kv_type_v: str | None = None
    active_loras: tuple[str, ...] = ()

    @field_validator("backend")
    @classmethod
    def _backend_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("backend must be non-empty")
        return value


class GenerationRequest(_BoundaryModel):
    """A validated generation request before it enters the token hot path."""

    prompt: str
    max_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    top_k: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stop: tuple[str, ...] = ()
    seed: int | None = None
    grammar_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(_BoundaryModel):
    """A completed generation with enough facts to reproduce its route."""

    text: str
    token_ids: tuple[int, ...] = ()
    prompt_token_count: int = Field(ge=0)
    completion_token_count: int = Field(ge=0)
    finish_reason: str
    request_id: str | None = None
    validated: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
