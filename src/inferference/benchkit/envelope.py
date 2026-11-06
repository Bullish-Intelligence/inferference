"""RunEnvelope v2 — the canonical run contract.

Records v1 (``benchmark.py``) stay the per-request artifact contract. The
*directory* contract evolves to envelope v2: one ``envelope.json`` per run dir
carrying the run identity, the workload, the full environment fingerprint
(LlamaEngineFingerprint + nvidia-smi + env), the baseline reference, the gate
verdict, a canonicalized summary, and the records dir. It replaces the ad-hoc
per-runner ``summary.json`` shapes and names ``git-status-before`` consistently
with what the run dir actually records.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GateKind = Literal["validation", "perf", "soak", "smoke"]
GateVerdict = Literal["pass", "fail", "skip"]


class RunEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 2
    run_id: str
    workload: dict[str, Any]
    environment: dict[str, Any]
    baseline_ref: str | None = None
    gate: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    records: str = "records/"
    command: list[str] = Field(default_factory=list)
