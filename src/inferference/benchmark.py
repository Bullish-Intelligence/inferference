"""Small, local benchmark records shared by the llama.cpp teaching pillars.

The harness deliberately measures named portions of an owned decode loop rather
than attempting to hide them behind one throughput number.  It has no GPU or
llama-cpp-python dependency, so examples and unit tests can exercise artifact
production on any development machine.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from time import perf_counter_ns
from typing import Any
from uuid import uuid4

TIMING_FIELDS = (
    "tokenizer_preparation",
    "grammar_compile",
    "matcher_creation",
    "prefill_enqueue",
    "prefill_wall",
    "generation_wall",
    "candidate_array",
    "mask_creation",
    "mask_application",
    "sampler_apply",
    "sampler_accept",
    "matcher_accept",
    "detokenize",
    "validation",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """A portable, JSON-serializable measurement from one local generation."""

    run_id: str
    started_at: str
    path: str
    prompt_tokens: int
    completion_tokens: int
    timings_ns: Mapping[str, int]
    token_latencies_ns: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        invalid = set(self.timings_ns).difference(TIMING_FIELDS)
        if invalid:
            raise ValueError(f"unknown timing field(s): {sorted(invalid)}")
        if any(value < 0 for value in self.timings_ns.values()) or any(value < 0 for value in self.token_latencies_ns):
            raise ValueError("durations must be non-negative")

    @property
    def total_ns(self) -> int:
        return sum(self.timings_ns.values())

    @property
    def prefill_tokens_per_second(self) -> float | None:
        duration = self.timings_ns.get("prefill_wall", 0)
        return self.prompt_tokens * 1_000_000_000 / duration if duration and self.prompt_tokens else None

    @property
    def decode_tokens_per_second(self) -> float | None:
        duration = self.timings_ns.get("generation_wall", 0)
        return self.completion_tokens * 1_000_000_000 / duration if duration and self.completion_tokens else None

    @property
    def ttft_ns(self) -> int | None:
        """Time to first emitted token, when token latency was observed."""
        if not self.token_latencies_ns:
            return None
        return self.timings_ns.get("prefill_wall", 0) + self.token_latencies_ns[0]

    def to_dict(self) -> dict[str, Any]:
        latency = sorted(self.token_latencies_ns)

        def percentile(fraction: float) -> int | None:
            if not latency:
                return None
            return latency[min(len(latency) - 1, ceil(len(latency) * fraction) - 1)]

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "path": self.path,
            "tokens": {"prompt": self.prompt_tokens, "completion": self.completion_tokens},
            "timings_ns": {name: self.timings_ns.get(name, 0) for name in TIMING_FIELDS},
            "token_latencies_ns": list(self.token_latencies_ns),
            "metrics": {
                "total_ns": self.total_ns,
                "prefill_tokens_per_second": self.prefill_tokens_per_second,
                "decode_tokens_per_second": self.decode_tokens_per_second,
                "ttft_ns": self.ttft_ns,
                "token_latency_p50_ns": percentile(0.50),
                "token_latency_p95_ns": percentile(0.95),
            },
            "metadata": dict(self.metadata),
        }


class BenchmarkTimer:
    """Accumulate nanosecond timings from a single generation without hot-path models."""

    def __init__(self, path: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.path = path
        self.metadata = dict(metadata or {})
        self.started_at = _utc_now()
        self._timings_ns = dict.fromkeys(TIMING_FIELDS, 0)
        self._token_latencies_ns: list[int] = []

    @contextmanager
    def measure(self, field: str) -> Iterator[None]:
        if field not in self._timings_ns:
            raise ValueError(f"unknown timing field: {field}")
        started = perf_counter_ns()
        try:
            yield
        finally:
            self._timings_ns[field] += perf_counter_ns() - started

    def add_ns(self, field: str, duration_ns: int) -> None:
        if field not in self._timings_ns:
            raise ValueError(f"unknown timing field: {field}")
        if duration_ns < 0:
            raise ValueError("duration_ns must be non-negative")
        self._timings_ns[field] += duration_ns

    def record_token_latency_ns(self, duration_ns: int) -> None:
        if duration_ns < 0:
            raise ValueError("duration_ns must be non-negative")
        self._token_latencies_ns.append(duration_ns)

    def record(self, *, prompt_tokens: int, completion_tokens: int) -> BenchmarkRecord:
        return BenchmarkRecord(
            run_id=uuid4().hex,
            started_at=self.started_at,
            path=self.path,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            timings_ns=dict(self._timings_ns),
            token_latencies_ns=tuple(self._token_latencies_ns),
            metadata=self.metadata,
        )


def write_benchmark_record(record: BenchmarkRecord, directory: Path) -> Path:
    """Atomically write one structured artifact and return its final path."""
    directory.mkdir(parents=True, exist_ok=True)
    safe_path = re.sub(r"[^a-zA-Z0-9_.-]+", "-", record.path).strip("-") or "generation"
    target = directory / f"{record.started_at.replace(':', '').replace('-', '')}-{safe_path}-{record.run_id}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target
