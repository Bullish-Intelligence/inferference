"""The token-exact validation gates.

Fail-closed, batch-1 ground truth, with known-flip reporting. This module holds
the gate-relevant machinery (``_run_once`` / ``aggregate_outcomes`` /
``_model_for_schema`` / ``_resolve_tokenizer`` / ``_synchronize_gpu``), lifted
from an earlier soak harness onto the library with its import coupling removed:
the library is imported as a package, not via a script directory or bare local
modules.

The gate loop runs the owned decoder (``inferference.decode.OwnedLlamaDecoder``)
with an optional ``JsonSchemaGrammar`` matcher, classifies every request
(valid / malformed / rejected / runtime_error / cutoff) fail-closed, and
aggregates them into one outcome shape.
"""

from __future__ import annotations

import json
import statistics
import traceback
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from inferference.benchmark import TIMING_FIELDS, BenchmarkRecord, BenchmarkTimer
from inferference.decode import FINISH_STOP, OwnedLlamaDecoder
from inferference.grammar import JsonSchemaGrammar

FAILURE_STATUSES = frozenset({"malformed", "rejected", "runtime_error", "error"})


class CapitalAnswer(BaseModel):
    """The small deterministic JSON contract used by the default smoke."""

    city: str
    country: str


DEFAULT_SCHEMA = CapitalAnswer.model_json_schema()


def _percentiles(values: list[int]) -> dict[str, float | None]:
    """Deterministic nearest-rank percentiles without NumPy."""
    if not values:
        return {"p50_ns": None, "p95_ns": None, "mean_ns": None, "total_ns": 0.0}
    ordered = sorted(values)
    return {
        "p50_ns": float(statistics.median(ordered)),
        "p95_ns": float(ordered[min(len(ordered) - 1, (len(ordered) * 95 + 99) // 100 - 1)]),
        "mean_ns": statistics.fmean(ordered),
        "total_ns": float(sum(ordered)),
    }


def aggregate_outcomes(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate request outcomes; intentionally GPU- and llama-free.

    ``invalid_count`` excludes cleanly identified cutoffs so an interrupted
    completion cannot disappear into a generic validation-failure bucket. A
    cutoff is a structured-correctness failure, but it is still a completed
    performance observation and must not invalidate throughput aggregates.
    """
    counts: dict[str, int] = {}
    phase_values = {field: [] for field in TIMING_FIELDS}
    wall_values = {"generation": [], "request": []}
    completion_tokens = 0
    prompt_tokens = 0
    mask_values: list[int] = []

    for outcome in outcomes:
        status = str(outcome["status"])
        counts[status] = counts.get(status, 0) + 1
        prompt_tokens += int(outcome.get("prompt_tokens", 0))
        completion_tokens += int(outcome.get("completion_tokens", 0))
        timings = outcome.get("timings_ns", {})
        for field in TIMING_FIELDS:
            phase_values[field].append(int(timings.get(field, 0)))
        mask_values.append(int(timings.get("mask_creation", 0)) + int(timings.get("mask_application", 0)))
        wall_timings = outcome.get("wall_timings_ns", {})
        for field in wall_values:
            wall_values[field].append(int(wall_timings.get(field, 0)))

    valid_count = counts.get("valid", 0)
    cutoff_count = counts.get("cutoff", 0)
    invalid_count = sum(count for status, count in counts.items() if status in FAILURE_STATUSES)
    mask_total_ns = sum(mask_values)
    generation_total_ns = sum(wall_values["generation"])
    request_total_ns = sum(wall_values["request"])
    return {
        "request_count": len(outcomes),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "cutoff_count": cutoff_count,
        "hard_failure_count": invalid_count,
        "failure_count": len(outcomes) - valid_count,
        "performance_result": "completed",
        "structured_correctness_result": "pass" if not invalid_count and not cutoff_count else "incomplete",
        "status_counts": counts,
        "token_counts": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "phase_timings_ns": {field: _percentiles(values) for field, values in phase_values.items()},
        "wall_timings_ns": {field: _percentiles(values) for field, values in wall_values.items()},
        "end_to_end": {
            "generation_tokens_per_second": (
                completion_tokens * 1_000_000_000 / generation_total_ns
                if generation_total_ns and completion_tokens
                else None
            ),
            "request_tokens_per_second": (
                completion_tokens * 1_000_000_000 / request_total_ns if request_total_ns and completion_tokens else None
            ),
            "generation_ns_per_completion_token": (
                generation_total_ns / completion_tokens if generation_total_ns and completion_tokens else None
            ),
            "request_ns_per_completion_token": (
                request_total_ns / completion_tokens if request_total_ns and completion_tokens else None
            ),
        },
        "mask_overhead": {
            **_percentiles(mask_values),
            "per_completion_token_ns": (mask_total_ns / completion_tokens if completion_tokens else None),
            "percent_of_generation_wall_ns": (
                mask_total_ns * 100 / generation_total_ns if generation_total_ns else None
            ),
        },
    }


def _synchronize_ctx(llm: Any) -> None:
    """Wait for THIS context's CUDA work before recording an outer wall duration.

    Uses ``llama_synchronize(ctx)`` (llama.h:1017 on the anchor) — the
    context-correct sync, stricter than ``torch.cuda.synchronize()`` (which
    waits on the default stream, not llama.cpp's compute stream). This is the "cudaEvent-based sync" the bench
    spec demands: hard-required, fail-closed — an absent
    symbol is a loud error, never a silent optional-sync. (CPU-only torch stays
    in the library venv only because xgrammar 0.2.1 imports it; it is not used
    for timing.)
    """
    import llama_cpp

    native = llama_cpp.llama_cpp
    if not hasattr(native, "llama_synchronize") or not hasattr(native._lib, "llama_synchronize"):
        raise RuntimeError("llama_synchronize absent on the loaded lib — cannot record wall durations (fail-closed)")
    native.llama_synchronize(llm._ctx.ctx)


def _model_for_schema(schema: dict[str, Any]) -> type[BaseModel]:
    """Build a small Pydantic object validator for a user-supplied JSON schema.

    Ported from an earlier soak-grammar script ``_model_for_schema``. The grammar
    can express more than this boundary validator; the CLI stays explicit about
    the supported (primitive property) subset.
    """
    if schema == DEFAULT_SCHEMA:
        return CapitalAnswer
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise ValueError("--schema-json/--schema-file must be an object schema with primitive properties")
    type_map: dict[str, type[Any]] = {"string": str, "integer": int, "number": float, "boolean": bool}
    required = set(schema.get("required", []))
    fields: dict[str, tuple[type[Any], Any]] = {}
    for name, property_schema in schema["properties"].items():
        json_type = property_schema.get("type")
        python_type = type_map.get(json_type)
        if python_type is None:
            raise ValueError(f"schema property {name!r} has unsupported type {json_type!r}")
        fields[name] = (python_type, ... if name in required else None)
    return create_model("SoakSchema", __config__=ConfigDict(extra="forbid"), **fields)


def _resolve_tokenizer(tokenizer: str, *, allow_network: bool) -> str:
    """Resolve a repository ID to a local snapshot before Transformers loads it.

    Ported from an earlier soak-grammar script ``_resolve_tokenizer``: keeps a
    soak reproducible offline (transformers can hit the network for tokenizer
    metadata even when files are cached). Also accepts a huggingface hub-cache
    REPO dir (the ``models--owner--name`` directory with ``snapshots/``): the
    pinned revision (``refs/main``) is resolved to its snapshot, so the
    ``TOKENIZER_DEFAULT`` var can point at the hub cache root as it sits on disk.
    """
    if Path(tokenizer).exists():
        repo_dir = Path(tokenizer)
        if (repo_dir / "snapshots").is_dir():
            refs = repo_dir / "refs"
            rev_file = refs / "main"
            if rev_file.is_file():
                rev = rev_file.read_text().strip()
                snapshot = repo_dir / "snapshots" / rev
                if snapshot.is_dir():
                    return str(snapshot)
        return tokenizer
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=tokenizer,
        allow_patterns=["config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "*.jinja"],
        local_files_only=not allow_network,
    )


_SPECIAL_TOKEN_KEYS = ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token")


def _special_token_text(value: Any) -> str | None:
    """The text of a special token as ``tokenizer_config.json`` stores it: a string or an AddedToken dict."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def load_tokenizer(tokenizer_dir: str) -> Any:
    """Load a local tokenizer directory for grammar work, for any model whose ``tokenizer.json`` is present.

    ``AutoTokenizer`` is the first choice. It fails on a directory whose
    ``tokenizer_config.json`` is newer than the installed transformers: version 5
    writes ``extra_special_tokens`` as a list, and version 4 reads it as a
    mapping (``'list' object has no attribute 'keys'``). The grammar compiler
    needs only the vocabulary and the end-of-sequence token. So on that failure,
    load ``tokenizer.json`` with ``PreTrainedTokenizerFast`` and take the special
    tokens from the config file, in either format.

    Fail-closed: when the directory has no ``tokenizer.json``, or the config names
    no end-of-sequence token (a grammar could then never stop), raise. The first
    error stays attached as the cause.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: PLC0415

    try:
        return AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    except Exception as first:
        root = Path(tokenizer_dir)
        tokenizer_file = root / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise
        try:
            config = json.loads((root / "tokenizer_config.json").read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot read tokenizer_config.json in {root}") from exc
        kwargs: dict[str, Any] = {}
        for key in _SPECIAL_TOKEN_KEYS:
            text = _special_token_text(config.get(key))
            if text is not None:
                kwargs[key] = text
        extra = config.get("extra_special_tokens") or config.get("additional_special_tokens") or []
        extra_values = extra.values() if isinstance(extra, dict) else extra
        additional = [t for t in (_special_token_text(v) for v in extra_values) if t is not None]
        if additional:
            kwargs["additional_special_tokens"] = additional
        if "eos_token" not in kwargs:
            raise RuntimeError(
                f"{root}/tokenizer_config.json names no eos_token; a grammar could never stop"
            ) from first
        return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), **kwargs)


def run_once(
    *,
    llm: Any,
    grammar: JsonSchemaGrammar | None,
    validator: type[BaseModel],
    prompt: str,
    prompt_tokens_count: int,
    request_index: int,
    max_tokens: int,
    constrained: bool,
    synchronize_gpu: bool,
) -> tuple[dict[str, Any], BenchmarkRecord]:
    """Run and classify one request, preserving the benchmark even on failure.

    Ported from an earlier soak-grammar script ``_run_once`` (the gate semantics —
    fail-closed classification, matcher never shared across requests, ``stop``
    is the only clean end). Returns ``(outcome, record)``.
    """
    benchmark = BenchmarkTimer(
        "grammar-soak",
        metadata={"request_index": request_index, "constrained": constrained, "vocab_size": llm.n_vocab()},
    )
    outcome: dict[str, Any] = {
        "request_index": request_index,
        "status": "error",
        "detail": None,
        "prompt_tokens": prompt_tokens_count,
        "completion_tokens": 0,
        "finish_reason": None,
    }
    request_started_ns = perf_counter_ns()
    generation_started_ns: int | None = None
    try:
        logits_hook = None
        token_hook = None
        if constrained:
            assert grammar is not None
            with benchmark.measure("matcher_creation"):
                matcher = grammar.new_matcher()  # Never share state across requests.
            logits_hook = grammar.logits_hook(matcher, benchmark=benchmark)
            token_hook = grammar.token_hook(matcher)
        generation_started_ns = perf_counter_ns()
        with OwnedLlamaDecoder(llm) as decoder:
            generated = decoder.generate_text(
                prompt,
                max_tokens=max_tokens,
                logits_hook=logits_hook,
                token_hook=token_hook,
                benchmark=benchmark,
            )
        if synchronize_gpu:
            _synchronize_ctx(llm)
        outcome["wall_timings_ns"] = {"generation": perf_counter_ns() - generation_started_ns}
        outcome["finish_reason"] = generated.finish_reason
        outcome["completion_tokens"] = generated.completion_token_count
        outcome["text"] = generated.text
        if generated.finish_reason != FINISH_STOP:
            outcome["status"] = "cutoff"
            outcome["detail"] = f"finish_reason={generated.finish_reason!r}"
        else:
            with benchmark.measure("validation"):
                validator.model_validate_json(generated.text)
            outcome["status"] = "valid"
    except ValidationError as exc:
        outcome["status"] = "malformed"
        outcome["detail"] = str(exc)
    except RuntimeError as exc:
        outcome["status"] = "rejected" if "rejected" in str(exc).lower() else "runtime_error"
        outcome["detail"] = str(exc)
    except Exception as exc:  # noqa: BLE001 -- a gate must account for every request.
        outcome["detail"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    wall_timings = outcome.setdefault("wall_timings_ns", {})
    if generation_started_ns is not None and "generation" not in wall_timings:
        wall_timings["generation"] = perf_counter_ns() - generation_started_ns
    wall_timings["request"] = perf_counter_ns() - request_started_ns
    record = benchmark.record(prompt_tokens=prompt_tokens_count, completion_tokens=outcome["completion_tokens"])
    outcome["timings_ns"] = dict(record.timings_ns)
    return outcome, record
