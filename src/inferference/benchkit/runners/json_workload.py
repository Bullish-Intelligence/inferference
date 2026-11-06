"""GPU-only sequential benchmark runner — the json_workload perf baseline.

Adapted from an earlier json-workload harness, with the import coupling removed:
it imports the library as a package and the benchkit gates/registry/env
directly — no script-directory reach-in, no bare local ``workload`` import.

Preserved semantics:

  * GPU-0-only isolation (``require_gpu0_only``) before and after model load;
  * CUDA_VISIBLE_DEVICES=0 + LLAMA_CPP_LIB_PATH hard requirement at entry;
  * records v1 via ``write_benchmark_record`` + the aggregate summary shape
    (``aggregate_outcomes``);
  * torch sync is hard-required;
  * artifacts must land under an ``artifacts/project17-*``-style ignored dir.

Added: envelope v2 (``envelope.json``) per run dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from inferference.benchmark import write_benchmark_record
from inferference.diagnostics import collect_runtime_diagnostics
from inferference.fingerprint import LlamaEngineFingerprint, register_artifact
from inferference.grammar import JsonSchemaGrammar

from ..env import driver_version, gpu_snapshot, require_ci_device, require_gpu0_only
from ..envelope import RunEnvelope
from ..gates import _model_for_schema, _resolve_tokenizer, aggregate_outcomes, run_once
from ..runenv import RunContext
from ..workload import load as load_workload

REPO = Path(__file__).resolve().parents[3]


def _require_artifacts_dir(directory: Path) -> None:
    resolved = directory.resolve()
    if resolved.parent.name != "artifacts" or not resolved.name.startswith("project17-"):
        raise ValueError("--artifacts must be an artifacts/project17-* directory")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--corpus", default="json_workload", help="workload name (registry)")
    parser.add_argument("--schema-registry", type=Path, default=None)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--same-grammar-repeated", action="store_true")
    parser.add_argument("--tokenizer", default=os.environ.get("TOKENIZER_DEFAULT"))
    parser.add_argument("--seed", type=int, default=17001)
    parser.add_argument("--n-ctx", type=int, default=512)
    args = parser.parse_args()

    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    # ---- workload (registry, sha256-verified at load) -------------------------
    workload = load_workload(args.corpus)
    registry = workload.schema_registry
    selected = workload.select(args.requests)

    # ---- run dir (envelope v2 + v1 artifacts) ---------------------------------
    _require_artifacts_dir(args.artifacts)
    run = RunContext(args.artifacts, name="project17-json-workload", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.log(f">> bench run {run.run_id}: workload={workload.name}@{workload.version} requests={args.requests}")

    gpu_before = gpu_snapshot()
    (run.dir / "gpu-before.json").write_text(json.dumps(gpu_before, indent=2) + "\n")

    from llama_cpp import Llama  # noqa: PLC0415 -- native import after env checks

    load_started = perf_counter_ns()
    llm = Llama(
        model_path=str(args.model), n_ctx=args.n_ctx, n_batch=128, n_gpu_layers=-1, seed=args.seed, verbose=False
    )
    # Context-correct CUDA sync for the model-load wall duration (see
    # benchkit/gates._synchronize_ctx — torch is CPU-only in this venv).
    import llama_cpp  # noqa: PLC0415

    llama_cpp.llama_cpp.llama_synchronize(llm._ctx.ctx)
    load_wall_ns = perf_counter_ns() - load_started
    after_load = gpu_snapshot()
    (run.dir / "gpu-after-load.json").write_text(json.dumps(after_load, indent=2) + "\n")
    require_gpu0_only(after_load)

    tokenizer_path = _resolve_tokenizer(args.tokenizer, allow_network=False)
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    grammars: dict[str, JsonSchemaGrammar] = {}
    compile_ns: dict[str, int] = {}
    if not args.baseline_only:
        for schema_id in sorted({str(item["schema_id"]) for item in selected}):
            schema = registry["schemas"][schema_id]
            started = perf_counter_ns()
            grammars[schema_id] = JsonSchemaGrammar.from_huggingface(tokenizer, schema, vocab_size=llm.n_vocab())
            compile_ns[schema_id] = perf_counter_ns() - started

    outcomes: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        schema_id = selected[0]["schema_id"] if args.same_grammar_repeated else item["schema_id"]
        started = perf_counter_ns()
        prompt_tokens = llm.tokenize(item["prompt"].encode("utf-8"), add_bos=False, special=True)
        prep_ns = perf_counter_ns() - started
        validator: type[Any] = _model_for_schema(registry["schemas"][schema_id])
        outcome, record = run_once(
            llm=llm,
            grammar=None if args.baseline_only else grammars[schema_id],
            validator=validator,
            prompt=item["prompt"],
            prompt_tokens_count=len(prompt_tokens),
            request_index=index,
            max_tokens=item["max_tokens"],
            constrained=not args.baseline_only,
            synchronize_gpu=True,
        )
        timings = dict(record.timings_ns)
        timings["tokenizer_preparation"] = prep_ns
        timings["grammar_compile"] = (
            compile_ns.get(schema_id, 0) if schema_id not in {o.get("schema_id") for o in outcomes} else 0
        )
        outcome.update(
            {
                "id": item["id"],
                "category": item["category"],
                "schema_id": schema_id,
                "input_metadata": {
                    "utf8_bytes": len(item["prompt"].encode("utf-8")),
                    "utf8_chars": len(item["prompt"]),
                    "prompt_tokens": len(prompt_tokens),
                },
            }
        )
        outcome["timings_ns"] = timings
        outcomes.append(outcome)
        write_benchmark_record(
            replace(record, timings_ns=timings, metadata={**record.metadata, **outcome}), run.dir / "records"
        )
        run.log(f"{index + 1}/{len(selected)} {outcome['status']} {item['id']}")

    summary = aggregate_outcomes(outcomes)
    runtime = collect_runtime_diagnostics().model_dump()
    summary.update(
        {
            "corpus": workload.name,
            "corpus_version": workload.version,
            "corpus_sha256_1000": workload.corpus_sha256("json_workload_1000.jsonl"),
            "runtime": runtime,
            "model_load_and_gpu_warmup_ns": load_wall_ns,
            "grammar_compile_ns_by_schema": compile_ns,
            "mode": "baseline"
            if args.baseline_only
            else ("same-grammar-repeated" if args.same_grammar_repeated else "corpus"),
            "gpu_after_load": after_load,
            "gpu_after_run": gpu_snapshot(),
            "async_enqueue_warning": (
                "llama_decode enqueue timings are deliberately not reported as throughput; "
                "rates use synchronized wall durations."
            ),
        }
    )
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # ---- envelope v2 ----------------------------------------------------------
    try:
        tokenizer_identity = register_artifact(Path(tokenizer_path) / "tokenizer.json")
        fingerprint = LlamaEngineFingerprint(
            model=register_artifact(args.model),
            tokenizer=tokenizer_identity,
            llama_cpp_python_version=runtime.get("llama_cpp_python_version") or "",
            llama_cpp_commit=runtime.get("llama_cpp_commit"),
            llama_cpp_build_id=runtime.get("llama_cpp_build_id"),
            seq_adapter_routing=bool(runtime.get("seq_adapter_routing")),
            backend="cuda",
            n_ctx=args.n_ctx,
        )
        fingerprint_key = fingerprint.cache_key()
    except Exception:  # noqa: BLE001 -- the envelope must not fail the run
        fingerprint_key = None
    envelope = RunEnvelope(
        run_id=run.run_id,
        workload={
            "name": workload.name,
            "version": workload.version,
            "corpus_sha256_1000": workload.corpus_sha256("json_workload_1000.jsonl"),
            "requests": args.requests,
        },
        environment={
            "fingerprint": fingerprint_key,
            "nvidia_smi": {
                "driver_version": driver_version(after_load),
                "gpu_after_load": after_load.get("stdout"),
            },
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "LLAMA_CPP_LIB_PATH": os.environ.get("LLAMA_CPP_LIB_PATH"),
            },
        },
        gate={"kind": "perf", "verdict": "pass" if summary["hard_failure_count"] == 0 else "fail"},
        summary={
            "generation_tokens_per_second": summary["end_to_end"]["generation_tokens_per_second"],
            "request_tokens_per_second": summary["end_to_end"]["request_tokens_per_second"],
            "hard_failure_count": summary["hard_failure_count"],
            "status_counts": summary["status_counts"],
        },
        command=sys.argv,
    )
    run.write_envelope(envelope)
    run.write_runtime_environment({"fingerprint": fingerprint_key})

    # Length cutoffs are retained as structured-correctness evidence but are
    # valid observed samples for the token-normalized performance baseline.
    return 0 if summary["hard_failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
