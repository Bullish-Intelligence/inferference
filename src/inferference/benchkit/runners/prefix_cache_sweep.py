"""Exact-prefix whole-state cache break-even sweep (run_prefix_cache sweep half).

Perf half of the dual runner. Port of
an earlier prefix-cache harness onto the library package: the
whole-state ``Llama.save_state``/``load_state`` path through
:class:`inferference.prefix_cache.PersistentPrefixCache`, with the CUDA sync
moved to ``llama_synchronize(ctx)`` (11-report Task-4 deviation 2) and the
run envelope via :class:`inferference.benchkit.runenv.RunContext`.

For each checkpoint length K: capture a persistent whole-state blob, restart a
fresh context, restore + decode the 1-token suffix, then measure (best-of-reps,
CUDA-synchronized) cold prefill vs the cache path phases. The continuation
token must match cold (fail-closed) or the run fails.

GPU-only: CUDA_VISIBLE_DEVICES=0, n_gpu_layers=-1, LLAMA_CPP_LIB_PATH pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from inferference.diagnostics import collect_runtime_diagnostics
from inferference.fingerprint import LlamaEngineFingerprint, register_artifact
from inferference.prefix_cache import PersistentPrefixCache, PrefixCacheKey

from ..env import gpu_snapshot, require_ci_device, require_gpu0_only
from ..runenv import RunContext

REPO = Path(__file__).resolve().parents[3]


def percentiles(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "total_ns": sum(values),
        "mean_ns": sum(values) / len(values),
        "p50_ns": ordered[(len(values) - 1) // 2],
        "p95_ns": ordered[min(len(values) - 1, int(np.ceil(len(values) * 0.95)) - 1)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument(
        "--cache-root", required=True, type=Path, help="PersistentPrefixCache root (the durable state store)"
    )
    parser.add_argument("--prefix-lengths", default="16,32,64,96")
    parser.add_argument("--repetitions", default=3, type=int)
    parser.add_argument("--n-ctx", default=512, type=int)
    parser.add_argument("--seed", default=17018, type=int)
    args = parser.parse_args()
    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    from llama_cpp import Llama, llama_cpp  # noqa: PLC0415

    gpu_before = gpu_snapshot()
    run = RunContext(args.artifacts, name="project17-prefix-cache-sweep", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()

    def mk() -> Any:
        return Llama(
            model_path=str(args.model), n_ctx=args.n_ctx, n_batch=128, n_gpu_layers=-1, seed=args.seed, verbose=False
        )

    def sync(llm: Any) -> None:
        llama_cpp.llama_synchronize(llm._ctx.ctx)

    def eval_tokens(model: Any, tokens: tuple[int, ...]) -> None:
        model.eval(list(tokens))

    def token(model: Any) -> int:
        return int(np.argmax(np.ctypeslib.as_array(model._ctx.get_logits(), shape=(model._n_vocab,))))

    llm = mk()
    after_load = gpu_snapshot()
    require_gpu0_only(after_load)
    runtime = collect_runtime_diagnostics()
    identity = register_artifact(args.model, metadata={"role": "model-and-gguf-tokenizer"})
    fingerprint = LlamaEngineFingerprint(
        model=identity,
        tokenizer=identity,
        llama_cpp_python_version=runtime.llama_cpp_python_version or "unknown",
        llama_cpp_commit=runtime.llama_cpp_commit,
        llama_cpp_build_id=runtime.llama_cpp_build_id,
        backend="cuda",
        n_ctx=args.n_ctx,
    )
    cache = PersistentPrefixCache(args.cache_root)
    tokens = tuple(llm.tokenize(b"Persistent exact prefix cache workload. " * 100, add_bos=False, special=True))
    records: list[dict[str, Any]] = []
    try:
        for length in (int(value) for value in args.prefix_lengths.split(",")):
            prefix, suffix = tokens[:length], (tokens[length],)
            key = PrefixCacheKey.from_fingerprint(
                namespace="prefix-cache-whole-state", fingerprint=fingerprint, prefix_token_ids=prefix
            )
            llm.reset()
            eval_tokens(llm, prefix)
            sync(llm)
            entry = cache.publish(
                key,
                pickle.dumps(llm.save_state(), protocol=pickle.HIGHEST_PROTOCOL),
                llama_state_version="llama-cpp-python-LlamaState-v1",
                runtime_facts=runtime.model_dump(mode="json", exclude_none=True),
            )
            # The 12 GiB 3060 cannot host two fully offloaded contexts at once.
            # Close the producer before the fresh context; this is also the
            # process/context-restart lifecycle the cache must prove.
            llm.close()
            restart = mk()
            hit = cache.lookup(key)
            if not hit.hit or hit.state is None:
                raise RuntimeError(f"cache setup failed: {hit.reason}")
            restart.load_state(pickle.loads(hit.state))
            eval_tokens(restart, suffix)
            sync(restart)
            restart_token = token(restart)
            llm = restart
            timings: dict[str, list[int]] = {
                name: []
                for name in (
                    "cold_prefill_wall",
                    "cache_lookup_wall",
                    "state_read_wall",
                    "state_restore_wall",
                    "suffix_prefill_wall",
                    "cache_end_to_end_wall",
                )
            }
            cold, restored = [], []
            for _ in range(args.repetitions):
                llm.reset()
                started = perf_counter_ns()
                eval_tokens(llm, prefix + suffix)
                sync(llm)
                timings["cold_prefill_wall"].append(perf_counter_ns() - started)
                cold.append(token(llm))
                started = perf_counter_ns()
                indexed, reason = cache.lookup_entry(key)
                timings["cache_lookup_wall"].append(perf_counter_ns() - started)
                if indexed is None:
                    raise RuntimeError(reason)
                request_started = perf_counter_ns()
                started = perf_counter_ns()
                loaded = cache.read_entry(indexed)
                timings["state_read_wall"].append(perf_counter_ns() - started)
                if not loaded.hit or loaded.state is None:
                    raise RuntimeError(loaded.reason)
                started = perf_counter_ns()
                llm.load_state(pickle.loads(loaded.state))
                sync(llm)
                timings["state_restore_wall"].append(perf_counter_ns() - started)
                started = perf_counter_ns()
                eval_tokens(llm, suffix)
                sync(llm)
                timings["suffix_prefill_wall"].append(perf_counter_ns() - started)
                timings["cache_end_to_end_wall"].append(perf_counter_ns() - request_started)
                restored.append(token(llm))
            if cold != restored or cold[0] != restart_token:
                raise RuntimeError(f"continuation mismatch after restore at K={length}")
            phases = {name: percentiles(values) for name, values in timings.items()}
            record = {
                "prefix_tokens": length,
                "suffix_tokens": 1,
                "generated_tokens": 1,
                "cache_hit": True,
                "cache_reason": "hit",
                "state_blob_bytes": entry.state_size_bytes,
                "state_sha256": entry.state_checksum_sha256,
                "continuation_token": cold[0],
                "phases": phases,
                "break_even_ns": phases["cold_prefill_wall"]["mean_ns"] - phases["cache_end_to_end_wall"]["mean_ns"],
                "completion_tokens_per_second": 1e9 / phases["cache_end_to_end_wall"]["mean_ns"],
            }
            records.append(record)
            run.log(
                f"[sweep] K={length}: cold={phases['cold_prefill_wall']['mean_ns'] / 1e6:.1f}ms "
                f"e2e-cache={phases['cache_end_to_end_wall']['mean_ns'] / 1e6:.1f}ms "
                f"beats_cold={record['break_even_ns'] > 0}"
            )
    finally:
        llm.close()

    summary = {
        "runtime": runtime.model_dump(),
        "fingerprint": fingerprint.model_dump(mode="json"),
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "records": records,
        "note": "llama_synchronize-synced. Validation half (semantic restore gates) lives in suites/prefix-cache.",
        "async_enqueue_warning": "llama_decode enqueue timing is not throughput; reported phases synchronize CUDA.",
        "retention": "manual MVP retention; no eviction",
    }
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": records}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
