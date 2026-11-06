"""Decomposition of whole-state restore cost: LlamaState vs native codec.

Port of an earlier state-decomposition harness onto the
library package. Separates pickle deserialization, the ``LlamaState`` score-array
apply, and the native ``llama_state_set_data`` host->device restore; also
measures a native-bytes codec that bypasses the ``LlamaState`` wrapper (no score
buffer). Both codecs restore the SAME native llama.cpp context state, so
continuation correctness is a hard, fail-closed gate: any restored token that
differs from the cold baseline raises. CUDA sync via ``llama_synchronize``.

GPU-only: CUDA_VISIBLE_DEVICES=0, n_gpu_layers=-1, LLAMA_CPP_LIB_PATH pinned.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
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

from ..env import gpu_snapshot, require_ci_device, require_gpu0_only
from ..runenv import RunContext

REPO = Path(__file__).resolve().parents[3]


def stats(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "mean_ns": sum(values) / len(values),
        "p50_ns": ordered[(len(values) - 1) // 2],
        "p95_ns": ordered[min(len(values) - 1, int(np.ceil(len(values) * 0.95)) - 1)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--prefix-lengths", default="16,64,128,256")
    parser.add_argument("--repetitions", default=3, type=int)
    parser.add_argument("--n-ctx", default=512, type=int)
    parser.add_argument("--n-batch", default=128, type=int)
    parser.add_argument("--seed", default=17018, type=int)
    args = parser.parse_args()
    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    from llama_cpp import Llama, llama_cpp  # noqa: PLC0415

    gpu_before = gpu_snapshot()
    run = RunContext(args.artifacts, name="project17-native-state-decompose", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()

    def mk() -> Any:
        return Llama(
            model_path=str(args.model),
            n_ctx=args.n_ctx,
            n_batch=args.n_batch,
            n_gpu_layers=-1,
            seed=args.seed,
            verbose=False,
        )

    def sync(model: Any) -> None:
        llama_cpp.llama_synchronize(model._ctx.ctx)

    def token(model: Any) -> int:
        return int(np.argmax(np.ctypeslib.as_array(model._ctx.get_logits(), shape=(model._n_vocab,))))

    def native_get(model: Any) -> bytes:
        ctx = model._ctx.ctx
        size = int(llama_cpp.llama_state_get_size(ctx))
        buf = (ctypes.c_uint8 * size)()
        copied = int(llama_cpp.llama_state_get_data(ctx, buf, size))
        return bytes(buf[:copied])

    def native_set(model: Any, blob: bytes, prefix: tuple[int, ...]) -> None:
        # Restore native context state, then replay the Python-side n_past/input
        # bookkeeping that Llama.load_state normally performs. Without this,
        # Llama.eval() would kv_cache_seq_rm from position 0 and wipe the restore.
        arr = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        loaded = int(llama_cpp.llama_state_set_data(model._ctx.ctx, arr, len(blob)))
        if loaded != len(blob):
            raise RuntimeError(f"native llama_state_set_data returned {loaded}, expected {len(blob)}")
        model.n_tokens = len(prefix)
        model.input_ids[: len(prefix)] = np.array(prefix, dtype=np.intc)

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
    tokens = tuple(llm.tokenize(b"Persistent exact prefix cache workload. " * 100, add_bos=False, special=True))
    records: list[dict[str, Any]] = []
    try:
        for length in (int(v) for v in args.prefix_lengths.split(",")):
            prefix, suffix = tokens[:length], (tokens[length],)

            # Capture both codecs from the same checkpoint.
            llm.reset()
            llm.eval(list(prefix))
            sync(llm)
            pickle_blob = pickle.dumps(llm.save_state(), protocol=pickle.HIGHEST_PROTOCOL)
            native_blob = native_get(llm)
            pickle_sha = hashlib.sha256(pickle_blob).hexdigest()
            native_sha = hashlib.sha256(native_blob).hexdigest()

            # Cold baseline (fresh prefill of prefix+suffix) establishes truth.
            llm.reset()
            llm.eval(list(prefix + suffix))
            sync(llm)
            cold_token = token(llm)

            timings: dict[str, list[int]] = {
                "cold_prefill_wall": [],
                "ls_pickle_loads_wall": [],
                "ls_load_state_wall": [],
                "ls_suffix_wall": [],
                "ls_restore_total_wall": [],
                "nat_set_data_wall": [],
                "nat_suffix_wall": [],
                "nat_restore_total_wall": [],
            }
            for _ in range(args.repetitions):
                llm.reset()
                t = perf_counter_ns()
                llm.eval(list(prefix + suffix))
                sync(llm)
                timings["cold_prefill_wall"].append(perf_counter_ns() - t)
                if token(llm) != cold_token:
                    raise RuntimeError("cold prefill nondeterministic")

                # --- LlamaState wrapper path ---
                llm.reset()
                r0 = perf_counter_ns()
                t = perf_counter_ns()
                state = pickle.loads(pickle_blob)
                timings["ls_pickle_loads_wall"].append(perf_counter_ns() - t)
                t = perf_counter_ns()
                llm.load_state(state)
                sync(llm)
                timings["ls_load_state_wall"].append(perf_counter_ns() - t)
                t = perf_counter_ns()
                llm.eval(list(suffix))
                sync(llm)
                timings["ls_suffix_wall"].append(perf_counter_ns() - t)
                timings["ls_restore_total_wall"].append(perf_counter_ns() - r0)
                if token(llm) != cold_token:
                    raise RuntimeError("LlamaState restore continuation diverged from cold baseline")

                # --- Native codec path ---
                llm.reset()
                r0 = perf_counter_ns()
                t = perf_counter_ns()
                native_set(llm, native_blob, prefix)
                sync(llm)
                timings["nat_set_data_wall"].append(perf_counter_ns() - t)
                t = perf_counter_ns()
                llm.eval(list(suffix))
                sync(llm)
                timings["nat_suffix_wall"].append(perf_counter_ns() - t)
                timings["nat_restore_total_wall"].append(perf_counter_ns() - r0)
                if token(llm) != cold_token:
                    raise RuntimeError("native codec restore continuation diverged from cold baseline")

            record = {
                "prefix_tokens": length,
                "suffix_tokens": 1,
                "continuation_token": cold_token,
                "pickle_blob_bytes": len(pickle_blob),
                "native_blob_bytes": len(native_blob),
                "wrapper_overhead_bytes": len(pickle_blob) - len(native_blob),
                "pickle_sha256": pickle_sha,
                "native_sha256": native_sha,
                "phases": {name: stats(values) for name, values in timings.items()},
            }
            records.append(record)
            run.log(
                f"[decompose] K={length}: pickle={len(pickle_blob)}B native={len(native_blob)}B "
                f"ls_restore={record['phases']['ls_restore_total_wall']['mean_ns'] / 1e6:.1f}ms "
                f"nat_restore={record['phases']['nat_restore_total_wall']['mean_ns'] / 1e6:.1f}ms"
            )
    finally:
        llm.close()

    summary = {
        "runtime": runtime.model_dump(),
        "fingerprint": fingerprint.model_dump(mode="json"),
        "gpu_before": gpu_before,
        "gpu_after_load": after_load,
        "gpu_after": gpu_snapshot(),
        "records": records,
        "note": (
            "In-memory restore isolates codec cost from disk I/O. "
            "native_* phases bypass the LlamaState score buffer. "
            "llama_synchronize-synced; enqueue timing is not throughput."
        ),
    }
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": records}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
