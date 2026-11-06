"""In-context cold-vs-restore break-even + n_seq_max batched throughput.

Port of an earlier breakeven harness onto the library's
seq-state surface (:class:`inferference.prefix_cache_live.LlamaSeqStateBridge`
wraps ``llama_state_seq_get_data``/``set_data`` and own-batch decode; the
custom-``n_seq_max`` context creation stays low-level). CUDA sync via
``llama_synchronize(ctx)`` (never torch).

  Part 1 — break-even *inside the same multi-sequence context*. For each
  checkpoint length K, in one ``n_seq_max=2`` context, compare (a) cold: decode
  the K-token prefix then one suffix token; (b) restore:
  ``llama_state_seq_set_data`` the matched cached blob then decode one suffix
  token. Reports the first K where restore is cheaper than recompute. Continuation
  tokens must match (fail-closed).

  Part 2 — ``n_seq_max`` scaling to 8/16 and concurrent batched decode: the
  matched-``n_seq_max`` portability rule at 8/16, then aggregate generation
  throughput when S sequences decode one token each per ``llama_decode`` step
  (the router's "batch many sequences in flight" path) for S in {1,2,4,8,16}.

GPU-only: CUDA_VISIBLE_DEVICES=0, n_gpu_layers=-1, LLAMA_CPP_LIB_PATH pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from inferference.prefix_cache_live import LlamaSeqStateBridge

from ..env import gpu_snapshot, require_ci_device, require_gpu0_only
from ..runenv import RunContext

REPO = Path(__file__).resolve().parents[3]


def stats(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {"mean_ns": sum(values) / len(values), "p50_ns": ordered[(len(values) - 1) // 2]}


class Harness:
    """Model + raw-ctx factory; the seq-state primitives ride on LlamaSeqStateBridge."""

    def __init__(self, model_path: str, n_ctx: int, n_batch: int, seed: int) -> None:
        from llama_cpp import Llama, llama_cpp  # noqa: PLC0415

        self.llama_cpp = llama_cpp
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_batch=n_batch, n_gpu_layers=-1, seed=seed, verbose=False)
        self.model = self.llm._model.model
        self.n_vocab = self.llm._n_vocab
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.bridge = LlamaSeqStateBridge(n_batch=n_batch, n_vocab=self.n_vocab, native=self.llama_cpp)

    def tokenize(self, text: bytes) -> list[int]:
        return list(self.llm.tokenize(text, add_bos=False, special=True))

    def make_ctx(self, n_seq_max: int) -> Any:
        p = self.llama_cpp.llama_context_default_params()
        p.n_ctx = self.n_ctx
        p.n_batch = self.n_batch
        p.n_ubatch = self.n_batch
        p.n_seq_max = n_seq_max
        return self.llama_cpp.llama_new_context_with_model(self.model, p)

    def free(self, ctx: Any) -> None:
        self.llama_cpp.llama_free(ctx)

    def sync(self, ctx: Any) -> None:
        self.llama_cpp.llama_synchronize(ctx)

    def seq_rm(self, ctx: Any, seq_id: int) -> None:
        mem = self.llama_cpp.llama_get_memory(ctx)
        self.llama_cpp.llama_memory_seq_rm(mem, seq_id, -1, -1)

    def decode(self, ctx: Any, tokens: list[int], seq_id: int, start_pos: int) -> None:
        self.bridge.decode_tokens(ctx, tokens, seq_id, start_pos)

    def last_token(self, ctx: Any) -> int:
        return self.bridge.last_token(ctx)

    def get_seq(self, ctx: Any, seq_id: int) -> bytes:
        return self.bridge.capture_seq_state(ctx, seq_id)

    def set_seq(self, ctx: Any, blob: bytes, seq_id: int) -> int:
        return self.bridge.restore_seq_state(ctx, blob, seq_id)

    def batched_step(self, ctx: Any, cur: list[int], pos: list[int]) -> list[int]:
        """Decode one token for each of len(cur) sequences in a single batch."""
        lcpp = self.llama_cpp
        s = len(cur)
        batch = lcpp.llama_batch_init(s, 0, 1)
        batch.n_tokens = s
        for i in range(s):
            batch.token[i] = cur[i]
            batch.pos[i] = pos[i]
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = i
            batch.logits[i] = 1
        rc = lcpp.llama_decode(ctx, batch)
        if rc != 0:
            lcpp.llama_batch_free(batch)
            raise RuntimeError(f"batched llama_decode rc={rc}")
        nxt = []
        for i in range(s):
            ptr = lcpp.llama_get_logits_ith(ctx, i)
            nxt.append(int(np.argmax(np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)))))
        lcpp.llama_batch_free(batch)
        return nxt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--n-ctx", default=2048, type=int)
    parser.add_argument("--n-batch", default=128, type=int)
    parser.add_argument("--seed", default=17018, type=int)
    parser.add_argument("--breakeven-lengths", default="16,32,64,96,128,192,256")
    parser.add_argument("--seq-counts", default="1,2,4,8,16")
    parser.add_argument("--gen-tokens", default=32, type=int)
    parser.add_argument("--prime-tokens", default=8, type=int)
    parser.add_argument("--repetitions", default=5, type=int)
    args = parser.parse_args()
    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    gpu_before = gpu_snapshot()
    run = RunContext(args.artifacts, name="project17-seq-batch-breakeven", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()

    h = Harness(str(args.model), args.n_ctx, args.n_batch, args.seed)
    after_load = gpu_snapshot()
    require_gpu0_only(after_load)

    toks = h.tokenize(b"Router batched throughput and break-even workload. " * 80)

    # ---- Part 1: in-context cold-vs-restore break-even (n_seq_max=2, seq 1) ----
    breakeven = []
    for length in (int(v) for v in args.breakeven_lengths.split(",")):
        if length + 1 >= len(toks):
            run.log(f"[p1] skip K={length}: corpus too short")
            continue
        prefix, suffix = toks[:length], [toks[length]]
        # Capture the matched-n_seq_max blob once.
        src = h.make_ctx(2)
        h.decode(src, prefix, 1, 0)
        h.sync(src)
        blob = h.get_seq(src, 1)
        h.free(src)
        ctx = h.make_ctx(2)
        cold_ns, restore_ns, cold_tok, restore_tok = [], [], [], []
        try:
            for _ in range(args.repetitions):
                h.seq_rm(ctx, 1)
                t = perf_counter_ns()
                h.decode(ctx, prefix + suffix, 1, 0)
                h.sync(ctx)
                cold_ns.append(perf_counter_ns() - t)
                cold_tok.append(h.last_token(ctx))
                h.seq_rm(ctx, 1)
                t = perf_counter_ns()
                if h.set_seq(ctx, blob, 1) <= 0:
                    raise RuntimeError("matched set_data failed")
                h.decode(ctx, suffix, 1, len(prefix))
                h.sync(ctx)
                restore_ns.append(perf_counter_ns() - t)
                restore_tok.append(h.last_token(ctx))
        finally:
            h.free(ctx)
        if cold_tok != restore_tok or len(set(cold_tok)) != 1:
            raise RuntimeError(f"continuation mismatch at K={length}: cold={cold_tok} restore={restore_tok}")
        c, r = stats(cold_ns), stats(restore_ns)
        breakeven.append(
            {
                "checkpoint_tokens": length,
                "seq_blob_bytes": len(blob),
                "continuation_token": cold_tok[0],
                "cold_ns": c,
                "restore_ns": r,
                "restore_beats_cold": r["mean_ns"] < c["mean_ns"],
                "delta_ns": c["mean_ns"] - r["mean_ns"],
            }
        )
        run.log(
            f"[p1] K={length}: cold={c['mean_ns'] / 1e6:.1f}ms restore={r['mean_ns'] / 1e6:.1f}ms "
            f"beats={breakeven[-1]['restore_beats_cold']}"
        )

    # ---- Part 2: n_seq_max scaling + concurrent batched-decode throughput ----
    throughput = []
    prime = toks[: args.prime_tokens]
    for s in (int(v) for v in args.seq_counts.split(",")):
        if args.n_ctx // s < args.prime_tokens + args.gen_tokens + 2:
            throughput.append({"seq_count": s, "skipped": "n_ctx/n_seq_max too small"})
            continue
        ctx = h.make_ctx(s)
        try:
            # Diagonal portability confirmation at this n_seq_max.
            set_ok = None
            if s in (8, 16):
                src = h.make_ctx(s)
                h.decode(src, prime, s - 1, 0)
                h.sync(src)
                b = h.get_seq(src, s - 1)
                h.free(src)
                set_ok = h.set_seq(ctx, b, s - 1) > 0
                h.seq_rm(ctx, s - 1)
            for i in range(s):
                h.decode(ctx, prime, i, 0)
            h.sync(ctx)
            cur = [prime[-1]] * s
            pos = [args.prime_tokens] * s
            step_ns = []
            gen_started = perf_counter_ns()
            for _ in range(args.gen_tokens):
                t = perf_counter_ns()
                nxt = h.batched_step(ctx, cur, pos)
                h.sync(ctx)
                step_ns.append(perf_counter_ns() - t)
                for i in range(s):
                    pos[i] += 1
                cur = nxt
            total_wall = perf_counter_ns() - gen_started
        finally:
            h.free(ctx)
        tokens_generated = s * args.gen_tokens
        throughput.append(
            {
                "seq_count": s,
                "diagonal_set_ok": set_ok,
                "gen_tokens_per_seq": args.gen_tokens,
                "tokens_generated": tokens_generated,
                "total_wall_ns": total_wall,
                "aggregate_tok_per_s": tokens_generated * 1e9 / total_wall,
                "per_seq_tok_per_s": args.gen_tokens * 1e9 / total_wall,
                "step_ns_mean": sum(step_ns) / len(step_ns),
                "step_ns_p50": sorted(step_ns)[(len(step_ns) - 1) // 2],
            }
        )
        run.log(f"[p2] S={s}: agg={throughput[-1]['aggregate_tok_per_s']:.1f} tps diag_set_ok={set_ok}")

    h.llm.close()
    summary = {
        "gpu_before": gpu_before,
        "gpu_after_load": after_load,
        "gpu_after": gpu_snapshot(),
        "n_ctx": args.n_ctx,
        "n_batch": args.n_batch,
        "breakeven": breakeven,
        "throughput": throughput,
        "note": "llama_synchronize-synced. Success in Part 1 = identical continuation cold vs restore.",
    }
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"breakeven": breakeven, "throughput": throughput}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
