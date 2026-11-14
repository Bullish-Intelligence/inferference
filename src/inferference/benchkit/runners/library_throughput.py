"""Throughput re-measure through the shipping library path.

Adapted from an earlier throughput harness onto the ``inferference`` package.
Two changes: the import moved to ``inferference``, and the CUDA sync is
``llama_synchronize(ctx)`` — the earlier harness swallowed a torch ImportError
and relied on synchronous decode, so here the sync is hard-required and
context-correct.

Phase 1: fixed K=2 pool, sweep S (sequential / context_pool / seq_routed) with
the token-exact cross-check. Phase 2: fixed S, grow the pool K to find where
the masked K x LoRA overhead erodes the fork's structural win (each leg guarded
for VRAM, recorded as skipped rather than crashing the run).

Requires a P2 fork lib; GPU-only (one card pinned, n_gpu_layers=-1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import llama_cpp

from inferference.gpu_env import lib_backend_label
from inferference.models import EngineConfig
from inferference.router import AdapterSpec, MultiLoRARouter, RouterConfig, RouteRequest

from ..env import gpu_snapshot, require_ci_device, require_gpu0_only
from ..runenv import RunContext

REPO = Path(__file__).resolve().parents[3]

PROMPTS = [
    "The capital of France is",
    "In a distant galaxy,",
    "def add(a, b):",
    "Once upon a time",
    "The three primary colors are",
    "Water boils at",
    "The opposite of hot is",
    "To make tea you first",
]


def _sync(llm: Any) -> None:
    llama_cpp.llama_cpp.llama_synchronize(llm._ctx.ctx)


def _pool(k: int, lora_a: Path, lora_b: Path) -> tuple[AdapterSpec, ...]:
    paths = [str(lora_a), str(lora_b)]
    return tuple(AdapterSpec(name=f"probe-{i}", gguf_path=paths[i % 2]) for i in range(k))


def _make_router(args: argparse.Namespace, adapters: tuple[AdapterSpec, ...], backend: str) -> MultiLoRARouter:
    return MultiLoRARouter(
        RouterConfig(
            engine=EngineConfig(
                model_path=str(args.model),
                n_ctx=args.n_ctx,
                n_gpu_layers=-1,
                n_batch=128,
                seed=args.seed,
                backend=lib_backend_label(),
            ),
            adapters=adapters,
            n_seq_max=args.n_seq_max,
            include_base=False,
            backend=backend,
        )
    )


def _requests(s: int, pool_names: list[str], max_tokens: int) -> list[RouteRequest]:
    return [
        RouteRequest(
            prompt=PROMPTS[i % len(PROMPTS)],
            adapter=pool_names[i % len(pool_names)],
            max_tokens=max_tokens,
            request_id=f"r{i}",
        )
        for i in range(s)
    ]


def _timed(fn, llm: Any, reps: int) -> tuple[float, list[Any]]:
    fn()  # warmup
    best: float | None = None
    out: list[Any] = []
    for _ in range(reps):
        _sync(llm)
        t0 = perf_counter_ns()
        out = fn()
        _sync(llm)
        dt = perf_counter_ns() - t0
        best = dt if best is None else min(best, dt)
    return (best or 0.0) / 1e6, out


def _toks(results: list[Any]) -> list[tuple[int, ...]]:
    return [tuple(r.token_ids) for r in sorted(results, key=lambda r: r.request_id or "")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--lora-a", required=True, type=Path)
    ap.add_argument("--lora-b", required=True, type=Path)
    ap.add_argument("--artifacts", required=True, type=Path)
    ap.add_argument("--n-ctx", default=2048, type=int)
    ap.add_argument("--n-seq-max", default=16, type=int)
    ap.add_argument("--max-tokens", default=32, type=int)
    ap.add_argument("--batch-sizes", default="2,4,8")
    ap.add_argument("--k-sweep", default="2,4,8,16,24")
    ap.add_argument("--k-sweep-s", default=16, type=int)
    ap.add_argument("--reps", default=3, type=int)
    ap.add_argument("--seed", default=17018, type=int)
    args = ap.parse_args()

    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    gpu_before = gpu_snapshot()
    run = RunContext(args.artifacts, name="project17-library-throughput", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()

    sizes = [int(x) for x in args.batch_sizes.split(",")]
    pool2 = _pool(2, args.lora_a, args.lora_b)
    names2 = [a.name for a in pool2]

    # Phase 1: fixed K=2, sweep S (each backend opens its own full model copy —
    # one at a time, 12 GB 3060 constraint).
    fork_side: dict[int, tuple] = {}
    fork = _make_router(args, pool2, "seq_routed")
    try:
        after_load = gpu_snapshot()
        require_gpu0_only(after_load, before=gpu_before)
        for s in sizes:
            if s > args.n_seq_max:
                continue
            reqs = _requests(s, names2, args.max_tokens)
            seq_ms, seq_out = _timed(lambda rs=reqs: [g for r in rs for g in fork.run([r])], fork.llm, args.reps)
            fk_ms, fk_out = _timed(lambda rs=reqs: fork.run(rs), fork.llm, args.reps)
            fork_side[s] = (seq_ms, _toks(seq_out), fk_ms, _toks(fk_out))
    finally:
        fork.close()

    router_side: dict[int, tuple] = {}
    router = _make_router(args, pool2, "context_pool")
    try:
        for s in sizes:
            if s > args.n_seq_max:
                continue
            reqs = _requests(s, names2, args.max_tokens)
            rt_ms, rt_out = _timed(lambda rs=reqs: router.run(rs), router.llm, args.reps)
            router_side[s] = (rt_ms, _toks(rt_out))
    finally:
        router.close()

    phase1: list[dict[str, Any]] = []
    for s in sizes:
        if s > args.n_seq_max:
            phase1.append({"S": s, "skipped": f"S>{args.n_seq_max}"})
            continue
        seq_ms, seq_toks, fk_ms, fk_toks = fork_side[s]
        rt_ms, rt_toks = router_side[s]
        gen_tokens = s * args.max_tokens
        identical = seq_toks == rt_toks == fk_toks
        phase1.append(
            {
                "S": s,
                "gen_tokens": gen_tokens,
                "outputs_identical": identical,
                "sequential_ms": round(seq_ms, 1),
                "sequential_tok_s": round(gen_tokens / (seq_ms / 1e3), 1),
                "router_ms": round(rt_ms, 1),
                "router_tok_s": round(gen_tokens / (rt_ms / 1e3), 1),
                "fork_ms": round(fk_ms, 1),
                "fork_tok_s": round(gen_tokens / (fk_ms / 1e3), 1),
                "fork_vs_router": round(rt_ms / fk_ms, 2),
                "fork_vs_sequential": round(seq_ms / fk_ms, 2),
            }
        )
        run.log(
            f"[thru] S={s}: seq={phase1[-1]['sequential_tok_s']}tps "
            f"router={phase1[-1]['router_tok_s']}tps fork={phase1[-1]['fork_tok_s']}tps "
            f"(fork/router x{phase1[-1]['fork_vs_router']}) identical={identical}"
        )

    # Phase 2: fixed S, grow K (each leg VRAM-guarded -> recorded as skipped).
    phase2: list[dict[str, Any]] = []
    s = min(args.k_sweep_s, args.n_seq_max)
    for k in (int(x) for x in args.k_sweep.split(",")):
        poolk = _pool(k, args.lora_a, args.lora_b)
        namesk = [a.name for a in poolk]
        reqs = _requests(s, namesk, args.max_tokens)
        gen_tokens = s * args.max_tokens

        fk_ms: float | None = None
        try:
            fk = _make_router(args, poolk, "seq_routed")
            try:
                fk_ms, _ = _timed(lambda rs=reqs, rt=fk: rt.run(rs), fk.llm, args.reps)
            finally:
                fk.close()
        except Exception as exc:  # noqa: BLE001 — VRAM guard, recorded not fatal
            run.log(f"[ksweep] K={k}: fork skipped ({type(exc).__name__})")

        rk_ms: float | None = None
        try:
            rk = _make_router(args, poolk, "context_pool")
            try:
                rk_ms, _ = _timed(lambda rs=reqs, rt=rk: rt.run(rs), rk.llm, args.reps)
            finally:
                rk.close()
        except Exception as exc:  # noqa: BLE001
            run.log(f"[ksweep] K={k}: router skipped ({type(exc).__name__})")

        row: dict[str, Any] = {
            "K": k,
            "S": s,
            "gen_tokens": gen_tokens,
            "fork_tok_s": round(gen_tokens / (fk_ms / 1e3), 1) if fk_ms else None,
            "router_tok_s": round(gen_tokens / (rk_ms / 1e3), 1) if rk_ms else None,
        }
        if fk_ms and rk_ms:
            row["fork_vs_router"] = round(rk_ms / fk_ms, 2)
        phase2.append(row)
        run.log(f"[ksweep] K={k}: fork={row['fork_tok_s']}tps router={row['router_tok_s']}tps")

    all_identical = all(r.get("outputs_identical", True) for r in phase1 if "skipped" not in r)
    result = {
        "all_outputs_identical": all_identical,
        "config": {
            "n_ctx": args.n_ctx,
            "n_seq_max": args.n_seq_max,
            "max_tokens": args.max_tokens,
            "reps": args.reps,
            "seed": args.seed,
        },
        "phase1": phase1,
        "phase2": phase2,
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "model": str(args.model),
        "lib": os.environ.get("LLAMA_CPP_LIB_PATH"),
    }
    (run.dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"all_outputs_identical": all_identical, "phase1": phase1, "phase2": phase2}, indent=2), flush=True
    )
    return 0 if all_identical else 1


if __name__ == "__main__":
    sys.exit(main())
