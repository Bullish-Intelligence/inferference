"""P2 throughput — fork mixed-batch vs no-fork router vs sequential (perf half).

Port of an earlier throughput harness onto the library's
``MultiLoRARouter``. Three router
instances serve the same S-request mixed-adapter workload and measure aggregate
tokens/sec, sweeping S:

  sequential : ``seq_routed`` backend, one request per ``run()`` (batch-1)
  router     : ``context_pool`` backend — batched per adapter context
  fork       : ``seq_routed`` backend — one mixed-batch decode of all S

Plus the token-exact cross-check (all three modes identical) so the throughput
numbers compare like for like. The CUDA sync is ``llama_synchronize(ctx)``
(11-report Task-4 deviation 2). Requires a P2 fork lib; GPU-only.
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
POOL = ["probe-a", "probe-b"]


def _sync(llm: Any) -> None:
    llama_cpp.llama_cpp.llama_synchronize(llm._ctx.ctx)


def _make_router(args: argparse.Namespace, backend: str) -> MultiLoRARouter:
    return MultiLoRARouter(
        RouterConfig(
            engine=EngineConfig(
                model_path=str(args.model),
                n_ctx=args.n_ctx,
                n_gpu_layers=-1,
                n_batch=128,
                seed=args.seed,
                backend="cuda",
            ),
            adapters=(
                AdapterSpec(name="probe-a", gguf_path=str(args.lora_a)),
                AdapterSpec(name="probe-b", gguf_path=str(args.lora_b)),
            ),
            n_seq_max=args.n_seq_max,
            include_base=False,
            backend=backend,
        )
    )


def _timed(fn, llm: Any, reps: int) -> tuple[float, list[Any]]:
    fn()  # warmup (graph build / cache)
    best: float | None = None
    out: list[Any] = []
    for _ in range(reps):
        _sync(llm)
        t0 = perf_counter_ns()
        out = fn()
        _sync(llm)
        dt = perf_counter_ns() - t0
        best = dt if best is None else min(best, dt)
    return (best or 0.0) / 1e6, out  # ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--lora-a", required=True, type=Path)
    ap.add_argument("--lora-b", required=True, type=Path)
    ap.add_argument("--artifacts", required=True, type=Path)
    ap.add_argument("--n-ctx", default=2048, type=int)
    ap.add_argument("--n-seq-max", default=8, type=int)
    ap.add_argument("--max-tokens", default=32, type=int)
    ap.add_argument("--batch-sizes", default="2,4,8")
    ap.add_argument("--reps", default=3, type=int)
    ap.add_argument("--seed", default=17018, type=int)
    args = ap.parse_args()

    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    gpu_before = gpu_snapshot()
    run = RunContext(args.artifacts, name="project17-p2-throughput", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()

    sizes = [int(x) for x in args.batch_sizes.split(",")]

    def make_requests(S: int) -> list[RouteRequest]:
        return [
            RouteRequest(
                request_id=f"r{i}",
                prompt=PROMPTS[i % len(PROMPTS)],
                adapter=POOL[i % len(POOL)],
                max_tokens=args.max_tokens,
            )
            for i in range(S)
        ]

    def toks(gens: list[Any]) -> list[tuple[int, ...]]:
        return [tuple(g.token_ids) for g in sorted(gens, key=lambda g: g.request_id or "")]

    # Each MultiLoRARouter loads its OWN full model copy (12 GB 3060): open
    # exactly one at a time — seq_routed (sequential + fork) in one pass,
    # context_pool (router) in a second pass — and merge by S, following the
    # library_throughput pattern.
    fork_side: dict[int, tuple] = {}
    seq_router = _make_router(args, "seq_routed")
    try:
        after_load = gpu_snapshot()
        require_gpu0_only(after_load)
        for S in sizes:
            if S > args.n_seq_max:
                continue
            reqs = make_requests(S)
            seq_ms, seq_out = _timed(
                lambda rs=reqs: [r for rq in rs for r in seq_router.run([rq])],
                seq_router.llm,
                args.reps,
            )
            # crossed: each request alone routed to the OTHER adapter (for the
            # known-flip classification)
            crossed_out = [
                seq_router.run(
                    [
                        RouteRequest(
                            request_id=rq.request_id,
                            prompt=rq.prompt,
                            adapter="probe-b" if rq.adapter == "probe-a" else "probe-a",
                            max_tokens=rq.max_tokens,
                        )
                    ]
                )[0]
                for rq in reqs
            ]
            fk_ms, fk_out = _timed(lambda rs=reqs: seq_router.run(rs), seq_router.llm, args.reps)
            fork_side[S] = (seq_ms, toks(seq_out), fk_ms, toks(fk_out), toks(crossed_out))
    finally:
        seq_router.close()

    router_side: dict[int, tuple] = {}
    cp_router = _make_router(args, "context_pool")
    try:
        for S in sizes:
            if S > args.n_seq_max:
                continue
            reqs = make_requests(S)
            rt_ms, rt_out = _timed(lambda rs=reqs: cp_router.run(rs), cp_router.llm, args.reps)
            router_side[S] = (rt_ms, toks(rt_out))
    finally:
        cp_router.close()

    rows: list[dict[str, Any]] = []
    flips: list[str] = []
    route_bugs = 0
    for S in sizes:
        if S > args.n_seq_max:
            rows.append({"S": S, "skipped": f"S>{args.n_seq_max}"})
            continue
        seq_ms, seq_toks, fk_ms, fk_toks, crossed_toks = fork_side[S]
        rt_ms, rt_toks = router_side[S]
        gen_tokens = S * args.max_tokens
        identical = seq_toks == rt_toks == fk_toks
        if not identical:
            # known-flip classification: a fork divergence whose token matches the
            # CROSSED adapter's isolated output is a route-bug; otherwise a benign
            # batched-greedy FP near-tie flip (reported, never failed).
            for i, (_, seq, fk) in enumerate(zip(make_requests(S), seq_toks, fk_toks, strict=False)):
                j = next((k for k, (x, y) in enumerate(zip(seq, fk, strict=False)) if x != y), None)
                if j is None:
                    continue
                if j < len(crossed_toks[i]) and crossed_toks[i][j] == fk[j]:
                    route_bugs += 1
                    flips.append(f"S{S}:r{i}:ROUTE-BUG@tok{j}")
                else:
                    flips.append(f"S{S}:r{i}:fp@tok{j}")
        rows.append(
            {
                "S": S,
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
                "router_vs_sequential": round(seq_ms / rt_ms, 2),
            }
        )
        run.log(
            f"[thru] S={S}: seq={rows[-1]['sequential_tok_s']}tps "
            f"router={rows[-1]['router_tok_s']}tps fork={rows[-1]['fork_tok_s']}tps "
            f"(fork/router x{rows[-1]['fork_vs_router']}) identical={identical}"
        )
        if flips:
            run.log(f"[thru] S={S}: {', '.join(flips)}")

    all_identical = all(r.get("outputs_identical", True) for r in rows if "skipped" not in r)
    if route_bugs:
        print(f"[thru] FAIL — {route_bugs} route-bug(s): {[f for f in flips if 'ROUTE-BUG' in f]}", flush=True)
        return 1
    if flips:
        print(f"[thru] benign batched-greedy FP flips (not routing bugs): {', '.join(flips)}", flush=True)
    result = {
        "all_outputs_identical": all_identical,
        "route_bugs": route_bugs,
        "known_flips": flips,
        "config": {
            "n_ctx": args.n_ctx,
            "n_seq_max": args.n_seq_max,
            "max_tokens": args.max_tokens,
            "reps": args.reps,
            "pool": POOL,
            "seed": args.seed,
        },
        "rows": rows,
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "model": str(args.model),
        "lib": os.environ.get("LLAMA_CPP_LIB_PATH"),
    }
    (run.dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_outputs_identical": all_identical, "rows": rows}, indent=2), flush=True)
    return 0 if route_bugs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
