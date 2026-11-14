"""Measure batch-composition noise. The Vulkan gates do not read it.

On Vulkan, mixed-versus-isolated equivalence is a measured quantity, not a gate. This runner
measures it. It runs requests alone and in compositions, from a clean state, teacher-forced with
the same tokens, and records for each class (``batch/base``, ``batch/adapter``, ``cached/base``,
``cached/adapter``) the distribution of the per-step largest |dlogprob| (p50, p99, max) and the
worst cases. A worst case carries the forced token's log-probability in both runs, the
total-variation distance over the probe ids, and the top ids of both runs. The record carries its
provenance (backend, build, device, date, model, engine settings). Nothing reads it to pass or
fail a test.

Before it measures, it proves the premise of the exact gates. It runs a composition twice with
other work in between, each time from a reset state, and requires bit-identical results. It then
runs the same pair WITHOUT the reset and records how large the history effect is. If the reset
leaves any difference, the runner stops.

With ``--parity`` it also runs each request alone through ``seq_routed`` and compares it with
``context_pool`` alone, bit for bit, and reports the gap when there is one (item 4). The parity
GATE is in the suites.

``batch/*`` uses ``context_pool``. ``cached/*`` covers a cached restore against a cold full prompt.

GPU only. One card pinned (LLAMA_CI_GPU), LLAMA_CPP_LIB_PATH pinned. Needs an evicted card.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inferference.gpu_env import lib_backend_label
from inferference.mixed_batch_gates import (
    LOGIT_TOLERANCE,
    Pair,
    Trace,
    TracingEngine,
    check_exact,
    collect_isolated,
    collect_isolated_forced,
    collect_mixed,
    current_build,
    gate_mode,
    measure_pairs,
    require,
)
from inferference.models import EngineConfig
from inferference.router import AdapterSpec, RouterConfig, RouteRequest

from ..env import gpu_snapshot, require_ci_device, require_gpu0_only, visible_gpu_uuid
from ..runenv import RunContext
from ..trace_router import TraceRouter, cached_legs, cached_state_pairs, pairs_from_state
from .cached_router import PREFIX_TEXT

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
SUFFIXES = [
    " User request: book a flight to Paris. Decision:",
    " User request: what is 2 plus 2? Decision:",
    " User request: summarize the attached document. Decision:",
    " User request: turn off the living room lights. Decision:",
    " User request: set an alarm for six in the morning. Decision:",
    " User request: translate hello into French. Decision:",
    " User request: what is the weather in Oslo? Decision:",
    " User request: play some quiet music. Decision:",
    " User request: read my unread email. Decision:",
    " User request: how tall is Mount Everest? Decision:",
    " User request: remind me to call Anna at noon. Decision:",
    " User request: dim the bedroom lights to half. Decision:",
    " User request: convert ten miles to kilometres. Decision:",
    " User request: search for a recipe with lentils. Decision:",
    " User request: what is on my calendar tomorrow? Decision:",
    " User request: lock the front door. Decision:",
]
POOL = ("probe-a", "probe-b")
WIDTHS = (2, 4, 8)
SHIFTS = (0, 3)


def _adapter_of(prompt_index: int, with_adapters: bool) -> str | None:
    return POOL[prompt_index % len(POOL)] if with_adapters else None


def _batch_requests(indexes: list[int], with_adapters: bool, max_tokens: int) -> list[RouteRequest]:
    return [
        RouteRequest(
            request_id=f"p{p}",
            prompt=PROMPTS[p],
            adapter=_adapter_of(p, with_adapters),
            max_tokens=max_tokens,
        )
        for p in indexes
    ]


def batch_pairs(router: TracingEngine, with_adapters: bool, max_tokens: int) -> tuple[list[Pair], list[Trace]]:
    """Each prompt alone against the same prompt inside compositions of several widths.

    Returns the pairs and the isolated trace of every prompt."""
    iso = collect_isolated(router, _batch_requests(list(range(len(PROMPTS))), with_adapters, max_tokens))
    pairs: list[Pair] = []
    for width in WIDTHS:
        for shift in SHIFTS:
            for reverse in (False, True):
                members = [(k + shift) % len(PROMPTS) for k in range(width)]
                if reverse:
                    members.reverse()
                got = collect_mixed(
                    router, _batch_requests(members, with_adapters, max_tokens), [iso[p] for p in members]
                )
                label = f"w{width}s{shift}{'r' if reverse else 'f'}"
                pairs += [
                    Pair(PROMPTS[p], label, _adapter_of(p, with_adapters), iso[p], got[slot])
                    for slot, p in enumerate(members)
                ]
    return pairs, iso


def cached_pairs(router: TraceRouter, with_adapters: bool, max_tokens: int, bridge: Any) -> list[Pair]:
    """Cached restore against cold, by the state flow and by ``run_cached``."""
    prefix = router.tokenize(PREFIX_TEXT)
    suffix_lists = [router.tokenize(s) for s in SUFFIXES]
    adapters = list(POOL) if with_adapters else [None]
    pairs: list[Pair] = []
    for adapter in adapters:
        pairs += pairs_from_state(
            adapter, cached_state_pairs(router, bridge, adapter, prefix, suffix_lists), f"state/{adapter or 'base'}"
        )
    requests = [
        RouteRequest(request_id=f"{a or 'base'}-{i}", prompt=SUFFIXES[i], adapter=a, max_tokens=max_tokens)
        for a in adapters
        for i in range(len(SUFFIXES))
    ]
    _, run_pairs = cached_legs(router, requests, prefix)
    return pairs + run_pairs


def history_probe(router: TraceRouter, max_tokens: int) -> dict[str, Any]:
    """Prove the premise. A reset removes the history effect. Record the size of the effect without it."""
    reqs = _batch_requests([0, 1, 2, 3], True, max_tokens)
    junk = list(reversed(_batch_requests([4, 5, 6, 7, 0, 1], True, max_tokens)))
    first = collect_mixed(router, reqs)
    collect_mixed(router, junk)
    again = collect_mixed(router, reqs, first)
    findings: list[str] = []
    for i, (a, b) in enumerate(zip(first, again, strict=True)):
        findings += check_exact(f"history probe request {i}", a, b)
    # The same two runs without the reset.
    router.reset_state()
    plain_first = router.trace(reqs)
    router.trace(junk)
    plain_again = router.trace(reqs, reference=plain_first)
    effect = 0.0
    differing = 0
    for a, b in zip(plain_first, plain_again, strict=True):
        if check_exact("x", a, b):
            differing += 1
        for sa, sb in zip(a.steps, b.steps, strict=False):
            effect = max(
                effect, max((abs(y - x) for x, y in zip(sa.logprobs(), sb.logprobs(), strict=True)), default=0.0)
            )
    return {
        "reset_findings": findings,
        "without_reset_differing_requests": differing,
        "without_reset_max_abs_dlogprob": effect,
        "requests": len(reqs),
    }


def parity_probe(
    model: str, config_kwargs: dict[str, Any], lora_a: str, lora_b: str, max_tokens: int, ref: dict[str, Trace]
) -> dict[str, Any]:
    """One request alone: ``seq_routed`` against the ``context_pool`` reference traces (item 4).

    Reports bit-identity per request and, for the requests that differ, the distribution and
    the worst cases of the gap. The parity GATE is in the suites. This is the evidence."""
    router = TraceRouter(_config(model, lora_a, lora_b, "seq_routed", **config_kwargs))
    try:
        rows: list[dict[str, Any]] = []
        pairs: list[Pair] = []
        for with_adapters in (False, True):
            reqs = _batch_requests(list(range(len(PROMPTS))), with_adapters, max_tokens)
            refs = [ref[f"{r.adapter or 'base'}/{r.request_id}"] for r in reqs]
            got = collect_isolated_forced(router, reqs, refs)
            for r, a, b in zip(reqs, refs, got, strict=True):
                same = not check_exact("p", a, b)
                rows.append({"request": r.request_id, "adapter": r.adapter, "bit_identical": same})
                pairs.append(Pair(r.prompt, "parity", r.adapter, a, b))
        return {
            "rows": rows,
            "bit_identical": sum(1 for r in rows if r["bit_identical"]),
            "total": len(rows),
            "distribution": measure_pairs(pairs),
        }
    finally:
        router.close()


def _config(
    model: str, lora_a: str, lora_b: str, backend: str, *, n_ctx: int, n_seq_max: int, seed: int
) -> RouterConfig:
    return RouterConfig(
        engine=EngineConfig(
            model_path=model, n_ctx=n_ctx, n_gpu_layers=-1, n_batch=128, seed=seed, backend=lib_backend_label()
        ),
        adapters=(AdapterSpec(name=POOL[0], gguf_path=lora_a), AdapterSpec(name=POOL[1], gguf_path=lora_b)),
        n_seq_max=n_seq_max,
        include_base=True,
        backend=backend,  # type: ignore[arg-type]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--lora-a", required=True, type=Path)
    ap.add_argument("--lora-b", required=True, type=Path)
    ap.add_argument("--artifacts", required=True, type=Path)
    ap.add_argument("--n-seq-max", default=8, type=int)
    ap.add_argument("--max-tokens", default=32, type=int)
    ap.add_argument("--seed", default=17018, type=int)
    ap.add_argument("--parity", action="store_true", help="also measure the seq_routed against context_pool gap")
    args = ap.parse_args()

    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the lib dir")
    backend = lib_backend_label()
    if gate_mode(backend) != LOGIT_TOLERANCE:
        raise SystemExit(f"this measurement is for the logit gates; backend {backend!r} uses the token-exact gates")

    gpu_before = gpu_snapshot()
    run = RunContext(args.artifacts, name="batch-noise-measurement", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()

    record: dict[str, Any] = {
        "provenance": {
            "backend": backend,
            "build": current_build(),
            "device": visible_gpu_uuid() or "unknown",
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": args.model.name,
            "tool": "inferference.benchkit.runners.batch_noise_measurement",
            "note": "context_pool reference, greedy, teacher-forced, reset before every run. Measured, not gated.",
        },
        "engines": {
            "batch": {"n_ctx": 2048, "n_batch": 128, "n_seq_max": args.n_seq_max, "seed": args.seed},
            "cached": {"n_ctx": 4096, "n_batch": 128, "n_seq_max": args.n_seq_max, "seed": args.seed},
        },
        "classes": {},
    }
    iso_refs: dict[str, Trace] = {}

    def dump() -> None:
        (run.dir / "measurement.json").write_text(json.dumps(record, indent=2, default=str) + "\n")

    def measured(family: str, pairs: list[Pair]) -> None:
        for name, stats in measure_pairs(pairs).items():
            record["classes"][f"{family}/{name}"] = stats
            run.log(f"[measure] {family}/{name}: " + json.dumps({k: v for k, v in stats.items() if k != "worst"}))
        dump()

    router = TraceRouter(
        _config(
            str(args.model),
            str(args.lora_a),
            str(args.lora_b),
            "context_pool",
            n_ctx=2048,
            n_seq_max=args.n_seq_max,
            seed=args.seed,
        )
    )
    try:
        require_gpu0_only(gpu_snapshot(), before=gpu_before)
        probe = history_probe(router, args.max_tokens)
        record["history_probe"] = probe
        run.log(f"[measure] history probe: {probe}")
        dump()
        # The premise of every gate. If the reset leaves a difference, stop.
        require(probe["reset_findings"], "the reset does not remove the history effect")
        for with_adapters in (False, True):
            pairs, iso = batch_pairs(router, with_adapters, args.max_tokens)
            measured("batch", pairs)
            for index, trace in enumerate(iso):
                iso_refs[f"{_adapter_of(index, with_adapters) or 'base'}/p{index}"] = trace
    finally:
        router.close()

    if args.parity:
        record["parity"] = parity_probe(
            str(args.model),
            {"n_ctx": 2048, "n_seq_max": args.n_seq_max, "seed": args.seed},
            str(args.lora_a),
            str(args.lora_b),
            args.max_tokens,
            iso_refs,
        )
        run.log(f"[measure] parity: {record['parity']['bit_identical']} of {record['parity']['total']} bit-identical")
        dump()

    router = TraceRouter(
        _config(
            str(args.model),
            str(args.lora_a),
            str(args.lora_b),
            "context_pool",
            n_ctx=4096,
            n_seq_max=args.n_seq_max,
            seed=args.seed,
        )
    )
    try:
        from inferference.prefix_cache_live import LlamaSeqStateBridge

        bridge = LlamaSeqStateBridge(n_batch=128, n_vocab=router.n_vocab, native=router.C)
        for with_adapters in (False, True):
            measured("cached", cached_pairs(router, with_adapters, args.max_tokens, bridge))
    finally:
        router.close()

    dump()
    print(
        json.dumps({"measurement": str(run.dir / "measurement.json"), "classes": sorted(record["classes"])}, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
