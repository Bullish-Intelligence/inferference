"""Cached shared-prefix routing: prefill-once + per-request KV restore.

Port of an earlier cached-router harness and
``run_router_grammar_cached.py`` onto the library's
:meth:`inferference.router.MultiLoRARouter.run_cached`. A long
shared prefix is decoded ONCE per adapter and its per-sequence KV blob captured;
every request then restores that blob into its slot and decodes only its own
suffix. With ``--constrained`` the suffix decode is grammar-masked to a JSON
routing decision (the path-(a) MVP).

Gates (fail-closed, deterministic greedy), swept over base / probe-a / probe-b
(plain) or base / probe-a (constrained):
  1. Equivalence: the cached-restore continuation is token-exact vs the isolated
     batch-1 full-prompt baseline (plain), or vs the cold full re-prefill wave
     (constrained) — for EVERY sequence (also proves neighbour isolation across
     restored slots).
  2. Validity (constrained): every cached output is schema-valid JSON.
Throughput: cached wave vs cold wave (CUDA-synchronized wall time).

GPU-only: one card pinned (LLAMA_CI_GPU), n_gpu_layers=-1, LLAMA_CPP_LIB_PATH pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from inferference.gpu_env import lib_backend_label
from inferference.mixed_batch_gates import TOKEN_EXACT, gate_mode, report_measurement
from inferference.models import EngineConfig
from inferference.router import BASE, AdapterSpec, MultiLoRARouter, RouterConfig, RouteRequest

from ..env import gpu_snapshot, require_ci_device, require_gpu0_only
from ..gates import _resolve_tokenizer, load_tokenizer
from ..runenv import RunContext
from ..trace_router import TraceRouter, cached_legs

REPO = Path(__file__).resolve().parents[3]


class Route(BaseModel):
    tool: Literal["search", "calculator", "calendar", "smart_home", "none"]
    confidence: Literal["low", "medium", "high"]


# Long shared prefix (well past the ~192-token restore break-even) so restore is
# expected to beat cold re-prefill. Deterministic, no external data.
PREFIX_TEXT = (
    "You are a routing assistant for a tool-calling agent fleet. Follow these "
    "rules exactly. Always return a single well-formed decision. Consider the "
    "user's intent, the available tools, and the conversation so far. Do not "
    "invent tools that were not provided. Prefer the most specific tool. When no "
    "tool applies, say so plainly. Be concise, deterministic, and consistent. "
) * 3

SUFFIXES = [
    " User request: book a flight to Paris. Decision:",
    " User request: what is 2 plus 2? Decision:",
    " User request: summarize the attached document. Decision:",
    " User request: turn off the living room lights. Decision:",
]

PREFIX_TEXT_CONSTRAINED = (
    "You are the routing brain of a tool-calling agent fleet. Your only job is to "
    "select exactly one tool for each incoming user request and rate your confidence. "
    "Available tools: search (web/knowledge lookup), calculator (arithmetic and math), "
    "calendar (scheduling, reminders, availability), smart_home (device and light "
    "control), none (no tool is appropriate). Weigh the user's intent, the tool "
    "descriptions, and prior context. Always answer with a single JSON object of the "
    'form {"tool": <one of the tools>, "confidence": <low|medium|high>} and nothing '
    "else. Be deterministic and consistent across similar requests. "
) * 2

SUFFIXES_CONSTRAINED = [
    " Request: book a meeting for tomorrow at 3pm. Decision:",
    " Request: what is 17 times 23? Decision:",
    " Request: turn off the kitchen lights. Decision:",
    " Request: find the latest news about mars. Decision:",
]


def validate(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"json_ok": False, "schema_ok": False, "value": None, "raw": text}
    try:
        obj = json.loads(text)
        out["json_ok"] = True
    except Exception:  # noqa: BLE001
        return out
    try:
        out["value"] = Route.model_validate(obj).model_dump()
        out["schema_ok"] = True
    except ValidationError:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--lora-a", required=True, type=Path)
    ap.add_argument("--lora-b", required=True, type=Path)
    ap.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=os.environ.get("TOKENIZER_DEFAULT"),
        help="local tokenizer dir (required with --constrained; $TOKENIZER_DEFAULT)",
    )
    ap.add_argument("--artifacts", required=True, type=Path)
    ap.add_argument("--n-ctx", default=4096, type=int)
    ap.add_argument("--n-seq-max", default=8, type=int)
    ap.add_argument("--max-tokens", default=24, type=int)
    ap.add_argument("--seed", default=17018, type=int)
    ap.add_argument(
        "--constrained",
        action="store_true",
        help="grammar-mask every suffix to the Route schema (router_grammar_cached variant)",
    )
    args = ap.parse_args()

    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")
    if args.constrained and (args.tokenizer_dir is None or not Path(args.tokenizer_dir).exists()):
        raise SystemExit("--constrained requires --tokenizer-dir (local; $TOKENIZER_DEFAULT)")

    gpu_before = gpu_snapshot()
    run = RunContext(args.artifacts, name="project17-cached-router", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()

    prefix_text = PREFIX_TEXT_CONSTRAINED if args.constrained else PREFIX_TEXT
    suffixes = SUFFIXES_CONSTRAINED if args.constrained else SUFFIXES
    adapters: list[str | None] = [BASE, "probe-a"] if args.constrained else [BASE, "probe-a", "probe-b"]
    max_tokens = args.max_tokens

    # The gate family follows the loaded backend. An unknown backend raises here, before the load.
    mode = gate_mode()
    router_cls = MultiLoRARouter if mode == TOKEN_EXACT else TraceRouter
    router = router_cls(
        RouterConfig(
            engine=EngineConfig(
                model_path=str(args.model),
                n_ctx=args.n_ctx,
                n_gpu_layers=-1,
                n_batch=128,
                seed=args.seed,
                backend=lib_backend_label(),
            ),
            adapters=(
                AdapterSpec(name="probe-a", gguf_path=str(args.lora_a)),
                AdapterSpec(name="probe-b", gguf_path=str(args.lora_b)),
            ),
            n_seq_max=args.n_seq_max,
            include_base=True,
            backend="context_pool",
        )
    )
    try:
        if args.constrained:
            resolved = _resolve_tokenizer(str(args.tokenizer_dir), allow_network=False)
            tokenizer = load_tokenizer(resolved)
            router.enable_grammar(tokenizer, Route.model_json_schema())

        after_load = gpu_snapshot()
        require_gpu0_only(after_load, before=gpu_before)

        prefix_tokens = router.tokenize(prefix_text)
        n_ctx_seq = args.n_ctx // args.n_seq_max
        max_suffix = max(len(router.tokenize_suffix(s)) for s in suffixes)
        need = len(prefix_tokens) + max_suffix + max_tokens
        run.log(
            f"[cached] prefix={len(prefix_tokens)} tok, per-seq budget n_ctx_seq={n_ctx_seq}, worst-case need={need}"
        )
        if need >= n_ctx_seq:
            raise SystemExit(
                f"per-seq budget {n_ctx_seq} too small for need {need}; raise --n-ctx or lower --n-seq-max/prefix"
            )

        requests = [
            RouteRequest(
                request_id=f"{'base' if adapter is BASE else adapter}-{i}",
                prompt=suffixes[i],
                adapter=adapter,
                max_tokens=max_tokens,
            )
            for adapter in adapters
            for i in range(len(suffixes))
        ]
        res = router.run_cached(requests, prefix_tokens, constrained=args.constrained)
        logit_findings: list[str] = []
        if mode != TOKEN_EXACT:
            assert isinstance(router, TraceRouter)
            logit_findings, measured_pairs = cached_legs(router, requests, prefix_tokens, constrained=args.constrained)
            measurement = report_measurement(
                f"cached_router {'constrained' if args.constrained else 'plain'}: cached against reference",
                measured_pairs,
            )

        rows: list[dict[str, Any]] = []
        all_match = True
        all_valid = True
        for i, r in enumerate(requests):
            cached_toks = list(res.cached[i].token_ids)
            if args.constrained:
                cold_toks = list(res.cold[i].token_ids)
                eq = bool(cached_toks) and cached_toks == cold_toks[: len(cached_toks)]
                v = validate(res.cached[i].text or "")
                all_valid = all_valid and v["schema_ok"]
                rows.append(
                    {
                        "rid": r.request_id,
                        "adapter": r.adapter,
                        "schema_ok": v["schema_ok"],
                        "value": v["value"],
                        "raw": v["raw"],
                        "cached_eq_cold": eq,
                        "generated_tokens": len(cached_toks),
                    }
                )
            else:
                base_toks = list(res.baselines[i].token_ids)
                cold_toks = list(res.cold[i].token_ids)
                eq_cached = bool(cached_toks) and cached_toks == base_toks[: len(cached_toks)]
                eq_cold = bool(cached_toks) and cold_toks[: len(cached_toks)] == base_toks[: len(cached_toks)]
                all_match = all_match and eq_cached and eq_cold
                rows.append(
                    {
                        "rid": r.request_id,
                        "adapter": r.adapter,
                        "match_cached_vs_baseline": eq_cached,
                        "match_cold_vs_baseline": eq_cold,
                        "generated_tokens": len(cached_toks),
                    }
                )

        if mode != TOKEN_EXACT:
            # Token equality cannot hold when the logits depend on the batch composition.
            # The verdict takes the exact gates instead (replay and repeat bit-identical). The
            # difference to the reference is MEASURED, not gated. The rows keep their token columns as data.
            all_match = not logit_findings
        speedup = round(res.cold_wall_ns / res.cached_wall_ns, 2) if res.cached_wall_ns else None
        verdict = "GO" if (all_match and (all_valid if args.constrained else True)) else "NO-GO"
        result = {
            "verdict": verdict,
            "mode": "constrained" if args.constrained else "plain",
            "gates": {"all_equivalent": all_match, "all_schema_valid": all_valid},
            "equivalence_mode": mode,
            "equivalence_findings": logit_findings,
            "equivalence_measured_not_gated": measurement["classes"] if mode != TOKEN_EXACT else None,
            "prefix_tokens": len(prefix_tokens),
            "cached_wall_ms": round(res.cached_wall_ns / 1e6, 1),
            "cold_wall_ms": round(res.cold_wall_ns / 1e6, 1),
            "cached_speedup_vs_cold": speedup,
            "schema": Route.model_json_schema() if args.constrained else None,
            "config": {
                "n_ctx": args.n_ctx,
                "n_seq_max": args.n_seq_max,
                "max_tokens": max_tokens,
                "seed": args.seed,
            },
            "rows": rows,
            "gpu_before": gpu_before,
            "gpu_after": gpu_snapshot(),
            "model": str(args.model),
            "lib": os.environ.get("LLAMA_CPP_LIB_PATH"),
        }
        (run.dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "verdict": verdict,
                    "mode": result["mode"],
                    "gates": result["gates"],
                    "cached_speedup_vs_cold": speedup,
                    "cached_wall_ms": result["cached_wall_ms"],
                    "cold_wall_ms": result["cold_wall_ms"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0 if verdict == "GO" else 1
    finally:
        router.close()


if __name__ == "__main__":
    sys.exit(main())
