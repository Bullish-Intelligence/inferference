"""Grammar-constrained routing perf + validity gate (router_grammar port).

Port of an earlier router-grammar harness's perf+gate half onto
the library's ``MultiLoRARouter``. The gate half (constrained validity) rides
along as the run's verdict; the perf half measures the constrained vs
unconstrained run over the mixed base+adapter workload.

The reference's ``run_constrained`` maps to ``router.run(requests,
constrained=True)`` with ``router.enable_grammar(tokenizer, schema)``; the sync
is ``llama_synchronize(ctx)``. GPU-only; requires a fork lib (the grammar path
is P2-independent but the runner's adapter routing is the P2 capability).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from inferference.models import EngineConfig
from inferference.router import (
    AdapterSpec,
    MultiLoRARouter,
    RouterConfig,
    RouteRequest,
)

from ..env import gpu_snapshot, require_ci_device, require_gpu0_only
from ..gates import _resolve_tokenizer
from ..runenv import RunContext

REPO = Path(__file__).resolve().parents[3]


class Route(BaseModel):
    tool: Literal["search", "calculator", "calendar", "smart_home", "none"]
    confidence: Literal["low", "medium", "high"]


USER_TASKS = [
    "book a meeting for tomorrow at 3pm",
    "what is 17 times 23",
    "turn off the kitchen lights",
    "find the latest news about mars",
    "sing me a song",
]
INSTR = (
    "You are a tool router. For the user request, respond with ONLY a JSON "
    'object {"tool": ..., "confidence": ...}. User request: '
)


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
        help="local tokenizer dir (default: $TOKENIZER_DEFAULT — never network)",
    )
    ap.add_argument("--artifacts", required=True, type=Path)
    ap.add_argument("--n-ctx", default=4096, type=int)
    ap.add_argument("--n-seq-max", default=8, type=int)
    ap.add_argument("--max-tokens", default=48, type=int)
    ap.add_argument("--seed", default=17018, type=int)
    args = ap.parse_args()

    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    if args.tokenizer_dir is None or not args.tokenizer_dir.exists():
        raise SystemExit("no tokenizer dir: pass --tokenizer-dir or set TOKENIZER_DEFAULT (local only)")

    gpu_before = gpu_snapshot()
    run = RunContext(args.artifacts, name="project17-router-grammar", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()

    router = MultiLoRARouter(
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
            include_base=True,
            backend="seq_routed",
        )
    )
    try:
        after_load = gpu_snapshot()
        require_gpu0_only(after_load)
        from transformers import AutoTokenizer  # noqa: PLC0415

        resolved = _resolve_tokenizer(str(args.tokenizer_dir), allow_network=False)
        tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
        router.enable_grammar(tokenizer, Route.model_json_schema())

        # Mixed workload across base + probe-a (grammar is adapter-agnostic).
        requests = [
            RouteRequest(
                request_id=f"{'base' if a is None else a}-{i}",
                prompt=INSTR + USER_TASKS[i],
                adapter=a,
                max_tokens=args.max_tokens,
            )
            for a in (None, "probe-a")
            for i in range(len(USER_TASKS))
        ]
        run.log(f"[grammar] constrained run: {len(requests)} requests")
        constrained = router.run(requests, constrained=True)
        run.log("[grammar] unconstrained run (comparison)")
        unconstrained = router.run(requests)

        rows = []
        all_valid = True
        unconstrained_valid = 0
        for cg, ug in zip(constrained, unconstrained, strict=True):
            cv = validate(cg.text or "")
            uv = validate(ug.text or "")
            all_valid = all_valid and cv["schema_ok"]
            unconstrained_valid += 1 if uv["schema_ok"] else 0
            rows.append(
                {
                    "rid": cg.request_id,
                    "adapter": cg.adapter,
                    "constrained_schema_ok": cv["schema_ok"],
                    "constrained_value": cv["value"],
                    "unconstrained_schema_ok": uv["schema_ok"],
                }
            )

        verdict = "GO" if all_valid else "NO-GO"
        result = {
            "verdict": verdict,
            "gate_all_constrained_schema_valid": all_valid,
            "constrained_valid": sum(1 for r in rows if r["constrained_schema_ok"]),
            "unconstrained_valid": unconstrained_valid,
            "total": len(rows),
            "schema": Route.model_json_schema(),
            "config": {
                "n_ctx": args.n_ctx,
                "n_seq_max": args.n_seq_max,
                "max_tokens": args.max_tokens,
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
                    "constrained_valid": result["constrained_valid"],
                    "unconstrained_valid": unconstrained_valid,
                    "total": len(rows),
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
