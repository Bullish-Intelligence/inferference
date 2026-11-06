"""LoRA-delta go/no-go probe for the current arch/lib.

Greedy-decodes a fixed prompt on the SAME base model three ways and compares the
token continuations: base (no adapter), base + probe-a (synthetic large-
perturbation adapter), base + probe-b (a second, distinct synthetic adapter).

Verdict (deterministic greedy, temperature 0):
  GO   if base is reproducible AND base != probe-a AND probe-a != probe-b
       (delta applied + per-adapter distinct)
  NO-GO if base == probe-a (delta silently dropped on this arch/lib)

GPU-only: CUDA_VISIBLE_DEVICES=0, n_gpu_layers=-1, LLAMA_CPP_LIB_PATH pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..env import gpu_snapshot, require_ci_device
from ..runenv import RunContext

REPO = Path(__file__).resolve().parents[3]


def greedy_tokens(
    model_path: str, lora_path: str | None, prompt: str, n_tokens: int, n_ctx: int, seed: int
) -> list[int]:
    from llama_cpp import Llama  # noqa: PLC0415

    kwargs = dict(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1, seed=seed, verbose=False)
    if lora_path is not None:
        kwargs["lora_path"] = lora_path
    llm = Llama(**kwargs)
    try:
        toks = list(llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))
        out: list[int] = []
        for tok in llm.generate(toks, temp=0.0, top_k=1):
            out.append(int(tok))
            if len(out) >= n_tokens:
                break
        return out
    finally:
        llm.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--lora-a", required=True, type=Path)
    p.add_argument("--lora-b", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--n-tokens", default=48, type=int)
    p.add_argument("--n-ctx", default=2048, type=int)
    p.add_argument("--seed", default=17018, type=int)
    args = p.parse_args()

    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    gpu_before = gpu_snapshot()
    run = RunContext(args.artifacts, name="project17-lora-probe", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()

    def note(msg: str) -> None:
        run.log(f"[probe] {msg}")

    note("run 1/4: base")
    base = greedy_tokens(str(args.model), None, args.prompt, args.n_tokens, args.n_ctx, args.seed)
    note("run 2/4: base rerun")
    base2 = greedy_tokens(str(args.model), None, args.prompt, args.n_tokens, args.n_ctx, args.seed)
    note("run 3/4: base + probe-a")
    a = greedy_tokens(str(args.model), str(args.lora_a), args.prompt, args.n_tokens, args.n_ctx, args.seed)
    note("run 4/4: base + probe-b")
    b = greedy_tokens(str(args.model), str(args.lora_b), args.prompt, args.n_tokens, args.n_ctx, args.seed)
    # No require_gpu0_only here: each run closes its model (no resident snapshot);
    # GPU isolation is enforced by CUDA_VISIBLE_DEVICES=0 + the gpu-serialized flock.
    note("all runs complete; comparing")

    base_reproducible = base == base2
    delta_applied = base != a
    per_adapter_distinct = a != b
    verdict = "GO" if (delta_applied and per_adapter_distinct and base_reproducible) else "NO-GO"

    result = {
        "verdict": verdict,
        "checks": {
            "base_reproducible": base_reproducible,
            "delta_applied_base_vs_a": delta_applied,
            "per_adapter_distinct_a_vs_b": per_adapter_distinct,
        },
        "prompt": args.prompt,
        "n_tokens": args.n_tokens,
        "seed": args.seed,
        "tokens": {"base": base, "base_rerun": base2, "probe_a": a, "probe_b": b},
        "first_divergence_base_vs_a": next((i for i, (x, y) in enumerate(zip(base, a, strict=True)) if x != y), None),
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "model": str(args.model),
        "lora_a": str(args.lora_a),
        "lora_b": str(args.lora_b),
        "lib": os.environ.get("LLAMA_CPP_LIB_PATH"),
    }
    (run.dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": verdict, "checks": result["checks"]}, indent=2), flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
