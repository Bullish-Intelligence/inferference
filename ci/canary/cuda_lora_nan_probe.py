#!/usr/bin/env python3
"""Committed CUDA LoRA NaN reproducer (Task 1, step-6 kickoff prompt).

Characterizes the stock CUDA LoRA NaN precisely: a FULL-coverage adapter (all
22 nanbeige layers x ffn_up/ffn_gate pairs) produces all-NaN logits on the
CUDA LoRA path, while minimal-coverage adapters stay clean and CPU is always
correct. Reproduced on the STANDARD ``llama_set_adapters_lora`` path (no
P2/hats fork code) on both anchors (step-5 report 13-…-REPORT.md Task 3).

Given a lib dir + model + adapter, this probe decodes once (batch-1 prefill)
and reports ``nan_count`` / ``argmax`` of the first-token logits, optionally
repeating decodes in-process and/or forcing ``CUDA_LAUNCH_BLOCKING=1``.

Usage (inside ``devenv shell``; rig-env.sh sourced; GPU flock wrapper):

    ci/runner/gpu-serialized.sh ci/library/.venv/bin/python \\
        ci/canary/cuda_lora_nan_probe.py \\
        --lib <build out dir> --model <gguf> --adapter <lora.gguf> \\
        [--path standard|seq] [--reps N] [--expect clean|nan|any] \\
        [--launch-blocking] [--verbose] [--prompt "In a distant galaxy,"]

Exit codes: 0 = probe ran and (if --expect) matched expectation; 1 = NaN
reproduced but --expect clean, or clean but --expect nan; 2 = decode error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_NV = 0  # set after load


def _report(rep: int, rc: int, nan_count: int, n_vocab: int, argmax: object) -> dict[str, object]:
    return {
        "rep": rep,
        "decode_rc": rc,
        "nan_count": nan_count,
        "n_vocab": n_vocab,
        "all_nan": nan_count == n_vocab,
        "argmax": argmax,
    }


def _run_standard(lcpp: Any, llm: Any, toks: list[int], n_vocab: int) -> dict[str, object]:
    """Decode once via the standard path (llama_set_adapters_lora already pinned)."""
    import numpy as np

    arr = (lcpp.llama_token * len(toks))(*toks)
    batch = lcpp.llama_batch_get_one(arr, len(toks))
    rc = int(lcpp.llama_decode(llm._ctx.ctx, batch))
    lg = np.ctypeslib.as_array(lcpp.llama_get_logits_ith(llm._ctx.ctx, len(toks) - 1), shape=(n_vocab,)).copy()
    nan_count = int(np.isnan(lg).sum())
    argmax: object = int(np.nanargmax(lg)) if not np.all(np.isnan(lg)) else "NAN"
    return _report(0, rc, nan_count, n_vocab, argmax)


def _run_seq(lcpp: Any, llm: Any, toks: list[int], n_vocab: int) -> dict[str, object]:
    """Decode once via the P2 seq-routed path (fork surface; adapter pinned to seq 0)."""
    import numpy as np

    from inferference.seq_routing import SeqRoutingBinding

    adapter = llm._lora_adapter  # loaded via lora_path by the caller
    bind = SeqRoutingBinding(lcpp)
    ctx = llm._ctx.ctx
    bind.set_seq_adapters(ctx, [adapter])
    bind.set_seq_adapter(ctx, 0, 0)
    arr = (lcpp.llama_token * len(toks))(*toks)
    batch = lcpp.llama_batch_get_one(arr, len(toks))
    rc = int(lcpp.llama_decode(ctx, batch))
    lg = np.ctypeslib.as_array(lcpp.llama_get_logits_ith(ctx, len(toks) - 1), shape=(n_vocab,)).copy()
    nan_count = int(np.isnan(lg).sum())
    argmax: object = int(np.nanargmax(lg)) if not np.all(np.isnan(lg)) else "NAN"
    return _report(0, rc, nan_count, n_vocab, argmax)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", type=Path, default=None, help="build out dir with lib/ (default: LLAMA_CPP_LIB_PATH env)")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--prompt", default="In a distant galaxy,")
    ap.add_argument("--n-ctx", default=512, type=int)
    ap.add_argument("--n-batch", default=128, type=int)
    ap.add_argument("--path", default="standard", choices=("standard", "seq"))
    ap.add_argument("--reps", default=1, type=int, help="repeat decodes in-process (kv reset between reps)")
    ap.add_argument("--expect", default="any", choices=("any", "clean", "nan"))
    ap.add_argument("--launch-blocking", action="store_true", help="set CUDA_LAUNCH_BLOCKING=1 before import")
    ap.add_argument("--verbose", action="store_true", help="show llama context log (compute buffer sizes)")
    args = ap.parse_args()

    if args.launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if args.lib is not None:
        lib_dir = args.lib.resolve()
        os.environ["LLAMA_CPP_LIB_PATH"] = str(lib_dir / "lib")
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise SystemExit("no lib: pass --lib or set LLAMA_CPP_LIB_PATH")

    import llama_cpp  # noqa: PLC0415  (import after LLAMA_CPP_LIB_PATH is set)

    lcpp: Any = llama_cpp.llama_cpp
    llm: Any = llama_cpp.Llama(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_gpu_layers=-1,
        seed=17018,
        verbose=args.verbose,
        lora_path=str(args.adapter),
    )
    n_vocab = int(lcpp.llama_vocab_n_tokens(lcpp.llama_model_get_vocab(llm._model.model)))
    toks = list(llm.tokenize(args.prompt.encode("utf-8"), add_bos=False, special=True))

    reports: list[dict[str, object]] = []
    try:
        run = _run_seq if args.path == "seq" else _run_standard
        for rep in range(args.reps):
            if rep > 0:
                llm.reset()
            reports.append(run(lcpp, llm, toks, n_vocab))
    finally:
        llm.close()

    for report in reports:
        print(
            json.dumps(
                {
                    "lib": os.environ["LLAMA_CPP_LIB_PATH"],
                    "path": args.path,
                    "model": str(args.model),
                    "adapter": str(args.adapter),
                    **report,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.expect == "any":
        return 0
    nan_reproduced = any(r["all_nan"] for r in reports)
    if args.expect == "nan":
        return 0 if nan_reproduced else 1
    # expect clean
    return 1 if nan_reproduced else 0


if __name__ == "__main__":
    sys.exit(main())
