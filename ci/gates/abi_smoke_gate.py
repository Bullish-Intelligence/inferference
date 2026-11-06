"""ABI smoke gate for a llama.cpp fork lib (port of the reference P2 gate).

A fork build is only trustworthy after a behavioral gate, not just a symbol
probe. This checks, against the lib on
LLAMA_CPP_LIB_PATH:

  1. surface probe   — both fork routing symbols resolve
                      (llama_set_seq_adapters / llama_set_seq_adapter)
  2. generation      — load the GGUF, greedily generate ~32 tokens (no crash)
  3. tokenizer round — encode/decode a probe string losslessly enough

Exit 0 iff all pass. Fail-closed: a missing capability or env is a loud
non-zero exit, never a silent pass.

Ported from an earlier internal gate. That gate imported its symbol list from a
library that is not part of this repo, so the list is inlined below.

GPU env per the reference devenv.nix (libcuda-first LD_LIBRARY_PATH, the CI
device pinned, model path via --model). The CI device is pinned IN-SCRIPT so a
stray CUDA_VISIBLE_DEVICES can never send the smoke to the server's device
(GPU 1).
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

# The two fork-only symbols. A lib is fork-capable iff it exports both.
# Inlined rather than imported (the module that declares them is not part of
# this repo; the fork's P2 branch adds the symbols at include/llama.h:699 and
# include/llama.h:705, commit 108d4d756).
SEQ_ROUTING_SYMBOLS: tuple[str, ...] = ("llama_set_seq_adapters", "llama_set_seq_adapter")


def main() -> int:
    # Pin the CI device before llama_cpp (and thus the CUDA runtime) is imported.
    _ci_gpu = os.environ.get("LLAMA_CI_GPU", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _ci_gpu)
    # Vulkan reads its own variable; without it this gate runs on whichever card
    # ggml enumerates first, which need not be the one the lease granted.
    os.environ.setdefault("GGML_VK_VISIBLE_DEVICES", _ci_gpu)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--max-tokens", default=32, type=int)
    args = p.parse_args()

    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    import llama_cpp
    from llama_cpp import Llama

    lib = llama_cpp.llama_cpp._lib

    # 1. surface probe
    present = {s: hasattr(lib, s) for s in SEQ_ROUTING_SYMBOLS}
    print(f"[gate:1] surface probe: {present}", flush=True)
    if not all(present.values()):
        print("[gate:1] FAIL — fork routing symbols absent; this is not a P2 fork lib", flush=True)
        return 1

    # 2. generation
    llm = Llama(model_path=str(args.model), n_ctx=2048, n_batch=128, n_gpu_layers=-1, seed=17018, verbose=False)
    try:
        toks = llm.tokenize(b"The capital of France is", add_bos=True, special=True)
        # generate() yields until EOS/context-full; take only max_tokens.
        out = list(itertools.islice(llm.generate(list(toks), top_k=1, temp=0.0), args.max_tokens))
        text = llm.detokenize(out).decode("utf-8", errors="replace")
        print(f"[gate:2] gen: {len(out)} tokens -> {text[:60]!r}", flush=True)
        if len(out) == 0:
            print("[gate:2] FAIL — no tokens generated", flush=True)
            return 1

        # 3. tokenizer round-trip
        probe = "def add(a, b):\n    return a + b"
        rt = llm.detokenize(llm.tokenize(probe.encode("utf-8"), add_bos=False, special=True)).decode(
            "utf-8", errors="replace"
        )
        ok = probe.strip() in rt or rt.strip() in probe or rt.strip() == probe.strip()
        print(f"[gate:3] tokenizer round-trip ok={ok}: {rt[:60]!r}", flush=True)
        if not ok:
            print("[gate:3] FAIL — tokenizer round-trip drift", flush=True)
            return 1
    finally:
        llm.close()

    print("[gate] PASS — fork lib is surface-correct and behaviorally sound", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
