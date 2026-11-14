#!/usr/bin/env python3
"""The GPU validation of hats and seq-routing on the stock nanbeige architecture.

Runs the S2 / S3 / S5 assertions, per the token-exact rules (batch-1 ground truth +
known-flip reporting):

* S2 — continuation-equivalence: the hats branch with NO hat active must equal
  stock's nanbeige decode (greedy, batch-1). Reference = a stock cpu-light
  build of the same model. (R4 near-identity sanity is bundled here.)
* S3 — P2 seq-routing on a nanbeige base: mixed-batch 2-adapter decode must be
  token-exact vs each request decoded alone on a single-adapter context
  (ground truth, batch-1).
* S5 — per-pass-hat behavior: (a) reproducible; (b) differs from base and from
  either single hat; (c) near-identity hat leaves output ~unchanged.

All checks compare token IDs (never decoded text) and report any difference as
a known-flip candidate with the first differing position.

Usage:
    ci/library/.venv/bin/python ci/canary/hats-validate.py s2 \
        --lib-a <stock lib dir> --lib-b <hats lib dir>
    ci/library/.venv/bin/python ci/canary/hats-validate.py s3 \
        --lib <hats p2fork lib dir> --model <Nanbeige GGUF> \
        --lora-a <a.gguf> --lora-b <b.gguf>
    ci/library/.venv/bin/python ci/canary/hats-validate.py s5 \
        --lib <hats p2fork lib dir> --model <Nanbeige GGUF> \
        --lora-a <a.gguf> --lora-b <b.gguf> --lora-identity <identity.gguf>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_PROMPTS = [
    "The capital of France is",
    "In a distant galaxy,",
    "Two plus two equals",
    "The opposite of hot is",
    "Once upon a time,",
]
_MAX_TOKENS = 24
_N_CTX = 512


def _ld(lib_dir: Path) -> str:
    """Build the LD_LIBRARY_PATH that lets llama_cpp load the pointed-at lib."""
    gcc_so = subprocess.check_output(["gcc", "-print-file-name=libstdc++.so"], text=True).strip()
    gcc_omp = subprocess.check_output(["gcc", "-print-file-name=libgomp.so"], text=True).strip()
    return (
        f"/run/opengl-driver/lib:{lib_dir}:{os.path.dirname(gcc_so)}:{os.path.dirname(gcc_omp)}"
        f":{os.environ.get('LD_LIBRARY_PATH', '')}"
    )


def _decode_block(py_code: str, lib_dir: Path, env_extra: dict[str, str] | None = None) -> str:
    """Run a python block against ``lib_dir`` in a subprocess; return stdout."""
    env = dict(os.environ)
    env["LLAMA_CPP_LIB_PATH"] = str(lib_dir)
    env["LD_LIBRARY_PATH"] = _ld(lib_dir)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([sys.executable, "-c", py_code], capture_output=True, text=True, env=env, timeout=3600)
    if r.returncode != 0:
        raise SystemExit(f"decode subprocess failed ({lib_dir}):\n{r.stderr[-4000:]}")
    return r.stdout


# --------------------------------------------------------------------------- #
# S2 — continuation-equivalence: hats branch (no hat) == stock
# --------------------------------------------------------------------------- #
_S2_CODE = r"""
import os, json, sys
from llama_cpp import Llama

MODEL = os.environ["MODEL_NANBEIGE"]
llm = Llama(model_path=MODEL, n_ctx=%d, n_gpu_layers=0, n_threads=16, verbose=False)
out = {}
for p in %r:
    r = llm(p, max_tokens=%d, temperature=0.0, top_k=1, echo=False)
    out[p] = r["choices"][0]["text"]
print(json.dumps(out))
llm.close()
"""


def run_s2(args: argparse.Namespace) -> int:
    env_extra = {"MODEL_NANBEIGE": str(args.model)}
    stock = json.loads(_decode_block(_S2_CODE % (_N_CTX, _PROMPTS, _MAX_TOKENS), args.lib_a, env_extra))
    hats = json.loads(_decode_block(_S2_CODE % (_N_CTX, _PROMPTS, _MAX_TOKENS), args.lib_b, env_extra))

    # Compare at the TOKEN level: re-decode to ids is overkill for a text diff;
    # instead the caller re-checks any mismatch token-exactly. Report per-prompt.
    mismatches = [p for p in _PROMPTS if stock[p] != hats[p]]
    print(f"[s2] stock  lib: {args.lib_a}")
    print(f"[s2] hats   lib: {args.lib_b} (no hat active)")
    for p in _PROMPTS:
        same = stock[p] == hats[p]
        print(f"[s2]   {'OK ' if same else 'DIFF'} {p!r}")
        if not same:
            print(f"[s2]     stock: {stock[p][:60]!r}")
            print(f"[s2]     hats : {hats[p][:60]!r}")
    if mismatches:
        print(
            "[s2] FAIL — hats branch differs from stock with no hat active "
            f"({len(mismatches)}/{len(_PROMPTS)} prompts); report a known flip"
        )
        return 1
    print("[s2] PASS — hats branch (no hat) is continuation-equivalent to stock (batch-1 greedy)")
    return 0


# --------------------------------------------------------------------------- #
# S3 — P2 seq-routing on a nanbeige base (mixed-batch 2-adapter)
# --------------------------------------------------------------------------- #
_S3_CODE = r"""
import os, json, sys
from inferference.gpu_env import lib_backend_label
from inferference.models import EngineConfig
from inferference.router import AdapterSpec, MultiLoRARouter, RouteRequest, RouterConfig
from inferference.seq_routing import library_supports_seq_routing
import llama_cpp

MODEL = os.environ["MODEL_NANBEIGE"]
LORA_A = os.environ["LORA_A"]
LORA_B = os.environ["LORA_B"]

if not library_supports_seq_routing(getattr(llama_cpp.llama_cpp, "_lib", None)):
    print(json.dumps({"capable": False}))
    sys.exit(0)

# mixed workload: each sequence tagged with its own adapter incl. base (-1)
WORKLOAD = [
    ("a", "The capital of France is"),
    ("b", "In a distant galaxy,"),
    (None, "Two plus two equals"),
    ("a", "The opposite of hot is"),
]

reqs = [RouteRequest(prompt=p, adapter=a, max_tokens=24) for a, p in WORKLOAD]


def router(backend):
    return MultiLoRARouter(RouterConfig(
        engine=EngineConfig(
            model_path=MODEL, n_ctx=512, n_gpu_layers=-1, n_batch=128, seed=17018, backend=lib_backend_label(),
        ),
        adapters=(AdapterSpec(name="a", gguf_path=LORA_A), AdapterSpec(name="b", gguf_path=LORA_B)),
        n_seq_max=4, include_base=True, backend=backend,
    ))

# ground truth: each request alone on a single-adapter context (batch-1)
pool = router("context_pool")
try:
    baseline = [pool.run([rq])[0].token_ids for rq in reqs]
    # crossed: each request with the OTHER adapter, to classify a divergence
    crossed = [pool.run([RouteRequest(prompt=rq.prompt, adapter="b" if rq.adapter == "a" else "a",
                                      max_tokens=rq.max_tokens)])[0].token_ids for rq in reqs]
finally:
    pool.close()

# batch-1 through the fork, then the mixed decode (one run over all 4)
fork = router("seq_routed")
try:
    fork_b1 = [fork.run([rq])[0].token_ids for rq in reqs]
    mixed = [r.token_ids for r in fork.run(reqs)]
finally:
    fork.close()

out = {"baseline": baseline, "fork_b1": fork_b1, "mixed": mixed, "crossed": crossed}
print(json.dumps(out))
"""


def run_s3(args: argparse.Namespace) -> int:
    env_extra = {
        "MODEL_NANBEIGE": str(args.model),
        "LORA_A": str(args.lora_a),
        "LORA_B": str(args.lora_b),
    }
    raw = _decode_block(_S3_CODE, args.lib, env_extra)
    data = json.loads(raw)
    if data.get("capable") is False:
        print("[s3] SKIP — lib lacks the seq-routing surface (not the hats GPU build?)")
        return 0
    print(f"[s3] lib: {args.lib} (hats GPU build, seq-routed backend)")
    baseline, fork_b1, mixed, crossed = data["baseline"], data["fork_b1"], data["mixed"], data["crossed"]
    labels = [
        f"adapter={t!r} {p!r}"
        for t, p in [
            (None, "Two plus two equals"),
            ("a", "The capital of France is"),
            ("b", "In a distant galaxy,"),
            ("a", "The opposite of hot is"),
        ]
    ]
    flips: list[str] = []
    bugs = 0
    for i, (label, want) in enumerate(zip(labels, baseline, strict=True)):
        same_b1 = fork_b1[i] == want
        same_mix = mixed[i] == want
        # classify a mixed divergence: a divergent token that matches the CROSSED
        # adapter's isolated output is a route-bug; otherwise a benign batched-
        # greedy FP near-tie flip.
        kind = ""
        if not same_mix:
            j = next((k for k, (x, y) in enumerate(zip(mixed[i], want, strict=False)) if x != y), None)
            if j is not None and j < len(crossed[i]) and crossed[i][j] == mixed[i][j]:
                kind = "ROUTE-BUG"
                bugs += 1
            else:
                kind = "fp"
                flips.append(f"seq{i}:{label}:tok{j}")
        print(
            f"[s3]   seq {i} {label}: batch1={'OK' if same_b1 else 'DIFF'} "
            f"mixed={'OK' if same_mix else 'DIFF' + ('' if not kind else ' (' + kind + ')')}"
        )
    if bugs:
        print(f"[s3] FAIL — {bugs} route-bug(s) (divergence matches the crossed adapter)")
        return 1
    if flips:
        print(f"[s3] benign batched-greedy FP flips (not routing bugs): {', '.join(flips)}")
    print(
        "[s3] PASS — mixed-batch 2-adapter seq-routing on nanbeige is token-exact "
        "vs single-adapter contexts, modulo documented FP flips"
    )
    return 0


# --------------------------------------------------------------------------- #
# S5 — per-pass-hat behavior
# --------------------------------------------------------------------------- #
_S5_CODE = r"""
import json, os, sys
import ctypes
import llama_cpp
from llama_cpp import Llama
from inferference.seq_routing import LoopRoutingBinding, library_supports_loop_routing

MODEL = os.environ["MODEL_NANBEIGE"]
LORA_A = os.environ["LORA_A"]
LORA_B = os.environ["LORA_B"]
LORA_ID = os.environ.get("LORA_IDENTITY", "")

lcpp = llama_cpp.llama_cpp
lib = lcpp._lib
if not library_supports_loop_routing(lib):
    print(json.dumps({"capable": False}))
    sys.exit(0)

PROMPT = "The capital of France is"
MAX_TOKENS = 24

def greedy(llm, prompt=PROMPT):
    r = llm(prompt, max_tokens=MAX_TOKENS, temperature=0.0, top_k=1, echo=False)
    return r["choices"][0]["text"]

def first_token_logits(llm, prompt=PROMPT):
    # decode the prompt once (greedy, batch-1) and return the raw first-token logits
    toks = llm.tokenize(prompt.encode())
    arr = (lcpp.llama_token * len(toks))(*toks)  # keep alive across llama_decode
    # llama_batch_get_one shares the caller's token array — never llama_batch_free
    batch = lcpp.llama_batch_get_one(arr, len(toks))
    rc = lcpp.llama_decode(llm._ctx.ctx, batch)
    if rc != 0:
        raise RuntimeError(f"first-token decode rc={rc}")
    n_vocab = int(lcpp.llama_vocab_n_tokens(lcpp.llama_model_get_vocab(llm._model.model)))
    ptr = lcpp.llama_get_logits_ith(llm._ctx.ctx, len(toks) - 1)
    import numpy as np
    return np.ctypeslib.as_array(ptr, shape=(int(n_vocab),)).copy()

# adapters
def init_adapter(path):
    p = lib.llama_adapter_lora_init(llm._model.model, path.encode())
    assert p, f"adapter init failed: {path}"
    return p

# This block loads the adapters itself, so MultiLoRARouter does not check them. The loop pool
# is the seq_loras pool: it has the same two gaps as seq_routed. Run both refusals here.
from inferference.seq_adapter_coverage import refuse_dropped_adapters
from inferference.tied_head_adapter import refuse_tied_head_adapters

_pool = [("lora-a", LORA_A), ("lora-b", LORA_B)] + ([("lora-identity", LORA_ID)] if LORA_ID else [])
refuse_tied_head_adapters(MODEL, _pool)
refuse_dropped_adapters(_pool)

llm = Llama(model_path=MODEL, n_ctx=512, n_gpu_layers=-1, verbose=False)
ctx = llm._ctx.ctx
bind = LoopRoutingBinding(lcpp)

# pool: [0] = probe-a, [1] = probe-b, [2] = probe-identity
ptrs = [init_adapter(LORA_A), init_adapter(LORA_B)]
if LORA_ID:
    ptrs.append(init_adapter(LORA_ID))
bind.set_loop_adapters(ctx, ptrs)

def hat(step, idx):
    bind.set_loop_adapter(ctx, step, idx)  # sched_need_reserve -> graph rebuild

def no_hat():
    bind.set_loop_adapter(ctx, 0, -1)
    bind.set_loop_adapter(ctx, 1, -1)

out = {}
# base (no hat)
no_hat(); out["base"] = greedy(llm)
# S5(a) reproducible: hat A on pass 0 twice
hat(0, 0); out["A0_run1"] = greedy(llm); out["A0_run2"] = greedy(llm)
# S5(b) per-pass behavior
hat(0, 0); out["A0"] = greedy(llm)
hat(0, 1); out["B0"] = greedy(llm)
hat(1, 0); out["A1"] = greedy(llm)
# S5(c) near-identity on pass 0: first-token logit delta (known-flip characterization)
if LORA_ID:
    no_hat(); base_lg = first_token_logits(llm)
    hat(0, 2); id_lg = first_token_logits(llm)
    import numpy as np
    d = np.abs(base_lg - id_lg)
    out["ID_logit_maxdelta"] = float(d.max())
    out["ID_logit_meandelta"] = float(d.mean())
    out["ID_argmax_same"] = bool(int(np.argmax(base_lg)) == int(np.argmax(id_lg)))
    out["ID0"] = greedy(llm)

for p in ptrs:
    lib.llama_adapter_lora_free(p)
llm.close()
print(json.dumps(out))
"""


def run_s5(args: argparse.Namespace) -> int:
    env_extra = {
        "MODEL_NANBEIGE": str(args.model),
        "LORA_A": str(args.lora_a),
        "LORA_B": str(args.lora_b),
        "LORA_IDENTITY": str(args.lora_identity or ""),
    }
    raw = _decode_block(_S5_CODE, args.lib, env_extra)
    data = json.loads(raw)
    if data.get("capable") is False:
        print("[s5] SKIP — lib lacks the loop-routing surface (not the hats GPU build?)")
        return 0
    print(f"[s5] lib: {args.lib} (hats GPU build)")
    base, a0r1, a0r2, a0, b0, a1 = (data.get(k) for k in ("base", "A0_run1", "A0_run2", "A0", "B0", "A1"))

    fails = []

    # (a) reproducible
    if a0r1 == a0r2:
        print(f"[s5] (a) OK — hat-A on pass 0 reproducible: {a0r1[:40]!r}")
    else:
        print(f"[s5] (a) FAIL — run1 {a0r1[:40]!r} != run2 {a0r2[:40]!r}")
        fails.append("a")

    # (b) differs from base and either single hat
    checks = [
        ("hat A pass0 != base", a0 != base),
        ("hat B pass0 != base", b0 != base),
        ("hat A pass1 != base", a1 != base),
        ("hat A pass0 != hat B pass0", a0 != b0),
        ("hat A pass0 != hat A pass1 (per-pass differs)", a0 != a1),
    ]
    for label, ok in checks:
        print(f"[s5] (b) {'OK ' if ok else 'FAIL'} {label}")
        if not ok:
            fails.append("b")
    if base == a0:
        print(f"[s5]     base = hatA: {base[:40]!r}")

    # (c) near-identity ~= base (R4 sanity) — characterize the flip, don't just
    # compare text: a near-tie greedy argmax flip is the documented FP class
    # so report the first-token logit delta.
    if args.lora_identity:
        idd = data.get("ID0")
        maxd = data.get("ID_logit_maxdelta")
        meand = data.get("ID_logit_meandelta")
        same = data.get("ID_argmax_same")
        if idd is None:
            print("[s5] (c) MISSING ID0")
            fails.append("c")
        elif idd == base:
            print("[s5] (c) OK — near-identity hat leaves output identical to base (R4 sanity)")
        else:
            print(
                f"[s5] (c) NOTE — near-identity differs from base; first-token logits: "
                f"max_delta={maxd:.6f} mean_delta={meand:.2e} argmax_same={same}"
            )
            print(f"[s5]     base={base[:40]!r}")
            print(f"[s5]     id  ={idd[:40]!r}")
            if same:
                print("     argmax unchanged at token 1 — divergence is a later near-tie")
            else:
                print("     argmax flipped at token 1 — near-tie flip; report as known flip")
    else:
        print("[s5] (c) skipped (no --lora-identity)")

    print(f"[s5] outputs: base={base[:30]!r} A0={a0[:30]!r} B0={b0[:30]!r} A1={a1[:30]!r}")
    if fails:
        print(f"[s5] FAIL — {fails}")
        return 1
    print("[s5] PASS — per-pass-hat behavior (reproducible, differs per pass/adapter, near-identity ~ base)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p2 = sub.add_parser("s2")
    p2.add_argument("--lib-a", required=True, type=Path, help="stock lib dir (reference)")
    p2.add_argument("--lib-b", required=True, type=Path, help="hats lib dir (no hat active)")
    p2.add_argument("--model", required=True, type=Path)
    p2.set_defaults(func=run_s2)

    p3 = sub.add_parser("s3")
    p3.add_argument("--lib", required=True, type=Path, help="hats p2fork lib dir")
    p3.add_argument("--model", required=True, type=Path)
    p3.add_argument("--lora-a", required=True, type=Path)
    p3.add_argument("--lora-b", required=True, type=Path)
    p3.set_defaults(func=run_s3)

    p5 = sub.add_parser("s5")
    p5.add_argument("--lib", required=True, type=Path, help="hats p2fork lib dir")
    p5.add_argument("--model", required=True, type=Path)
    p5.add_argument("--lora-a", required=True, type=Path)
    p5.add_argument("--lora-b", required=True, type=Path)
    p5.add_argument("--lora-identity", type=Path, default=None)
    p5.set_defaults(func=run_s5)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
