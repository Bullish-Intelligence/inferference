"""Restore into a live multi-seq context: perf half (run_seq_reuse port).

Perf half of the dual runner. Port of
an earlier seq-reuse harness onto
:class:`inferference.benchkit.runners.seq_harness.SeqHarness` (which wraps
``LlamaSeqStateBridge``). Measures, with synchronized GPU evidence:

  * multi-sequence decode correctness (control) — independent sequences in one
    context match their isolated single-sequence baselines;
  * the ``n_seq_max`` portability matrix — a blob captured under one
    ``n_seq_max`` only loads into a context with the *same* ``n_seq_max``
    (``set_data`` returns 0 otherwise — a safe, detectable reject);
  * the router path — restore into seq 1 of a context already holding seq 0,
    verifying both the restored continuation and neighbour isolation, swept
    over checkpoint length, with synchronized restore timing;
  * cross-process restart — capture to disk in one process, restore in a fresh
    process/context and match the same continuation.

Success is deterministic greedy continuation equality only; ``set_data`` return
values and byte counts are recorded but never treated as proof.
GPU-only: CUDA_VISIBLE_DEVICES=0, n_gpu_layers=-1, LLAMA_CPP_LIB_PATH pinned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from ..env import gpu_snapshot, require_ci_device, require_gpu0_only
from ..runenv import RunContext
from .seq_harness import SeqHarness

REPO = Path(__file__).resolve().parents[3]


def restore_child(h: Any, blob_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Second-process half of the cross-process restart test."""
    blob = blob_path.read_bytes()
    ctx = h.make_ctx(meta["n_seq_max"])
    set_return = h.set_seq(ctx, blob, meta["seq_id"])
    # Own-batch decode replays position explicitly, so no Llama.n_tokens replay
    # is needed (that is only required when using the high-level Llama.eval path).
    h.decode(ctx, meta["suffix"], meta["seq_id"], meta["prefix_len"])
    h.sync(ctx)
    restored = h.last_token(ctx)
    h.free(ctx)
    return {
        "set_return": set_return,
        "blob_sha256": hashlib.sha256(blob).hexdigest(),
        "restored_token": restored,
        "match": restored == meta["baseline_token"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    # n_ctx is divided across sequences (n_ctx/n_seq_max cells each), so keep it
    # large enough that the longest prefix fits under every swept n_seq_max.
    parser.add_argument("--n-ctx", default=2048, type=int)
    parser.add_argument("--n-batch", default=128, type=int)
    parser.add_argument("--seed", default=17018, type=int)
    parser.add_argument("--prefix-lengths", default="32,64,128,256")
    parser.add_argument("--repetitions", default=3, type=int)
    # Internal: cross-process restore half.
    parser.add_argument("--restore-meta", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    if args.restore_meta is not None:
        meta = json.loads(args.restore_meta.read_text())
        h = SeqHarness(str(args.model), args.n_ctx, args.n_batch, args.seed)
        result = restore_child(h, Path(meta["blob_path"]), meta)
        print(json.dumps({"crossprocess": result}, sort_keys=True))
        (args.restore_meta.parent / "crossprocess-child.json").write_text(json.dumps(result, indent=2) + "\n")
        h.llm.close()
        return 0 if result["match"] else 1

    gpu_before = gpu_snapshot()
    run = RunContext(args.artifacts, name="project17-seq-reuse", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()

    h = SeqHarness(str(args.model), args.n_ctx, args.n_batch, args.seed)
    after_load = gpu_snapshot()
    require_gpu0_only(after_load)

    toks = h.tokenize(b"Multi sequence router reuse workload. " * 60)
    pa, sa = toks[:24], [toks[24]]  # sequence A: a live neighbour
    lengths = [int(v) for v in args.prefix_lengths.split(",")]

    report: dict[str, Any] = {"n_ctx": args.n_ctx, "n_batch": args.n_batch}

    # 1. Multi-sequence decode control (independent sequences in one context).
    pb0, sb0 = toks[40:72], [toks[72]]
    base_a = h.baseline(1, pa, sa)
    base_b = h.baseline(1, pb0, sb0)
    ctx = h.make_ctx(2)
    h.decode(ctx, pa, 0, 0)
    h.decode(ctx, pb0, 1, 0)
    h.decode(ctx, sa, 0, len(pa))
    ma = h.last_token(ctx)
    h.decode(ctx, sb0, 1, len(pb0))
    mb = h.last_token(ctx)
    h.free(ctx)
    report["multiseq_control"] = {
        "baseline_A": base_a,
        "baseline_B": base_b,
        "multiseq_A": ma,
        "multiseq_B": mb,
        "A_ok": ma == base_a,
        "B_ok": mb == base_b,
    }

    # 2. n_seq_max portability matrix (capture nsm -> restore nsm).
    matrix = []
    for cap in (1, 2, 4):
        src = h.make_ctx(cap)
        h.decode(src, pb0, min(1, cap - 1), 0)
        blob = h.get_seq(src, min(1, cap - 1))
        h.free(src)
        for tgt_nsm in (1, 2, 4):
            tgt = h.make_ctx(tgt_nsm)
            seq = min(1, tgt_nsm - 1)
            ret = h.set_seq(tgt, blob, seq)
            match = None
            if ret > 0:
                h.decode(tgt, sb0, seq, len(pb0))
                h.sync(tgt)
                match = h.last_token(tgt) == base_b
            h.free(tgt)
            matrix.append(
                {
                    "capture_nsm": cap,
                    "restore_nsm": tgt_nsm,
                    "blob_bytes": len(blob),
                    "set_return": ret,
                    "loaded": ret > 0,
                    "match": match,
                }
            )
    report["nsm_portability_matrix"] = matrix

    # 3. Router path: restore B into seq 1 of a live n_seq_max=2 context holding
    #    A on seq 0, swept over checkpoint length, with synchronized restore time.
    router = []
    for length in lengths:
        pb = toks[40 : 40 + length]
        if 40 + length >= len(toks):
            continue
        sb = [toks[40 + length]]
        base = h.baseline(2, pb, sb)
        # capture B under matched n_seq_max=2 from seq 1
        src = h.make_ctx(2)
        h.decode(src, pb, 1, 0)
        h.sync(src)
        blob = h.get_seq(src, 1)
        h.free(src)
        set_returns, restore_ns, restored_ok, isolation_ok = [], [], [], []
        for _ in range(args.repetitions):
            tgt = h.make_ctx(2)
            h.decode(tgt, pa, 0, 0)  # live neighbour
            t = perf_counter_ns()
            ret = h.set_seq(tgt, blob, 1)
            h.sync(tgt)
            restore_ns.append(perf_counter_ns() - t)
            set_returns.append(ret)
            h.decode(tgt, sb, 1, len(pb))
            h.sync(tgt)
            restored_ok.append(h.last_token(tgt) == base)
            h.decode(tgt, sa, 0, len(pa))
            h.sync(tgt)
            isolation_ok.append(h.last_token(tgt) == base_a)
            h.free(tgt)
        router.append(
            {
                "checkpoint_tokens": length,
                "seq_blob_bytes": len(blob),
                "blob_sha256": hashlib.sha256(blob).hexdigest(),
                "baseline_token": base,
                "set_return": set_returns[0],
                "restore_ns_mean": sum(restore_ns) / len(restore_ns),
                "restore_ns_p50": sorted(restore_ns)[(len(restore_ns) - 1) // 2],
                "restored_ok_all": all(restored_ok),
                "isolation_ok_all": all(isolation_ok),
            }
        )
    report["router_path"] = router

    # 4. Cross-process restart: capture to disk here, restore in a fresh process.
    pb = toks[40 : 40 + 64]
    sb = [toks[40 + 64]]
    base = h.baseline(2, pb, sb)
    src = h.make_ctx(2)
    h.decode(src, pb, 1, 0)
    h.sync(src)
    blob = h.get_seq(src, 1)
    h.free(src)
    blob_path = run.dir / "crossprocess-seq.bin"
    blob_path.write_bytes(blob)
    meta = {
        "blob_path": str(blob_path),
        "n_seq_max": 2,
        "seq_id": 1,
        "prefix_len": len(pb),
        "suffix": sb,
        "baseline_token": base,
    }
    meta_path = run.dir / "crossprocess-meta.json"
    meta_path.write_text(json.dumps(meta))
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "inferference.benchkit.runners.seq_reuse",
            "--model",
            str(args.model),
            "--artifacts",
            str(args.artifacts),
            "--n-ctx",
            str(args.n_ctx),
            "--n-batch",
            str(args.n_batch),
            "--restore-meta",
            str(meta_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=os.environ.copy(),
    )
    report["crossprocess"] = {
        "child_exit": child.returncode,
        "child_stdout": child.stdout.strip(),
        "captured_baseline": base,
    }
    if child.returncode != 0:
        run.log(f"[reuse] cross-process child failed: {child.stderr.strip()[:500]}")

    h.llm.close()
    summary = {
        "gpu_before": gpu_before,
        "gpu_after_load": after_load,
        "gpu_after": gpu_snapshot(),
        "report": report,
        "note": (
            "Success = greedy continuation match. set_data return values are "
            "recorded, not trusted. llama_synchronize-synced."
        ),
    }
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
