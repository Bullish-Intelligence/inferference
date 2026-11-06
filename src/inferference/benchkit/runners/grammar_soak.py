"""Soak leg — long grammar-constrained run with no correctness verdict.

The soak produces artifacts only (perf/robustness, no gate verdict beyond the
run's own fail-closed accounting): a long json_workload-style run at a fixed
requests count, reusing the benchkit gates' ``run_once`` / ``aggregate_outcomes``
(the ported ``soak_grammar.py`` machinery). The CPU accounting tests for that
machinery already live in ``tests/test_benchkit_contracts.py``; this runner is
the nightly soak body (bench.yml soak leg / manual).

Requires a fork lib; GPU-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..env import gpu_snapshot, require_ci_device, require_gpu0_only
from ..gates import _model_for_schema, _resolve_tokenizer, aggregate_outcomes, run_once
from ..runenv import RunContext
from ..workload import load as load_workload

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--requests", type=int, default=200, help="soak length (the perf leg uses ~10)")
    ap.add_argument("--artifacts", required=True, type=Path)
    ap.add_argument("--tokenizer", default=os.environ.get("TOKENIZER_DEFAULT"))
    ap.add_argument("--n-ctx", type=int, default=512)
    ap.add_argument("--seed", type=int, default=17001)
    args = ap.parse_args()

    require_ci_device()
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    gpu_before = gpu_snapshot()
    run = RunContext(args.artifacts, name="project17-grammar-soak", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()

    workload = load_workload("json_workload")
    registry = workload.schema_registry
    selected = workload.select(args.requests)

    from llama_cpp import Llama  # noqa: PLC0415

    llm = Llama(
        model_path=str(args.model), n_ctx=args.n_ctx, n_batch=128, n_gpu_layers=-1, seed=args.seed, verbose=False
    )
    after_load = gpu_snapshot()
    require_gpu0_only(after_load)
    tokenizer = _resolve_tokenizer(args.tokenizer, allow_network=False)
    # tokenizer path resolution is a hard gate (local_files_only); the object
    # itself is not needed by the unconstrained soak path
    del tokenizer
    outcomes = []
    for index, item in enumerate(selected):
        schema_id = item["schema_id"]
        validator = _model_for_schema(registry["schemas"][schema_id])
        outcome, _ = run_once(
            llm=llm,
            grammar=None,
            validator=validator,
            prompt=item["prompt"],
            prompt_tokens_count=len(llm.tokenize(item["prompt"].encode("utf-8"), add_bos=False, special=True)),
            request_index=index,
            max_tokens=item["max_tokens"],
            constrained=False,
            synchronize_gpu=True,
        )
        outcomes.append(outcome)
        run.log(f"{index + 1}/{len(selected)} {outcome['status']}")

    summary = aggregate_outcomes(outcomes)
    summary.update(
        {
            "corpus": workload.name,
            "corpus_version": workload.version,
            "mode": "soak",
            "gpu_before": gpu_before,
            "gpu_after_load": after_load,
            "gpu_after_run": gpu_snapshot(),
        }
    )
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "mode": "soak",
                "status_counts": summary["status_counts"],
                "hard_failure_count": summary["hard_failure_count"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if summary["hard_failure_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
