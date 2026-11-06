#!/usr/bin/env python3
"""Canary per-branch report — suite-driven: run every per-branch SUITE against a
STOCK upstream build and classify pass / fail / skip.

The nightly canary (ci/workflows/canary.yml, cron "0 6 * * *") builds stock
llama.cpp master (cpu-light, `--repo` override) and runs each suite in
`suites/` against that stock lib through the same fail-closed executor as
suites.yml (ci/runner/run-suite.sh — capability probe → RUN/SKIP/FAIL). This is
the "should we bump the anchor?" decision input:

  - PASS  — the suite passed on stock: upstream adopted equivalent semantics;
            the branch may be droppable after its own tests confirm.
  - SKIP  — capability absent on stock (normal): the probe correctly gates
            (skip_on_stock suites, e.g. P2).
  - FAIL  — the suite failed on stock: this modification broke against new
            upstream (fix it) or the stock lib itself is broken.

Expected on current stock master: `feat/p2-mixed-batch-lora: skip` (upstream
has NOT adopted the P2 routing surface) and `feat/nanbeige-arch: pass` (upstream
HAS natively adopted the nanbeige arch — LLM_ARCH_NANBEIGE at
src/llama-arch.h:151 on stock). That pass-on-stock is the "branch may be
droppable" signal — report it as such, it is a decision input, not an action.

Plus the ABI canary: compile the cffi binding (our 0.3.34-anchored cdef)
against the stock headers + lib. A COMPILE failure means upstream changed the
ABI surface (renamed/removed symbol, signature change) — that is the red
canary: "an anchor bump will need cdef reconciliation".

Usage:
    ci/library/.venv/bin/python ci/canary/canary-report.py --out <stock-out-dir> \
        [--gate-python ci/gates/.venv/bin/python] [--library-python ci/library/.venv/bin/python]
    env: rig-env.sh sourced (MODEL_NANBEIGE), LLAMA_CPP_LIB_PATH / LD_LIBRARY_PATH
         for the gate python
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent  # inferference repo root
SUITES = sorted(p.name for p in (REPO / "suites").iterdir() if (p / "suite.json").exists())


def _run(cmd: list[str], env: dict[str, str] | None = None, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, check=False)


def abi_canary(out: Path, gate_python: Path, env: dict[str, str]) -> str:
    """Compile the cffi binding against the STOCK headers+lib (cpu-light).

    Run with --no-fork-symbols: the fork-paired cdef must NOT be compiled
    against stock (upstream lacks the fork decls, so a plain bindgen fails on
    implicit declarations — capability absence, not drift). Stripping the fork
    section leaves a pure ABI-surface probe: a compile failure then means
    upstream moved a struct/signature vs our 0.3.34-anchored cdef — the red
    canary. A successful compile + import means no drift.
    """
    env = dict(env)
    env.pop("LLAMA_CPP_LIB_PATH", None)
    r = _run(
        [
            str(gate_python),
            str(REPO / "ci" / "gates" / "cffi_bindgen.py"),
            "--build-dir",
            str(out),
            "--profile",
            "cpu-light",
            "--no-fork-symbols",
            "--out-dir",
            str(out / "canary-bindgen"),
        ],
        env=env,
    )
    return "none" if r.returncode == 0 else "compile-failure"


def run_suite(suite: str, out: Path, library_python: Path, env: dict[str, str]) -> tuple[str, str, str]:
    """Run one suite against ``out`` via run-suite.sh; return (suite, branch, verdict)."""
    branch = suite
    manifest = json.loads((out / "build-manifest.json").read_text())
    suite_dir = REPO / "suites" / suite
    suite_json = json.loads((suite_dir / "suite.json").read_text())
    # A retired fork branch is reported as a STOCK capability: the suite.json's
    # ``source`` key ("stock") marks suites that no longer map to a fork branch
    # (nanbeige — upstream natively owns the arch since the b10233 anchor bump,
    # the fork branch was retired). The branch key reads "<suite> (stock
    # capability)" so the report never implies a fork branch that no longer
    # exists (field finding 2026-08-02: feat/nanbeige-arch deleted).
    if suite_json.get("source") == "stock":
        branch = f"{suite} (stock capability)"
    else:
        branch = suite_json.get("branch", suite)

    r = _run(
        [str(REPO / "ci" / "runner" / "run-suite.sh"), "--suite", suite, "--out", str(out)],
        env=env,
        timeout=3600,
    )
    verdict = "fail"
    for line in r.stdout.splitlines():
        if line.startswith("[suite-verdict]") and suite in line:
            verdict = line.split(":", 2)[-1].strip().lower()
    if verdict == "fail" and r.returncode == 0:
        # a SKIP exits 0 with no verdict line edge case -> re-read the classifier
        if "-> SKIP" in r.stdout:
            verdict = "skip"
    print(f"  suite {suite} vs stock {manifest.get('ref_built')}: {verdict}")
    if r.stdout.strip():
        for line in r.stdout.splitlines()[-8:]:
            print(f"    {line}")
    return suite, branch, verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path, help="stock cpu-light build out dir")
    ap.add_argument("--gate-python", type=Path, default=Path("ci/gates/.venv/bin/python"))
    ap.add_argument("--library-python", type=Path, default=Path("ci/library/.venv/bin/python"))
    ap.add_argument("--report", type=Path, default=Path("ci/.ci-artifacts/canary-report.json"))
    args = ap.parse_args()

    if not (args.out / "build-manifest.json").is_file():
        raise SystemExit(f"no build-manifest.json in {args.out} — build stock first")

    manifest = json.loads((args.out / "build-manifest.json").read_text())
    env = dict(os.environ)
    env["LIBRARY_PYTHON"] = str(args.library_python)

    report: dict[str, object] = {
        "stock": {
            "repo": manifest.get("repo"),
            "ref_built": manifest.get("ref_built"),
            "ref_requested": manifest.get("ref_requested"),
            "profile": manifest.get("profile"),
        },
        "abi_drift": abi_canary(args.out, args.gate_python, env),
        "branches": {},
        "suites": {},
    }

    print(f">> canary: running {len(SUITES)} suite(s) against stock {manifest.get('ref_built')}")
    for suite in SUITES:
        name, branch, verdict = run_suite(suite, args.out, args.library_python, env)
        report["branches"][branch] = verdict  # type: ignore[index]
        report["suites"][name] = verdict  # type: ignore[index]

    # The nanbeige pass-on-stock signal is a decision input; with the fork branch
    # retired (b10233 anchor bump) it is now a stock-capability confirmation.
    nanbeige = report["branches"].get("nanbeige-arch (stock capability)")
    if nanbeige == "pass":
        print(
            "\n>> canary finding: nanbeige-arch PASSED on stock — upstream natively owns "
            "the arch (LLM_ARCH_NANBEIGE); the fork branch was retired at the b10233 "
            "anchor bump, the suite is now the stock drift detector."
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("\n>> canary report written to", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
