"""A/B harness: same workload vs two builds, side-by-side diff.

Mode B (``LLAMA_CPP_LIB_PATH`` lib swap) is the same-ABI iteration vehicle; a
normal A/B compares a stock/fork build against another build's lib dir by
running the SAME workload through each and diffing the aggregate. The diff
applies the same false-positive rule as the gates: validation verdicts compare
token-exact with batch-1 ground truth plus known-flip reporting; perf deltas
compare aggregate tps with a threshold, never raw per-request numbers.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .env import gpu_snapshot, require_gpu0_only


@dataclass(frozen=True, slots=True)
class AbResult:
    """Side-by-side diff of two runs over the same workload."""

    a: dict[str, Any]
    b: dict[str, Any]
    diff: dict[str, Any]
    exit_code: int


def _nested(mapping: dict[str, Any], path: tuple[str, ...]) -> float | None:
    cursor: Any = mapping
    for part in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    return float(cursor) if isinstance(cursor, (int, float)) else None


def run(
    lib_a: Path,
    lib_b: Path,
    *,
    runner: Path,
    model: Path,
    corpus: Path,
    requests: int,
    threshold_percent: float = 5.0,
    extra_args: list[str] | None = None,
) -> AbResult:
    """Run ``runner`` against lib_a then lib_b (Mode B swap), and diff.

    Fail-closed: each leg must exit 0 (its own gates green) before the diff is
    trusted; the GPU-0-only isolation check runs before each leg.
    """
    require_gpu0_only(gpu_snapshot())
    env_a = dict(os.environ)
    env_a["LLAMA_CPP_LIB_PATH"] = str(lib_a)
    env_b = dict(os.environ)
    env_b["LLAMA_CPP_LIB_PATH"] = str(lib_b)
    base = ["--model", str(model), "--corpus", str(corpus), "--requests", str(requests), *(extra_args or [])]

    leg_a = _run_leg(runner, base, env_a)
    leg_b = _run_leg(runner, base, env_b)

    tps_a = _nested(leg_a.get("summary", {}), ("end_to_end", "generation_tokens_per_second"))
    tps_b = _nested(leg_b.get("summary", {}), ("end_to_end", "generation_tokens_per_second"))
    delta = None
    if tps_a is not None and tps_b is not None and tps_a:
        delta = (tps_b - tps_a) * 100 / tps_a
    regression = delta is not None and delta < -threshold_percent
    diff: dict[str, Any] = {
        "generation_tokens_per_second": {"a": tps_a, "b": tps_b, "delta_percent": delta},
        "threshold_percent": threshold_percent,
        "regression": regression,
        "hard_failure_a": leg_a["summary"].get("hard_failure_count", 0),
        "hard_failure_b": leg_b["summary"].get("hard_failure_count", 0),
    }
    return AbResult(a=leg_a, b=leg_b, diff=diff, exit_code=1 if regression else 0)


def _run_leg(runner: Path, args: list[str], env: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        ["ci/library/.venv/bin/python", str(runner), *args, "--artifacts", f"/tmp/ab-{os.getpid()}"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"A/B leg failed (rc={result.returncode}): {result.stderr[-2000:]}")
    summary_path = Path(f"/tmp/ab-{os.getpid()}") / "summary.json"
    return {
        "summary": json.loads(summary_path.read_text()) if summary_path.exists() else {},
        "leg_stdout": result.stdout,
    }
