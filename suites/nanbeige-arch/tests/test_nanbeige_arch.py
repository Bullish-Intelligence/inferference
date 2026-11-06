"""Nanbeige arch suite — the arch-recognition gate as pytest (the canary finding).

Runs the permanent gate ``ci/gates/nanbeige_arch_gate.py`` against the pointed-at
lib (LLAMA_CPP_LIB_PATH) and the Nanbeige GGUF. CPU-only (n_gpu_layers=0). The
capability probe (probes/capability.sh) already established the arch is
recognized; this test re-runs the FULL gate — symbols (integration builds) or
load-only semantics, arch metadata, block_count/num_loops, expanded n_layer —
so the suite's verdict is the gate's verdict.

NOTE (canary finding): upstream stock master has natively adopted the nanbeige
arch, so this suite must PASS against stock too (skip_on_stock: false). The
gate's [arch:1] symbol check is fork-specific; the load checks [arch:2..4] are
the stock-valid portion. The probe already gates on the load checks; if the
symbol check fails on stock, the gate's exit code still reflects it — record it
as informational, and let the load semantics carry the verdict (matching the
canary's ARCH_LOAD_OK probe).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.asset_paths import asset_path

_MODEL_PATH = asset_path("LLAMA_TEST_MODEL")
_GATE = Path(__file__).resolve().parents[3] / "ci" / "gates" / "nanbeige_arch_gate.py"


def _arch_ready() -> bool:
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        return False
    if not _MODEL_PATH.exists():
        return False
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _arch_ready(),
    reason="requires LLAMA_CPP_LIB_PATH + the Nanbeige GGUF",
)


def test_nanbeige_arch_gate_passes_against_pointed_lib() -> None:
    """The nanbeige arch gate exits 0 against this lib (arch recognized + expanded)."""
    python = sys.executable
    r = subprocess.run(
        [python, str(_GATE), "--model", str(_MODEL_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    print(r.stdout)
    if r.stderr:
        print(r.stderr, file=sys.stderr)
    assert r.returncode == 0, (
        f"nanbeige_arch_gate failed (rc={r.returncode}); the lib does not recognize "
        f"the nanbeige arch with the expected 22x2 layer expansion"
    )
