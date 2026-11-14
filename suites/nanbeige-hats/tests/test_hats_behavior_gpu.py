"""GPU token-exact hats behavior gate (S5).

Drives the hats surface (``LoopRoutingBinding``) against a pointed-at hats lib
(``LLAMA_CPP_LIB_PATH``) and asserts the per-pass-hat behavior:

  (a) reproducible — the same hat on pass 0 gives the same output twice;
  (b) differs from base and from either single hat — a pass-0 hat, a pass-1 hat,
      and the two different adapters on pass 0 all produce distinct outputs;
  (c) near-identity — a near-zero-scale hat leaves the first-token logits and
      the output ~unchanged (R4 sanity).

The behavioral assertions run through ``ci/canary/hats-validate.py s5`` (the
same harness) against the minimal nanbeige
adapters in the rig cache (``nanbeige-lora-probe/suite-*.gguf``). The minimal
coverage (blk.3 x attn_k/v/output) is deliberate: a full-coverage adapter
exposes a pre-existing stock CUDA LoRA-path NaN (both b10103 and b10233 anchors,
standard llama_set_adapters_lora path); the
hats mechanism itself is what this gate proves, and the minimal adapter keeps
the gate on the healthy CUDA path.

GPU-only: skips (never fails) when the env lacks the hats lib / GPU / model.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.asset_paths import asset_path

from inferference.gpu_env import visible_devices

_REPO = Path(__file__).resolve().parents[3]
_MODEL = asset_path("MODEL_NANBEIGE")
_LORA_DIR = asset_path("MODEL_NANBEIGE_LORA_PROBE")
_LORA_A = _LORA_DIR / "suite-probe-a.gguf"
_LORA_B = _LORA_DIR / "suite-probe-b.gguf"
_LORA_ID = _LORA_DIR / "suite-probe-identity.gguf"


def _hats_gpu_ready() -> bool:
    lib_dir = os.environ.get("LLAMA_CPP_LIB_PATH")
    # Backend-agnostic device visibility; see inferference.gpu_env.
    if not lib_dir or visible_devices() is None:
        return False
    if not (_MODEL.exists() and _LORA_A.exists() and _LORA_B.exists() and _LORA_ID.exists()):
        return False
    try:
        import llama_cpp

        from inferference.seq_routing import library_supports_loop_routing
    except Exception:
        return False
    return library_supports_loop_routing(getattr(llama_cpp.llama_cpp, "_lib", None))


pytestmark = pytest.mark.skipif(
    not _hats_gpu_ready(),
    reason=(
        "requires hats fork GPU env (LLAMA_CPP_LIB_PATH -> hats lib, a GPU device, "
        "MODEL_NANBEIGE, MODEL_NANBEIGE_LORA_PROBE adapters)"
    ),
)


def test_per_pass_hat_behavior() -> None:
    """S5 — the full per-pass-hat behavior gate (a/b/c)."""
    r = subprocess.run(
        [
            sys.executable,
            str(_REPO / "ci/canary/hats-validate.py"),
            "s5",
            "--lib",
            os.environ["LLAMA_CPP_LIB_PATH"],
            "--model",
            str(_MODEL),
            "--lora-a",
            str(_LORA_A),
            "--lora-b",
            str(_LORA_B),
            "--lora-identity",
            str(_LORA_ID),
        ],
        capture_output=True,
        text=True,
        timeout=2400,
    )
    assert r.returncode == 0, f"hats S5 gate failed:\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}"
    assert "[s5] PASS" in r.stdout
