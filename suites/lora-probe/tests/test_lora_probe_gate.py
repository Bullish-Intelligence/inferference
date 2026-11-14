"""LoRA-delta go/no-go gate (run_lora_probe validation half).

The probe's validation half: this suite drives the benchkit runner end-to-end.
The probe greedy-decodes a fixed prompt on the SAME base model three ways and
compares continuations: base (no adapter), base + probe-a, base + probe-b.
GO requires base reproducible AND base != probe-a (delta applied on this
arch/lib) AND probe-a != probe-b (per-adapter distinct). NO-GO means the delta
is silently dropped — the healthy-GPU-path tripwire.

GPU-only: skips (never fails) when the env lacks GPU-offload lib / model /
probe adapters.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from tests.asset_paths import asset_path

from inferference.gpu_env import gpu_env_ready

_REPO = Path(__file__).resolve().parents[3]
_MODEL = asset_path("MODEL_DEFAULT")
_PROBE_DIR = asset_path("MODEL_PROBE_DIR")


def _ready() -> bool:
    # Backend-agnostic: a vendor device variable means nothing to a build for another backend,
    # and requiring it here SKIPS this gate on a rig that can run it.
    if not gpu_env_ready():
        return False
    if not (_MODEL.exists() and (_PROBE_DIR / "probe-a.gguf").exists() and (_PROBE_DIR / "probe-b.gguf").exists()):
        return False
    try:
        import llama_cpp
    except Exception:
        return False
    return bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())


pytestmark = pytest.mark.skipif(
    not _ready(),
    reason="requires GPU-offload lib + MODEL_DEFAULT + probe adapters (GPU env)",
)


def test_lora_probe_verdict_go(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "inferference.benchkit.runners.lora_probe",
        "--model",
        str(_MODEL),
        "--lora-a",
        str(_PROBE_DIR / "probe-a.gguf"),
        "--lora-b",
        str(_PROBE_DIR / "probe-b.gguf"),
        "--artifacts",
        str(tmp_path),
    ]
    r = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"lora_probe rc={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert '"verdict": "GO"' in r.stdout, f"lora_probe did not end GO:\n{r.stdout}"
