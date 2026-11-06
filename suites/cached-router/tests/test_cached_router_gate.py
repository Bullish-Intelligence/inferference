"""Cached shared-prefix routing gate (validation half of the cached compositions).

The validation half of the cached compositions; the perf half is the runner's
own cached-vs-cold timing. Drives the benchkit runner end-to-end (CLI + verdict + artifact) in both
modes:

  plain       — the cached-restore continuation is token-exact vs the isolated
                batch-1 full-prompt baseline, per adapter (GO/NO-GO);
  constrained — cached+grammar outputs are schema-valid AND token-exact vs the
                cold full re-prefill wave (GO/NO-GO).

GPU-only: skips (never fails) when the env lacks CUDA-offload lib / model /
probe adapters / tokenizer.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from tests.asset_paths import asset_path

from inferference.gpu_env import gpu_env_ready

_REPO = Path(__file__).resolve().parents[3]
_MODEL = asset_path("MODEL_ORNITH")
_PROBE_DIR = asset_path("MODEL_PROBE_DIR")
_TOKENIZER = asset_path("TOKENIZER_ORNITH")


def _ready() -> bool:
    # Backend-agnostic: CUDA_VISIBLE_DEVICES means nothing to the Vulkan build,
    # and requiring it here SKIPS this gate on a rig that can run it.
    if not gpu_env_ready():
        return False
    if not (
        _MODEL.exists()
        and (_PROBE_DIR / "probe-a.gguf").exists()
        and (_PROBE_DIR / "probe-b.gguf").exists()
        and _TOKENIZER.exists()
    ):
        return False
    try:
        import llama_cpp
    except Exception:
        return False
    return bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())


pytestmark = pytest.mark.skipif(
    not _ready(),
    reason="requires CUDA-offload lib + Ornith model + probe adapters (GPU env)",
)


def _run_runner(artifacts: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "inferference.benchkit.runners.cached_router",
        "--model",
        str(_MODEL),
        "--lora-a",
        str(_PROBE_DIR / "probe-a.gguf"),
        "--lora-b",
        str(_PROBE_DIR / "probe-b.gguf"),
        "--artifacts",
        str(artifacts),
        *extra,
    ]
    return subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, check=False)


def test_cached_router_plain_and_constrained_verdict_go(tmp_path: Path) -> None:
    """Both modes must end GO (equivalence + validity), not just exit 0."""
    for extra in ((), ("--constrained", "--tokenizer-dir", str(_TOKENIZER))):
        name = "constrained" if extra else "plain"
        r = _run_runner(tmp_path / name, *extra)
        assert r.returncode == 0, f"cached_router {name} rc={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        assert '"verdict": "GO"' in r.stdout, f"cached_router {name} did not end GO:\n{r.stdout}"
