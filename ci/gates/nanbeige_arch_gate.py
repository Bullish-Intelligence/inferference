"""Nanbeige-architecture recognition gate for a fork lib (off-GPU, CPU-only load).

A fork lib built from a tree that contains the nanbeige arch feature must
recognize GGUFs of that architecture. This gate proves, fail-closed, against the lib on
LLAMA_CPP_LIB_PATH:

  1. the BUILT lib exports the nanbeige model machinery (mangled C++ symbols,
     readelf — llama_cpp's ctypes cannot see non-C-API symbols);
  2. general.architecture == "nanbeige" in the loaded model metadata (no
     "unknown architecture" fallback);
  3. the arch keys report num_loops=2 over block_count=22;
  4. the C++ side expanded n_layer to 22*2 = 44 (print_info: n_layer in the
     llama.cpp load log: logical layers are
     expanded to n_layer_phys * n_loops with shared physical weights).

Exits 0 only if all four hold. GPU-free by design (n_gpu_layers=0); the model
path is a GGUF of the nanbeige architecture (22 blocks, 2 loops) from the
machine's model inventory (`MODEL_NANBEIGE`, declared in `ci/runner/rig-local.sh`).
"""

from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
from contextlib import redirect_stderr
from pathlib import Path

_ci_gpu = os.environ.get("LLAMA_CI_GPU", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", _ci_gpu)
# Vulkan reads its own variable; see ci/runner/gpu-lease.sh.
os.environ.setdefault("GGML_VK_VISIBLE_DEVICES", _ci_gpu)

# The llama_model_nanbeige machinery as exported (mangled) by the BUILT lib.
MODEL_SYMBOLS: tuple[str, ...] = (
    "_ZN20llama_model_nanbeige17load_arch_hparamsER18llama_model_loader",
    "_ZN20llama_model_nanbeige17load_arch_tensorsER18llama_model_loader",
    "_ZN20llama_model_nanbeige5graphC1ERK11llama_modelRK16llm_graph_params",
)

EXPECTED = {"block_count": 22, "num_loops": 2, "expanded_n_layer": 44}


def _lib_symbols(lib_dir: Path) -> set[str]:
    so = next(lib_dir.glob("libllama.so.*"))
    out = subprocess.run(["readelf", "-Ws", str(so)], capture_output=True, text=True, check=False)
    return {line.split()[-1] for line in out.stdout.splitlines() if len(line.split()) >= 8}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    args = p.parse_args()

    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the fork lib dir")

    from llama_cpp import Llama

    # 1. built lib exports the nanbeige model machinery
    lib_dir = Path(os.environ["LLAMA_CPP_LIB_PATH"])
    present = {s: s in _lib_symbols(lib_dir) for s in MODEL_SYMBOLS}
    print(f"[arch:1] nanbeige symbols in lib: {present}", flush=True)
    if not all(present.values()):
        print("[arch:1] FAIL — nanbeige model machinery absent from this lib", flush=True)
        return 1

    # 2-4. load with verbose stderr capture (llama.cpp load log)
    err = io.StringIO()
    llm = None
    try:
        with redirect_stderr(err):
            llm = Llama(
                model_path=str(args.model),
                n_ctx=256,
                n_batch=32,
                n_gpu_layers=0,
                verbose=True,
            )
    finally:
        if llm is not None:
            llm.close()

    log = err.getvalue()
    md = getattr(llm, "metadata", {}) if llm else {}

    arch = md.get("general.architecture")
    print(f"[arch:2] general.architecture = {arch!r}", flush=True)
    if arch != "nanbeige":
        print(f"[arch:2] FAIL — arch {arch!r}; expected 'nanbeige'", flush=True)
        return 1

    bc = int(md.get("nanbeige.block_count", 0))
    nl = int(md.get("nanbeige.num_loops", 0))
    print(f"[arch:3] block_count={bc} num_loops={nl}", flush=True)
    if (bc, nl) != (EXPECTED["block_count"], EXPECTED["num_loops"]):
        print(f"[arch:3] FAIL — expected {EXPECTED['block_count']} x {EXPECTED['num_loops']}", flush=True)
        return 1

    m = re.search(r"print_info: n_layer\s*=\s*(\d+)", log)
    expanded = int(m.group(1)) if m else None
    print(f"[arch:4] expanded n_layer from load log = {expanded}", flush=True)
    if expanded != EXPECTED["expanded_n_layer"]:
        print(
            f"[arch:4] FAIL — n_layer={expanded}; expected {EXPECTED['expanded_n_layer']} (22 physical x 2 loops)",
            flush=True,
        )
        return 1

    if "unknown architecture" in log:
        print("[arch:2] FAIL — 'unknown architecture' in load log", flush=True)
        return 1

    print("[arch] PASS — nanbeige arch recognized on this lib", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
