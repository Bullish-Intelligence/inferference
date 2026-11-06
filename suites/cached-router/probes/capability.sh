#!/usr/bin/env bash
# cached-router suite capability probe: does the pointed-at lib (1) load,
# (2) support GPU offload (the gates are CUDA-mechanism gates), and are the rig
# assets the runner needs present (Ornith model, probe LoRAs, Ornith tokenizer)?
#
# This is a library-mechanism gate (not a fork-symbol gate): on the integration
# CUDA build it must PASS (the suite RUNS); on a CPU stock build it FAILS and
# the suite SKIPs via skip_on_stock (the canary's stock build is CPU-only and
# cannot exercise CUDA decode gates).
#
# Usage: probes/capability.sh [<lib-dir>]   (default: $LLAMA_CPP_LIB_PATH)
set -euo pipefail

lib_dir="${1:-${LLAMA_CPP_LIB_PATH:-}}"
if [[ -z "${lib_dir}" || ! -d "${lib_dir}" ]]; then
  echo "!! cached-router capability.sh: no lib dir (pass one or set LLAMA_CPP_LIB_PATH)" >&2
  exit 1
fi

export LLAMA_CPP_LIB_PATH="${lib_dir}"
export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LLAMA_CPP_LIB_PATH}:$(dirname "$(gcc -print-file-name=libstdc++.so)"):$(dirname "$(gcc -print-file-name=libgomp.so)"):${LD_LIBRARY_PATH:-}"
python="${LIBRARY_PYTHON:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/ci/library/.venv/bin/python}"

"${python}" - <<'PY'
import os
import sys
from pathlib import Path


def _required(name: str) -> Path:
    """A model-inventory path. Unset is a loud failure, never a default."""
    value = os.environ.get(name, "").strip()
    if not value:
        print(
            f"!! cached-router probe: {name} is not set; source ci/runner/rig-env.sh "
            "(roles come from the untracked ci/runner/rig-local.sh)",
            file=sys.stderr,
        )
        sys.exit(1)
    return Path(value)


try:
    import llama_cpp
except Exception as exc:
    print(f"!! cached-router probe: llama_cpp import failed: {exc}", file=sys.stderr)
    sys.exit(1)

lib_dir = os.environ["LLAMA_CPP_LIB_PATH"]
if not llama_cpp.llama_cpp.llama_supports_gpu_offload():
    print(">> [cached-router probe] ABSENT: lib has no GPU offload (CPU build) — suite skips on stock")
    sys.exit(1)

model = _required("MODEL_ORNITH")
probe_dir = _required("MODEL_PROBE_DIR")
tokenizer = _required("TOKENIZER_ORNITH")
missing = [p for p in (model, probe_dir / "probe-a.gguf", probe_dir / "probe-b.gguf", tokenizer) if not p.exists()]
if missing:
    print(f"!! cached-router probe: missing assets: {missing}", file=sys.stderr)
    sys.exit(1)

print(f">> [cached-router probe] PASS: {lib_dir} loads, GPU offload, assets present")
sys.exit(0)
PY
