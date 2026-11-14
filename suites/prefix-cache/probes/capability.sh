#!/usr/bin/env bash
# Prefix-cache suite capability probe: does the pointed-at lib (1) load,
# (2) support GPU offload, and are the rig assets present (default-role model)?
#
# Library-mechanism gate: on the integration GPU build it must PASS (RUN); on
# a CPU stock build it FAILs and the suite SKIPs via skip_on_stock (the canary's
# stock build is CPU-only and cannot exercise GPU decode gates).
#
# Usage: probes/capability.sh [<lib-dir>]   (default: $LLAMA_CPP_LIB_PATH)
set -euo pipefail

lib_dir="${1:-${LLAMA_CPP_LIB_PATH:-}}"
if [[ -z "${lib_dir}" || ! -d "${lib_dir}" ]]; then
  echo "!! prefix-cache capability.sh: no lib dir (pass one or set LLAMA_CPP_LIB_PATH)" >&2
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
            f"!! prefix-cache probe: {name} is not set; source ci/runner/rig-env.sh "
            "(declare the model role in the untracked ci/runner/rig-local.sh; "
            "see ci/runner/rig-local.example.sh)",
            file=sys.stderr,
        )
        sys.exit(1)
    return Path(value)


try:
    import llama_cpp
except Exception as exc:
    print(f"!! prefix-cache probe: llama_cpp import failed: {exc}", file=sys.stderr)
    sys.exit(1)

if not llama_cpp.llama_cpp.llama_supports_gpu_offload():
    print(">> [prefix-cache probe] ABSENT: lib has no GPU offload (CPU build) — suite skips on stock")
    sys.exit(1)

model = _required("MODEL_DEFAULT")
if not model.exists():
    print(f"!! prefix-cache probe: missing model {model}", file=sys.stderr)
    sys.exit(1)

print(f">> [prefix-cache probe] PASS: {os.environ['LLAMA_CPP_LIB_PATH']} loads, GPU offload, model present")
sys.exit(0)
PY
