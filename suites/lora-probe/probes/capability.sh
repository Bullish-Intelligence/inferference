#!/usr/bin/env bash
# lora-probe suite capability probe: does the pointed-at lib (1) load,
# (2) support GPU offload (the probe decodes on a GPU), and are the rig assets
# present (default-role model, probe LoRAs)?
#
# Library-mechanism gate (not a fork-symbol gate): on the integration GPU build
# it must PASS (the suite RUNS); on a CPU stock build it FAILS and the suite
# SKIPs via skip_on_stock (the canary's stock build is CPU-only).
#
# Usage: probes/capability.sh [<lib-dir>]   (default: $LLAMA_CPP_LIB_PATH)
set -euo pipefail

lib_dir="${1:-${LLAMA_CPP_LIB_PATH:-}}"
if [[ -z "${lib_dir}" || ! -d "${lib_dir}" ]]; then
  echo "!! lora-probe capability.sh: no lib dir (pass one or set LLAMA_CPP_LIB_PATH)" >&2
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
            f"!! lora-probe probe: {name} is not set; source ci/runner/rig-env.sh "
            "(declare the model role in the untracked ci/runner/rig-local.sh; "
            "see ci/runner/rig-local.example.sh)",
            file=sys.stderr,
        )
        sys.exit(1)
    return Path(value)


try:
    import llama_cpp
except Exception as exc:
    print(f"!! lora-probe probe: llama_cpp import failed: {exc}", file=sys.stderr)
    sys.exit(1)

lib_dir = os.environ["LLAMA_CPP_LIB_PATH"]
if not llama_cpp.llama_cpp.llama_supports_gpu_offload():
    print(">> [lora-probe probe] ABSENT: lib has no GPU offload (CPU build) — suite skips on stock")
    sys.exit(1)

model = _required("MODEL_DEFAULT")
probe_dir = _required("MODEL_PROBE_DIR")
missing = [p for p in (model, probe_dir / "probe-a.gguf", probe_dir / "probe-b.gguf") if not p.exists()]
if missing:
    print(f"!! lora-probe probe: missing assets: {missing}", file=sys.stderr)
    sys.exit(1)

print(f">> [lora-probe probe] PASS: {lib_dir} loads, GPU offload, assets present")
sys.exit(0)
PY
