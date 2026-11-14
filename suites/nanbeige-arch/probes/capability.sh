#!/usr/bin/env bash
# Nanbeige-arch capability probe: does the pointed-at lib RECOGNIZE the nanbeige
# architecture? (The canary finding: upstream master has natively adopted
# nanbeige — LLM_ARCH_NANBEIGE at src/llama-arch.h:151 on stock — so this probe
# must pass on BOTH the fork integration build and stock. It is the load-based
# probe: load the Nanbeige GGUF (CPU, n_gpu_layers=0) and require the arch to be
# recognized with the expected layer expansion.)
#
# Usage: probes/capability.sh [<lib-dir>]   (default: $LLAMA_CPP_LIB_PATH)
# Env: MODEL_NANBEIGE (the nanbeige-architecture GGUF path) — from the nanbeige role in rig-local.sh.
set -euo pipefail

lib_dir="${1:-${LLAMA_CPP_LIB_PATH:-}}"
if [[ -z "${lib_dir}" || ! -d "${lib_dir}" ]]; then
  echo "!! capability.sh: no lib dir (pass one or set LLAMA_CPP_LIB_PATH)" >&2
  exit 1
fi

model="${MODEL_NANBEIGE:-}"
if [[ -z "${model}" ]]; then
  echo "!! capability.sh: MODEL_NANBEIGE is not set; source ci/runner/rig-env.sh (declare the nanbeige role in the untracked ci/runner/rig-local.sh; see ci/runner/rig-local.example.sh)" >&2
  exit 1
fi
if [[ ! -f "${model}" ]]; then
  echo "!! capability.sh: Nanbeige GGUF not found: ${model}" >&2
  exit 1
fi

python="${LIBRARY_PYTHON:-ci/library/.venv/bin/python}"
if [[ ! -x "${python}" ]]; then
  echo "!! capability.sh: library venv python missing — run ci/runner/library-venv.sh first" >&2
  exit 1
fi

# Load the GGUF through the pointed-at lib and require arch=nanbeige + the 22x2
# layer expansion (the same checks nanbeige_arch_gate.py makes at [arch:2..4]).
code=$("${python}" - "$lib_dir" "$model" <<'PY'
import io
import os
import re
import sys
from contextlib import redirect_stderr

os.environ["LLAMA_CPP_LIB_PATH"] = sys.argv[1]
from llama_cpp import Llama

err = io.StringIO()
llm = None
try:
    with redirect_stderr(err):
        llm = Llama(model_path=sys.argv[2], n_ctx=256, n_batch=32, n_gpu_layers=0, verbose=True)
except Exception as exc:
    print(f"LOAD_ERR {exc}", flush=True)
    sys.exit(2)
finally:
    if llm is not None:
        llm.close()

log = err.getvalue()
md = getattr(llm, "metadata", {}) if llm else {}
arch = md.get("general.architecture")
if arch != "nanbeige":
    print(f"NOT_RECOGNIZED arch={arch!r}", flush=True)
    sys.exit(1)
if "unknown architecture" in log or "unknown arch" in log:
    print("NOT_RECOGNIZED unknown-arch-in-log", flush=True)
    sys.exit(1)
print(f"RECOGNIZED arch=nanbeige", flush=True)
PY
)

echo ">> [nanbeige probe] ${code}"
if [[ "${code}" == "RECOGNIZED arch=nanbeige" ]]; then
  exit 0
fi
exit 1
