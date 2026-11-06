#!/usr/bin/env bash
# P2-mixed-batch-lora capability probe: does the pointed-at lib export the P2
# fork's per-sequence LoRA routing surface?
#
# Usage: probes/capability.sh [<lib-dir>]   (default: $LLAMA_CPP_LIB_PATH)
set -euo pipefail

lib_dir="${1:-${LLAMA_CPP_LIB_PATH:-}}"
if [[ -z "${lib_dir}" || ! -d "${lib_dir}" ]]; then
  echo "!! capability.sh: no lib dir (pass one or set LLAMA_CPP_LIB_PATH)" >&2
  exit 1
fi

lib="$(find "${lib_dir}" -maxdepth 1 -name 'libllama.so.*' | head -1)"
if [[ -z "${lib}" ]]; then
  echo "!! capability.sh: no libllama.so.* in ${lib_dir}" >&2
  exit 1
fi

# Capture first, then match with bash string comparison (no pipe — a
# `grep -q` inside a pipe would SIGPIPE the producer under `set -o pipefail`
# when it exits on the first match, a spurious ABSENT).
symbols="$(nm -D "${lib}" 2>/dev/null || true)"
if [[ "${symbols}" == *"llama_set_seq_adapter"* ]]; then
  echo ">> [p2 probe] PASS: ${lib} exports llama_set_seq_adapter"
  exit 0
fi

echo ">> [p2 probe] ABSENT: ${lib} has no llama_set_seq_adapter (stock/integration)"
exit 1
