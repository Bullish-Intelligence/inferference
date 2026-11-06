#!/usr/bin/env bash
# Nanbeige-hats capability probe: does the pointed-at lib export the hats
# loop-routing surface? (`nm -D libllama.so | grep llama_set_loop_adapter` —
# hats are fork-only, never on stock.)
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
# `grep -q` inside a pipe would SIGPIPE the producer under `set -o pipefail`).
symbols="$(nm -D "${lib}" 2>/dev/null || true)"
if [[ "${symbols}" == *"llama_set_loop_adapter"* ]]; then
  echo ">> [hats probe] PASS: ${lib} exports llama_set_loop_adapters/llama_set_loop_adapter"
  exit 0
fi

echo ">> [hats probe] ABSENT: ${lib} has no llama_set_loop_adapter (integration/stock)"
exit 1
