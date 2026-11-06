#!/usr/bin/env bash
# p2-server-seq-routing capability probe: does the pointed-at build carry the
# stage-4 server surface? Two things, both read from the build, never assumed:
#
#   1. the lib exports llama_set_seq_adapter_scaled (D2 option B). An
#      `integration` lib carries the P2 pair and NOT this symbol, so the pair
#      alone is not enough.
#   2. the build ships a llama-server that knows --lora-seq-routing.
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

# Capture first, then match in bash: a `grep -q` inside a pipe would SIGPIPE the
# producer under `set -o pipefail` and report a spurious ABSENT.
symbols="$(nm -D "${lib}" 2>/dev/null || true)"
if [[ "${symbols}" != *"llama_set_seq_adapter_scaled"* ]]; then
  echo ">> [server-routing probe] ABSENT: ${lib} has no llama_set_seq_adapter_scaled"
  exit 1
fi

server="$(dirname "${lib_dir}")/bin/llama-server"
if [[ ! -x "${server}" ]]; then
  echo ">> [server-routing probe] ABSENT: no llama-server at ${server}"
  exit 1
fi

if ! "${server}" --help 2>&1 | grep -q -- "--lora-seq-routing"; then
  echo ">> [server-routing probe] ABSENT: ${server} does not know --lora-seq-routing"
  exit 1
fi

echo ">> [server-routing probe] PASS: ${lib} + ${server} carry stage-4 routing"
exit 0
