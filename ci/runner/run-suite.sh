#!/usr/bin/env bash
# Run one per-branch suite against a build out dir — the suite executor for
# suites.yml and the canary.
#
# Classification (fail-closed, per suite.json):
#   probe PASS                          -> run the suite's tests (pass/fail)
#   probe FAIL + skip_on_stock + stock  -> SKIP  (stock build, capability absent
#                                             is expected — e.g. P2 on stock)
#   probe FAIL + skip_on_stock + fork   -> FAIL  (a fork build missing its own
#                                             capability = merge block)
#   probe FAIL + required               -> FAIL
#   probe FAIL + !required              -> SKIP
#
# "stock build" is read from the build manifest's `features` key ([] for stock,
# non-empty for fork builds — ci/build/build-manifest.md). A missing capability
# is never a silent pass.
#
# Usage:
#   ci/runner/run-suite.sh --suite <name> --out <build-out-dir> [--library-python <py>]
# Env: rig-env.sh must be sourced first (model paths, GPU 0, LD_LIBRARY_PATH).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd "${here}/.." && pwd)"

suite=""
out_dir=""
library_python="${LIBRARY_PYTHON:-${repo}/ci/library/.venv/bin/python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) suite="$2"; shift 2 ;;
    --out) out_dir="$2"; shift 2 ;;
    --library-python) library_python="$2"; shift 2 ;;
    *) echo "!! unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${suite}" ]] || { echo "!! --suite required" >&2; exit 2; }
[[ -n "${out_dir}" ]] || { echo "!! --out required" >&2; exit 2; }

suite_dir="${repo}/suites/${suite}"
[[ -d "${suite_dir}" ]] || { echo "!! suite not found: ${suite_dir}" >&2; exit 2; }
[[ -f "${out_dir}/build-manifest.json" ]] || { echo "!! no build-manifest.json in ${out_dir}" >&2; exit 2; }
[[ -x "${library_python}" ]] || { echo "!! library python missing (${library_python}) — run ci/runner/library-venv.sh" >&2; exit 2; }

export LLAMA_CPP_LIB_PATH="${out_dir}/lib"
# Preserve the inherited LD_LIBRARY_PATH (devenv enterShell exports zlib + gcc
# libs for native Python extensions; rig-env.sh appends /run/opengl-driver/lib).
export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LLAMA_CPP_LIB_PATH}:$(dirname "$(gcc -print-file-name=libstdc++.so)"):$(dirname "$(gcc -print-file-name=libgomp.so)"):${LD_LIBRARY_PATH:-}"
export LIBRARY_PYTHON="${library_python}"

# Read suite.json + build manifest (python; jq is not guaranteed).
meta="$("${library_python}" - "${suite_dir}" "${out_dir}" <<'PY'
import json
import os
import sys

suite_dir, out_dir = sys.argv[1], sys.argv[2]
with open(os.path.join(out_dir, "build-manifest.json")) as fh:
    manifest = json.load(fh)
with open(os.path.join(suite_dir, "suite.json")) as fh:
    suite = json.load(fh)

stock = not manifest.get("features")  # [] for stock builds
print(suite["capability"]["probe"])
print(json.dumps(suite["capability"].get("required", True)))
print(json.dumps(suite.get("skip_on_stock", False)))
print(json.dumps(suite.get("tests", [])))
print(json.dumps(stock))
PY
)"
mapfile -t meta_lines <<<"${meta}"
probe="${meta_lines[0]}"
required=$(printf '%s' "${meta_lines[1]}" | tr -d '"')
skip_on_stock=$(printf '%s' "${meta_lines[2]}" | tr -d '"')
tests_json="${meta_lines[3]}"
stock_build=$(printf '%s' "${meta_lines[4]}" | tr -d '"')

# ---- capability probe -------------------------------------------------------
if "${suite_dir}/${probe}" "${LLAMA_CPP_LIB_PATH}"; then
  probe_ok=1
else
  probe_ok=0
fi

# ---- classification ---------------------------------------------------------
verdict=""
if [[ "${probe_ok}" == "1" ]]; then
  verdict="RUN"
elif [[ "${skip_on_stock}" == "true" && "${stock_build}" == "true" ]]; then
  verdict="SKIP"
elif [[ "${required}" == "true" ]]; then
  verdict="FAIL"
else
  verdict="SKIP"
fi

echo ">> [suite] ${suite}: probe=$([ "${probe_ok}" == 1 ] && echo PASS || echo ABSENT)"
echo ">> [suite] ${suite}: stock_build=${stock_build} skip_on_stock=${skip_on_stock} required=${required} -> ${verdict}"

case "${verdict}" in
  SKIP)
    echo ">> [suite] ${suite}: SKIP (capability absent on stock — expected)"
    echo "[suite-verdict] ${suite}: SKIP"
    exit 0
    ;;
  FAIL)
    echo ">> [suite] ${suite}: FAIL (capability probe failed on a fork build — merge block)" >&2
    echo "[suite-verdict] ${suite}: FAIL"
    exit 1
    ;;
esac

# ---- run the tests ----------------------------------------------------------
echo ">> [suite] ${suite}: running tests against ${LLAMA_CPP_LIB_PATH}"
set +e
"${library_python}" -m pytest "${suite_dir}/tests" -q "$@"
rc=$?
set -e
if [[ "${rc}" == "0" ]]; then
  echo "[suite-verdict] ${suite}: PASS"
else
  echo "[suite-verdict] ${suite}: FAIL"
fi
exit "${rc}"
