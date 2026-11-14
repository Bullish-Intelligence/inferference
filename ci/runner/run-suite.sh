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
# Model inventory comes before the probe. Each suite.json names a model ROLE in
# env.MODEL_ROLE (never a model file). The role resolves to $MODEL_<ROLE>, set by
# rig-env.sh from the untracked ci/runner/rig-local.sh. A suite can need more
# inventory: it lists the extra variable names in env.MODEL_VARS (for example
# MODEL_PROBE_DIR, TOKENIZER_DEFAULT). Every listed variable must be set and
# name a readable file or directory. A role or variable with no readable path is
# an inventory gap. It FAILS, and the message names each variable. The one
# exception is a suite that skips on a stock build: a stock build cannot run it
# anyway, so the gap is reported and the verdict is SKIP.
#
# GPU suites: a suite with env.GPU true must pass at least one GPU test. The GPU
# tests are env.GPU_TESTS (test file names); the default is every test. A run
# where every GPU test skipped is a SKIP, never a PASS. A suite with no GPU test
# that passed gives no evidence about the GPU.
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
env = suite.get("env", {})
print(env.get("MODEL_ROLE", ""))
print(env.get("MODEL_NEEDS", ""))
print(" ".join(env.get("MODEL_VARS", [])))
print(json.dumps(bool(env.get("GPU", False))))
# GPU test modules: env.GPU_TESTS, else every test file of the suite.
gpu_tests = env.get("GPU_TESTS", suite.get("tests", []))
print(" ".join(os.path.splitext(os.path.basename(t))[0] for t in gpu_tests))
PY
)"
mapfile -t meta_lines <<<"${meta}"
probe="${meta_lines[0]}"
required=$(printf '%s' "${meta_lines[1]}" | tr -d '"')
skip_on_stock=$(printf '%s' "${meta_lines[2]}" | tr -d '"')
tests_json="${meta_lines[3]}"
stock_build=$(printf '%s' "${meta_lines[4]}" | tr -d '"')
model_role="${meta_lines[5]:-}"
model_needs="${meta_lines[6]:-}"
model_vars="${meta_lines[7]:-}"
gpu_suite=$(printf '%s' "${meta_lines[8]:-false}" | tr -d '"')
gpu_modules="${meta_lines[9]:-}"

# ---- model inventory --------------------------------------------------------
if [[ -z "${model_role}" ]]; then
  echo "!! [suite] ${suite}: suite.json has no env.MODEL_ROLE; every suite names the model role it needs" >&2
  echo "[suite-verdict] ${suite}: FAIL"
  exit 2
fi
if [[ ! "${model_role}" =~ ^[a-z][a-z0-9]*$ ]]; then
  echo "!! [suite] ${suite}: env.MODEL_ROLE '${model_role}' is not a role name (lowercase letters and digits; the router launcher takes the same names)" >&2
  echo "[suite-verdict] ${suite}: FAIL"
  exit 2
fi
model_var="MODEL_${model_role^^}"
inventory_gap=""
# Check the role variable first, then each variable the suite lists. An unset
# variable and an unreadable path are both gaps. Report every gap, not the first.
for var in "${model_var}" ${model_vars}; do
  if [[ ! "${var}" =~ ^[A-Z][A-Z0-9_]*$ ]]; then
    gap="env.MODEL_VARS entry '${var}' is not a variable name"
  elif [[ -z "${!var:-}" ]]; then
    gap="${var} is not set"
  elif [[ ! -r "${!var}" ]]; then
    gap="${var} names no readable file: ${!var}"
  else
    continue
  fi
  inventory_gap="${inventory_gap:+${inventory_gap}; }${gap}"
done
if [[ -n "${inventory_gap}" ]]; then
  echo "!! [suite] ${suite}: inventory gap: ${inventory_gap}" >&2
  echo "!! [suite] ${suite}: needs role '${model_role}': ${model_needs}" >&2
  echo "!! [suite] ${suite}: declare the role in ci/runner/rig-local.sh (see ci/runner/rig-local.example.sh)" >&2
fi

# ---- capability probe -------------------------------------------------------
# With an inventory gap the probe cannot give a true answer. Skip it.
probe_ok=0
if [[ -z "${inventory_gap}" ]] && "${suite_dir}/${probe}" "${LLAMA_CPP_LIB_PATH}"; then
  probe_ok=1
fi

# ---- classification ---------------------------------------------------------
verdict=""
if [[ -n "${inventory_gap}" ]]; then
  if [[ "${skip_on_stock}" == "true" && "${stock_build}" == "true" ]]; then
    verdict="SKIP"
  else
    verdict="FAIL"
  fi
elif [[ "${probe_ok}" == "1" ]]; then
  verdict="RUN"
elif [[ "${skip_on_stock}" == "true" && "${stock_build}" == "true" ]]; then
  verdict="SKIP"
elif [[ "${required}" == "true" ]]; then
  verdict="FAIL"
else
  verdict="SKIP"
fi

if [[ -n "${inventory_gap}" ]]; then
  echo ">> [suite] ${suite}: probe=NOT RUN (inventory gap)"
else
  echo ">> [suite] ${suite}: probe=$([ "${probe_ok}" == 1 ] && echo PASS || echo ABSENT)"
fi
echo ">> [suite] ${suite}: stock_build=${stock_build} skip_on_stock=${skip_on_stock} required=${required} -> ${verdict}"

case "${verdict}" in
  SKIP)
    if [[ -n "${inventory_gap}" ]]; then
      echo ">> [suite] ${suite}: SKIP (stock build cannot run this suite; inventory gap: ${inventory_gap})"
    else
      echo ">> [suite] ${suite}: SKIP (capability absent on stock — expected)"
    fi
    echo "[suite-verdict] ${suite}: SKIP"
    exit 0
    ;;
  FAIL)
    if [[ -n "${inventory_gap}" ]]; then
      echo ">> [suite] ${suite}: FAIL (inventory gap: ${inventory_gap})" >&2
    else
      echo ">> [suite] ${suite}: FAIL (capability probe failed on a fork build — merge block)" >&2
    fi
    echo "[suite-verdict] ${suite}: FAIL"
    exit 1
    ;;
esac

# ---- run the tests ----------------------------------------------------------
echo ">> [suite] ${suite}: running tests against ${LLAMA_CPP_LIB_PATH}"
# pytest exits 0 when every test skipped. A non-GPU suite whose probe passed must
# run at least one test, so an all-skipped run is a FAIL, never a silent pass. `-rs`
# prints each skip reason.
test_log="$(mktemp "${TMPDIR:-/tmp}/run-suite.${suite}.XXXXXX")"
junit_xml="${test_log}.xml"
trap 'rm -f "${test_log}" "${junit_xml}"' EXIT
set +e
# `-o addopts=` clears the `-q` in pyproject.toml. With both, pytest runs at -qq and
# prints no "N passed" summary, so the check below fails every passing suite.
"${library_python}" -m pytest "${suite_dir}/tests" -o addopts= -q -rs --junitxml="${junit_xml}" "$@" 2>&1 | tee "${test_log}"
rc=${PIPESTATUS[0]}
set -e
if [[ "${rc}" == "0" && "${gpu_suite}" != "true" ]] && ! grep -Eq '(^|[ ,])[0-9]+ passed' "${test_log}"; then
  echo "!! [suite] ${suite}: the probe passed but no test passed (all skipped). See the skip reasons above." >&2
  rc=1
fi
# A GPU suite can pass its non-GPU tests, or no test, while every GPU gate skips.
# Count the GPU tests that passed. None passed: the suite gave no GPU evidence,
# so the verdict is SKIP (a visible non-pass), never PASS.
if [[ "${rc}" == "0" && "${gpu_suite}" == "true" ]]; then
  gpu_passed="$("${library_python}" - "${junit_xml}" ${gpu_modules} <<'PY'
import sys
import xml.etree.ElementTree as ET

xml_path, modules = sys.argv[1], set(sys.argv[2:])
passed = 0
for case in ET.parse(xml_path).iter("testcase"):
    if modules.isdisjoint(case.get("classname", "").split(".")):
        continue
    if not any(child.tag in ("skipped", "failure", "error") for child in case):
        passed += 1
print(passed)
PY
)"
  if [[ "${gpu_passed}" == "0" ]]; then
    echo "!! [suite] ${suite}: every GPU test skipped (${gpu_modules:-none listed}). See the skip reasons above." >&2
    echo "[suite-verdict] ${suite}: SKIP"
    exit 0
  fi
fi
if [[ "${rc}" == "0" ]]; then
  echo "[suite-verdict] ${suite}: PASS"
else
  echo "[suite-verdict] ${suite}: FAIL"
fi
exit "${rc}"
