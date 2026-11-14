#!/usr/bin/env bash
# Rig environment facts for jobs that need the GPU/build env. Idempotent.
#
# THIS FILE HOLDS NO MODEL NAMES AND NO MODEL PATHS. Inferference is a
# multi-model inference platform: which models a machine happens to hold is
# machine inventory, not a property of the platform, and baking two of them in
# here made every consumer read as if the project were about those two models.
#
# Roles live in ci/runner/rig-local.sh — untracked, one per machine, sourced
# below when it exists. It declares roles with the `inferference_role` helper,
# which exports the two variables every consumer already expects:
#
#   inferference_role <name> <card-index> <model-path> [key=value ...]
#     -> LLAMA_GPU_<NAME>   the card that role serves
#     -> MODEL_<NAME>       that role's GGUF
#     -> any extra key=value as <KEY>_<NAME>, for role-scoped extras such as
#        adapter pools or a tokenizer dir
#
# Copy ci/runner/rig-local.example.sh to get started. A machine with no
# rig-local.sh still gets a working build/CI environment — it simply declares no
# roles, and any job that needs one fails loudly on the missing variable rather
# than falling back to a model nobody chose.
#
# Why LD_LIBRARY_PATH is built here and not in a workflow env block: the
# libstdc++/libgomp paths come from the devenv gcc, which is a runtime query.
#
# Rules (load-bearing):
#   * a real GPU driver library must win over any nix stub that built libs
#     resolve via rpath — so /run/opengl-driver/lib comes FIRST.
#   * gcc libstdc++ + libgomp are needed at runtime.
#   * every device is named by a ROLE variable, never by a literal index. The
#     index is a function of the visible-device list, so only the role names a
#     card, and a job asserts the variable it was assigned.
set -u

export NIXPKGS_ALLOW_UNFREE="${NIXPKGS_ALLOW_UNFREE:-1}"
export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/llamacpp-ccache}"
mkdir -p "$CCACHE_DIR"

# --- role declaration ---------------------------------------------------------
# FAIL-CLOSED ON THE MODEL PATH. A role whose GGUF is absent is not declared at
# all, so a consumer hits "unset variable" instead of launching a server against
# a path that does not exist. A missing capability is a skip or a loud fail,
# never a silent pass.
INFERFERENCE_ROLES="${INFERFERENCE_ROLES:-}"
inferference_role() {
  local name="$1" card="$2" model="$3"; shift 3
  local upper="${name^^}"
  if [[ ! -r "$model" ]]; then
    echo ">> [rig-env] role '${name}' NOT declared: no readable model at ${model}" >&2
    return 0
  fi
  export "LLAMA_GPU_${upper}=${card}"
  export "MODEL_${upper}=${model}"
  local kv
  for kv in "$@"; do
    [[ "$kv" == *=* ]] || { echo ">> [rig-env] role '${name}': ignoring '${kv}' (want key=value)" >&2; continue; }
    local k="${kv%%=*}" v="${kv#*=}"
    export "${k^^}_${upper}=${v}"
  done
  INFERFERENCE_ROLES="${INFERFERENCE_ROLES:+${INFERFERENCE_ROLES} }${name}"
  export INFERFERENCE_ROLES
}

# The machine file is sourced BEFORE the defaults below, and it is the
# authority. Defaulting first would make the platform's guess win over the
# machine's own answer — which it did on the first run of this file, silently
# pointing every role at an empty models dir.
_rig_local="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rig-local.sh"
# shellcheck disable=SC1090
[[ -r "$_rig_local" ]] && source "$_rig_local"
unset _rig_local

# --- the default role ---------------------------------------------------------
# Gates and bench workloads that exercise the PLATFORM rather than a particular
# model need some model to run against. They resolve $MODEL_DEFAULT, so no gate
# and no workflow has to name a role. The machine picks it with
# INFERFERENCE_DEFAULT_ROLE in rig-local.sh; otherwise the first role declared.
INFERFERENCE_DEFAULT_ROLE="${INFERFERENCE_DEFAULT_ROLE:-}"
if [[ -z "$INFERFERENCE_DEFAULT_ROLE" && -n "${INFERFERENCE_ROLES:-}" ]]; then
  INFERFERENCE_DEFAULT_ROLE="${INFERFERENCE_ROLES%% *}"
fi
if [[ -n "$INFERFERENCE_DEFAULT_ROLE" ]]; then
  _u="${INFERFERENCE_DEFAULT_ROLE^^}"
  _m="MODEL_${_u}"; _g="LLAMA_GPU_${_u}"
  # Only when the role really resolved. A default pointing at an undeclared role
  # would hand every generic gate an empty model path.
  if [[ -n "${!_m:-}" ]]; then
    export INFERFERENCE_DEFAULT_ROLE MODEL_DEFAULT="${!_m}"
    [[ -n "${!_g:-}" ]] && export LLAMA_GPU_DEFAULT="${!_g}"
    # Grammar work needs a LOCAL tokenizer and must never reach the network.
    _t="TOKENIZER_${_u}"
    [[ -n "${!_t:-}" ]] && export TOKENIZER_DEFAULT="${!_t}"
    unset _t
  else
    echo ">> [rig-env] INFERFERENCE_DEFAULT_ROLE='${INFERFERENCE_DEFAULT_ROLE}' is not a declared role; \$MODEL_DEFAULT left unset" >&2
  fi
  unset _u _m _g
fi

# --- defaults, for anything the machine file did not set ----------------------
# Where the host keeps GGUFs. Roles may name absolute paths and ignore it.
export INFERFERENCE_MODELS_DIR="${INFERFERENCE_MODELS_DIR:-$HOME/.cache/inferference/models}"

# Persistent bench baselines root — OUTSIDE any CI workdir so the baseline store
# survives cache trims. baseline_gate.py resolves this first; repo
# ci/.ci-artifacts is the documented fallback. Key = workload@<built-commit>-
# <featureset>@<backend>, never a date. A baseline without a backend suffix
# (an older key) never matches a run. Moving this root orphans every baseline.
export BASELINES_ROOT="${BASELINES_ROOT:-$HOME/.cache/inferference/baselines}"
mkdir -p "$BASELINES_ROOT"

# --- the card this job was assigned -------------------------------------------
# LLAMA_CI_GPU means "the card this job runs on", and ci/runner/gpu-lease.sh
# overwrites it when it places the job. The default below is what an UNLEASED
# command gets, not what a leased one gets.
export LLAMA_CI_GPU="${LLAMA_CI_GPU:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$LLAMA_CI_GPU}"
# The Vulkan device filter, derived from the same role variable. ggml IGNORES
# CUDA_VISIBLE_DEVICES, so without this an unleased job on this rig sees BOTH
# cards and llama.cpp's default `n_gpu_layers = -1` will happily split a model
# across them.
export GGML_VK_VISIBLE_DEVICES="${GGML_VK_VISIBLE_DEVICES:-$LLAMA_CI_GPU}"
# A ROCm build reads HIP_VISIBLE_DEVICES. Same default, same reason.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$LLAMA_CI_GPU}"

# --- driver-first library path (see header) -----------------------------------
# Append the build lib dir when a specific lib set is being loaded (the gates do
# that themselves via LLAMA_CPP_LIB_PATH + an LD_LIBRARY_PATH prepend).
_build_ld() {
  local lib omp
  lib="$(dirname "$(gcc -print-file-name=libstdc++.so)")"
  omp="$(dirname "$(gcc -print-file-name=libgomp.so)")"
  # Preserve the inherited LD_LIBRARY_PATH: the devenv shell already exports
  # zlib + gcc lib dirs for native Python extensions in the library venv.
  # Clobbering them here breaks `import numpy` under a fresh-env CI executor.
  export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LLAMA_CPP_LIB_PATH:-}:${lib}:${omp}:${LD_LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | sed 's/::*/:/g; s/:$//')"
}
_build_ld
