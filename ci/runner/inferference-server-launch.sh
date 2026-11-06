#!/usr/bin/env bash
# Resolve and start one routed llama-server instance, by ROLE.
#
# Stage 7 item 2. This is what inferference-server@<role>.service execs.
# The resolution lives here and not in the unit file for one reason: a unit
# cannot compute anything. Pinning `Environment="CUDA_VISIBLE_DEVICES=0"` was
# which predates 19's two-model decision and is now exactly the literal index
# AGENTS.md forbids. A unit instantiated per ROLE can hold no index at all — the
# card comes from that role's variable, resolved here.
#
#   inferference-server-launch.sh <role> [--check]
#
# <role> is any role this machine declares in ci/runner/rig-local.sh. It is the
# instance name AND the suffix of the two variables that resolve the role:
#
#   <role> -> $LLAMA_GPU_<ROLE>   the card it serves
#          -> $MODEL_<ROLE>       its GGUF
#
# The platform names no model. For scale: a routed ~3B server measured 5.3 GiB
# + 54.4 MiB per adapter, a ~9B server 6.3 GiB + 59.3 MiB per adapter.
#
# --check resolves everything, prints it, and exits without starting. Use it to
# verify an install without holding a card.
#
# RUNTIME CONFIGURATION IS YAML, AND ONLY YAML. One role, one file:
# ~/.config/inferference/<role>.yaml, resolved by inferference-config.py. There
# is no environment override for a port, a model, a context size or an argument.
# Two mechanisms mean a server can run a port one file names and another file
# does not, and no one can tell which won by reading either file. The schema is
# ci/runner/server-role.example.yaml.
#
# The only environment this file reads is install plumbing — WHERE things are,
# never WHAT the server runs:
#   INFERFERENCE_CONFIG_DIR  where the per-role YAML lives.
#                            Default ~/.config/inferference.
#   INFERFERENCE_SERVER_DIR  Mode A artefact dir. Default ci/modea/server/current.
#   INFERFERENCE_PYTHON      interpreter for the resolver. Default the library venv.
# Plus $LLAMA_GPU_<ROLE> and $MODEL_<ROLE> from rig-env.sh, which are the rig's
# own inventory rather than this server's configuration.
#
# This unit takes NO GPU lock. Preemption is the arbitration: a CI job
# whose budget does not fit beside it stops it inside the lock and restarts it on
# exit. Taking a lock here would deadlock all CI for the unit's lifetime.
set -euo pipefail

role="${1:-}"; shift || true
check=0
[[ "${1:-}" == "--check" ]] && check=1
[[ -n "$role" ]] || { echo "usage: inferference-server-launch.sh <role> [--check]" >&2; exit 2; }
[[ "$role" =~ ^[a-z][a-z0-9]*$ ]] || { echo "!! role '${role}' must be lowercase alphanumeric" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # ci/runner
repo="$(cd "${here}/../.." && pwd)"
# shellcheck disable=SC1091
source "${here}/rig-env.sh"

# EX_CONFIG. systemd reports it distinctly, so a missing rig fact never looks
# like a crashed server.
_cfg() { echo "!! [inferference-server:${role}] $*" >&2; exit 78; }

# --- the card, from the ROLE variable and nothing else ------------------------
upper="${role^^}"
gpu_var="LLAMA_GPU_${upper}"
card="${!gpu_var:-}"
[[ -n "$card" ]] || _cfg "\$${gpu_var} is unset. A role with no card variable is not a role this rig serves; declare it in ci/runner/rig-local.sh (see rig-local.example.sh). The card is never a literal index (AGENTS.md, GPU discipline)."
[[ "$card" =~ ^[0-9]+$ ]] || _cfg "\$${gpu_var}='${card}' is not a device index"

# A server pins its ROLE's card, so it must never run inside a GPU lease: the
# lease locks one card and this would then compute on another, holding a lock
# over a card it does not use and using a card no lock covers. Measured
# 2026-08-26 — `gpu-lease.sh 6000 -- inferference-server-launch.sh <role>`
# leased card 0 and served 5294 MiB on card 1, with the lease's own assertion
# still reading 0. Refuse rather than let the two disagree silently.
if [[ -n "${GPU_LEASE_CARD:-}" && "${GPU_LEASE_CARD}" != "$card" ]]; then
  _cfg "started inside a GPU lease on card ${GPU_LEASE_CARD}, but role '${role}' serves card ${card} (\$${gpu_var}). The lease would hold a lock over a card this server does not use. This unit takes no lock and must not run under one."
fi

export CUDA_VISIBLE_DEVICES="$card"

# --- the artefact -------------------------------------------------------------
server_dir="${INFERFERENCE_SERVER_DIR:-${repo}/ci/modea/server/current}"
[[ -d "$server_dir" ]] || _cfg "no Mode A server artefact at ${server_dir}. Publish one with ci/build/mode-a-server.sh."
server_dir="$(cd "$server_dir" && pwd -P)"
bin="${server_dir}/bin/llama-server"
man="${server_dir}/build-manifest.json"
[[ -x "$bin" ]] || _cfg "no executable at ${bin}"
[[ -r "$man" ]] || _cfg "no build-manifest.json beside ${server_dir}/lib. An artefact that cannot name its provenance is not a release artefact."

case "$server_dir" in
  *.llamacpp-builds/*) _cfg "${server_dir} is inside the prunable build garage. ci/build/prune-builds.sh deletes it; publish with ci/build/mode-a-server.sh first." ;;
esac

routing="$(python3 -c "import json;print(json.load(open('${man}')).get('server_seq_routing'))")"
[[ "$routing" == "True" ]] || _cfg "the artefact's manifest says server_seq_routing=${routing}. This repository serves the ROUTED server; a binary that does not call the routing API is not it."
build_id="$(python3 -c "import json;print(json.load(open('${man}')).get('build_id',''))")"

# The model is resolved and checked by inferference-config.py, from the YAML's
# `model:` or this role's $MODEL_<ROLE>. It is not resolved twice.

# --- libcuda first, and PROVE it ---------------------------------------------
# "/run/opengl-driver/lib first or it silently serves on CPU". The response R10
# records is "assert the device, never grep the log" — so assert the loader's
# answer here, before the server starts, rather than read a log afterwards.
export LD_LIBRARY_PATH="/run/opengl-driver/lib:${server_dir}/lib:${LD_LIBRARY_PATH:-}"
[[ -e /run/opengl-driver/lib/libcuda.so.1 ]] || _cfg "/run/opengl-driver/lib/libcuda.so.1 is absent. The built libs would resolve libcuda through their RUNPATH to the nix CUDA STUB and this server would serve on CPU with no error anywhere."
cuda_res="$(ldd "${server_dir}/lib/libggml-cuda.so" 2>/dev/null | sed -n 's/.*libcuda\.so\.1 => \([^ ]*\).*/\1/p' | head -1 || true)"
case "$cuda_res" in
  /run/opengl-driver/lib/*) : ;;
  "") _cfg "libggml-cuda.so does not resolve libcuda.so.1 at all" ;;
  *stubs*) _cfg "libggml-cuda.so resolves libcuda.so.1 to the nix STUB (${cuda_res}). This server would serve on CPU." ;;
  *) echo ">> [inferference-server:${role}] note: libcuda.so.1 resolves to ${cuda_res}" >&2 ;;
esac

# --- the YAML config — REQUIRED, and the only source --------------------------
# Runtime config lives in ~/.config/inferference/<role>.yaml, NOT in the Nix
# module: changing a model or a pool should be an edit and a restart, never a
# NixOS rebuild. inferference-config.py resolves it to an argv and fails closed
# (78) on anything it cannot verify — a missing adapter included, because a
# server that starts without an adapter it was told to load answers with the
# wrong weights and no error anywhere.
#
# There is no second path. An environment-variable fallback used to live here so
# an ad-hoc `INFERFERENCE_PORT=... launch.sh <role>` would run without a config
# file. It is removed: two mechanisms let a server run a port one file names and
# another does not, and neither file shows which won. A missing YAML is now a
# refusal that names the file to write.
config_dir="${INFERFERENCE_CONFIG_DIR:-${HOME}/.config/inferference}"
cfg_file="${config_dir}/${role}.yaml"
resolver="${here}/inferference-config.py"
config_py="${INFERFERENCE_PYTHON:-${repo}/ci/library/.venv/bin/python}"

[[ -f "$cfg_file" ]] || _cfg "no config at ${cfg_file}. Write it — see ci/runner/server-role.example.yaml. YAML is the only configuration mechanism; there is no environment fallback, on purpose."
[[ -x "$config_py" ]] || _cfg "no python at ${config_py} to read ${cfg_file}. Set \$INFERFERENCE_PYTHON, or run ci/runner/library-venv.sh."

# Command substitution, not process substitution: `mapfile < <(...)` reports
# mapfile's status, so the resolver's EX_CONFIG 78 was swallowed and became a
# confusing "produced no argv". This propagates the real exit code.
# Capture the status EXPLICITLY. `if ! cmd; then exit $?` exits 0, because `!`
# has already inverted the status by the time the branch runs — so the
# resolver's EX_CONFIG 78 printed its reason and then reported success, which
# systemd would have read as a clean start. Found by running the refusal cases.
set +e
resolved="$("$config_py" "$resolver" "$role" --config-dir "$config_dir" --binary "$bin" --print-summary)"
rc_cfg=$?
set -e
(( rc_cfg == 0 )) || exit "$rc_cfg"
mapfile -t argv <<<"$resolved"
[[ ${#argv[@]} -gt 0 ]] || _cfg "the resolver produced no argv for ${cfg_file}"

echo ">> [inferference-server:${role}] card ${card} (from \$${gpu_var}), artefact ${server_dir}"
echo ">>   build_id        ${build_id}"
echo ">>   libcuda         ${cuda_res}"
echo ">>   GPU lock        none — this unit is preemptible by policy, but nothing preempts it (AGENTS.md)"
(( check )) && { echo ">> --check: resolved, not started."; exit 0; }
exec "${argv[@]}"
