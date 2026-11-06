#!/usr/bin/env bash
# Start the llama.cpp ROUTER — one process, no `-m`, that spawns one child
# `llama-server` per resident model from a generated preset INI.
#
# This replaces the retired single-role design (inferference-server-launch.sh /
# inferference-server@<role>.service), which pinned one model to one card
# through $LLAMA_GPU_<ROLE> and started that model's own llama-server directly.
# AGENTS.md declares that layer SUPERSEDED: "THIS LIBRARY OWNS ALL LLM SERVING.
# There is no other server." Every model this machine serves is a child of the
# ONE router this script starts.
#
#   inferference-router-launch.sh [--check]
#
# --check resolves everything, renders the preset INI, and exits without
# starting. Use it to verify an install without holding a card.
#
# RUNTIME CONFIGURATION IS YAML, AND ONLY YAML — the same rule the single-role
# server followed, now at the router level instead of per-role:
#
#   $INFERFERENCE_CONFIG_DIR/placement.yaml      which models are resident,
#                                                 where, and how the router
#                                                 itself is launched (host,
#                                                 port, --models-max)
#   $INFERFERENCE_CONFIG_DIR/models/<name>.yaml   what each model is
#
# See ci/runner/placement.example.yaml and ci/runner/model.example.yaml.
#
# The card is never named here, and never a literal index: each model's
# `device:` in placement.yaml is `VulkanN`, read by inferference-reconcile.py
# when it renders the preset INI this script starts the router against.
#
# The only environment this file reads is install plumbing — WHERE things are,
# never WHAT is served:
#   INFERFERENCE_CONFIG_DIR   where placement.yaml + models/ live.
#                             Default ~/.config/inferference.
#   INFERFERENCE_PRESET_INI   where the generated preset INI is written.
#                             Default $INFERFERENCE_CONFIG_DIR/models-preset.ini.
#   INFERFERENCE_SERVER_DIR   Mode A artefact dir. Default ci/modea/server/current.
#   INFERFERENCE_PYTHON       interpreter for the reconciler. Default the
#                             library venv.
#
# THIS UNIT TAKES NO GPU LEASE. It is the one long-lived owner of every card it
# places a model on — there is nothing to preempt it in favour of, and a lease
# taken here would hold every listed card for the router's entire lifetime.
# ci/runner/gpu-lease.sh still governs BUILD and GATE jobs; a lease that finds no
# room because the router is resident is not a bug (AGENTS.md, GPU discipline).
set -euo pipefail

check=0
[[ "${1:-}" == "--check" ]] && check=1

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # ci/runner
repo="$(cd "${here}/../.." && pwd)"

_cfg() { echo "!! [inferference-router] $*" >&2; exit 78; }

config_dir="${INFERFERENCE_CONFIG_DIR:-${HOME}/.config/inferference}"
[[ -f "${config_dir}/placement.yaml" ]] || _cfg "no placement at ${config_dir}/placement.yaml. See ci/runner/placement.example.yaml."
[[ -d "${config_dir}/models" ]] || _cfg "no model configs at ${config_dir}/models. See ci/runner/model.example.yaml."

reconciler="${here}/inferference-reconcile.py"
config_py="${INFERFERENCE_PYTHON:-${repo}/ci/library/.venv/bin/python}"
[[ -x "$config_py" ]] || _cfg "no python at ${config_py} to read ${config_dir}. Set \$INFERFERENCE_PYTHON, or run ci/runner/library-venv.sh."

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

build_id="$("$config_py" -c "import json;print(json.load(open('${man}')).get('build_id',''))")"
residency_patch="$("$config_py" -c "import json;print(json.load(open('${man}')).get('vulkan_residency_patch_sha256',''))")"
expected_patch="$(sha256sum "${repo}/ci/build/patches/vulkan-residency.patch" | cut -d ' ' -f 1)"
[[ "$residency_patch" == "$expected_patch" ]] || _cfg "server does not carry the current residency patch"

# --- the backend: Vulkan/RADV, not CUDA ---------------------------------------
# The rig is two AMD MI25 now (AGENTS.md, "RIG HARDWARE CHANGED 2026-09-04");
# the served artefact is a vulkan-mi25-server build, and its RUNPATH already
# resolves libvulkan.so.1 straight out of the nix store — measured: `ldd` on
# both llama-server and libggml-vulkan.so needs no LD_LIBRARY_PATH help at all.
# There is no libcuda/CUDA-stub failure mode to check for on this backend, and
# forcing an LD_LIBRARY_PATH here would be solving a problem this artefact does
# not have.
#
# What CAN mislabel a build is a profile name that promises Vulkan but was not
# actually built with it (ci/gates/vulkan_mi25_smoke.sh checks the same thing):
# a `vulkan-mi25-server` artefact with no libggml-vulkan.so would silently run
# every model on the CPU backend with no error anywhere.
ls "${server_dir}"/lib/libggml-vulkan.so* >/dev/null 2>&1 \
  || _cfg "${server_dir}/lib has no libggml-vulkan.so — this artefact is not a Vulkan build. Every model the router loads would run on the CPU backend with no error anywhere."

# RADV needs its ICD advertised (`hardware.graphics.enable`, AGENTS.md). A
# process on the bare host finds it via the loader's own default search of
# /run/opengl-driver/share/vulkan/icd.d — confirmed with a clean environment,
# no VK_ICD_FILENAMES needed (an earlier note about exporting it by hand was
# about a devenv/bubblewrap shell, which remaps /run and is not what this unit
# runs under). This checks the precondition for that default search to work at
# all, without hardcoding the nix store hash underneath it.
[[ -e /run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json ]] \
  || _cfg "/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json is absent. hardware.graphics.enable is not wired, or /run/opengl-driver has not been populated by this generation — every model would fall back to the CPU backend with no error anywhere."

# --- render the preset INI, and read how the router itself is launched -------
# The INI is a generated artefact (placement.py:render_preset_ini) — never
# hand-edited, always regenerated from placement.yaml + models/*.yaml. Resident
# models carry `load-on-startup = true`, so starting the router against this
# file with --no-models-autoload loads exactly the declared set: no reconcile
# call is needed just to bring the router up.
ini_path="${INFERFERENCE_PRESET_INI:-${config_dir}/models-preset.ini}"
mkdir -p "$(dirname "$ini_path")"

set +e
render_out="$("$config_py" "$reconciler" render --out "$ini_path" 2>&1)"
rc_render=$?
set -e
if (( rc_render != 0 )); then
  echo "$render_out" >&2
  exit "$rc_render"
fi
echo ">> [inferference-router] $render_out"

read -r host port models_max <<<"$("$config_py" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${repo}/src')
from inferference.reconcile import read_config_dir
placement, _ = read_config_dir(Path('${config_dir}'))
print(placement.router.host, placement.router.port, placement.router.models_max)
")"
[[ -n "$host" && -n "$port" && -n "$models_max" ]] || _cfg "could not resolve router.host/port/models_max from ${config_dir}/placement.yaml"

echo ">> [inferference-router] artefact ${server_dir}"
echo ">>   build_id        ${build_id}"
echo ">>   backend         Vulkan (libggml-vulkan.so present, RADV ICD advertised)"
echo ">>   listen          ${host}:${port}"
echo ">>   models-max      ${models_max}"
echo ">>   preset          ${ini_path}"
echo ">>   thermal guard   pause at ${GPU_THERMAL_MAX_C:-85} C, resume at ${GPU_THERMAL_RESUME_C:-70} C, kill at ${GPU_THERMAL_HARD_C:-90} C"
echo ">>   GPU lease       none — this process is the long-lived owner of every card it places a model on"

argv=(
  "$bin"
  --host "$host"
  --port "$port"
  --models-preset "$ini_path"
  --no-models-autoload
  --models-max "$models_max"
)

(( check )) && { echo ">> --check: resolved, not started."; exit 0; }
# Keep the resident router and its child model servers under the thermal guard
# in PAUSE mode. The guard owns a separate process group. At 85 C it freezes
# the whole router tree (SIGSTOP). It thaws the tree when every card has held
# 70 C or less for 30 s and the fan watchdog reads OK. There is no time limit
# on a pause, so a long overnight run survives a hot spell. Only a card that
# reaches 90 C while paused (cooling failed) kills the tree, with exit 75.
# Owner decision 2026-10-28; the earlier limit was 80 C with a kill.
# An environment value from the unit overrides these defaults.
export GPU_THERMAL_MAX_C="${GPU_THERMAL_MAX_C:-85}"
export GPU_THERMAL_RESUME_C="${GPU_THERMAL_RESUME_C:-70}"
export GPU_THERMAL_HARD_C="${GPU_THERMAL_HARD_C:-90}"
export INFERFERENCE_RESIDENCY_REQUIRED=1
exec "${here}/gpu-thermal-guard.sh" run --on-limit pause -- "${argv[@]}"
