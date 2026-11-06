#!/usr/bin/env bash
# Baseline functionality gate for the AMD MI25 (Vega 10 / gfx900) Vulkan build.
#
# This is the spike's definition of "it works": the vulkan-mi25-server artifact
# loads a real model onto ONE NAMED MI25 and generates tokens over HTTP.
#
# WHY THE GATE IS THIS HEAVY. ggml falls back to the CPU backend silently: a
# server that never saw a GPU still answers /health 200 and still completes a
# prompt. A smoke test that only curls the API therefore passes identically on a
# rig with no cards, which is the silent pass AGENTS.md forbids. So the gate
# proves the placement three independent ways, and any one of them failing is a
# failure:
#
#   1. the server names the Vulkan device it used, AND its PCI address is the
#      one amdgpu sysfs gives for the card this job was assigned;
#   2. every repeating layer is assigned to THAT Vulkan device — none to the CPU
#      and none to the other card;
#   3. that card's own sysfs VRAM accounting rises while the server is resident,
#      AND its GTT (host memory over PCIe) does NOT -- see the note on spilling.
#
# (3) is the one that cannot be faked by a log format change, and this build
# reports "model buffer size = 0.00 MiB" for the Vulkan buffers, so the log is
# not a usable source for how much memory was really placed.
#
# ON THE VULKAN INDEX. The card is named by $LLAMA_CI_GPU — the role variable
# the lease writes, never a literal index (AGENTS.md, GPU discipline). That is a
# DRM/sysfs index. The gate exposes one Vulkan device and verifies its PCI
# address against sysfs. The server must report that address for Vulkan0.
# A future driver or loader that
# reorders devices fails here loudly instead of quietly pinning the wrong card.
#
# Usage:
#   vulkan_mi25_smoke.sh [--out DIR] [--model PATH] [--port N] [--device-index N]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # ci/gates
repo="$(cd "${here}/../.." && pwd)"
# shellcheck disable=SC1091
source "${repo}/ci/runner/rig-env.sh"
# shellcheck disable=SC1091
source "${here}/gtt-check.sh"

_die() { echo "!! [vulkan-mi25-smoke] $*" >&2; exit 1; }

OUT=""; MODEL="${VULKAN_SMOKE_MODEL:-}"; PORT="${VULKAN_SMOKE_PORT:-8099}"
IDX="${LLAMA_CI_GPU:-0}"
MIN_VRAM_MIB="${VULKAN_SMOKE_MIN_VRAM_MIB:-1024}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)          OUT="$2";   shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --port)         PORT="$2";  shift 2 ;;
    --device-index) IDX="$2";   shift 2 ;;
    *) _die "unknown arg: $1" ;;
  esac
done
[[ "$IDX" =~ ^[0-9]+$ ]] || _die "device index must be a non-negative integer, got '${IDX}'"

# --- resolve the artifact -----------------------------------------------------
if [[ -z "$OUT" ]]; then
  OUT="$(ls -d "${repo}"/ci/build/.llamacpp-builds/out-vulkan-mi25-server-* 2>/dev/null \
         | sort | tail -1 || true)"
fi
[[ -n "$OUT" && -d "$OUT" ]] \
  || _die "no vulkan-mi25-server build found. Build one first: ci/build/build-llamacpp.sh --ref 0ab9d6fed --profile vulkan-mi25-server"
server="${OUT}/bin/llama-server"
[[ -x "$server" ]] || _die "${server} is missing or not executable"

# A vulkan-mi25 artifact without libggml-vulkan.so is a mislabelled CPU build,
# and every check below would then be measuring the CPU backend.
ls "${OUT}"/lib/libggml-vulkan.so* >/dev/null 2>&1 \
  || _die "${OUT}/lib has no libggml-vulkan.so — this artifact is not a Vulkan build despite its profile name"

# Any declared role will do — this gate tests the BACKEND, not a model. Take
# the first role the machine declares (ci/runner/rig-local.sh) so the gate needs
# no model of its own and names none.
if [[ -z "$MODEL" ]]; then
  for _r in ${INFERFERENCE_ROLES:-}; do
    _v="MODEL_${_r^^}"
    if [[ -n "${!_v:-}" ]]; then MODEL="${!_v}"; echo ">> role:     ${_r} (first declared)"; break; fi
  done
fi
[[ -n "$MODEL" ]] || _die "no model to smoke with. Pass --model, set VULKAN_SMOKE_MODEL, or declare a role in ci/runner/rig-local.sh (see rig-local.example.sh)."
[[ -r "$MODEL" ]] || _die "model not readable: ${MODEL}"

# --- the card this job was assigned, from sysfs -------------------------------
_cards() { printf '%s\n' /sys/class/drm/card[0-9]*/device | sort -V; }
sysfs_dev=""
i=0
for d in $(_cards); do
  [[ -r "${d}/mem_info_vram_total" ]] || continue
  if (( i == IDX )); then sysfs_dev="$d"; break; fi
  i=$(( i + 1 ))
done
[[ -n "$sysfs_dev" ]] || _die "no amdgpu card at index ${IDX} (sysfs reports ${i} card(s)). LLAMA_CI_GPU names a card this rig does not have."
pci="$(basename "$(readlink -f "$sysfs_dev")")"
_vram_used_mib() { echo $(( $(cat "${sysfs_dev}/mem_info_vram_used") / 1048576 )); }
# GTT is host memory the GPU reaches over PCIe. RADV spills into it when VRAM is
# exhausted, and NOTHING in llama.cpp's output says so: the model still reports
# every layer "assigned to VulkanN", still loads, still answers. Measured
# 2026-09-04: the same 9B at the same ctx ran 30.51 tok/s resident
# and 2.18 tok/s spilled -- a 14x loss reported as success. So a spill is the
# CPU-fallback failure in its real form on this hardware, and this gate must see
# it. AGENTS.md: NO CPU FALLBACK, EVER.
vram_before="$(_vram_used_mib)"
card="${sysfs_dev%/device}"
card="${card##*card}"

log="$(mktemp -t vulkan-mi25-smoke-XXXXXX.log)"
pid=""
cleanup() {
  if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
}
trap cleanup EXIT

echo ">> artifact: ${OUT}"
echo ">> model:    ${MODEL}"
echo ">> device:   Vulkan0 -> ${pci} (${vram_before} MiB VRAM in use before start)"
echo ">> log:      ${log}"

# -lv 5 is load-bearing: at the default verbosity this build prints NEITHER the
# device line NOR the per-layer assignment, so there is nothing to check.
# --device pins one card, per the one-role-one-card rule.
GGML_VK_VISIBLE_DEVICES="$IDX" INFERFERENCE_GTT_BASELINE_CARD="$card" "$server" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
          --device "Vulkan0" --n-gpu-layers 99 --ctx-size 2048 -lv 5 \
          > "$log" 2>&1 &
pid=$!

deadline=$(( SECONDS + ${VULKAN_SMOKE_TIMEOUT:-180} ))
ready=0
while (( SECONDS < deadline )); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "--- server log (tail) ---" >&2; tail -40 "$log" >&2
    _die "llama-server exited before it became healthy"
  fi
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
(( ready == 1 )) || { echo "--- server log (tail) ---" >&2; tail -40 "$log" >&2; _die "server never answered /health"; }
baseline="$(grep -E '^INFERFERENCE_BASELINE card=[0-9]+ pci=[0-9a-fA-F:.]+ gtt=[0-9]+ vram=[0-9]+$' "$log" || true)"
[[ "$baseline" =~ ^INFERFERENCE_BASELINE\ card=([0-9]+)\ pci=([0-9a-fA-F:.]+)\ gtt=([0-9]+)\ vram=([0-9]+)$ ]] || _die "missing or invalid server baseline"
[[ "${BASH_REMATCH[1]}" == "$card" && "${BASH_REMATCH[2]}" == "$pci" ]] || _die "server baseline names the wrong card"
gtt_before="${BASH_REMATCH[3]}"
vram_before=$(( BASH_REMATCH[4] / 1048576 ))
IDX=0

# --- 1. the device it used, and that it is the card we meant ------------------
used="$(grep -Eom1 "using device Vulkan${IDX} \(.*\) \([0-9a-fA-F:.]+\)" "$log" || true)"
[[ -n "$used" ]] \
  || { echo "--- server log (tail) ---" >&2; tail -40 "$log" >&2;
       _die "the server never reported using Vulkan${IDX}. It is on the CPU backend, or on a device this gate did not ask for; a passing completion here would be a silent CPU pass."; }
used_pci="$(printf '%s' "$used" | grep -oE '\([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]\)' | tr -d '()')"
[[ "$used_pci" == "$pci" ]] \
  || _die "device-index mismatch: sysfs index ${IDX} is ${pci}, but the server's Vulkan${IDX} is ${used_pci}. The Vulkan enumeration order no longer matches the DRM order, so LLAMA_CI_GPU would pin the WRONG card. Refusing."

# --- 2. every layer on that device --------------------------------------------
on_ours="$(grep -cE "assigned to device Vulkan${IDX}\b" "$log" || true)"
elsewhere="$(grep -E 'assigned to device ' "$log" | grep -cvE "assigned to device Vulkan${IDX}\b" || true)"
(( on_ours > 0 )) || _die "no layer was assigned to Vulkan${IDX} (see ${log})"
(( elsewhere == 0 )) \
  || _die "${elsewhere} layer(s) were assigned to a device other than Vulkan${IDX}: $(grep -E 'assigned to device ' "$log" | grep -vE "Vulkan${IDX}\b" | grep -oE 'device [A-Za-z0-9]+' | sort -u | tr '\n' ' '). A baseline must place the whole model on its own card."
offload="$(sed -nE 's/.*offloaded ([0-9]+)\/([0-9]+) layers to GPU.*/\1 \2/p' "$log" | tail -1)"
[[ -n "$offload" ]] || _die "the server did not report the number of offloaded layers"
read -r offloaded total_layers <<< "$offload"
(( total_layers > 0 && offloaded == total_layers && on_ours >= total_layers )) \
  || _die "layer placement is incomplete: ${offloaded}/${total_layers} offloaded, ${on_ours} assignments to Vulkan${IDX}"

# --- 3. the card's own VRAM accounting ----------------------------------------
vram_after="$(_vram_used_mib)"
delta=$(( vram_after - vram_before ))
(( delta >= MIN_VRAM_MIB )) \
  || _die "VRAM in use on ${pci} rose by only ${delta} MiB (before ${vram_before}, after ${vram_after}); at least ${MIN_VRAM_MIB} MiB was expected. The logs claim a GPU placement the card does not show."

# --- 3b. and did NOT spill to host memory -------------------------------------
gtt_after="$(gtt_read_bytes "${sysfs_dev}/mem_info_gtt_used")" || _die "cannot read GTT usage on ${pci}"
gtt_delta=$(( gtt_after - gtt_before ))
gtt_did_not_increase "$gtt_before" "$gtt_after" \
  || _die "GTT on ${pci} rose by ${gtt_delta} bytes (before ${gtt_before}, after ${gtt_after}). The gate cannot prove VRAM residency."

# --- generate ------------------------------------------------------------------
resp="$(curl -sf -X POST "http://127.0.0.1:${PORT}/completion" \
         -H 'Content-Type: application/json' \
         -d '{"prompt":"The capital of France is","n_predict":16,"temperature":0}' || true)"
[[ -n "$resp" ]] || { echo "--- server log (tail) ---" >&2; tail -40 "$log" >&2; _die "/completion returned nothing"; }
if [[ -n "${VULKAN_SMOKE_RESPONSE_FILE:-}" ]]; then
  printf '%s\n' "$resp" > "$VULKAN_SMOKE_RESPONSE_FILE"
fi
text="$(printf '%s' "$resp" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content",""))')"
[[ -n "${text//[[:space:]]/}" ]] || _die "/completion returned empty content: ${resp}"
gtt_final="$(gtt_read_bytes "${sysfs_dev}/mem_info_gtt_used")" || _die "cannot read GTT usage after completion on ${pci}"
gtt_did_not_increase "$gtt_before" "$gtt_final" || _die "GTT rose during completion (before ${gtt_before}, after ${gtt_final})"

echo ">> device:     ${used}"
echo ">> layers:     ${offloaded}/${total_layers} offloaded, ${on_ours} assignments to Vulkan${IDX}, 0 elsewhere"
echo ">> VRAM:       +${delta} MiB on ${pci} (${vram_before} -> ${vram_after})"
echo ">> GTT:        ${gtt_delta} bytes (no increase)"
echo ">> completion: ${text}"
echo "[gate] PASS  vulkan-mi25  (MI25 ${pci} loads the model and generates)"
