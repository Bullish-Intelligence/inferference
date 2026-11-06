#!/usr/bin/env bash
# Measure a model's VRAM footprint and print the YAML block the reconciler needs.
#
#   measure-footprint.sh --model <gguf> --device VulkanN --card N [--ctx-per-lane 4096]
#                        [--extra '<arg>' ...]
#
# --extra passes one argument through to llama-server, verbatim, and may repeat.
# It exists because a model's footprint is a property of the ARGV it is served
# with, not of the GGUF: a routing-enabled model (--lora ... --lora-seq-routing,
# carries its adapters on the card and builds a wider mask, so a
# footprint measured without them under-reports what the reconciler must budget.
# Whatever a model's YAML puts in `loras:` and `extra_args:` belongs here too, or
# the measurement describes a different server than the one that will run.
#
# Loads the model twice — at 1 lane and at 4 — and reads the CARD'S OWN
# accounting (mem_info_vram_used) before and after each load, then fits
#
#     total = base_mib + lanes * per_lane_mib
#
# WHY SYSFS AND NOT THE SERVER LOG. llama.cpp reports its model / KV / compute
# buffer sizes, and summing them understates what the card commits by an amount
# that grows per lane (+67 MiB at 1 lane, +259 at 4, on a 12B here). Budgeting
# from the log declared a 16-lane placement that then spilled into host memory
# and ran ~14x slower while reporting success. The card's accounting is the only
# honest source.
#
# The fit is a line in the lane count, which is what the hardware does: on this
# rig it reproduces its own points exactly, predicts an independently measured
# 8-lane load to within 8 MiB, and correctly refuses 16 lanes.
set -euo pipefail

SRV="${SRV:-}"; MODEL=""; DEV=""; CARD=""; CTX=4096; PORT="${PORT:-8176}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL="$2"; shift 2 ;;
    --device)       DEV="$2";   shift 2 ;;
    --card)         CARD="$2";  shift 2 ;;
    --ctx-per-lane) CTX="$2";   shift 2 ;;
    --server)       SRV="$2";   shift 2 ;;
    --extra)        EXTRA+=("$2"); shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" && -n "$DEV" && -n "$CARD" ]] || {
  echo "usage: measure-footprint.sh --model <gguf> --device VulkanN --card N [--ctx-per-lane N]" >&2
  exit 2
}
[[ -r "$MODEL" ]] || { echo "!! model not readable: $MODEL" >&2; exit 1; }
if [[ -z "$SRV" ]]; then
  SRV="$(ls -d "$(dirname "${BASH_SOURCE[0]}")/../build/.llamacpp-builds"/out-*-server-*/bin/llama-server 2>/dev/null | sort | tail -1 || true)"
fi
[[ -x "$SRV" ]] || { echo "!! no llama-server; pass --server" >&2; exit 1; }

base_sys="/sys/class/drm/card${CARD}/device"
[[ -r "${base_sys}/mem_info_vram_used" ]] || { echo "!! no amdgpu sysfs at ${base_sys}" >&2; exit 1; }

_used() { echo $(( $(cat "${base_sys}/mem_info_vram_used") / 1048576 )); }
_gtt()  { echo $(( $(cat "${base_sys}/mem_info_gtt_used")  / 1048576 )); }

measure() { # lanes -> MiB delta on stdout
  local lanes="$1" v0 v1 g0 g1 log pid
  v0="$(_used)"; g0="$(_gtt)"
  log="$(mktemp)"
  # ${EXTRA[@]+"${EXTRA[@]}"} and not "${EXTRA[@]}": an empty array under
  # `set -u` is an unbound-variable abort on older bash, which would break every
  # existing caller that passes no --extra.
  "$SRV" --model "$MODEL" --host 127.0.0.1 --port "$PORT" --device "$DEV" \
         --n-gpu-layers 999 --ctx-size "$(( CTX * lanes ))" --parallel "$lanes" \
         ${EXTRA[@]+"${EXTRA[@]}"} \
         > "$log" 2>&1 &
  pid=$!
  local ok=0
  for _ in $(seq 1 120); do
    kill -0 "$pid" 2>/dev/null || break
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 2
  done
  sleep 3
  v1="$(_used)"; g1="$(_gtt)"
  kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
  if (( ok == 0 )); then
    echo "!! ${lanes}-lane load never became healthy; see $log" >&2; exit 1
  fi
  # A measurement taken on a spilled load describes host RAM, not the card.
  if (( g1 - g0 > 512 )); then
    echo "!! ${lanes}-lane load SPILLED to host memory (GTT +$(( g1 - g0 )) MiB)." >&2
    echo "!! Its footprint would be meaningless. Free the card or lower --ctx-per-lane." >&2
    exit 1
  fi
  rm -f "$log"
  echo $(( v1 - v0 ))
  sleep 3
}

echo ">> measuring $(basename "$MODEL") on ${DEV} (card ${CARD}) at ctx_per_lane=${CTX}" >&2
(( ${#EXTRA[@]} )) && echo ">>   extra argv: ${EXTRA[*]}" >&2
one="$(measure 1)"
echo ">>   1 lane : ${one} MiB" >&2
four="$(measure 4)"
echo ">>   4 lanes: ${four} MiB" >&2

python3 - "$one" "$four" "$CTX" <<'PY'
import sys
one, four, ctx = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
per = (four - one) / 3
base = one - per
print("footprint:")
print(f"  ctx_per_lane: {ctx}")
print(f"  base_mib: {base:.1f}")
print(f"  per_lane_mib: {per:.1f}")
print("  measured_by: sysfs-vram-delta")
PY
