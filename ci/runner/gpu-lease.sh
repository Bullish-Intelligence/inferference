#!/usr/bin/env bash
# GPU lease — pick a card by free VRAM, hold that card's own lock, run, restore.
#
# The lease's rule: a job that fits beside a resident server must not evict it.
# This replaces the single rig-wide flock in gpu-serialized.sh, which took one
# exclusive untimed lock over one card while the other card was unmanaged.
#
#   gpu-lease.sh <mib> -- <command> [args...]
#     1. pick the card with the most free VRAM (per card; the vendor tool is
#        chosen by the detected GPU vendor)
#     2. flock that card's own lock file
#     3. run with every device variable pinned to that card
#        (CUDA_VISIBLE_DEVICES, GGML_VK_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES, LLAMA_CI_GPU)
#     4. propagate the child's exit code
#
# Step 3.5 — preemption — is NOT in this stage. No long-running unit is installed
# on this rig, so there is nothing to preempt. When
# the units land, the stop/restart goes between the lock and the run, and the
# restore goes after `rc=$?`. This script is already a supervisor and NOT an
# `exec` wrapper for exactly that reason: `exec` replaces the shell, so an EXIT
# trap never fires and a stopped server would never come back. Do not turn the last lines back into `exec "$@"`.
#
# Environment:
#   GPU_LEASE_CARDS          physical indexes to consider, e.g. "0 1". Default:
#                            every card the vendor tool reports.
#   GPU_LEASE_LOCK_DIR       lock directory (default /tmp). One lock per card:
#                            <dir>/llama-gpu-card<N>.lock
#   GPU_LEASE_TIMEOUT        seconds to wait for a card. 0 = wait forever, which
#                            is today's gpu-serialized.sh behaviour and the
#                            default.
#   GPU_LEASE_POLL           seconds between placement attempts (default 5)
#   GPU_LEASE_FREE_OVERRIDE  TEST HOOK. Forced free MiB per card, e.g.
#                            "0=6174,1=5112". Two cards cannot be filled to
#                            within a budget of full reproducibly on this rig,
#                            so the gate forces the input — the same reason the
#                            server carries TEST_FORCE=vram_tight.
#                            Everything downstream of the forced figure is
#                            shipped code.
#
# On free VRAM: nvidia-smi's memory.free already excludes the driver-reserved
# region — it reports 11911/11875 MiB on cards whose nameplate is 12288, which is
# the measured "usable" figure, not the nameplate. So the budget is
# compared against memory.free directly, with no margin constant.
set -euo pipefail

# The rig env first: the real libcuda must precede the nix stub. It
# also sets CUDA_VISIBLE_DEVICES from $LLAMA_CI_GPU; this script overrides both
# after the placement, so there is still exactly one writer per run.
# shellcheck disable=SC1091
source "$(dirname "$0")/rig-env.sh"

_die() { echo "!! [gpu-lease] $*" >&2; exit 2; }

# --- arguments ----------------------------------------------------------------
[[ $# -ge 2 ]] || { echo "usage: gpu-lease.sh <mib> -- <command> [args...]" >&2; exit 3; }
budget_mib="$1"; shift
[[ "$budget_mib" =~ ^[1-9][0-9]*$ ]] || _die "budget '<mib>' must be a positive integer, got '${budget_mib}'"
[[ "${1:-}" == "--" ]] && shift
[[ $# -ge 1 ]] || { echo "usage: gpu-lease.sh <mib> -- <command> [args...]" >&2; exit 3; }

lock_dir="${GPU_LEASE_LOCK_DIR:-/tmp}"
poll_s="${GPU_LEASE_POLL:-5}"
# A BOUNDED wait, by default. This was 0 (wait forever) to match the flock it
# replaced, and that is wrong once a server is always on: a resident server never
# releases its card on its own, so "wait" becomes "hang". 900s is long enough for
# a legitimate queue behind another gate and short enough to be a failure rather
# than a mystery. Set GPU_LEASE_TIMEOUT=0 to restore the old unbounded wait.
timeout_s="${GPU_LEASE_TIMEOUT:-900}"

# --- vendor dispatch ----------------------------------------------------------
# The rig's cards decide how free VRAM is read. NVIDIA answers through
# nvidia-smi; AMD answers through amdgpu's sysfs, because this rig's MI25s are
# gfx900 and ROCm (rocm-smi) dropped that target after 5.7 — the Vulkan baseline
# deliberately installs no ROCm. Detection is by which reader is actually
# present, never by which cards are expected. Override with GPU_LEASE_VENDOR.

# --- per-card VRAM ------------------------------------------------------------
# Every reader emits the SAME four fields, in MiB: index,uuid,total,free. The
# placement loop below is vendor-blind by construction.
#
# CUDA_VISIBLE_DEVICES is cleared for the query. nvidia-smi does report every
# card regardless (checked on this rig), but the query must not depend on that:
# it is the one read whose answer must be about physical cards.
_query_nvidia() {
  # nvidia-smi prints its driver errors on STDOUT, in the same stream as the
  # CSV. Keep only rows shaped like a real answer, so a broken driver yields NO
  # cards (which detection and _read_cards both treat as fatal) instead of one
  # card made of the error message.
  env -u CUDA_VISIBLE_DEVICES nvidia-smi \
    --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader,nounits \
    2>/dev/null | grep -E '^[[:space:]]*[0-9]+,[^,]+,[[:space:]]*[0-9]+,[[:space:]]*[0-9]+[[:space:]]*$' || true
}

# amdgpu sysfs. mem_info_vram_used is the driver's own accounting of allocated
# VRAM, so total-used is the same quantity nvidia-smi calls memory.free.
#
# The "uuid" field carries the card's PCI DOMAIN:BUS:DEV.FN. AMD exposes no UUID
# here, and the PCI address is the property this rule actually wants: a name for
# the physical card that does not move when enumeration order does.
# The index is the position in sorted DRM card order, which is what
# LLAMA_CI_GPU has always meant.
_query_amd() {
  local dev idx=0 total used pci
  for dev in $(printf '%s\n' /sys/class/drm/card[0-9]*/device | sort -V); do
    [[ -r "${dev}/mem_info_vram_total" ]] || continue
    total="$(cat "${dev}/mem_info_vram_total" 2>/dev/null)" || continue
    used="$(cat "${dev}/mem_info_vram_used" 2>/dev/null)" || continue
    [[ "$total" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
    pci="$(basename "$(readlink -f "$dev")")"
    printf '%s,%s,%s,%s\n' \
      "$idx" "$pci" "$(( total / 1048576 ))" "$(( (total - used) / 1048576 ))"
    idx=$(( idx + 1 ))
  done
}

_query() {
  case "$GPU_LEASE_VENDOR" in
    nvidia) _query_nvidia ;;
    amd)    _query_amd ;;
    *)      _die "GPU_LEASE_VENDOR must be nvidia, amd, or auto; got '${GPU_LEASE_VENDOR}'" ;;
  esac
}

# --- vendor dispatch ----------------------------------------------------------
GPU_LEASE_VENDOR="${GPU_LEASE_VENDOR:-auto}"

# Detection asks each reader for A CARD, never whether its binary is installed.
# Presence is not capability: after the NVIDIA cards left this rig, the
# nvidia-smi WRAPPER stayed on PATH and `command -v` still found it. It then
# printed its driver error on STDOUT, which the CSV parser read as a card whose
# name was the error text and whose VRAM was empty — a phantom card that made
# every budget "exceed the largest candidate card (0 MiB)". A reader counts only
# if it returns a row this script can actually place a job on.
_reader_works() { [[ -n "$("$1" 2>/dev/null | head -1)" ]]; }

if [[ "$GPU_LEASE_VENDOR" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && _reader_works _query_nvidia; then
    GPU_LEASE_VENDOR=nvidia
  elif _reader_works _query_amd; then
    GPU_LEASE_VENDOR=amd
  else
    _die "no working per-card VRAM reader (nvidia-smi returned no card, and amdgpu sysfs under /sys/class/drm/card*/device/mem_info_vram_total reported none). A lease cannot place a job without a per-card VRAM figure; refusing rather than guessing a card."
  fi
fi

declare -A CARD_UUID CARD_TOTAL CARD_FREE
CARDS_ALL=()
_read_cards() {
  local idx uuid total free
  CARDS_ALL=()
  while IFS=, read -r idx uuid total free; do
    idx="${idx// /}"; uuid="${uuid// /}"; total="${total// /}"; free="${free// /}"
    [[ -n "$idx" ]] || continue
    # Fail-closed on a malformed row: a card with no VRAM figure cannot be
    # ranked or budget-checked, and admitting it as 0 MiB silently poisons the
    # placement loop.
    [[ "$total" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] \
      || _die "the ${GPU_LEASE_VENDOR} reader returned a row with no usable VRAM figures: '${idx},${uuid},${total},${free}'"
    CARD_UUID[$idx]="$uuid"; CARD_TOTAL[$idx]="$total"; CARD_FREE[$idx]="$free"
    CARDS_ALL+=("$idx")
  done < <(_query)
  [[ ${#CARDS_ALL[@]} -gt 0 ]] || _die "the ${GPU_LEASE_VENDOR} reader reported no cards. Refusing to place a GPU job with no device."

  # TEST HOOK — forced free figures, applied over the real read.
  if [[ -n "${GPU_LEASE_FREE_OVERRIDE:-}" ]]; then
    local pair k v
    for pair in ${GPU_LEASE_FREE_OVERRIDE//,/ }; do
      k="${pair%%=*}"; v="${pair##*=}"
      [[ -n "${CARD_TOTAL[$k]:-}" ]] || _die "GPU_LEASE_FREE_OVERRIDE names card '${k}', which the ${GPU_LEASE_VENDOR} reader does not report"
      [[ "$v" =~ ^[0-9]+$ ]] || _die "GPU_LEASE_FREE_OVERRIDE value for card '${k}' is not an integer: '${v}'"
      CARD_FREE[$k]="$v"
    done
  fi
}
_read_cards

# The candidate set: every card, or the GPU_LEASE_CARDS subset.
CARDS=()
if [[ -n "${GPU_LEASE_CARDS:-}" ]]; then
  for c in ${GPU_LEASE_CARDS//,/ }; do
    [[ -n "${CARD_TOTAL[$c]:-}" ]] || _die "GPU_LEASE_CARDS names card '${c}', which the ${GPU_LEASE_VENDOR} reader does not report"
    CARDS+=("$c")
  done
else
  CARDS=("${CARDS_ALL[@]}")
fi

# A budget no card could ever hold is a wait that never ends. Fail loudly now.
max_total=0
for c in "${CARDS[@]}"; do (( CARD_TOTAL[$c] > max_total )) && max_total="${CARD_TOTAL[$c]}"; done
(( budget_mib <= max_total )) || _die "budget ${budget_mib} MiB exceeds the largest candidate card (${max_total} MiB). No card can ever satisfy it."

# --- nested lease: reuse the outer one, never take a second lock ---------------
# A lease inside a lease would block on a lock its own ancestor holds and hang
# forever. Measured 2026-08-26: the gate battery ran under a lease, and
# a gate script called gpu-serialized.sh, which is a lease — it waited on
# card 1's lock until it was killed. The old single-flock gpu-serialized.sh had
# the same deadlock, so this is not new; it is now fixed rather than avoided.
#
# The outer lease already placed the job, and the card it chose is the card the
# work runs on. The nested call re-checks that card against ITS budget, so a
# nested request the card cannot hold is a loud failure and not a silent
# over-commit.
if [[ -n "${GPU_LEASE_CARD:-}" ]]; then
  card="$GPU_LEASE_CARD"
  _read_cards
  free_now="${CARD_FREE[$card]:-0}"
  if (( free_now < budget_mib )); then
    _die "nested lease asks ${budget_mib} MiB on card ${card}, which the outer lease holds and which now has ${free_now} MiB free. Raise the outer lease's budget (it reserved ${GPU_LEASE_BUDGET_MIB:-?} MiB) or run this command outside it."
  fi
  echo ">> [gpu-lease] nested on card ${card} (outer lease holds the lock; ${free_now} MiB free, budget ${budget_mib} MiB)"
  echo ">> [gpu-lease] running: $*"
  export GPU_LEASE_BUDGET_MIB="$budget_mib"
  set +e
  "$@"
  rc=$?
  set -e
  exit "$rc"
fi

# --- placement ----------------------------------------------------------------
# A locked card is skipped, so the lock IS the reservation: an unlocked card's
# memory.free is not about to be consumed by another lease. Non-blocking flock in
# free-VRAM order, then poll — never a blocking flock on one card, because the
# card that frees first may not be the one with the most free VRAM now.
card=""
waited=0
announced=0
while :; do
  _read_cards
  ranked=$(for c in "${CARDS[@]}"; do echo "${CARD_FREE[$c]} $c"; done | sort -rn | awk '{print $2}')
  for c in $ranked; do
    (( CARD_FREE[$c] >= budget_mib )) || continue
    lock="${lock_dir}/llama-gpu-card${c}.lock"
    exec 9>"$lock"
    if flock -n 9; then card="$c"; break; fi
    exec 9>&-
  done
  [[ -n "$card" ]] && break

  # Say WHY each card was rejected, not just that it was. "Locked" resolves
  # itself when the other job ends; "not enough free" does not, if a resident
  # server is holding the memory. The two need different reactions, so they are
  # never reported as one line.
  if (( announced == 0 )); then
    echo ">> [gpu-lease] waiting up to ${timeout_s}s for a card with ${budget_mib} MiB free:" >&2
    for c in "${CARDS[@]}"; do
      if (( CARD_FREE[$c] < budget_mib )); then
        why="only ${CARD_FREE[$c]} MiB free — something resident is holding it"
      else
        why="${CARD_FREE[$c]} MiB free, but LOCKED by another job"
      fi
      echo "   card ${c}: ${why}  (${CARD_UUID[$c]})" >&2
    done
    echo "   a resident server does NOT yield on its own: preemption is not implemented." >&2
    echo "   stop it (systemctl --user stop inferference-server@<role>) or lower the budget." >&2
    announced=1
  fi
  if (( timeout_s > 0 )) && (( waited >= timeout_s )); then
    for c in "${CARDS[@]}"; do
      (( CARD_FREE[$c] < budget_mib )) && short=1
    done
    _die "no card offered ${budget_mib} MiB within ${timeout_s}s.$(
      [[ -n "${short:-}" ]] && printf '%s' " At least one card is short of free VRAM, which a resident model server would explain — this rig runs ONE always-on server and keeps the other card free for gates (AGENTS.md, GPU discipline). Stop the server, run on the free card, or lower the budget."
    ) Refusing to run on a card that cannot hold the job."
  fi
  sleep "$poll_s"
  waited=$(( waited + poll_s ))
done

# --- pin, run, propagate ------------------------------------------------------
# Every name is written: CUDA_VISIBLE_DEVICES is what a CUDA build reads,
# GGML_VK_VISIBLE_DEVICES is what a Vulkan build reads,
# HIP_VISIBLE_DEVICES is what a ROCm build reads (all three are
# VISIBLE_DEVICE_VARS in src/inferference/gpu_env.py), and LLAMA_CI_GPU is the
# ROLE variable every existing job asserts against — it means "the card this CI
# job was assigned" (rig-env.sh), which is exactly what the lease just decided.
# Re-sourcing rig-env.sh inside the child keeps them, because it derives the
# vendor variables from LLAMA_CI_GPU.
#
# GGML_VK_VISIBLE_DEVICES WAS MISSING AND THE LEASE DID NOT ACTUALLY PIN ANYTHING
# ON THIS RIG. The rig has been Vulkan since 2026-09-04, and
# ggml ignores CUDA_VISIBLE_DEVICES entirely, so a leased job saw BOTH cards. The
# lease still took the right lock and still refused an over-budget placement, so
# nothing looked wrong — but the child was free to run on, or auto-split across,
# a card the lease had not granted it. With llama.cpp's `n_gpu_layers = -1` that
# is the DEFAULT behaviour, not an edge case: a 9B measured this session put
# layers 0-16 on one card and 17-32 on the other with no flag asked for it.
#
# The value is the same DRM index the rest of the lease uses. That index is only
# valid as a Vulkan device index if the two enumerations agree, which is NOT
# guaranteed — ci/gates/vulkan_mi25_smoke.sh checks the PCI address for exactly
# this reason and fails loudly if they ever diverge.
export LLAMA_CI_GPU="$card"
export CUDA_VISIBLE_DEVICES="$card"
export GGML_VK_VISIBLE_DEVICES="$card"
export HIP_VISIBLE_DEVICES="$card"
export GPU_LEASE_CARD="$card"
export GPU_LEASE_VENDOR
export GPU_LEASE_CARD_UUID="${CARD_UUID[$card]}"
export GPU_LEASE_BUDGET_MIB="$budget_mib"

echo ">> [gpu-lease] card ${card} (${CARD_UUID[$card]}), ${CARD_FREE[$card]} MiB free, budget ${budget_mib} MiB; lock ${lock_dir}/llama-gpu-card${card}.lock"
echo ">> [gpu-lease] running: $*"

# STAGE 2 INSERTION POINT: if this card's free VRAM
# is below the budget, stop this card's inferference-gpu@<card>.service HERE,
# inside the lock, and restore it after `rc=$?` below — on every exit path,
# including a non-zero exit and SIGINT.

set +e
"$@"
rc=$?
set -e

exit "$rc"
