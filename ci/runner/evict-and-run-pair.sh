#!/usr/bin/env bash
#
# evict-and-run-pair.sh -- free two cards from the router, then run a dual-card
# experiment spec on both cards at once.
#
# WHY THIS EXISTS. A dual-card run starts two llama-server processes, one whole
# model instance on each card. The residency proof refuses a card with 1,024 MiB
# or more of VRAM in use before load (src/inferference/experiment/residency.py).
# The supported way to free a card is `ci/runner/inferference-reconcile.py
# evict`, which leaves the placement config untouched, so a later `apply`
# restores serving exactly.
#
# This script is evict-and-run-eval.sh for two cards. It evicts both cards,
# polls until both have drained below the residency floor, then execs
# `python -m inferference.experiment run-pair`. Preflight runs first inside the
# CLI and refuses on any mismatch.
#
# The script never uploads a pp_table, never writes pp_features or power1_cap,
# never changes GPU_THERMAL_MAX_C, and never touches the router's placement
# config. If a card does not carry the cap the spec declares, the run refuses
# at preflight -- fail closed, on purpose.
#
# Usage:
#   ci/runner/evict-and-run-pair.sh [<spec.yaml>] [--cards C0,C1] [--no-evict] [-- <extra run-pair args>]
#
#   <spec.yaml>  defaults to the 160 W dual-card arm, the one arm that needs no
#                power1_cap write today (both cards default to 160 W).
#   --cards C0,C1  the two DRM cards (default 0,1). They must match the spec's
#                fixed.placement.cards.
#   --no-evict   skip the router eviction, for a box whose router is down. The
#                drain poll still runs, so a busy card still stops the run.
#   -- ...       forwarded verbatim to `python -m inferference.experiment run-pair`.
#
# The machine-limit guard (80 C) is box-wide. If any card passes it, both runs
# end. The 75 C arm bound stops only the card that reaches it.
#
# Environment (all optional; test overrides first):
#   EVICT_CONFIG_ROOT      placement config dir with one resident model per
#                          card. Used for the evict device lookup and for the
#                          run's --config-root. When set, the script exports it
#                          as INFERFERENCE_CONFIG_DIR for the reconcile tool.
#                          When unset, the script sets no config variable:
#                          $INFERFERENCE_CONFIG_DIR passes through, and the
#                          reconcile tool picks the dir. The script asks
#                          `inferference-reconcile.py config-dir` for it. That
#                          tool owns the default order:
#                          $INFERFERENCE_CONFIG_DIR, then /etc/inferference
#                          when it exists, then ~/.config/inferference with a
#                          warning. The script never sets
#                          INFERFERENCE_PRESET_INI; `evict` does not read it.
#   EVICT_ROUTER           router base URL forwarded as --router to evict.
#   EVICT_SERVER_BIN       server binary for --server-bin. Default:
#                          ci/modea/server/current/bin/llama-server.
#   EVICT_STOCK_PP_TABLE   checked-in stock blob for --stock-pp-table.
#   EVICT_SYSFS_ROOT       drm sysfs root for the drain poll and the PCI probe.
#   EVICT_VRAM_MAX_MIB     drain threshold (default 1024, the residency floor).
#   EVICT_DRAIN_TIMEOUT_S  give up waiting for the drain after this long
#                          (default 300).
#   EVICT_DRAIN_POLL_S     poll interval while draining (default 3).
#   EVICT_PORTS            use these two ports, "P0,P1", instead of a free-port
#                          probe.
#   EVICT_RUNS_DIR         artifact-root parent (default: <repo>/runs).
#   PYTHON_BIN             python to invoke (test override).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_SPEC=""   # pass a spec path; the repository ships no experiment specs
# The stock pp_table is AMD firmware data and is NOT redistributed with
# this repository. Point INFERFERENCE_PP_TABLE_DIR (or --stock-pp-table)
# at the directory that holds it.
STOCK_BLOB_DEFAULT="${INFERFERENCE_PP_TABLE_DIR:-$HOME/.local/share/inferference/pp-tables}/stock-card0.pp_table"
SERVER_BIN_DEFAULT="$REPO_ROOT/ci/modea/server/current/bin/llama-server"

die() { echo "ABORT: $*" >&2; exit 1; }

usage() {
    sed -n '2,56p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---- arguments ---------------------------------------------------------
SPEC="$DEFAULT_SPEC"
CARDS="0,1"
EVICT=1
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --cards) [ $# -ge 2 ] || die "--cards needs a value"; CARDS="$2"; shift 2 ;;
        --no-evict) EVICT=0; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        -*) die "unknown option: $1" ;;
        *) SPEC="$1"; shift ;;
    esac
done

[[ "$CARDS" =~ ^([0-9]+),([0-9]+)$ ]] || die "--cards must be two comma-separated card numbers, such as 0,1 (got '$CARDS')"
CARD_A="${BASH_REMATCH[1]}"
CARD_B="${BASH_REMATCH[2]}"
[ "$CARD_A" != "$CARD_B" ] || die "--cards must name two different cards (got '$CARDS')"
CARD_LIST=("$CARD_A" "$CARD_B")

SYSFS_ROOT="${EVICT_SYSFS_ROOT:-/sys/class/drm}"
VRAM_MAX_MIB="${EVICT_VRAM_MAX_MIB:-1024}"
DRAIN_TIMEOUT_S="${EVICT_DRAIN_TIMEOUT_S:-300}"
DRAIN_POLL_S="${EVICT_DRAIN_POLL_S:-3}"
STOCK_BLOB="${EVICT_STOCK_PP_TABLE:-$STOCK_BLOB_DEFAULT}"
SERVER_BIN="${EVICT_SERVER_BIN:-$SERVER_BIN_DEFAULT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# One source of truth for the default config dir: the reconcile CLI. This
# script sets INFERFERENCE_CONFIG_DIR only when the caller sets EVICT_CONFIG_ROOT.
if [ -n "${EVICT_CONFIG_ROOT:-}" ]; then
    CONFIG_ROOT="$EVICT_CONFIG_ROOT"
    export INFERFERENCE_CONFIG_DIR="$CONFIG_ROOT"
else
    CONFIG_ROOT="$("$PYTHON_BIN" "$REPO_ROOT/ci/runner/inferference-reconcile.py" config-dir)" \
        || die "the reconcile tool could not name the config dir (set EVICT_CONFIG_ROOT or INFERFERENCE_CONFIG_DIR)"
fi
export EVICT_RESOLVED_CONFIG_DIR="$CONFIG_ROOT"

[ -n "$SPEC" ] || die "no experiment spec given: pass one as the first argument (this repository ships no experiment specs)"
[ -f "$SPEC" ] || die "spec not found: $SPEC (pass an absolute path, or run from the spec tree)"
# The script changes to the repository root below. Keep a relative spec valid.
case "$SPEC" in /*) ;; *) SPEC="$PWD/$SPEC" ;; esac
[ -d "$CONFIG_ROOT" ] || die "placement config dir not found: $CONFIG_ROOT (set EVICT_CONFIG_ROOT)"
[ -r "$STOCK_BLOB" ] || die "stock pp_table blob not readable: $STOCK_BLOB (set EVICT_STOCK_PP_TABLE)"
[ -x "$SERVER_BIN" ] || die "server binary not executable: $SERVER_BIN (set EVICT_SERVER_BIN)"

for card in "${CARD_LIST[@]}"; do
    [ -r "$SYSFS_ROOT/card$card/device/mem_info_vram_used" ] \
        || die "cannot read $SYSFS_ROOT/card$card/device/mem_info_vram_used -- is DRM card $card present? (set EVICT_SYSFS_ROOT for tests)"
done

cd "$REPO_ROOT"

# ---- 1. evict: the supported route; the router stays up, config untouched --
if [ "$EVICT" = "1" ]; then
    for card in "${CARD_LIST[@]}"; do
        DEVICE_ID="$(EVICT_CARD="$card" "$PYTHON_BIN" -c '
import os
from pathlib import Path
from inferference.reconcile import read_config_dir
placement, _ = read_config_dir(Path(os.environ["EVICT_RESOLVED_CONFIG_DIR"]))
card = int(os.environ["EVICT_CARD"])
for device in placement.devices:
    if device.card == card:
        print(device.id)
        raise SystemExit(0)
raise SystemExit(f"no placement device declares card {card}")
' )" || die "could not resolve the placement device for card $card from $CONFIG_ROOT"
        echo ">> evicting card $card (placement device: $DEVICE_ID)"
        EVICT_ARGS=(ci/runner/inferference-reconcile.py evict "$DEVICE_ID")
        [ -n "${EVICT_ROUTER:-}" ] && EVICT_ARGS=(ci/runner/inferference-reconcile.py --router "$EVICT_ROUTER" evict "$DEVICE_ID")
        "$PYTHON_BIN" "${EVICT_ARGS[@]}" || die "eviction of card $card failed -- the run is not started; fix the router state and retry"
    done
else
    echo ">> --no-evict: the router is not asked to free either card"
fi

# ---- 2. wait for both drains; do not trust an unload POST as fact ---------
echo ">> waiting for cards ${CARD_A} and ${CARD_B} to drain below ${VRAM_MAX_MIB} MiB"
deadline=$(( $(date +%s) + DRAIN_TIMEOUT_S ))
for card in "${CARD_LIST[@]}"; do
    vram_file="$SYSFS_ROOT/card$card/device/mem_info_vram_used"
    while :; do
        vram_bytes="$(cat "$vram_file")" || die "cannot read $vram_file while polling"
        [[ "$vram_bytes" =~ ^[0-9]+$ ]] || die "$vram_file returned '$vram_bytes', not a byte count"
        vram_mib=$(( vram_bytes / 1048576 ))
        if [ "$vram_mib" -le "$VRAM_MAX_MIB" ]; then
            echo ">> card $card drained: ${vram_mib} MiB in use"
            break
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            die "card $card did not drain below ${VRAM_MAX_MIB} MiB within ${DRAIN_TIMEOUT_S}s (still ${vram_mib} MiB) -- the run is not started; the router may still hold the card (reconcile 'apply' restores serving)"
        fi
        sleep "$DRAIN_POLL_S"
    done
done

# ---- 3. the run: two fresh ports, fresh artifact root, provenance flags ----
if [ -n "${EVICT_PORTS:-}" ]; then
    PORTS="$EVICT_PORTS"
else
    PORTS="$("$PYTHON_BIN" -c '
import socket
probes = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(2)]
for probe in probes:
    probe.bind(("127.0.0.1", 0))
print(",".join(str(probe.getsockname()[1]) for probe in probes))
for probe in probes:
    probe.close()
')" || die "could not probe two free ports"
fi
[[ "$PORTS" =~ ^[0-9]+,[0-9]+$ ]] || die "ports must be two comma-separated numbers (got '$PORTS')"

PCI_ARGS=()
PCI_SUMMARY=""
for card in "${CARD_LIST[@]}"; do
    pci="$(basename "$(readlink -f "$SYSFS_ROOT/card$card/device")")"
    [ -n "$pci" ] || die "could not resolve the PCI address of DRM card $card"
    PCI_ARGS+=(--expect-pci "$card=$pci")
    PCI_SUMMARY+=" card $card ($pci)"
done

SPEC_STEM="$(basename "$SPEC")"; SPEC_STEM="${SPEC_STEM%.yaml}"
RUNS_DIR="${EVICT_RUNS_DIR:-$REPO_ROOT/runs}"
ARTIFACT_ROOT="${RUNS_DIR}/${SPEC_STEM}-pair-$(date -u +%Y%m%dT%H%M%SZ)"
[ -e "$ARTIFACT_ROOT" ] && die "artifact root already exists: $ARTIFACT_ROOT (write-once; do not reuse)"

echo ">> running: $SPEC ${PCI_SUMMARY}  ports $PORTS  artifacts $ARTIFACT_ROOT"
echo ">> note: keep the router idle during the arm -- the machine limit is box-wide and ends both runs; the arm bound ends only its own card."
exec "$PYTHON_BIN" -m inferference.experiment run-pair "$SPEC" \
    --cards "$CARDS" \
    --ports "$PORTS" \
    --server-bin "$SERVER_BIN" \
    --config-root "$CONFIG_ROOT" \
    --stock-pp-table "$STOCK_BLOB" \
    "${PCI_ARGS[@]}" \
    --artifact-root "$ARTIFACT_ROOT" \
    "${EXTRA_ARGS[@]}"
