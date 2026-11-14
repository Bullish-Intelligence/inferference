#!/usr/bin/env bash
#
# evict-and-run-eval.sh -- free a card from the router, then immediately run a
# experiment spec against it.
#
# WHY THIS EXISTS. Serving owns the cards (AGENTS.md), but a measured
# experiment runs its own llama-server and needs the measured card idle: the
# residency proof refuses a card with 1,024 MiB or more of VRAM in use before
# load (src/inferference/experiment/residency.py). The supported way to free a
# card is `ci/runner/inferference-reconcile.py evict`, which leaves the
# placement config untouched so a later `apply` restores serving exactly.
#
# This script chains the evict step and the run step, with the waiting built
# in: evict, poll until the card has actually
# drained below the runner's own residency floor, then exec the experiment CLI
# (preflight runs first inside the CLI and refuses on any mismatch).
#
# The script never uploads a pp_table, never writes pp_features, never changes
# GPU_THERMAL_MAX_C, and never touches the router's placement config. If the
# spec's declared table is not already on the card, the run refuses at
# preflight -- fail closed, on purpose.
#
# Usage:
#   ci/runner/evict-and-run-eval.sh <spec.yaml> [--card N] [--no-evict] [-- <extra run args>]
#
#   <spec.yaml>  the experiment spec to run. It is required: this repository
#                ships no experiment specs.
#   --card N     measured card (default 0).
#   --no-evict   skip the evict step. Use it only when the router is not
#                running. The drain check still runs, so a card that holds
#                1,024 MiB or more still stops the run.
#   -- ...       forwarded verbatim to `python -m inferference.experiment run`.
#
# Environment (all optional; test overrides first):
#   EVICT_CONFIG_ROOT      placement config dir. Used for the evict device
#                          lookup and for the run's --config-root. When set,
#                          the script exports it as INFERFERENCE_CONFIG_DIR for
#                          the reconcile tool. When unset, the script sets no
#                          config variable: $INFERFERENCE_CONFIG_DIR passes
#                          through, and the reconcile tool picks the dir. The
#                          script asks `inferference-reconcile.py config-dir`
#                          for it. That tool owns the default order:
#                          $INFERFERENCE_CONFIG_DIR, then /etc/inferference
#                          when it exists, then ~/.config/inferference with a
#                          warning. The script never sets
#                          INFERFERENCE_PRESET_INI; `evict` does not read it.
#   EVICT_ROUTER           router base URL forwarded as --router to evict.
#   EVICT_SERVER_BIN       server binary for --server-bin. Default:
#                          ci/modea/server/current/bin/llama-server (the
#                          repository's Mode A default, machine-local).
#   EVICT_STOCK_PP_TABLE   checked-in stock blob for --stock-pp-table.
#   EVICT_SYSFS_ROOT       drm sysfs root for the drain poll and the PCI probe.
#   EVICT_VRAM_MAX_MIB     drain threshold (default 1024, the residency floor).
#   EVICT_DRAIN_TIMEOUT_S  give up waiting for the drain after this long
#                          (default 300).
#   EVICT_DRAIN_POLL_S     poll interval while draining (default 3).
#   EVICT_PORT             use this fixed port instead of a free-port probe.
#   EVICT_RUNS_DIR          artifact-root parent (default: <repo>/runs).
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
    sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---- arguments ---------------------------------------------------------
SPEC="$DEFAULT_SPEC"
CARD="0"
EVICT="1"
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --card) [ $# -ge 2 ] || die "--card needs a value"; CARD="$2"; shift 2 ;;
        --no-evict) EVICT="0"; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        -*) die "unknown option: $1" ;;
        *) SPEC="$1"; shift ;;
    esac
done

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
export EVICT_CARD="$CARD"

[ -n "$SPEC" ] || die "no experiment spec given: pass one as the first argument (this repository ships no experiment specs)"
[ -f "$SPEC" ] || die "spec not found: $SPEC (pass an absolute path, or run from the spec tree)"
[ -d "$CONFIG_ROOT" ] || die "placement config dir not found: $CONFIG_ROOT (set EVICT_CONFIG_ROOT)"
[ -r "$STOCK_BLOB" ] || die "stock pp_table blob not readable: $STOCK_BLOB (set EVICT_STOCK_PP_TABLE)"
[ -x "$SERVER_BIN" ] || die "server binary not executable: $SERVER_BIN (set EVICT_SERVER_BIN)"

VRAM_FILE="$SYSFS_ROOT/card$CARD/device/mem_info_vram_used"
[ -r "$VRAM_FILE" ] || die "cannot read $VRAM_FILE -- is DRM card $CARD present? (set EVICT_SYSFS_ROOT for tests)"

# ---- 1. evict: the supported route; the router stays up, config untouched --
cd "$REPO_ROOT"
if [ "$EVICT" = "1" ]; then
DEVICE_ID="$("$PYTHON_BIN" -c '
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
' )" || die "could not resolve the placement device for card $CARD from $CONFIG_ROOT"
echo ">> evicting card $CARD (placement device: $DEVICE_ID)"
EVICT_ARGS=(ci/runner/inferference-reconcile.py evict "$DEVICE_ID")
[ -n "${EVICT_ROUTER:-}" ] && EVICT_ARGS=(ci/runner/inferference-reconcile.py --router "$EVICT_ROUTER" evict "$DEVICE_ID")
"$PYTHON_BIN" "${EVICT_ARGS[@]}" || die "eviction failed -- the run is not started; fix the router state and retry"
else
    echo ">> --no-evict: skipping the router evict; the drain check below still gates the run"
fi

# ---- 2. wait for the drain; do not trust an unload POST as fact -----------
echo ">> waiting for card $CARD to drain below ${VRAM_MAX_MIB} MiB"
deadline=$(( $(date +%s) + DRAIN_TIMEOUT_S ))
while :; do
    vram_bytes="$(cat "$VRAM_FILE")" || die "cannot read $VRAM_FILE while polling"
    [[ "$vram_bytes" =~ ^[0-9]+$ ]] || die "$VRAM_FILE returned '$vram_bytes', not a byte count"
    vram_mib=$(( vram_bytes / 1048576 ))
    if [ "$vram_mib" -le "$VRAM_MAX_MIB" ]; then
        echo ">> card $CARD drained: ${vram_mib} MiB in use"
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        die "card $CARD did not drain below ${VRAM_MAX_MIB} MiB within ${DRAIN_TIMEOUT_S}s (still ${vram_mib} MiB) -- the run is not started; the router may still hold the card (reconcile 'apply' restores serving)"
    fi
    sleep "$DRAIN_POLL_S"
done

# ---- 3. the run: fresh port, fresh artifact root, provenance flags ---------
if [ -n "${EVICT_PORT:-}" ]; then
    PORT="$EVICT_PORT"
else
    PORT="$("$PYTHON_BIN" -c '
import socket
probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
probe.bind(("127.0.0.1", 0))
print(probe.getsockname()[1])
probe.close()
')" || die "could not probe a free port"
fi

PCI="$(basename "$(readlink -f "$SYSFS_ROOT/card$CARD/device")")"
[ -n "$PCI" ] || die "could not resolve the PCI address of DRM card $CARD"

SPEC_STEM="$(basename "$SPEC")"; SPEC_STEM="${SPEC_STEM%.yaml}"
# When the spec directory name starts with a number (such as 101-name), prefix
# the run with it, so arms of the same name from different experiments differ.
SPEC_PROJECT="$(basename "$(dirname "$SPEC")")"; SPEC_PROJECT="${SPEC_PROJECT%%-*}"
if [[ "$SPEC_PROJECT" =~ ^[0-9]+$ ]]; then SPEC_STEM="${SPEC_PROJECT}-${SPEC_STEM}"; fi
RUNS_DIR="${EVICT_RUNS_DIR:-$REPO_ROOT/runs}"
ARTIFACT_ROOT="${RUNS_DIR}/${SPEC_STEM}-$(date -u +%Y%m%dT%H%M%SZ)"
[ -e "$ARTIFACT_ROOT" ] && die "artifact root already exists: $ARTIFACT_ROOT (write-once; do not reuse)"

echo ">> running: $SPEC  card $CARD ($PCI)  port $PORT  artifacts $ARTIFACT_ROOT"
echo ">> note: keep the router idle during the arm -- the safety monitor aborts any run on the hottest sensor of ANY card."
exec "$PYTHON_BIN" -m inferference.experiment run "$SPEC" \
    --card "$CARD" \
    --port "$PORT" \
    --server-bin "$SERVER_BIN" \
    --config-root "$CONFIG_ROOT" \
    --stock-pp-table "$STOCK_BLOB" \
    --expect-pci "$PCI" \
    --artifact-root "$ARTIFACT_ROOT" \
    "${EXTRA_ARGS[@]}"
