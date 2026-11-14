#!/usr/bin/env bash
# Reconcile the running router to the YAML config. This is the ExecReload of the
# router unit: `systemctl reload inferference-router` runs it as the unit user.
#
#   inferference-router-reload.sh
#
# It runs `inferference-reconcile.py apply` and nothing else. `apply` does the
# whole reload, in this order:
#   1. read the config dir and refuse a bad config;
#   2. refuse a placement that does not fit a card (no eviction to make room);
#   3. write the preset file ($INFERFERENCE_PRESET_INI) by temp file + rename;
#   4. GET /models?reload=1, then load only the models whose preset changed;
#   5. prove residency per card: PCI address, VRAM rise, GTT flat.
# When the plan has no change, apply stops after step 2: it writes nothing,
# reloads nothing, verifies nothing, and unloads nothing.
# The unit user owns the preset file, so step 3 works here. A login
# user cannot do it.
#
# EXIT STATUS. Zero only when the reconcile succeeded: residency proven for any
# change, or nothing to change. Any other status fails the reload. systemd then
# keeps the router running and `systemctl reload` reports the failure. This script
# never stops, restarts, or signals the router. If `apply` fails after it changed
# a model, it unloads the models that it changed and no others.
#
# ONE APPLY AT A TIME. The script takes an exclusive lock (flock) before it runs
# `apply`. A second reload that finds the lock held fails at once with status 75
# and runs nothing. The lock file is, in this order:
#   $INFERFERENCE_RELOAD_LOCK       an explicit path. Set it in the unit and in any
#                                   manual run, so that both use one file;
#   $RUNTIME_DIRECTORY/reload.lock  systemd sets it with RuntimeDirectory= (the
#                                   first entry when it lists several);
#   $XDG_RUNTIME_DIR/inferference-reload.lock
#   ${XDG_STATE_HOME:-~/.local/state}/inferference/reload.lock
# The lock belongs to the open file, so it goes when the process ends. A crash
# leaves no stale lock.
#
# TIME BOUND. The unit timeout of a reload is TimeoutStartSec (systemd applies it
# to ExecReload). One deadline of $INFERFERENCE_RELOAD_LOAD_TIMEOUT seconds covers
# every load wait and router call of the whole apply. It is not one limit per wait.
# After the deadline, apply can use at most 240 s more (the settle wait, the
# unload cleanup after a failure, and four router reads). So the worst case is
#   load timeout + 240 s   (default 600 + 240 = 840 s, under 900 s).
# The script refuses a load timeout that breaks this bound against
# $INFERFERENCE_RELOAD_UNIT_TIMEOUT (default 900, the TimeoutStartSec of the
# units). It runs apply with `exec`, so the python process is the main process of
# the reload, and a systemd timeout kills the real worker.
#
# ENVIRONMENT. The same plumbing as the launcher, and no model name:
#   INFERFERENCE_CONFIG_DIR   where placement.yaml + models/ live. Unset: the
#                             reconcile CLI chooses (see its header).
#   INFERFERENCE_PRESET_INI   the preset file the router reads. The unit sets it.
#                             Unset: the reconcile CLI chooses, as for the launcher.
#   INFERFERENCE_PYTHON       interpreter for the reconciler. Default the
#                             library venv of the checkout, as the launcher.
#   INFERFERENCE_RELOAD_LOAD_TIMEOUT
#                             the one deadline for the whole apply, in seconds.
#                             Default 600.
#   INFERFERENCE_RELOAD_UNIT_TIMEOUT
#                             the TimeoutStartSec of the unit that runs this
#                             script. Default 900.
#   INFERFERENCE_RELOAD_LOCK  the lock file. See above.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # ci/runner
repo="$(cd "${here}/../.." && pwd)"

fail() {
  echo "!! [inferference-router-reload] $1" >&2
  exit "${2:-1}"
}

# Keep in step with APPLY_OVERHEAD_S in inferference-reconcile.py.
reload_overhead_s=240

reconciler="${here}/inferference-reconcile.py"
python="${INFERFERENCE_PYTHON:-${repo}/ci/library/.venv/bin/python}"
load_timeout="${INFERFERENCE_RELOAD_LOAD_TIMEOUT:-600}"
unit_timeout="${INFERFERENCE_RELOAD_UNIT_TIMEOUT:-900}"

[[ -x "$python" ]] || fail "no python at ${python}. Set \$INFERFERENCE_PYTHON, or build the router venv: ci/runner/library-venv.sh." 78
[[ -f "$reconciler" ]] || fail "no reconcile CLI at ${reconciler}." 78
[[ "$load_timeout" =~ ^[0-9]+$ ]] || fail "INFERFERENCE_RELOAD_LOAD_TIMEOUT must be whole seconds, got '${load_timeout}'." 78
[[ "$unit_timeout" =~ ^[0-9]+$ ]] || fail "INFERFERENCE_RELOAD_UNIT_TIMEOUT must be whole seconds, got '${unit_timeout}'." 78
(( load_timeout > 0 )) || fail "INFERFERENCE_RELOAD_LOAD_TIMEOUT must be more than zero." 78
(( load_timeout + reload_overhead_s <= unit_timeout )) \
  || fail "load timeout ${load_timeout}s + ${reload_overhead_s}s of overhead is more than the unit timeout ${unit_timeout}s. Lower INFERFERENCE_RELOAD_LOAD_TIMEOUT to $(( unit_timeout - reload_overhead_s )) or less, or raise TimeoutStartSec of the unit." 78

command -v flock >/dev/null || fail "flock not found. Put util-linux on the PATH of the unit." 78

lock="${INFERFERENCE_RELOAD_LOCK:-}"
if [[ -z "$lock" ]]; then
  if [[ -n "${RUNTIME_DIRECTORY:-}" ]]; then
    lock="${RUNTIME_DIRECTORY%%:*}/reload.lock"
  elif [[ -n "${XDG_RUNTIME_DIR:-}" && -d "${XDG_RUNTIME_DIR}" ]]; then
    lock="${XDG_RUNTIME_DIR}/inferference-reload.lock"
  else
    state_root="${XDG_STATE_HOME:-${HOME:+${HOME}/.local/state}}"
    [[ -n "$state_root" ]] || fail "no place for the reload lock: set \$INFERFERENCE_RELOAD_LOCK." 78
    lock="${state_root}/inferference/reload.lock"
  fi
fi
mkdir -p "$(dirname "$lock")" || fail "cannot create the dir of the reload lock ${lock}." 78
exec 9>"$lock" || fail "cannot open the reload lock ${lock}." 78
flock -n 9 || fail "another reload holds ${lock}. This reload ran nothing." 75

echo ">> [inferference-router-reload] reconcile apply (config ${INFERFERENCE_CONFIG_DIR:-default}, preset ${INFERFERENCE_PRESET_INI:-default}, deadline ${load_timeout}s, lock ${lock})"
echo ">> [inferference-router-reload] if apply fails, the router keeps running; read the lines below, the models may not match the YAML."

# exec: the python process replaces this shell, keeps fd 9 (the lock), and is the
# process that systemd signals on a timeout. Its status is the status of the reload.
exec "$python" "$reconciler" --timeout "$load_timeout" apply
