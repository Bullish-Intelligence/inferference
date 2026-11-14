#!/usr/bin/env bash
# GATE — the router stop path delivers exactly one signal and tears down clean.
#
# WHY THIS EXISTS. Measured on this project:
# the unit set KillSignal=SIGINT and left KillMode at the systemd default
# control-group, so systemd broadcast SIGINT to EVERY process in the cgroup. The
# main process (gpu-thermal-guard.sh) trapped nothing and died at once, systemd
# saw it gone, entered final-sigterm and re-broadcast the signal — so each
# llama-server took two SIGINTs about 1.5 ms apart and called exit() from inside
# its own signal handler. Measured over the journal 2026-09-06 -> 2026-10-06:
# 38 stop jobs, 63 "Received second interrupt" lines, 4 hangs. About 11%.
#
# THE FIX IS TWO HALVES AND THEY MUST LAND TOGETHER. KillMode=mixed (unit, from
# the Nix module) signals only the main process; the guard's relay-once trap
# (working tree, ci/runner/gpu-thermal-guard.sh) forwards it to the child group.
# Deploying only one half leaves TWO signals reaching llama-server by a
# different route, so this gate checks both halves before it cycles anything.
#
# ONE CLEAN STOP IS NOT PROOF. The defect is a race that fired on about 11% of
# stops. Run enough cycles to mean something: 10 is the floor.
#
# Usage:
#   ci/gates/router_stop_gate.sh                 # read-only checks, no restart
#   ci/gates/router_stop_gate.sh --cycles 10     # checks, then 10 stop/start cycles
#   ci/gates/router_stop_gate.sh --cycles 10 --json-out report.json
#
# THE CYCLE TEST INTERRUPTS SERVING. It stops and starts the router N times and
# unloads the resident set each time. It is safe but it is not read-only. The
# default runs no cycles for exactly that reason.
#
# Exit: 0 all checks (and cycles, if asked) passed · 1 a check failed ·
#       2 the environment is wrong (no user bus, no unit, no router)
set -euo pipefail

UNIT="inferference-router.service"
GUARD_RELAY_MARK="relaying stop to process group"
SECOND_INT_MARK="Received second interrupt"
cycles=0
json_out=""
start_timeout="${ROUTER_GATE_START_TIMEOUT:-180}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cycles) cycles="$2"; shift 2 ;;
    --json-out) json_out="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "!! unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ "$cycles" =~ ^[0-9]+$ ]] || { echo "!! --cycles needs a non-negative integer" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # ci/
repo="$(cd "${here}/.." && pwd)"
guard="${here}/runner/gpu-thermal-guard.sh"

# The user manager needs these; a non-login shell (and an agent session) does
# not inherit them, and systemctl --user then reports nothing at all rather
# than failing loudly. Set them before the first query.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"

fails=0
checks_json=""

_ok()   { printf '>> PASS  %-34s %s\n' "$1" "${2:-}"; checks_json+="{\"check\":\"$1\",\"state\":\"pass\",\"detail\":\"${2//\"/}\"},"; }
_bad()  { printf '!! FAIL  %-34s %s\n' "$1" "${2:-}" >&2; fails=$((fails+1)); checks_json+="{\"check\":\"$1\",\"state\":\"fail\",\"detail\":\"${2//\"/}\"},"; }
_info() { printf '   ..    %-34s %s\n' "$1" "${2:-}"; }
_die()  { echo "!! $*" >&2; exit 2; }

_show() { systemctl --user show "$UNIT" -p "$1" --value 2>/dev/null; }

command -v systemctl >/dev/null || _die "systemctl is not on PATH"
[[ -n "$(systemctl --user show "$UNIT" -p Id --value 2>/dev/null)" ]] \
  || _die "user unit ${UNIT} is not loaded. Is this the right user, and has the generation been activated?"

echo "== router stop gate =="
echo "   unit            ${UNIT}"
echo "   fragment        $(_show FragmentPath)"
echo

# ---------------------------------------------------------------- half 1: unit
kill_mode="$(_show KillMode)"
[[ "$kill_mode" == "mixed" ]] \
  && _ok "KillMode" "mixed" \
  || _bad "KillMode" "got '${kill_mode}', want 'mixed' — the unit is stale; update the inferference pin of the host configuration and rebuild FROM THE TRUNK REF"

stop_usec="$(_show TimeoutStopUSec)"
[[ "$stop_usec" == "1min 30s" || "$stop_usec" == "90s" ]] \
  && _ok "TimeoutStopSec" "$stop_usec" \
  || _bad "TimeoutStopSec" "got '${stop_usec}', want 90s"

rpes="$(_show RestartPreventExitStatus)"
[[ "$rpes" == *"75"* && "$rpes" == *"76"* ]] \
  && _ok "RestartPreventExitStatus" "$rpes" \
  || _bad "RestartPreventExitStatus" "got '${rpes}', want '75 76' (stops the thermal kill loop)"

# --------------------------------------------------------------- half 2: guard
grep -q '_forward_stop' "$guard" \
  && _ok "guard carries the relay" "$(basename "$guard")" \
  || _bad "guard carries the relay" "no _forward_stop in ${guard}"

main_pid="$(_show MainPID)"
if [[ -z "$main_pid" || "$main_pid" == "0" ]]; then
  _bad "router is running" "MainPID=${main_pid:-unset}; ActiveState=$(_show ActiveState)"
else
  _info "MainPID" "$main_pid"
  # The RUNNING process must have the trap installed, not just the file on disk.
  # A router started before the guard was patched shows no SIGTERM bit.
  #
  # SigCgt is a mask over bit (signal - 1), so SIGTERM (15) is bit 14 = 0x4000.
  # SIGINT (2) is bit 1 = 0x2, but bash catches SIGINT internally even with no
  # user trap, so SIGINT alone proves nothing. SIGTERM is the discriminator:
  # the patched guard traps INT and TERM, the old one trapped neither.
  #   patched:   0x14002 = SIGCHLD 0x10000 | SIGTERM 0x4000 | SIGINT 0x2
  #   unpatched: 0x10002 = SIGCHLD 0x10000 |                  SIGINT 0x2
  sigcgt="$(awk '/^SigCgt:/ {print $2}' "/proc/${main_pid}/status" 2>/dev/null || true)"
  if [[ -z "$sigcgt" ]]; then
    _bad "running guard traps the stop" "could not read /proc/${main_pid}/status"
  elif (( 0x${sigcgt} & 0x4000 )); then
    _ok "running guard traps the stop" "SigCgt=0x${sigcgt} catches SIGTERM"
  else
    _bad "running guard traps the stop" "SigCgt=0x${sigcgt} does not catch SIGTERM (0x4000) — this process predates the patched guard. Restart the unit."
  fi
fi

# ------------------------------------------------------------------- the router
# The config dir follows the reconcile CLI default (the one source of truth):
# $INFERFERENCE_CONFIG_DIR, then /etc/inferference, then ~/.config/inferference
# with a warning. Python needs the library to run the CLI: set
# $INFERFERENCE_PYTHON when python3 lacks it. On failure the gate says so and
# uses the router default port.
cfg_dir="$("${INFERFERENCE_PYTHON:-python3}" "${here}/runner/inferference-reconcile.py" config-dir 2>/dev/null)" || cfg_dir=""
[[ -n "$cfg_dir" ]] || _info "config dir" "unresolved: using router port 8100 (set INFERFERENCE_PYTHON)"
port="$(INFERFERENCE_GATE_CFG="$cfg_dir" python3 -I -c "
import os,sys,pathlib
try:
    import yaml
except Exception:
    print(8100); sys.exit()
d = os.environ.get('INFERFERENCE_GATE_CFG')
if not d:
    print(8100); sys.exit()
p = pathlib.Path(d)/'placement.yaml'
try:
    print((yaml.safe_load(p.read_text()).get('router') or {}).get('port', 8100))
except Exception:
    print(8100)
" 2>/dev/null || echo 8100)"
base="http://127.0.0.1:${port}"
_info "router endpoint" "$base"

_health() { curl -fsS --max-time 5 "${base}/health" 2>/dev/null; }
_loaded_count() {
  curl -fsS --max-time 10 "${base}/models" 2>/dev/null | python3 -I -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print(-1); sys.exit()
print(sum(1 for m in d.get('data',[]) if (m.get('status') or {}).get('value')=='loaded'))
" 2>/dev/null || echo -1
}

if _health >/dev/null; then
  resident="$(_loaded_count)"
  _ok "router healthy" "${resident} model(s) loaded"
else
  _bad "router healthy" "GET ${base}/health did not return 200"
  resident=-1
fi

echo
echo "-- journal history (whole-unit, for context) --"
hist_second="$(journalctl --user -u "$UNIT" --no-pager 2>/dev/null | grep -c "$SECOND_INT_MARK" || true)"
hist_relay="$(journalctl --user -u "$UNIT" --no-pager 2>/dev/null | grep -c "$GUARD_RELAY_MARK" || true)"
hist_final="$(journalctl --user -u "$UNIT" --no-pager 2>/dev/null | grep -c 'final-sigterm' || true)"
_info "second-interrupt lines (all time)" "$hist_second"
_info "relay lines (all time)"            "$hist_relay"
_info "final-sigterm (all time)"          "$hist_final"
echo

if (( fails > 0 )); then
  echo "!! ${fails} pre-check(s) failed. NOT cycling the router." >&2
  echo "!! Fix the deployment first: both halves must land together, or a stop still" >&2
  echo "!! delivers two signals by a different route." >&2
  exit 1
fi

if (( cycles == 0 )); then
  echo ">> pre-checks passed. No cycles requested."
  echo "   The stop path is NOT yet proven. Run: $0 --cycles 10"
  echo "   That interrupts serving 10 times and unloads the resident set each time."
  exit 0
fi

# ------------------------------------------------------------------ the cycles
echo "== cycling the router ${cycles}x (this interrupts serving) =="
clean=0
cycle_json=""

for ((i=1; i<=cycles; i++)); do
  cursor="$(journalctl --user -u "$UNIT" -n 0 --show-cursor --no-pager 2>/dev/null \
            | sed -n 's/^-- cursor: //p' | tail -1)"
  [[ -n "$cursor" ]] || _die "could not capture a journal cursor"

  t0=$(date +%s.%N)
  stop_rc=0
  systemctl --user stop "$UNIT" || stop_rc=$?
  t1=$(date +%s.%N)
  dur="$(python3 -I -c "print(f'{${t1}-${t0}:.2f}')")"

  result="$(_show Result)"
  since="$(journalctl --user -u "$UNIT" --after-cursor "$cursor" --no-pager 2>/dev/null || true)"
  n_second="$(printf '%s' "$since" | grep -c "$SECOND_INT_MARK" || true)"
  n_relay="$(printf '%s' "$since"  | grep -c "$GUARD_RELAY_MARK" || true)"
  n_final="$(printf '%s' "$since"  | grep -c 'final-sigterm' || true)"
  n_kill="$(printf '%s' "$since"   | grep -c 'Killing process' || true)"

  bad=""
  (( stop_rc == 0 ))        || bad+=" stop-rc=${stop_rc}"
  [[ "$result" == "success" ]] || bad+=" Result=${result}"
  (( n_second == 0 ))       || bad+=" second-interrupt=${n_second}"
  (( n_relay == 1 ))        || bad+=" relay=${n_relay}(want 1)"
  (( n_final == 0 ))        || bad+=" final-sigterm=${n_final}"
  (( n_kill == 0 ))         || bad+=" killing-process=${n_kill}"

  # Restart and wait for the declared resident set to come back.
  systemctl --user start "$UNIT" || bad+=" start-failed"
  deadline=$(( $(date +%s) + start_timeout ))
  healthy=0
  while (( $(date +%s) < deadline )); do
    if _health >/dev/null; then healthy=1; break; fi
    sleep 2
  done
  (( healthy == 1 )) || bad+=" unhealthy-after-start"

  if (( healthy == 1 && resident > 0 )); then
    back=-1
    while (( $(date +%s) < deadline )); do
      back="$(_loaded_count)"
      (( back >= resident )) && break
      sleep 2
    done
    (( back >= resident )) || bad+=" loaded=${back}(want>=${resident})"
  fi

  if [[ -z "$bad" ]]; then
    clean=$((clean+1))
    printf '>> cycle %2d/%d  CLEAN   stop %ss  relay=1 second=0\n' "$i" "$cycles" "$dur"
  else
    printf '!! cycle %2d/%d  DIRTY   stop %ss %s\n' "$i" "$cycles" "$dur" "$bad" >&2
  fi
  cycle_json+="{\"cycle\":$i,\"stop_s\":${dur},\"result\":\"${result}\",\"second_interrupt\":${n_second},\"relay\":${n_relay},\"final_sigterm\":${n_final},\"killing\":${n_kill},\"clean\":$([[ -z "$bad" ]] && echo true || echo false),\"detail\":\"${bad# }\"},"
done

echo
echo "== summary =="
echo "   clean cycles: ${clean}/${cycles}"

if [[ -n "$json_out" ]]; then
  printf '{"unit":"%s","utc":"%s","checks":[%s],"cycles":[%s],"clean":%d,"requested":%d}\n' \
    "$UNIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${checks_json%,}" "${cycle_json%,}" "$clean" "$cycles" \
    > "$json_out"
  echo "   report: ${json_out}"
fi

if (( clean == cycles )); then
  echo ">> PASS — ${cycles}/${cycles} stops delivered exactly one signal and tore down clean."
  exit 0
fi
echo "!! FAIL — ${clean}/${cycles} clean. The stop path is not fixed." >&2
exit 1
