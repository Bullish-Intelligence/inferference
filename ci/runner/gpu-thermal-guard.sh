#!/usr/bin/env bash
# Thermal admission control and watchdog for the GPUs of the host.
#
# LIMIT: guarded GPU commands default to a maximum of 80 C across all
# detected amdgpu temperature sensors. A command that crosses the limit is killed.
#
# TWO LIMIT ACTIONS. `--on-limit kill` (the default) is the hard stop that
# builds, gates, and experiments need: the process group dies, exit 75.
# `--on-limit pause` is for the long-lived serving router (a thermal stop must
# not ruin an overnight run). The guard freezes
# the whole process group with SIGSTOP at MAX_C, and thaws it with SIGCONT when
# the hottest sensor stays at or below RESUME_C for RESUME_HOLD_S seconds and
# the fan watchdog reads OK. A pause has no time limit. Only HARD_C kills a
# paused group: the GPUs are idle, so a rising temperature means cooling failed.
#
# WHY THE WHOLE GROUP. The llama.cpp router never health-checks its children.
# It learns of a child exit only from end-of-file on the child's stdout pipe,
# and it checks liveness with waitpid(WNOHANG), which ignores a stopped child.
# A stopped child is neither dead nor restarted. A stopped router cannot act
# on a frozen child, and its listen socket keeps queued requests. So the guard
# stops the router and its children together.
#
# SENSOR SELECTION. Read every available amdgpu sensor and gate on the hottest
# reading. This catches cards where junction or memory temperature leads edge.
#
# SCOPE. This script guards commands started with run and processes passed to
# watch. Changing its default does not monitor processes started without it.
# The router launcher starts the router under this guard.
#
# Usage:
#   gpu-thermal-guard.sh check                 # print temps; exit 1 if any is over
#   gpu-thermal-guard.sh watch <pid>           # poll until <pid> exits; kill it if over
#   gpu-thermal-guard.sh run [--on-limit kill|pause] -- <cmd> [args]
#                                              # run <cmd> under the watchdog
#   gpu-thermal-guard.sh cool                  # wait until every card is under the resume mark
#
# Environment:
#   GPU_THERMAL_MAX_C     hard limit, default 80
#   GPU_THERMAL_RESUME_C  `cool` waits for this, default 70
#   GPU_THERMAL_POLL_S    poll interval, default 5
#   GPU_THERMAL_COOL_TIMEOUT_S  give up cooling after this, default 600
#   GPU_THERMAL_SYSFS_ROOT  sensor root, default /sys/class/drm
#   GPU_THERMAL_FAN_STATUS_FILE  optional fan watchdog status file
#   GPU_THERMAL_FAN_WATCHDOG_UNIT  system unit, default arctic-fan-watchdog.service
#   GPU_THERMAL_TERM_GRACE_TENTHS  TERM grace in 0.1s steps, default 50
#   GPU_THERMAL_ON_LIMIT  kill (default) or pause; `run --on-limit` overrides
#   GPU_THERMAL_HARD_C    pause mode: kill a paused group at this, default 90
#   GPU_THERMAL_RESUME_HOLD_S  pause mode: hold RESUME_C this long, default 30
#   GPU_THERMAL_STATUS_FILE  pause mode: state file, default
#                            $XDG_RUNTIME_DIR/inferference-thermal-guard.status
#   GPU_THERMAL_FAN_GRACE_S  tolerate an INDETERMINATE fan subsystem for this
#                            long while watching, default 10. A definite bad
#                            state still stops the job at once.
#   GPU_THERMAL_SENDER_TRACE  `run` starts under gpu-thermal-signal-relay.py,
#                            which logs the PID and command that sent each
#                            stop signal. Default 1. Set 0 to turn it off. It
#                            turns itself off, with one log line, when python3
#                            is missing or broken. It changes no safety rule.
#   GPU_THERMAL_TRACE_PYTHON  the interpreter for the relay helper, default
#                            python3 from PATH
set -euo pipefail

MAX_C="${GPU_THERMAL_MAX_C:-80}"
RESUME_C="${GPU_THERMAL_RESUME_C:-70}"
POLL_S="${GPU_THERMAL_POLL_S:-5}"
COOL_TIMEOUT_S="${GPU_THERMAL_COOL_TIMEOUT_S:-600}"
SYSFS_ROOT="${GPU_THERMAL_SYSFS_ROOT:-/sys/class/drm}"
FAN_STATUS_FILE="${GPU_THERMAL_FAN_STATUS_FILE:-}"
FAN_WATCHDOG_UNIT="${GPU_THERMAL_FAN_WATCHDOG_UNIT:-arctic-fan-watchdog.service}"
TERM_GRACE_TENTHS="${GPU_THERMAL_TERM_GRACE_TENTHS:-50}"
FAN_GRACE_S="${GPU_THERMAL_FAN_GRACE_S:-10}"
ON_LIMIT="${GPU_THERMAL_ON_LIMIT:-kill}"
HARD_C="${GPU_THERMAL_HARD_C:-90}"
RESUME_HOLD_S="${GPU_THERMAL_RESUME_HOLD_S:-30}"
STATUS_FILE="${GPU_THERMAL_STATUS_FILE:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/inferference-thermal-guard.status}"
THERMAL_STOP_EXIT=75
SAFETY_GATE_EXIT=76
[[ "$TERM_GRACE_TENTHS" =~ ^[0-9]+$ ]] || { echo "!! [thermal] invalid TERM grace" >&2; exit 2; }
[[ "$FAN_GRACE_S" =~ ^[0-9]+$ ]] || { echo "!! [thermal] invalid fan grace" >&2; exit 2; }
[[ "$HARD_C" =~ ^[0-9]+$ ]] || { echo "!! [thermal] invalid hard limit" >&2; exit 2; }
[[ "$RESUME_HOLD_S" =~ ^[0-9]+$ ]] || { echo "!! [thermal] invalid resume hold" >&2; exit 2; }
# systemd gives the router a narrow PATH that omits util-linux's setsid.
THERMAL_GUARD_SETSID_BIN="$(command -v setsid 2>/dev/null || true)"
if [[ -z "$THERMAL_GUARD_SETSID_BIN" && -x /run/current-system/sw/bin/setsid ]]; then
  THERMAL_GUARD_SETSID_BIN=/run/current-system/sw/bin/setsid
fi

_die() { echo "!! [thermal] $*" >&2; exit 2; }
_safety_error() { echo "!! [thermal] $*" >&2; return "$SAFETY_GATE_EXIT"; }

# Exit 0 healthy, INDETERMINATE_FAN when the fan subsystem cannot be read at
# all, SAFETY_GATE_EXIT when it definitely reports bad cooling.
#
# INDETERMINATE means the watchdog is absent, not that cooling has failed. A
# `nixos-rebuild switch` that changes the watchdog restarts it, and systemd
# deletes its RuntimeDirectory while it is down, so the status file disappears
# for a few seconds. A live watchdog reporting DEGRADED or NO_AIRFLOW is a
# definite failure and never indeterminate.
INDETERMINATE_FAN=70
_check_fan_state_now() {
  [[ -n "$FAN_STATUS_FILE" ]] || return 0
  [[ -r "$FAN_STATUS_FILE" ]] || { echo "!! [thermal] fan watchdog status is missing or unreadable at $FAN_STATUS_FILE" >&2; return "$INDETERMINATE_FAN"; }
  local state
  state="$(cut -d= -f2 "$FAN_STATUS_FILE" | cut -d' ' -f1)" \
    || { echo "!! [thermal] could not read fan watchdog state" >&2; return "$INDETERMINATE_FAN"; }
  [[ "$state" == "OK" ]] || { _safety_error "fan watchdog state is ${state:-missing}; refusing to run"; return "$SAFETY_GATE_EXIT"; }
  command -v systemctl >/dev/null 2>&1 \
    || { _safety_error "systemctl is unavailable; refusing to trust fan status"; return "$SAFETY_GATE_EXIT"; }
  systemctl is-active --quiet "$FAN_WATCHDOG_UNIT" \
    || { echo "!! [thermal] fan watchdog unit $FAN_WATCHDOG_UNIT is not active" >&2; return "$INDETERMINATE_FAN"; }
}

# Strict form, for admission control. A job must not START unless cooling is
# known good, so indeterminate is a refusal here.
_check_fan_state() {
  local rc=0
  _check_fan_state_now || rc=$?
  if (( rc == 0 )); then
    return 0
  fi
  if (( rc == INDETERMINATE_FAN )); then
    _safety_error "fan watchdog state is indeterminate; refusing to start"
    return "$SAFETY_GATE_EXIT"
  fi
  return "$rc"
}

# Graced form, for the watch loop on an ALREADY RUNNING job. Tolerate an
# indeterminate fan subsystem for FAN_GRACE_S, then fail closed. Killing
# production for a three-second watchdog restart is its own hazard, and the
# temperature limit below keeps watching throughout the window.
_FAN_INDETERMINATE_SINCE=""
_check_fan_state_watch() {
  local rc=0 now
  _check_fan_state_now || rc=$?
  if (( rc == 0 )); then
    if [[ -n "$_FAN_INDETERMINATE_SINCE" ]]; then
      echo ">> [thermal] fan watchdog is readable again" >&2
      _FAN_INDETERMINATE_SINCE=""
    fi
    return 0
  fi
  (( rc == INDETERMINATE_FAN )) || return "$rc"
  now="$(date +%s)"
  if [[ -z "$_FAN_INDETERMINATE_SINCE" ]]; then
    _FAN_INDETERMINATE_SINCE="$now"
    echo ">> [thermal] fan watchdog is indeterminate; tolerating up to ${FAN_GRACE_S}s" >&2
    return 0
  fi
  if (( now - _FAN_INDETERMINATE_SINCE < FAN_GRACE_S )); then
    return 0
  fi
  _safety_error "fan watchdog stayed indeterminate for ${FAN_GRACE_S}s; stopping the job"
  return "$SAFETY_GATE_EXIT"
}

# Every amdgpu hwmon temperature input on the box, as "card<N>/<label>=<C>".
_read_temps() {
  local found=0 card base h t label val raw
  for base in "${SYSFS_ROOT}"/card[0-9]*/device; do
    [[ -d "$base" ]] || continue
    card="$(basename "$(dirname "$base")")"
    for h in "$base"/hwmon/hwmon*/; do
      [[ -r "${h}name" ]] || continue
      [[ "$(cat "${h}name")" == "amdgpu" ]] || continue
      for t in "$h"temp*_input; do
        [[ -r "$t" ]] || continue
        raw="$(cat "$t" 2>/dev/null)" || { _safety_error "could not read temperature sensor $t"; return "$SAFETY_GATE_EXIT"; }
        [[ "$raw" =~ ^[0-9]+$ ]] || { _safety_error "invalid temperature sensor value at $t"; return "$SAFETY_GATE_EXIT"; }
        val=$(( raw / 1000 ))
        label="$(cat "${t%_input}_label" 2>/dev/null || basename "$t")"
        echo "${card}/${label}=${val}"
        found=1
      done
    done
  done
  # No sensor is not "cool enough" — it is no thermal protection at all.
  [[ "$found" == 1 ]] || { _safety_error "no amdgpu temperature sensor found; refusing to run a GPU job unguarded"; return "$SAFETY_GATE_EXIT"; }
}

# Hottest reading on the box.
_hottest() {
  local readings hot
  readings="$(_read_temps)" || return $?
  hot="$(printf '%s\n' "$readings" | sed 's/.*=//' | sort -n | tail -1)"
  [[ "$hot" =~ ^[0-9]+$ ]] || { _safety_error "could not determine hottest GPU temperature"; return "$SAFETY_GATE_EXIT"; }
  printf '%s\n' "$hot"
}

_report() {
  local readings
  readings="$(_read_temps)" || return $?
  echo ">> [thermal] $(printf '%s\n' "$readings" | tr '\n' ' ')(limit ${MAX_C} C)"
}

cmd_check() {
  local hot
  _report
  hot="$(_hottest)"
  if (( hot > MAX_C )); then
    echo "!! [thermal] a card is at ${hot} C, over the ${MAX_C} C limit. Not starting." >&2
    return 1
  fi
  return 0
}

cmd_cool() {
  local hot deadline=$(( SECONDS + COOL_TIMEOUT_S ))
  while :; do
    hot="$(_hottest)" || return $?
    (( hot <= RESUME_C )) && { echo ">> [thermal] cards at ${hot} C, under the ${RESUME_C} C resume mark"; return 0; }
    echo ">> [thermal] waiting to cool: hottest ${hot} C > ${RESUME_C} C"
    if (( SECONDS >= deadline )); then
      _safety_error "cards did not cool below ${RESUME_C} C within ${COOL_TIMEOUT_S}s"
      return "$SAFETY_GATE_EXIT"
    fi
    sleep "$POLL_S"
  done
}

# Stop every process in the guarded command's private process group.
_terminate_process_group() {
  local pid="$1" attempt
  # A stopped process holds TERM pending. Thaw the group first.
  kill -CONT -- "-$pid" 2>/dev/null || true
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for (( attempt=0; attempt<TERM_GRACE_TENTHS; attempt++ )); do
    kill -0 -- "-$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

# One line per poll: a timestamp plus every sensor reading. GPU_THERMAL_POLL_S
# is the only knob, so this is the whole time series a 5 s failsafe can afford
# -- a line before and after a run told nothing about what happened between.
_log_tick() {
  printf '>> [thermal] tick %s %s(limit %s C)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" "$(printf '%s\n' "$1" | tr '\n' ' ')" "$MAX_C"
}

# Poll while <pid> lives. Thermal or safety failures stop its whole process tree.
cmd_watch() {
  local pid="$1" hot readings rc
  [[ -n "$pid" ]] || _die "watch needs a pid"
  if [[ "$ON_LIMIT" == pause ]]; then
    _watch_pause "$pid"
    return $?
  fi
  while kill -0 "$pid" 2>/dev/null; do
    if readings="$(_read_temps)"; then :; else
      rc=$?
      _terminate_process_group "$pid"
      return "$rc"
    fi
    hot="$(printf '%s\n' "$readings" | sed 's/.*=//' | sort -n | tail -1)"
    _log_tick "$readings"
    if _check_fan_state_watch; then :; else
      rc=$?
      _terminate_process_group "$pid"
      return "$rc"
    fi
    if (( hot > MAX_C )); then
      echo "!! [thermal] OVER LIMIT: ${hot} C > ${MAX_C} C — stopping process group ${pid}" >&2
      printf '%s\n' "$readings" | sed 's/^/!!   /' >&2
      _terminate_process_group "$pid"
      echo "!! [thermal] thermal stop exit status ${THERMAL_STOP_EXIT}" >&2
      return "$THERMAL_STOP_EXIT"
    fi
    sleep "$POLL_S"
  done
  return 0
}

# --- pause mode ---------------------------------------------------------------

# Write the current state for the operator. One line of key=value pairs, replaced
# whole so a reader never sees half a line. A missing runtime directory must
# not stop the job, so a failed write is silent.
_write_status() {
  local state="$1" reason="$2" hot="$3" since="$4" tmp
  tmp="${STATUS_FILE}.$$"
  {
    printf 'state=%s hottest_c=%s limit_c=%s resume_c=%s hard_c=%s since=%s updated=%s pid=%s' \
      "$state" "$hot" "$MAX_C" "$RESUME_C" "$HARD_C" "$since" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$"
    [[ -z "$reason" ]] || printf ' reason="%s"' "$reason"
    printf '\n'
  } >"$tmp" 2>/dev/null && mv -f "$tmp" "$STATUS_FILE" 2>/dev/null || rm -f "$tmp" 2>/dev/null || true
}

# Fan watchdog state for pause mode. Sets FAN_REASON to "ok", or to the reason
# the fan is bad. An unreadable watchdog (status file missing, unit inactive)
# counts as bad only after FAN_GRACE_S, so a three second watchdog restart does
# not freeze the job. A definite bad state counts at once. Call it directly,
# not in a command substitution: it keeps the grace clock in a global.
_PAUSE_FAN_BAD_SINCE=""
FAN_REASON=ok
_pause_fan_reason() {
  local rc=0 msg
  FAN_REASON=ok
  msg="$(_check_fan_state_now 2>&1)" || rc=$?
  if (( rc == 0 )); then
    _PAUSE_FAN_BAD_SINCE=""
    return 0
  fi
  msg="${msg#\!\! \[thermal\] }"
  if (( rc == INDETERMINATE_FAN )); then
    [[ -n "$_PAUSE_FAN_BAD_SINCE" ]] || _PAUSE_FAN_BAD_SINCE="$SECONDS"
    (( SECONDS - _PAUSE_FAN_BAD_SINCE >= FAN_GRACE_S )) || return 0
  fi
  FAN_REASON="fan watchdog not OK: ${msg:-unreadable}"
}

_watch_stopping=0
_watch_note_stopping() { _watch_stopping=1; }

# Pause-mode watch loop. See the header for the rules.
_watch_pause() {
  local pid="$1" hot readings rc fan reason
  local paused=0 since="" paused_at=0 ok_since="" bad_since=""
  trap _watch_note_stopping USR1
  while kill -0 "$pid" 2>/dev/null; do
    if readings="$(_read_temps)"; then :; else
      rc=$?
      _write_status killed "temperature sensors unreadable" "" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      _terminate_process_group "$pid"
      return "$rc"
    fi
    hot="$(printf '%s\n' "$readings" | sed 's/.*=//' | sort -n | tail -1)"
    _log_tick "$readings"

    # Fail-safe. The GPUs are idle while paused, so a card at HARD_C means the
    # cooling failed. Kill, as the default mode does.
    if (( hot >= HARD_C )); then
      if (( paused )); then
        echo "!! [thermal] HARD LIMIT: ${hot} C >= ${HARD_C} C while paused. The GPUs are idle, so cooling has failed. Stopping process group ${pid}" >&2
      else
        echo "!! [thermal] HARD LIMIT: ${hot} C >= ${HARD_C} C. Stopping process group ${pid}" >&2
      fi
      printf '%s\n' "$readings" | sed 's/^/!!   /' >&2
      _write_status killed "hard limit ${HARD_C} C reached at ${hot} C" "$hot" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      _terminate_process_group "$pid"
      echo "!! [thermal] thermal stop exit status ${THERMAL_STOP_EXIT}" >&2
      return "$THERMAL_STOP_EXIT"
    fi

    _pause_fan_reason
    fan="$FAN_REASON"

    # The guard is stopping the job. Never freeze it again, or the clean stop
    # would hang. The HARD_C check above still guards it.
    if (( _watch_stopping )); then
      paused=0
      _write_status stopping "guard received a stop signal" "$hot" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      sleep "$POLL_S"
      continue
    fi

    if (( ! paused )); then
      reason=""
      if (( hot >= MAX_C )); then
        reason="hottest sensor ${hot} C >= ${MAX_C} C limit"
      elif [[ "$fan" != ok ]]; then
        reason="$fan"
      fi
      if [[ -n "$reason" ]]; then
        kill -STOP -- "-$pid" 2>/dev/null || kill -STOP "$pid" 2>/dev/null || true
        paused=1
        since="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        paused_at="$SECONDS"
        ok_since=""
        echo "!! [thermal] PAUSED process group ${pid}: ${reason}; resume at <= ${RESUME_C} C held ${RESUME_HOLD_S}s with the fan OK; kill at ${HARD_C} C" >&2
        printf '%s\n' "$readings" | sed 's/^/!!   /' >&2
        _write_status paused "$reason" "$hot" "$since"
      else
        _write_status running "" "$hot" ""
      fi
    else
      if (( hot <= RESUME_C )) && [[ "$fan" == ok ]]; then
        [[ -n "$ok_since" ]] || ok_since="$SECONDS"
        if (( SECONDS - ok_since >= RESUME_HOLD_S )); then
          kill -CONT -- "-$pid" 2>/dev/null || kill -CONT "$pid" 2>/dev/null || true
          echo ">> [thermal] RESUMED process group ${pid} after $(( SECONDS - paused_at ))s: ${hot} C <= ${RESUME_C} C for ${RESUME_HOLD_S}s, fan OK" >&2
          paused=0
          ok_since=""
          _write_status running "" "$hot" ""
          sleep "$POLL_S"
          continue
        fi
      else
        ok_since=""
      fi
      reason="hottest sensor ${hot} C"
      [[ "$fan" == ok ]] || reason="$fan"
      _write_status paused "$reason" "$hot" "$since"
    fi
    sleep "$POLL_S"
  done
  return 0
}

# Pause-mode admission. Wait with no time limit until the fan reads OK and the
# hottest sensor is at or below RESUME_C. A missing sensor is still a refusal:
# the guard cannot protect what it cannot read.
_wait_until_safe() {
  local hot fan last="" last_at=-100 shown
  while :; do
    hot="$(_hottest)" || return $?
    _pause_fan_reason
    fan="$FAN_REASON"
    if (( hot <= RESUME_C )) && [[ "$fan" == ok ]]; then
      _write_status running "" "$hot" ""
      return 0
    fi
    shown="hottest ${hot} C (start at <= ${RESUME_C} C)"
    [[ "$fan" == ok ]] || shown="$fan"
    _write_status waiting "$shown" "$hot" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ "$shown" != "$last" ]] || (( SECONDS - last_at >= 60 )); then
      echo ">> [thermal] waiting to start: ${shown}" >&2
      last="$shown"
      last_at="$SECONDS"
    fi
    sleep "$POLL_S"
  done
}

# Relay a stop signal into the guarded process group, ONCE.
#
# The unit sets KillMode=mixed, so systemd signals only this script. Without
# this relay the guarded tree gets no stop signal at all and systemd SIGKILLs
# it. With the systemd default KillMode=control-group instead, every process in
# the cgroup gets KillSignal, this script dies at once because it traps
# nothing, and systemd re-broadcasts KillSignal in final-sigterm — so
# llama-server sees two interrupts about 1.5 ms apart and calls exit() from its
# own signal handler, which deadlocked 4 of 38 stops. Both measurements:
# the router-stop hang investigation.
_GUARD_CHILD=0
_GUARD_DOG=0
_GUARD_FORWARDED=0
_forward_stop() {
  (( _GUARD_FORWARDED )) && return 0
  _GUARD_FORWARDED=1
  echo ">> [thermal] relaying stop to process group ${_GUARD_CHILD}" >&2
  # Tell the pause-mode watcher not to freeze the group again, then thaw it.
  # A stopped process holds the stop signal pending and would never exit.
  if [[ "$ON_LIMIT" == pause ]] && (( _GUARD_DOG )); then
    kill -USR1 "$_GUARD_DOG" 2>/dev/null || true
  fi
  kill -CONT -- "-${_GUARD_CHILD}" 2>/dev/null || true
  kill -INT -- "-${_GUARD_CHILD}" 2>/dev/null \
    || kill -INT "${_GUARD_CHILD}" 2>/dev/null || true
}

# Last-resort thaw. Whatever ends the guard, never leave a stopped group
# frozen with no one to wake it. Harmless when the group runs or is gone.
_thaw_on_exit() {
  (( _GUARD_CHILD )) || return 0
  # Stop the watcher first. An orphaned pause-mode watcher would freeze the
  # group again.
  if (( _GUARD_DOG )); then
    kill "$_GUARD_DOG" 2>/dev/null || true
    _GUARD_DOG=0
  fi
  if kill -0 -- "-${_GUARD_CHILD}" 2>/dev/null; then
    kill -CONT -- "-${_GUARD_CHILD}" 2>/dev/null || true
  fi
}

cmd_run() {
  while (( $# > 0 )); do
    case "$1" in
      --on-limit)
        (( $# >= 2 )) || _die "--on-limit needs kill or pause"
        ON_LIMIT="$2"; shift 2 ;;
      --) shift; break ;;
      *) break ;;
    esac
  done
  case "$ON_LIMIT" in
    kill) ;;
    pause) (( HARD_C > MAX_C )) || _die "pause mode needs GPU_THERMAL_HARD_C (${HARD_C}) above GPU_THERMAL_MAX_C (${MAX_C})" ;;
    *) _die "unknown limit action '${ON_LIMIT}' (use kill or pause)" ;;
  esac
  (( $# > 0 )) || _die "run needs a command"
  [[ -n "$THERMAL_GUARD_SETSID_BIN" && -x "$THERMAL_GUARD_SETSID_BIN" ]] \
    || _die "setsid is unavailable; refusing to start without process-group protection"
  if [[ "$ON_LIMIT" == pause ]]; then
    # A long-lived job waits for a safe start. It does not give up.
    _wait_until_safe || return $?
  else
    _check_fan_state || return "$SAFETY_GATE_EXIT"
    cmd_cool || return $?
    _check_fan_state || return "$SAFETY_GATE_EXIT"
    cmd_check || return "$SAFETY_GATE_EXIT"
  fi
  # Own process group, so the watchdog can take the whole child tree down.
  "$THERMAL_GUARD_SETSID_BIN" "$@" &
  local child=$!
  _GUARD_CHILD="$child"
  trap _forward_stop INT TERM
  trap _thaw_on_exit EXIT
  cmd_watch "$child" &
  local dog=$!
  _GUARD_DOG="$dog"
  local rc=0 watch_rc=0
  # A trap makes `wait` return 128+signo with the child still running, so keep
  # waiting until the process group is really gone. Then rc is the child's own
  # status, which RestartPreventExitStatus=75 76 depends on. Do NOT write this
  # as `if wait "$child"; then` — that form loses the 128+signo result.
  while :; do
    set +e; wait "$child"; rc=$?; set -e
    (( rc > 128 )) || break
    kill -0 "$child" 2>/dev/null || break
  done
  if wait "$dog"; then :; else watch_rc=$?; fi
  _GUARD_DOG=0
  if (( watch_rc == THERMAL_STOP_EXIT || watch_rc == SAFETY_GATE_EXIT )); then
    rc="$watch_rc"
  elif (( rc == THERMAL_STOP_EXIT || rc == SAFETY_GATE_EXIT )); then
    # Reserve these statuses for the guard; child crashes still restart normally.
    rc=1
  fi
  if [[ "$ON_LIMIT" == pause ]] && (( watch_rc == 0 )); then
    _write_status exited "command exited with status ${rc}" "" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
  if (( rc != 0 )); then
    local hot
    if hot="$(_hottest)"; then
      (( hot > MAX_C )) && echo "!! [thermal] the command died with a card at ${hot} C — treat the run as void" >&2
    else
      rc="$SAFETY_GATE_EXIT"
    fi
  fi
  _report || return "$SAFETY_GATE_EXIT"
  return "$rc"
}

# Sender trace. A Bash trap cannot name the process that sent a signal. For
# `run`, replace this process with gpu-thermal-signal-relay.py. It starts this
# script again as its child (GPU_THERMAL_TRACE_ACTIVE marks the child), logs the
# sender of each signal, and relays the signal to the child once. The child
# holds every safety rule. Return normally when the helper cannot run, so the
# guard starts as before and never refuses a start because of the trace.
_exec_with_sender_trace() {
  if [[ "${GPU_THERMAL_SENDER_TRACE:-1}" == 0 || -n "${GPU_THERMAL_TRACE_ACTIVE:-}" ]]; then
    return 0
  fi
  local self="${BASH_SOURCE[0]}" relay py
  relay="$(dirname -- "$self")/gpu-thermal-signal-relay.py"
  if [[ ! -r "$relay" ]]; then
    echo ">> [thermal] sender trace is off: $relay is missing" >&2
    return 0
  fi
  py="${GPU_THERMAL_TRACE_PYTHON:-$(command -v python3 2>/dev/null || true)}"
  if [[ -z "$py" && -x /run/current-system/sw/bin/python3 ]]; then
    py=/run/current-system/sw/bin/python3
  fi
  if [[ -z "$py" ]] || ! "$py" -I -S -c 'import signal, subprocess, datetime' 2>/dev/null; then
    echo ">> [thermal] sender trace is off: no working python3" >&2
    return 0
  fi
  export GPU_THERMAL_TRACE_ACTIVE=1
  shopt -s execfail
  exec "$py" -I -S "$relay" -- "$BASH" "$self" run -- "$@" || {
    shopt -u execfail
    echo ">> [thermal] sender trace is off: cannot start $py" >&2
    return 0
  }
}

case "${1:-}" in
  check) shift; cmd_check "$@" ;;
  cool)  shift; cmd_cool  "$@" ;;
  watch) shift; cmd_watch "$@" ;;
  run)   shift; [[ "${1:-}" == "--" ]] && shift; _exec_with_sender_trace "$@"; cmd_run "$@" ;;
  *) echo "usage: gpu-thermal-guard.sh {check|cool|watch <pid>|run [--on-limit kill|pause] -- <cmd>}" >&2; exit 3 ;;
esac
