#!/usr/bin/env python3
"""Run a command as a child. Log who sends each signal to this process, then relay it.

The thermal guard is a Bash script. A Bash `trap` cannot name the process that
sent a signal. This helper becomes the process that systemd and the launchers
signal. It blocks the stop signals and reads each one with `sigtimedwait`. The
kernel then gives the sender PID and UID (`si_pid`, `si_uid`). The helper reads
`/proc/<pid>/cmdline` at once, because the sender may exit a moment later.
Then it sends the same signal to the child, once.

The child (the guard) keeps all safety behavior: when it pauses, stops or
kills, and the one-SIGINT relay to the guarded process group. This helper only
adds the log line and passes each signal on.

Usage:
  gpu-thermal-signal-relay.py -- <command> [args...]

Exit status: the child's status. A child that a signal killed makes this
helper die from the same signal, so the caller sees the same wait status.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

# The signals that stop the guard by default. A signal to this process must act
# on the guard in the same way as a signal to the guard itself.
_RELAYED = (
    signal.SIGHUP,
    signal.SIGINT,
    signal.SIGQUIT,
    signal.SIGTERM,
    signal.SIGUSR1,
    signal.SIGUSR2,
    signal.SIGALRM,
)
# The guard relays only the first of these to its process group. So does this helper.
_STOP_ONCE = frozenset({signal.SIGINT, signal.SIGTERM})
_CORE_SIGNALS = frozenset({signal.SIGQUIT, signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS, signal.SIGFPE, signal.SIGILL})

# Linux `si_code` values for a signal that a process or the kernel sent.
_SI_CODES = {0: "kill", 0x80: "kernel", -1: "sigqueue", -2: "timer", -3: "mesgq", -4: "aio", -5: "sigio", -6: "tkill"}

_MAX_TEXT = 240
_WAIT_S = 1.0


def _clean(raw: bytes) -> str:
    text = raw.replace(b"\0", b" ").decode(errors="replace").strip()
    return text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT] + "..."


class _Process(NamedTuple):
    comm: str
    state: str
    ppid: int
    start_s: float
    cmdline: str


def _read_process(pid: int) -> _Process | None:
    """Read what /proc says about one process now. Return None when it is gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(errors="replace")
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    open_paren, close_paren = stat.find("("), stat.rfind(")")
    if open_paren < 0 or close_paren < open_paren:
        return None
    fields = stat[close_paren + 1 :].split()
    try:
        ticks = int(fields[19])
        parent = int(fields[1])
    except (IndexError, ValueError):
        return None
    return _Process(
        comm=stat[open_paren + 1 : close_paren],
        state=fields[0],
        ppid=parent,
        start_s=ticks / os.sysconf("SC_CLK_TCK"),
        cmdline=_clean(cmdline),
    )


def describe_sender(pid: int, uid: int, code: int, received_boot_s: float) -> str:
    """Return one line that names the sender of a signal.

    `received_boot_s` is CLOCK_BOOTTIME at the moment the signal arrived. A
    process that started after that moment cannot be the sender. Then the PID
    was reused and the line says so, never a wrong name.
    """
    kind = _SI_CODES.get(code, f"si_code {code}")
    if pid == 0:
        return f"from the kernel or a terminal (si_code {kind}, no sender pid)"
    head = f"from pid {pid} uid {uid} ({kind})"
    info = _read_process(pid)
    if info is None:
        return f"{head}; that process had already exited, so its command is unknown"
    start_s = info.start_s
    tick_s = 1.0 / os.sysconf("SC_CLK_TCK")
    if start_s > received_boot_s:
        return f"{head}; that pid now belongs to a newer process, so the sender's command is unknown"
    caveat = ""
    if start_s + tick_s > received_boot_s:
        caveat = " (it started within one clock tick of the signal; the pid may be reused)"
    if info.state == "Z" or not info.cmdline:
        what = f'comm "{info.comm}", already exited' if info.state == "Z" else f'comm "{info.comm}"'
        return f"{head} {what}{caveat}"
    text = f'{head} comm "{info.comm}" cmdline "{info.cmdline}"'
    parent = _read_process(info.ppid)
    if parent is not None and parent.cmdline:
        text += f' parent pid {info.ppid} cmdline "{parent.cmdline}"'
    return text + caveat


def _log(message: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        sys.stderr.write(f">> [thermal] {message} at {stamp}\n")
        sys.stderr.flush()
    except OSError:
        # A closed log must never stop the relay.
        pass


def _die_like(returncode: int) -> int:
    """Return the child's status. Die from the same signal when a signal killed it."""
    if returncode >= 0:
        return returncode
    number = -returncode
    if signal.Signals(number) not in _CORE_SIGNALS:
        signal.pthread_sigmask(signal.SIG_SETMASK, ())
        try:
            signal.signal(number, signal.SIG_DFL)
        except (OSError, ValueError):
            pass  # SIGKILL has no handler to reset
        os.kill(os.getpid(), number)
        time.sleep(1)
    return 128 + number


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "--":
        print("usage: gpu-thermal-signal-relay.py -- <command> [args...]", file=sys.stderr)
        return 2
    command = argv[1:]
    watched = {*_RELAYED, signal.SIGCHLD}
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched)

    def restore_mask() -> None:
        # The child must not inherit the blocked signals.
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)

    try:
        child = subprocess.Popen(command, preexec_fn=restore_mask)  # noqa: PLW1509 - no threads here
    except OSError as exc:
        print(f"!! [thermal] cannot start {command[0]}: {exc}", file=sys.stderr)
        return 127
    stop_relayed = False
    while True:
        info = signal.sigtimedwait(watched, _WAIT_S)
        received = time.clock_gettime(time.CLOCK_BOOTTIME)
        if info is not None and info.si_signo != signal.SIGCHLD:
            number = signal.Signals(info.si_signo)
            try:
                text = describe_sender(info.si_pid, info.si_uid, info.si_code, received)
            except Exception as exc:  # a trace failure must never stop the relay
                text = f"from pid {info.si_pid} (the trace failed: {exc})"
            if number in _STOP_ONCE and stop_relayed:
                _log(f"stop signal {number.name} {text}; already relayed, not sending it again")
            else:
                _log(f"stop signal {number.name} {text}; relaying it to the guard (pid {child.pid})")
                try:
                    os.kill(child.pid, number)
                except ProcessLookupError:
                    pass
                if number in _STOP_ONCE:
                    stop_relayed = True
        returncode = child.poll()
        if returncode is not None:
            return _die_like(returncode)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
