"""Test that the thermal guard names the sender of a stop signal.

Every test starts its own helper, guard and sender processes, and signals only
those. Nothing signals the pytest process or any process that the test did not
start.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from .test_gpu_thermal_guard import GUARD, ROOT, _environment

RELAY = ROOT / "ci/runner/gpu-thermal-signal-relay.py"

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the guard and its relay need Linux: /proc, process groups, signals"
)

# A child that counts every signal it gets and writes the count to a file.
_COUNTING_CHILD = """
import os, signal, sys, time
log = sys.argv[1]
ready = sys.argv[2]
seen = []
def handler(sig, frame):
    seen.append(sig)
    with open(log, "a") as handle:
        handle.write(f"signal {sig}\\n")
    if sig in (signal.SIGINT, signal.SIGTERM):
        time.sleep(0.3)
        os._exit(7)
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR2):
    signal.signal(sig, handler)
open(ready, "w").write("1")
while True:
    time.sleep(0.02)
"""


def _wait_for(path: Path, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not path.exists():
        time.sleep(0.01)
    assert path.exists(), f"{path} never appeared"


def _start_relay(tmp_path: Path, *command: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-I", str(RELAY), "--", *command],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _signal_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if "stop signal" in line]


def test_the_relay_names_a_live_sender_by_pid_and_command(tmp_path: Path) -> None:
    log, ready = tmp_path / "child.log", tmp_path / "child.ready"
    relay = _start_relay(tmp_path, sys.executable, "-I", "-c", _COUNTING_CHILD, str(log), str(ready))
    _wait_for(ready)
    # The sender stays alive after it signals, so /proc still names it.
    sender = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            "import os, signal, sys, time; os.kill(int(sys.argv[1]), signal.SIGTERM); time.sleep(3)",
            str(relay.pid),
            "marker-sender-live",
        ]
    )
    try:
        output = relay.communicate(timeout=20)[0]
    finally:
        sender.kill()
        sender.wait()

    lines = _signal_lines(output)
    assert len(lines) == 1, output
    assert f"stop signal SIGTERM from pid {sender.pid} uid {os.getuid()}" in lines[0], output
    assert "marker-sender-live" in lines[0], output
    assert "relaying it to the guard" in lines[0]
    assert log.read_text().splitlines() == [f"signal {int(signal.SIGTERM)}"]
    assert relay.returncode == 7


def test_the_relay_never_names_the_wrong_process_for_a_sender_that_exits_at_once(tmp_path: Path) -> None:
    """A sender that exits before the helper reads /proc must not get another process's name."""
    log, ready = tmp_path / "child.log", tmp_path / "child.ready"
    relay = _start_relay(tmp_path, sys.executable, "-I", "-c", _COUNTING_CHILD, str(log), str(ready))
    _wait_for(ready)
    sender_pids: list[int] = []
    for index in range(25):
        sender = subprocess.Popen(["sh", "-c", f"kill -USR2 {relay.pid} # marker-short-{index}"])
        sender_pids.append(sender.pid)
        sender.wait()
    time.sleep(0.3)
    relay.send_signal(signal.SIGTERM)
    output = relay.communicate(timeout=20)[0]

    usr2_lines = [line for line in output.splitlines() if "SIGUSR2" in line]
    assert len(usr2_lines) >= 1, output
    for line in usr2_lines:
        match = re.search(r"from pid (\d+)", line)
        assert match is not None, line
        assert int(match.group(1)) in sender_pids, line
        if 'cmdline "' in line:
            # When the helper names a command, it is the sender's own.
            assert "marker-short-" in line, line
        else:
            assert "exited" in line or "newer process" in line or 'comm "' in line, line


def test_the_relay_passes_the_child_status_through(tmp_path: Path) -> None:
    ok = _start_relay(tmp_path, sys.executable, "-I", "-c", "raise SystemExit(7)")
    ok.communicate(timeout=20)
    assert ok.returncode == 7

    killed = _start_relay(
        tmp_path, sys.executable, "-I", "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
    )
    killed.communicate(timeout=20)
    assert killed.returncode == -int(signal.SIGKILL)


def test_the_relay_dies_from_the_signal_that_killed_its_child(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    relay = _start_relay(
        tmp_path,
        sys.executable,
        "-I",
        "-c",
        f"import time; open({str(ready)!r}, 'w').write('1'); time.sleep(60)",
    )
    _wait_for(ready)
    relay.send_signal(signal.SIGTERM)
    relay.communicate(timeout=20)

    assert relay.returncode == -int(signal.SIGTERM)


def test_the_relay_refuses_a_call_without_a_command() -> None:
    result = subprocess.run([sys.executable, "-I", str(RELAY)], capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "usage" in result.stderr


# ---- the guard, with the relay in front of it ---------------------------------


def _guard_worker(tmp_path: Path) -> tuple[Path, Path, Path]:
    log, ready = tmp_path / "worker.log", tmp_path / "worker.ready"
    worker = tmp_path / "worker.py"
    worker.write_text(_COUNTING_CHILD)
    return worker, log, ready


def _start_guard(
    tmp_path: Path, env: dict[str, str], worker: Path, log: Path, ready: Path, start_new_session: bool = False
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [str(GUARD), "run", "--", sys.executable, "-I", str(worker), str(log), str(ready)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=start_new_session,
    )


def test_the_guard_log_names_the_sender_of_a_stop_signal(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    worker, log, ready = _guard_worker(tmp_path)
    guard = _start_guard(tmp_path, env, worker, log, ready)
    _wait_for(ready)
    sender = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            "import os, signal, sys, time; os.kill(int(sys.argv[1]), signal.SIGINT); time.sleep(3)",
            str(guard.pid),
            "marker-guard-sender",
        ]
    )
    try:
        output = guard.communicate(timeout=30)[0]
    finally:
        sender.kill()
        sender.wait()

    lines = _signal_lines(output)
    assert len(lines) == 1, output
    assert f"stop signal SIGINT from pid {sender.pid} uid {os.getuid()}" in lines[0], output
    assert "marker-guard-sender" in lines[0], output
    # The guard still relays to the guarded group once, and the worker's own status comes back.
    assert output.count("relaying stop to process group") == 1, output
    assert log.read_text().splitlines() == [f"signal {int(signal.SIGINT)}"]
    assert guard.returncode == 7, output


def test_a_repeated_stop_signal_reaches_the_worker_once_through_the_relay(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    worker, log, ready = _guard_worker(tmp_path)
    guard = _start_guard(tmp_path, env, worker, log, ready)
    _wait_for(ready)

    guard.send_signal(signal.SIGINT)
    time.sleep(0.05)
    guard.send_signal(signal.SIGINT)
    guard.send_signal(signal.SIGTERM)
    output = guard.communicate(timeout=30)[0]

    assert log.read_text().splitlines() == [f"signal {int(signal.SIGINT)}"], output
    assert output.count("relaying stop to process group") == 1, output
    assert len(_signal_lines(output)) == 3, output


def test_a_signal_to_the_whole_group_reaches_the_worker_once(tmp_path: Path) -> None:
    """A terminal interrupt signals the relay and the guard together. The worker still gets one."""
    env, _ = _environment(tmp_path)
    worker, log, ready = _guard_worker(tmp_path)
    guard = _start_guard(tmp_path, env, worker, log, ready, start_new_session=True)
    _wait_for(ready)

    # The group holds only processes that this test started.
    os.killpg(guard.pid, signal.SIGINT)
    output = guard.communicate(timeout=30)[0]

    assert log.read_text().splitlines() == [f"signal {int(signal.SIGINT)}"], output
    assert output.count("relaying stop to process group") == 1, output
    assert guard.returncode == 7, output


def test_the_trace_can_be_turned_off(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    env["GPU_THERMAL_SENDER_TRACE"] = "0"
    worker, log, ready = _guard_worker(tmp_path)
    guard = _start_guard(tmp_path, env, worker, log, ready)
    _wait_for(ready)

    guard.send_signal(signal.SIGINT)
    output = guard.communicate(timeout=30)[0]

    assert "stop signal" not in output
    assert output.count("relaying stop to process group") == 1, output
    assert guard.returncode == 7, output


def test_the_guard_still_starts_when_the_trace_python_is_broken(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    env["GPU_THERMAL_TRACE_PYTHON"] = str(tmp_path / "no-such-python")
    worker, log, ready = _guard_worker(tmp_path)
    guard = _start_guard(tmp_path, env, worker, log, ready)
    _wait_for(ready)

    guard.send_signal(signal.SIGINT)
    output = guard.communicate(timeout=30)[0]

    assert "sender trace is off: no working python3" in output
    assert output.count("relaying stop to process group") == 1, output
    assert guard.returncode == 7, output
