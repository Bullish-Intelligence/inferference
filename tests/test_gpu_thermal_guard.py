from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "ci/runner/gpu-thermal-guard.sh"
UNIT_TEMPLATE = ROOT / "ci/runner/inferference-router.service.in"
NIX_MODULE = ROOT / "nix/nixos-module.nix"


def _environment(tmp_path: Path, temperature_mc: int = 65_000) -> tuple[dict[str, str], Path]:
    sysfs = tmp_path / "sys"
    hwmon = sysfs / "card1/device/hwmon/hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "name").write_text("amdgpu\n")
    for sensor, label in ((1, "edge"), (2, "junction"), (3, "mem")):
        (hwmon / f"temp{sensor}_label").write_text(f"{label}\n")
        (hwmon / f"temp{sensor}_input").write_text(f"{temperature_mc}\n")

    fan_status = tmp_path / "fan-status"
    fan_status.write_text("state=OK target_pwm=255 max_gpu_junction_mC=65000\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    real_systemctl = shutil.which("systemctl")
    assert real_systemctl is not None
    (fake_bin / "systemctl").write_text("#!/usr/bin/env bash\n[[ ${GPU_THERMAL_TEST_FAN_ACTIVE:-0} == 1 ]]\n")
    (fake_bin / "systemctl").chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "GPU_THERMAL_SYSFS_ROOT": str(sysfs),
            "GPU_THERMAL_FAN_STATUS_FILE": str(fan_status),
            "GPU_THERMAL_FAN_WATCHDOG_UNIT": "arctic-fan-watchdog.service",
            "GPU_THERMAL_TEST_FAN_ACTIVE": "1",
            "GPU_THERMAL_MAX_C": "80",
            "GPU_THERMAL_RESUME_C": "70",
            "GPU_THERMAL_POLL_S": "0.01",
            "GPU_THERMAL_TERM_GRACE_TENTHS": "2",
        }
    )
    return env, hwmon


def _pid_is_running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    state = stat.rsplit(")", 1)[-1].strip().split()[0]
    return state != "Z"


def test_thermal_trip_returns_distinct_status_and_kills_process_group(tmp_path: Path) -> None:
    env, hwmon = _environment(tmp_path)
    parent_pid_file = tmp_path / "parent.pid"
    child_pid_file = tmp_path / "child.pid"
    target = tmp_path / "worker.sh"
    target.write_text(
        "#!/usr/bin/env bash\n"
        "trap '' TERM\n"
        f"printf '%s\\n' \"$BASHPID\" > {parent_pid_file}\n"
        "while :; do\n"
        "  sleep 60 &\n"
        "  child=$!\n"
        f"  printf '%s\\n' \"$child\" > {child_pid_file}\n"
        '  wait "$child" || true\n'
        "done\n"
    )
    target.chmod(0o755)

    process = subprocess.Popen(
        [str(GUARD), "run", "--", "bash", str(target)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not child_pid_file.exists():
        time.sleep(0.01)
    assert parent_pid_file.exists()
    assert child_pid_file.exists()
    parent_pid = int(parent_pid_file.read_text().strip())
    child_pid = int(child_pid_file.read_text().strip())

    (hwmon / "temp2_input").write_text("81000\n")
    output, _ = process.communicate(timeout=8)

    assert process.returncode == 75, output
    assert "OVER LIMIT: 81 C > 80 C" in output
    assert "thermal stop exit status 75" in output
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and (_pid_is_running(parent_pid) or _pid_is_running(child_pid)):
        time.sleep(0.02)
    assert not _pid_is_running(parent_pid)
    assert not _pid_is_running(child_pid)


def test_watch_logs_one_timestamped_tick_per_poll(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    target = tmp_path / "worker.sh"
    target.write_text("#!/usr/bin/env bash\nexec sleep 2\n")
    target.chmod(0o755)

    result = subprocess.run(
        [str(GUARD), "run", "--", "bash", str(target)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    ticks = [line for line in result.stdout.splitlines() if "[thermal] tick " in line]
    assert len(ticks) >= 2, result.stdout
    for line in ticks:
        timestamp = line.split("tick ", 1)[1].split(" ", 1)[0]
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", timestamp), line
        assert "edge=" in line
        assert "(limit 80 C)" in line


def test_failed_fan_check_blocks_start(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    (tmp_path / "fan-status").write_text("state=DEGRADED target_pwm=255\n")
    marker = tmp_path / "started"

    result = subprocess.run(
        [str(GUARD), "run", "--", "touch", str(marker)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 76
    assert "fan watchdog state is DEGRADED" in result.stderr
    assert not marker.exists()


def test_missing_fan_status_blocks_start(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    (tmp_path / "fan-status").unlink()
    marker = tmp_path / "started"

    result = subprocess.run(
        [str(GUARD), "run", "--", "touch", str(marker)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 76
    assert "fan watchdog status is missing" in result.stderr
    assert not marker.exists()


def test_inactive_fan_watchdog_blocks_start(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    env["GPU_THERMAL_TEST_FAN_ACTIVE"] = "0"
    marker = tmp_path / "started"

    result = subprocess.run(
        [str(GUARD), "run", "--", "touch", str(marker)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 76
    assert "fan watchdog unit arctic-fan-watchdog.service is not active" in result.stderr
    assert not marker.exists()


def test_fan_failure_during_run_returns_safety_status_and_cleans_process_group(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    parent_pid_file = tmp_path / "parent.pid"
    target = tmp_path / "worker.sh"
    target.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$BASHPID\" > {parent_pid_file}\nexec sleep 60\n")
    target.chmod(0o755)

    process = subprocess.Popen(
        [str(GUARD), "run", "--", "bash", str(target)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not parent_pid_file.exists():
        time.sleep(0.01)
    assert parent_pid_file.exists()
    parent_pid = int(parent_pid_file.read_text().strip())

    (tmp_path / "fan-status").write_text("state=DEGRADED target_pwm=255\n")
    output, _ = process.communicate(timeout=8)

    assert process.returncode == 76, output
    assert "fan watchdog state is DEGRADED" in output
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and _pid_is_running(parent_pid):
        time.sleep(0.02)
    assert not _pid_is_running(parent_pid)


def test_child_cannot_claim_reserved_guard_status(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path)
    for status in (75, 76):
        result = subprocess.run(
            [str(GUARD), "run", "--", "bash", "-c", f"exit {status}"],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1


def test_missing_sensor_blocks_start(tmp_path: Path) -> None:
    env, hwmon = _environment(tmp_path)
    for sensor_file in hwmon.iterdir():
        sensor_file.unlink()
    marker = tmp_path / "started"

    result = subprocess.run(
        [str(GUARD), "run", "--", "touch", str(marker)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 76
    assert "no amdgpu temperature sensor found" in result.stderr
    assert not marker.exists()


def test_hot_card_must_cool_below_resume_mark_before_start(tmp_path: Path) -> None:
    env, _ = _environment(tmp_path, temperature_mc=71_000)
    env["GPU_THERMAL_COOL_TIMEOUT_S"] = "0"
    marker = tmp_path / "started"

    result = subprocess.run(
        [str(GUARD), "run", "--", "touch", str(marker)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 76
    assert "cards did not cool below 70 C" in result.stderr
    assert not marker.exists()


def test_router_unit_prevents_restart_only_for_guard_statuses() -> None:
    template = UNIT_TEMPLATE.read_text()
    module = NIX_MODULE.read_text()

    assert "Restart=on-failure" in template
    assert "RestartPreventExitStatus=75 76" in template
    assert 'Restart = "on-failure";' in module
    assert 'RestartPreventExitStatus = "75 76";' in module


def _start_guarded_sleeper(tmp_path: Path, env: dict[str, str]) -> tuple[subprocess.Popen[str], int]:
    """Run a long-lived child under the guard and return it with its own pid."""
    parent_pid_file = tmp_path / "parent.pid"
    target = tmp_path / "worker.sh"
    target.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$BASHPID\" > {parent_pid_file}\nexec sleep 60\n")
    target.chmod(0o755)
    process = subprocess.Popen(
        [str(GUARD), "run", "--", "bash", str(target)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not parent_pid_file.exists():
        time.sleep(0.01)
    assert parent_pid_file.exists()
    return process, int(parent_pid_file.read_text().strip())


def test_transient_missing_fan_status_does_not_stop_a_running_job(tmp_path: Path) -> None:
    """A watchdog restart deletes its RuntimeDirectory for a few seconds.

    That gap is indeterminate, not a cooling failure, so a running job must
    survive it. Measurement showed the opposite behavior killing production.
    """
    env, _ = _environment(tmp_path)
    env["GPU_THERMAL_FAN_GRACE_S"] = "10"
    fan_status = tmp_path / "fan-status"
    process, parent_pid = _start_guarded_sleeper(tmp_path, env)

    fan_status.unlink()
    time.sleep(0.3)
    fan_status.write_text("state=OK target_pwm=255 max_gpu_junction_mC=65000\n")
    time.sleep(0.3)

    assert process.poll() is None
    assert _pid_is_running(parent_pid)

    # The guard runs its child in its own process group, so terminating the
    # wrapper alone leaves the worker alive. Tear the group down instead.
    os.killpg(parent_pid, signal.SIGKILL)
    output, _ = process.communicate(timeout=8)
    assert "fan watchdog is indeterminate" in output
    assert "fan watchdog is readable again" in output


def test_fan_status_missing_past_the_grace_window_stops_the_job(tmp_path: Path) -> None:
    """The grace is bounded. A fan subsystem that never returns fails closed."""
    env, _ = _environment(tmp_path)
    env["GPU_THERMAL_FAN_GRACE_S"] = "1"
    fan_status = tmp_path / "fan-status"
    process, parent_pid = _start_guarded_sleeper(tmp_path, env)

    fan_status.unlink()
    output, _ = process.communicate(timeout=15)

    assert process.returncode == 76, output
    assert "stayed indeterminate" in output
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and _pid_is_running(parent_pid):
        time.sleep(0.02)
    assert not _pid_is_running(parent_pid)


def test_degraded_fan_state_is_never_graced(tmp_path: Path) -> None:
    """A live watchdog reporting bad cooling is definite, so it stops at once."""
    env, _ = _environment(tmp_path)
    env["GPU_THERMAL_FAN_GRACE_S"] = "600"
    process, parent_pid = _start_guarded_sleeper(tmp_path, env)

    (tmp_path / "fan-status").write_text("state=NO_AIRFLOW target_pwm=255\n")
    output, _ = process.communicate(timeout=8)

    assert process.returncode == 76, output
    assert "fan watchdog state is NO_AIRFLOW" in output
    assert "stayed indeterminate" not in output
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and _pid_is_running(parent_pid):
        time.sleep(0.02)
    assert not _pid_is_running(parent_pid)


def test_indeterminate_fan_state_still_blocks_a_start(tmp_path: Path) -> None:
    """Admission control stays strict: a job must not start on unknown cooling."""
    env, _ = _environment(tmp_path)
    env["GPU_THERMAL_FAN_GRACE_S"] = "600"
    (tmp_path / "fan-status").unlink()

    result = subprocess.run(
        [str(GUARD), "run", "--", "true"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 76, result.stderr
    assert "refusing to start" in result.stderr


def test_unit_pairs_killmode_mixed_with_the_guard_relay() -> None:
    """The three halves of the fix only work together.

    With KillMode=mixed and no relay, systemd signals the guard alone and
    SIGKILLs the router about a second later, with no graceful unload. With
    the default control-group and no relay, every process in the cgroup gets
    KillSignal, the trapless guard dies, systemd re-sends KillSignal in
    final-sigterm, and llama-server calls exit() from its own signal handler.
    Measured: 4 hangs in 38 stops.
    """
    template = UNIT_TEMPLATE.read_text()
    module = NIX_MODULE.read_text()
    guard = GUARD.read_text()

    assert "KillMode=mixed" in template
    assert 'KillMode = "mixed";' in module
    assert "_forward_stop()" in guard
    assert "trap _forward_stop INT TERM" in guard


def test_guard_relays_one_stop_signal_and_waits_for_the_child(tmp_path: Path) -> None:
    """KillMode=mixed signals only the guard, so the guard must relay — once.

    The worker installs its own SIGINT handler the way llama-server does
    (sigaction overrides the SIG_IGN bash gives an async job), counts every
    delivery, and takes a measurable teardown. Exactly one delivery must
    arrive, the guard must outlive the teardown, and the worker's own exit
    status must reach the caller — RestartPreventExitStatus=75 76 depends on
    that status being the child's.
    """
    env, _ = _environment(tmp_path)
    log = tmp_path / "worker.log"
    ready = tmp_path / "worker.ready"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, signal, time\n"
        "seen = 0\n"
        "def handler(sig, frame):\n"
        "    global seen\n"
        "    seen += 1\n"
        f"    open({str(log)!r}, 'a').write(f'signal {{sig}} #{{seen}}\\n')\n"
        "    if seen > 1:\n"
        "        os._exit(1)\n"
        "    time.sleep(0.5)\n"
        f"    open({str(log)!r}, 'a').write('teardown done\\n')\n"
        "    os._exit(7)\n"
        "signal.signal(signal.SIGINT, handler)\n"
        "signal.signal(signal.SIGTERM, handler)\n"
        f"open({str(ready)!r}, 'w').write('1')\n"
        "while True:\n"
        "    time.sleep(0.02)\n"
    )

    process = subprocess.Popen(
        [str(GUARD), "run", "--", sys.executable, "-I", str(worker)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.01)
    assert ready.exists(), "worker never started under the guard"

    # Signal the guard ALONE, which is what KillMode=mixed does.
    process.send_signal(signal.SIGINT)
    output = process.communicate(timeout=20)[0]

    lines = log.read_text().splitlines()
    assert [line for line in lines if line.startswith("signal")] == ["signal 2 #1"], lines
    assert "teardown done" in lines, "the guard exited before the child finished its teardown"
    assert process.returncode == 7, output
    assert "relaying stop to process group" in output


def test_guard_relays_a_repeated_stop_signal_only_once(tmp_path: Path) -> None:
    """A second signal to the guard must not become a second one to the child.

    That is the whole defect: llama-server treats a second interrupt as
    "terminate immediately" and calls exit() from the handler.
    """
    env, _ = _environment(tmp_path)
    log = tmp_path / "worker.log"
    ready = tmp_path / "worker.ready"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, signal, time\n"
        "seen = 0\n"
        "def handler(sig, frame):\n"
        "    global seen\n"
        "    seen += 1\n"
        f"    open({str(log)!r}, 'a').write(f'signal {{sig}} #{{seen}}\\n')\n"
        "    if seen == 1:\n"
        "        time.sleep(1.0)\n"
        "        os._exit(7)\n"
        "signal.signal(signal.SIGINT, handler)\n"
        "signal.signal(signal.SIGTERM, handler)\n"
        f"open({str(ready)!r}, 'w').write('1')\n"
        "while True:\n"
        "    time.sleep(0.02)\n"
    )

    process = subprocess.Popen(
        [str(GUARD), "run", "--", sys.executable, "-I", str(worker)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.01)
    assert ready.exists()

    process.send_signal(signal.SIGINT)
    time.sleep(0.2)
    process.send_signal(signal.SIGINT)
    process.send_signal(signal.SIGTERM)
    output = process.communicate(timeout=20)[0]

    lines = [line for line in log.read_text().splitlines() if line.startswith("signal")]
    assert lines == ["signal 2 #1"], lines
    assert output.count("relaying stop to process group") == 1, output


# --- pause mode (serving router) ---------------------------------------------

LAUNCHER = ROOT / "ci/runner/inferference-router-launch.sh"


def _pause_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    env, hwmon = _environment(tmp_path)
    env.update(
        {
            "GPU_THERMAL_MAX_C": "85",
            "GPU_THERMAL_RESUME_C": "70",
            "GPU_THERMAL_HARD_C": "90",
            "GPU_THERMAL_RESUME_HOLD_S": "2",
            "GPU_THERMAL_POLL_S": "0.05",
            "GPU_THERMAL_FAN_GRACE_S": "0",
            "GPU_THERMAL_STATUS_FILE": str(tmp_path / "guard.status"),
        }
    )
    return env, hwmon


def _set_temp(hwmon: Path, celsius: int) -> None:
    for sensor in (1, 2, 3):
        (hwmon / f"temp{sensor}_input").write_text(f"{celsius * 1000}\n")


def _proc_state(pid: int) -> str:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return "gone"
    return stat.rsplit(")", 1)[-1].strip().split()[0]


def _wait_for(predicate, timeout: float = 6.0) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _start_pause_worker(
    tmp_path: Path, env: dict[str, str], body: str = "exec sleep 60"
) -> tuple[subprocess.Popen[str], int, Path]:
    """Run a worker under `run --on-limit pause`. Return the guard, the worker pid, and the log."""
    pid_file = tmp_path / "worker.pid"
    target = tmp_path / "worker.sh"
    target.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$BASHPID\" > {pid_file}\n{body}\n")
    target.chmod(0o755)
    log = tmp_path / "guard.log"
    process = subprocess.Popen(
        [str(GUARD), "run", "--on-limit", "pause", "--", "bash", str(target)],
        cwd=ROOT,
        env=env,
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert _wait_for(pid_file.exists), log.read_text()
    return process, int(pid_file.read_text().strip()), log


def _status(tmp_path: Path) -> str:
    path = tmp_path / "guard.status"
    return path.read_text() if path.exists() else ""


def _finish(process: subprocess.Popen[str], pid: int) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_pause_mode_freezes_at_the_limit_and_resumes_after_the_hold(tmp_path: Path) -> None:
    env, hwmon = _pause_environment(tmp_path)
    process, pid, log = _start_pause_worker(tmp_path, env)
    try:
        assert _wait_for(lambda: "state=running" in _status(tmp_path)), log.read_text()
        assert _proc_state(pid) != "T"

        _set_temp(hwmon, 85)  # at the limit, not over it
        assert _wait_for(lambda: _proc_state(pid) == "T"), log.read_text()
        assert _wait_for(lambda: "state=paused" in _status(tmp_path))
        status = _status(tmp_path)
        assert "hottest_c=85" in status
        assert "since=" in status
        assert "reason=" in status
        assert process.poll() is None, "a pause must not exit the guard"
        assert "PAUSED process group" in log.read_text()

        _set_temp(hwmon, 70)  # at the resume mark, so the hold starts
        time.sleep(0.6)
        assert _proc_state(pid) == "T", "must not resume before the hold passes"
        assert _wait_for(lambda: _proc_state(pid) != "T"), log.read_text()
        assert _wait_for(lambda: "state=running" in _status(tmp_path))
        assert "RESUMED process group" in log.read_text()
        assert process.poll() is None
    finally:
        _finish(process, pid)


def test_pause_mode_hold_restarts_when_the_card_warms_again(tmp_path: Path) -> None:
    env, hwmon = _pause_environment(tmp_path)
    process, pid, log = _start_pause_worker(tmp_path, env)
    try:
        _set_temp(hwmon, 86)
        assert _wait_for(lambda: _proc_state(pid) == "T"), log.read_text()
        _set_temp(hwmon, 60)
        time.sleep(1.0)
        _set_temp(hwmon, 75)  # above the resume mark: the hold clock resets
        time.sleep(1.5)
        assert _proc_state(pid) == "T"
        _set_temp(hwmon, 60)
        assert _wait_for(lambda: _proc_state(pid) != "T", timeout=8), log.read_text()
    finally:
        _finish(process, pid)


def test_pause_mode_does_not_resume_while_the_fan_is_not_ok(tmp_path: Path) -> None:
    env, hwmon = _pause_environment(tmp_path)
    process, pid, log = _start_pause_worker(tmp_path, env)
    fan_status = tmp_path / "fan-status"
    try:
        _set_temp(hwmon, 86)
        assert _wait_for(lambda: _proc_state(pid) == "T"), log.read_text()
        _set_temp(hwmon, 40)
        fan_status.write_text("state=DEGRADED target_pwm=255\n")
        time.sleep(3.0)  # well past the hold
        assert _proc_state(pid) == "T"
        assert process.poll() is None, "a bad fan state pauses, it does not kill"
        assert "fan watchdog not OK" in _status(tmp_path)

        fan_status.write_text("state=OK target_pwm=255\n")
        assert _wait_for(lambda: _proc_state(pid) != "T", timeout=8), log.read_text()
    finally:
        _finish(process, pid)


def test_pause_mode_pauses_a_cool_job_when_the_fan_status_goes_missing(tmp_path: Path) -> None:
    env, _ = _pause_environment(tmp_path)
    process, pid, log = _start_pause_worker(tmp_path, env)
    fan_status = tmp_path / "fan-status"
    try:
        assert _wait_for(lambda: "state=running" in _status(tmp_path)), log.read_text()
        fan_status.unlink()
        assert _wait_for(lambda: _proc_state(pid) == "T"), log.read_text()
        assert process.poll() is None
        fan_status.write_text("state=OK target_pwm=255\n")
        assert _wait_for(lambda: _proc_state(pid) != "T", timeout=8), log.read_text()
    finally:
        _finish(process, pid)


def test_pause_mode_kills_at_the_hard_limit_while_paused(tmp_path: Path) -> None:
    env, hwmon = _pause_environment(tmp_path)
    process, pid, log = _start_pause_worker(tmp_path, env)
    try:
        _set_temp(hwmon, 86)
        assert _wait_for(lambda: _proc_state(pid) == "T"), log.read_text()
        _set_temp(hwmon, 90)  # still rising with the GPUs idle: cooling failed
        assert process.wait(timeout=8) == 75, log.read_text()
        output = log.read_text()
        assert "HARD LIMIT: 90 C >= 90 C while paused" in output
        assert "cooling has failed" in output
        assert "thermal stop exit status 75" in output
        assert _wait_for(lambda: _proc_state(pid) in ("gone", "Z"))
        assert "state=killed" in _status(tmp_path)
    finally:
        _finish(process, pid)


def test_pause_mode_thaws_the_group_when_the_guard_is_told_to_stop(tmp_path: Path) -> None:
    env, hwmon = _pause_environment(tmp_path)
    marker = tmp_path / "got-int"
    pid_file = tmp_path / "worker.pid"
    worker = tmp_path / "worker.py"
    # The worker handles INT the way llama-server does. A stopped worker holds INT pending.
    worker.write_text(
        "import os, signal, sys, time\n"
        f"def handler(sig, frame):\n    open({str(marker)!r}, 'w').write('1')\n    os._exit(0)\n"
        "signal.signal(signal.SIGINT, handler)\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "while True:\n    time.sleep(0.02)\n"
    )
    log = tmp_path / "guard.log"
    process = subprocess.Popen(
        [str(GUARD), "run", "--on-limit", "pause", "--", sys.executable, "-I", str(worker)],
        cwd=ROOT,
        env=env,
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert _wait_for(pid_file.exists), log.read_text()
    pid = int(pid_file.read_text())
    try:
        _set_temp(hwmon, 88)
        assert _wait_for(lambda: _proc_state(pid) == "T"), log.read_text()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=8)
        assert marker.exists(), log.read_text()
        assert _wait_for(lambda: _proc_state(pid) in ("gone", "Z"))
    finally:
        _finish(process, pid)


def test_pause_mode_thaws_the_group_when_the_guard_exits_on_hangup(tmp_path: Path) -> None:
    env, hwmon = _pause_environment(tmp_path)
    process, pid, log = _start_pause_worker(tmp_path, env)
    try:
        _set_temp(hwmon, 88)
        assert _wait_for(lambda: _proc_state(pid) == "T"), log.read_text()
        process.send_signal(signal.SIGHUP)
        process.wait(timeout=8)
        assert _wait_for(lambda: _proc_state(pid) not in ("T",)), "the worker must not stay frozen"
        time.sleep(0.5)  # an orphaned watcher would freeze it again
        assert _proc_state(pid) != "T"
    finally:
        _finish(process, pid)


def test_pause_mode_waits_for_a_safe_start_with_no_time_limit(tmp_path: Path) -> None:
    env, hwmon = _pause_environment(tmp_path)
    _set_temp(hwmon, 80)  # over the resume mark, under the limit
    pid_file = tmp_path / "worker.pid"
    target = tmp_path / "worker.sh"
    target.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$BASHPID\" > {pid_file}\nexec sleep 60\n")
    target.chmod(0o755)
    log = tmp_path / "guard.log"
    process = subprocess.Popen(
        [str(GUARD), "run", "--on-limit", "pause", "--", "bash", str(target)],
        cwd=ROOT,
        env=env,
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert _wait_for(lambda: "waiting to start" in log.read_text()), log.read_text()
        assert "state=waiting" in _status(tmp_path)
        assert not pid_file.exists()
        _set_temp(hwmon, 60)
        assert _wait_for(pid_file.exists), log.read_text()
    finally:
        _finish(process, int(pid_file.read_text()) if pid_file.exists() else 0)


def test_pause_mode_refuses_a_hard_limit_at_or_below_the_pause_limit(tmp_path: Path) -> None:
    env, _ = _pause_environment(tmp_path)
    env["GPU_THERMAL_HARD_C"] = "85"
    result = subprocess.run(
        [str(GUARD), "run", "--on-limit", "pause", "--", "true"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert result.returncode == 2
    assert "GPU_THERMAL_HARD_C" in result.stderr


def test_default_mode_still_kills_at_the_limit(tmp_path: Path) -> None:
    env, hwmon = _pause_environment(tmp_path)
    env["GPU_THERMAL_ON_LIMIT"] = ""  # unset the pause default, as an experiment run does
    del env["GPU_THERMAL_ON_LIMIT"]
    env["GPU_THERMAL_HARD_C"] = "90"
    pid_file = tmp_path / "worker.pid"
    target = tmp_path / "worker.sh"
    target.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$BASHPID\" > {pid_file}\nexec sleep 60\n")
    target.chmod(0o755)
    process = subprocess.Popen(
        [str(GUARD), "run", "--", "bash", str(target)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert _wait_for(pid_file.exists)
    pid = int(pid_file.read_text())
    _set_temp(hwmon, 86)  # over the 85 C limit
    output, _ = process.communicate(timeout=8)
    assert process.returncode == 75, output
    assert "OVER LIMIT: 86 C > 85 C" in output
    assert "PAUSED" not in output
    assert _wait_for(lambda: _proc_state(pid) in ("gone", "Z"))
    assert not (tmp_path / "guard.status").exists(), "kill mode writes no pause status"


def test_router_launcher_runs_the_guard_in_pause_mode_at_85_c() -> None:
    text = LAUNCHER.read_text()
    assert 'GPU_THERMAL_MAX_C="${GPU_THERMAL_MAX_C:-85}"' in text
    assert 'GPU_THERMAL_RESUME_C="${GPU_THERMAL_RESUME_C:-70}"' in text
    assert 'GPU_THERMAL_HARD_C="${GPU_THERMAL_HARD_C:-90}"' in text
    assert 'gpu-thermal-guard.sh" run --on-limit pause --' in text
    # Experiments and builds keep the hard stop: the guard default stays kill at 80 C.
    guard = GUARD.read_text()
    assert 'ON_LIMIT="${GPU_THERMAL_ON_LIMIT:-kill}"' in guard
    assert 'MAX_C="${GPU_THERMAL_MAX_C:-80}"' in guard
