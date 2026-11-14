"""The ExecReload of the router unit: ci/runner/inferference-router-reload.sh.

No GPU, no router, no real config. A stub python stands in for the library venv
and records how the script calls the reconcile CLI. The tests prove these things.

1. The script runs `inferference-reconcile.py --timeout N apply` with the python
   the launcher would use, and nothing else.
2. A non-zero reconcile status (overcommit refusal, failed residency proof, an
   unreachable router) is the status of the reload.
3. A missing python or a bad timeout fails before any call.
4. The CLI writes the preset file by temp file + rename: no partial file, no
   temp file left behind, and the mode stays readable.
5. The python process replaces the script (`exec`), so systemd signals the worker.
6. A lock stops a second reload. The lock is released when the first one ends.
7. The load deadline must fit the unit timeout, and the unit templates agree.
8. One deadline covers both waits of `apply`.
"""

from __future__ import annotations

import importlib.util
import os
import re
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELOAD = ROOT / "ci/runner/inferference-router-reload.sh"
CLI = ROOT / "ci/runner/inferference-reconcile.py"

_STUB_PYTHON = """#!/usr/bin/env bash
echo "$*" >> "$STUB_CALL_LOG"
exit "${STUB_RC:-0}"
"""


def _run(tmp_path: Path, rc: int = 0, **extra: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    stub = tmp_path / "stub-python"
    stub.write_text(_STUB_PYTHON)
    stub.chmod(0o755)
    log = tmp_path / "calls.log"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("INFERFERENCE_", "STUB_"))}
    env.update(
        {
            "INFERFERENCE_PYTHON": str(stub),
            "STUB_CALL_LOG": str(log),
            "STUB_RC": str(rc),
            "INFERFERENCE_RELOAD_LOCK": str(tmp_path / "reload.lock"),
        }
    )
    env.update(extra)
    result = subprocess.run(["bash", str(RELOAD)], env=env, capture_output=True, text=True, timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


def test_reload_runs_apply_once_with_the_load_timeout(tmp_path: Path) -> None:
    result, calls = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert calls == [f"{CLI} --timeout 600 apply"]
    assert "deadline 600s" in result.stdout


def test_reload_takes_the_load_timeout_from_the_environment(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, INFERFERENCE_RELOAD_LOAD_TIMEOUT="45")
    assert result.returncode == 0, result.stderr
    assert calls == [f"{CLI} --timeout 45 apply"]


@pytest.mark.parametrize("rc", [1, 78])
def test_reload_fails_with_the_status_of_the_reconcile(tmp_path: Path, rc: int) -> None:
    result, calls = _run(tmp_path, rc=rc)
    assert result.returncode == rc
    assert len(calls) == 1
    # The script `exec`s python, so the CLI prints the failure. The script says
    # before it runs that a failed reload leaves the router running.
    assert "the router keeps running" in result.stdout


def test_reload_fails_closed_without_python(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, INFERFERENCE_PYTHON=str(tmp_path / "absent"))
    assert result.returncode == 78
    assert calls == []
    assert "no python at" in result.stderr


def test_reload_refuses_a_bad_timeout(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, INFERFERENCE_RELOAD_LOAD_TIMEOUT="soon")
    assert result.returncode == 78
    assert calls == []
    assert "whole seconds" in result.stderr


# ------------------------------------------------------------------ exec ----
_PID_STUB = """#!/usr/bin/env bash
echo "$$" > "$STUB_PID_FILE"
echo "$*" >> "$STUB_CALL_LOG"
if [ -n "${STUB_HOLD:-}" ]; then
  : > "$STUB_STARTED"
  while [ ! -e "$STUB_HOLD" ]; do sleep 0.05; done
fi
exit 0
"""


def _start(tmp_path: Path, name: str, **extra: str) -> tuple[subprocess.Popen[str], Path]:
    stub = tmp_path / "stub-pid-python"
    stub.write_text(_PID_STUB)
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("INFERFERENCE_", "STUB_"))}
    env.update(
        {
            "INFERFERENCE_PYTHON": str(stub),
            "INFERFERENCE_RELOAD_LOCK": str(tmp_path / "reload.lock"),
            "STUB_CALL_LOG": str(tmp_path / f"{name}.calls"),
            "STUB_PID_FILE": str(tmp_path / f"{name}.pid"),
            "STUB_STARTED": str(tmp_path / f"{name}.started"),
        }
    )
    env.update(extra)
    proc = subprocess.Popen(["bash", str(RELOAD)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc, tmp_path / f"{name}.pid"


def _wait_for(path: Path, seconds: float = 20.0) -> None:
    end = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < end, f"{path} did not appear"
        time.sleep(0.02)


def test_reload_execs_python_so_systemd_signals_the_worker(tmp_path: Path) -> None:
    proc, pid_file = _start(tmp_path, "one")
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 0, err
    assert pid_file.read_text().strip() == str(proc.pid)  # the script process became python


# ------------------------------------------------------------------- lock ----
def test_a_second_reload_fails_while_the_first_holds_the_lock(tmp_path: Path) -> None:
    hold = tmp_path / "release"
    first, _ = _start(tmp_path, "first", STUB_HOLD=str(hold))
    try:
        _wait_for(tmp_path / "first.started")
        second, _ = _start(tmp_path, "second")
        out, err = second.communicate(timeout=30)
        assert second.returncode == 75
        assert "another reload holds" in err
        assert not (tmp_path / "second.calls").exists()  # it ran nothing
    finally:
        hold.write_text("")
        first.communicate(timeout=30)
    assert first.returncode == 0
    # The lock goes with the first process. A third reload runs.
    third, _ = _start(tmp_path, "third")
    third.communicate(timeout=30)
    assert third.returncode == 0
    assert (tmp_path / "third.calls").exists()


def test_the_lock_path_follows_runtime_directory_then_state_dir(tmp_path: Path) -> None:
    runtime = tmp_path / "rt"
    runtime.mkdir()
    proc, _ = _start(tmp_path, "rt", INFERFERENCE_RELOAD_LOCK="", RUNTIME_DIRECTORY=f"{runtime}:/other")
    proc.communicate(timeout=30)
    assert (runtime / "reload.lock").exists()
    state = tmp_path / "state"
    proc, _ = _start(tmp_path, "st", INFERFERENCE_RELOAD_LOCK="", XDG_RUNTIME_DIR="", XDG_STATE_HOME=str(state))
    proc.communicate(timeout=30)
    assert (state / "inferference" / "reload.lock").exists()


# --------------------------------------------------------------- deadline ----
def test_reload_refuses_a_deadline_that_does_not_fit_the_unit_timeout(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, INFERFERENCE_RELOAD_LOAD_TIMEOUT="700")  # 700 + 240 > 900
    assert result.returncode == 78
    assert calls == []
    assert "more than the unit timeout" in result.stderr
    assert "660 or less" in result.stderr


def test_reload_accepts_the_largest_deadline_that_fits(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, INFERFERENCE_RELOAD_LOAD_TIMEOUT="660")
    assert result.returncode == 0, result.stderr
    assert calls == [f"{CLI} --timeout 660 apply"]
    other = tmp_path / "other"
    other.mkdir()
    result, calls = _run(
        other,
        INFERFERENCE_RELOAD_LOAD_TIMEOUT="100",
        INFERFERENCE_RELOAD_UNIT_TIMEOUT="340",
    )
    assert result.returncode == 0, result.stderr


def _number(pattern: str, text: str, flags: int = 0) -> int:
    match = re.search(pattern, text, flags)
    assert match, pattern
    return int(match.group(1))


def test_the_default_deadline_and_the_units_agree() -> None:
    script = RELOAD.read_text()
    overhead = _number(r"^reload_overhead_s=(\d+)", script, re.M)
    cli = _load_cli()
    assert overhead == cli.APPLY_OVERHEAD_S
    # The worst case after the deadline: settle, cleanup, four router reads of 30 s.
    assert cli.APPLY_OVERHEAD_S >= cli.SETTLE_S + cli.CLEANUP_S + 4 * 30
    default = _number(r"INFERFERENCE_RELOAD_LOAD_TIMEOUT:-(\d+)", script)
    for unit in (ROOT / "ci/runner/inferference-router.service.in", ROOT / "nix/nixos-module.nix"):
        text = unit.read_text()
        assert "ExecReload" in text, unit.name
        timeout = _number(r"TimeoutStartSec\s*=\s*(\d+)", text)
        assert default + overhead <= timeout, unit.name


def test_one_deadline_covers_both_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli()
    clock = [1000.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(cli, "router_models", lambda _router: [{"id": "a", "status": {"value": "loading"}}])
    deadline = cli.Deadline(10.0)
    with pytest.raises(SystemExit):
        cli._wait_loaded("http://router", ["a"], deadline=deadline)
    spent = clock[0] - 1000.0
    assert 10.0 <= spent < 12.5
    # The deadline has gone. A second wait in the same apply gives up at once.
    before = clock[0]
    with pytest.raises(SystemExit):
        cli._wait_loaded("http://router", ["a"], deadline=deadline)
    assert clock[0] == before
    assert deadline.expired()
    assert deadline.cap(600.0) == 1.0  # a final router call still gets one second


def test_the_deadline_caps_router_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli()
    clock = [0.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    deadline = cli.Deadline(100.0)
    assert deadline.cap(600.0) == 100.0
    clock[0] = 70.0
    assert deadline.cap(600.0) == 30.0
    assert deadline.cap(10.0) == 10.0


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location("inferference_reconcile_cli_reload", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preset_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    cli = _load_cli()
    target = tmp_path / "run" / "models-preset.ini"
    cli._write_atomic(target, "[a]\nx = 1\n")
    assert target.read_text() == "[a]\nx = 1\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    inode = target.stat().st_ino
    cli._write_atomic(target, "[a]\nx = 2\n")
    assert target.read_text() == "[a]\nx = 2\n"
    assert target.stat().st_ino != inode  # a rename, not an in-place write
    assert [p.name for p in target.parent.iterdir()] == ["models-preset.ini"]


def test_failed_preset_write_keeps_the_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli()
    target = tmp_path / "models-preset.ini"
    target.write_text("old\n")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="rename failed"):
        cli._write_atomic(target, "new\n")
    assert target.read_text() == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["models-preset.ini"]
