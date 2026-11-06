"""Tests for the fan lease and power-profile readback, against fabricated files.

No hardware and no background thread. A "fake watchdog" is one step function,
`_watchdog_step`, that the test calls explicitly: it reads the request file
(if any), applies `arbitrate` -- the module's own reference rule -- and writes
the status file atomically, exactly as the real nix-meta watchdog must.
`FanLease`'s injected `sleep` callback drives that step function, so a poll
loop in `acquire()`/`release()` advances the fake watchdog without any real
waiting or threading.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from inferference.experiment.control import (
    ControlError,
    FanLease,
    LeaseRequest,
    SharedFanLease,
    arbitrate,
    normalize_power_profile,
    parse_request,
    parse_status,
    read_power_state,
    verify_power_profile,
)


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    os.replace(tmp_name, path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _watchdog_step(
    request_file: Path,
    status_file: Path,
    *,
    allowlist: frozenset[str],
    default_profile: str,
    fault: bool = False,
) -> None:
    """One pass of a fake watchdog: read the request, arbitrate, write status."""
    request = parse_request(request_file.read_text()) if request_file.exists() else None
    owner_alive = request is not None and _pid_alive(request.owner_pid)
    active, _reason = arbitrate(
        request,
        now=datetime.now(UTC),
        owner_alive=owner_alive,
        allowlist=allowlist,
        fault=fault,
        default_profile=default_profile,
    )
    _atomic_write_text(status_file, f"state=OK target_pwm=255 max_gpu_junction_mC=36000 profile={active}\n")


def _make_power_card(drm_root: Path, card: int, *, active_profile: str, performance_level: str = "manual") -> None:
    device = drm_root / f"card{card}" / "device"
    device.mkdir(parents=True)
    (device / "power_dpm_force_performance_level").write_text(f"{performance_level}\n")
    (device / "pp_power_profile_mode").write_text(
        f"PROFILE_INDEX(NAME) CLOCK_TYPE(NAME)\n 0 BOOTUP_DEFAULT :\n 2   {active_profile}*:\n"
    )


# ---- LeaseRequest render / parse ---------------------------------------------


def test_lease_request_render_and_parse_round_trip() -> None:
    request = LeaseRequest(profile="flat-max", expires_utc=datetime(2026, 10, 7, 12, 0, 0, tzinfo=UTC), owner_pid=4242)

    text = request.render()

    assert text == "profile=flat-max\nexpires=2026-10-07T12:00:00Z\nowner_pid=4242\n"
    assert parse_request(text) == request


def test_parse_request_refuses_a_missing_key() -> None:
    with pytest.raises(ControlError, match="missing"):
        parse_request("profile=flat-max\nexpires=2026-10-07T12:00:00Z\n")


def test_parse_status_reads_space_separated_tokens_on_one_line() -> None:
    tokens = parse_status("state=OK target_pwm=255 max_gpu_junction_mC=36000 profile=flat-max")

    assert tokens == {"state": "OK", "target_pwm": "255", "max_gpu_junction_mC": "36000", "profile": "flat-max"}


# ---- arbitrate: the pure reference rule --------------------------------------


def test_arbitrate_with_no_request_returns_default() -> None:
    active, reason = arbitrate(
        None,
        now=datetime.now(UTC),
        owner_alive=True,
        allowlist=frozenset({"flat-max"}),
        fault=False,
        default_profile="curve-default",
    )

    assert (active, reason) == ("curve-default", "no request")


def test_arbitrate_fault_beats_a_live_lease() -> None:
    request = LeaseRequest(
        profile="flat-max", expires_utc=datetime.now(UTC) + timedelta(seconds=60), owner_pid=os.getpid()
    )

    active, reason = arbitrate(
        request,
        now=datetime.now(UTC),
        owner_alive=True,
        allowlist=frozenset({"flat-max"}),
        fault=True,
        default_profile="curve-default",
    )

    assert (active, reason) == ("full-duty-fault", "fault override")


# ---- the fake watchdog step, exercised through the lease files --------------


def test_watchdog_step_expired_lease_falls_back_to_default(tmp_path: Path) -> None:
    request_file = tmp_path / "request"
    status_file = tmp_path / "status"
    expired = LeaseRequest(
        profile="flat-max", expires_utc=datetime.now(UTC) - timedelta(seconds=5), owner_pid=os.getpid()
    )
    _atomic_write_text(request_file, expired.render())

    _watchdog_step(request_file, status_file, allowlist=frozenset({"flat-max"}), default_profile="curve-default")

    assert parse_status(status_file.read_text())["profile"] == "curve-default"


def test_watchdog_step_dead_owner_falls_back_to_default(tmp_path: Path) -> None:
    request_file = tmp_path / "request"
    status_file = tmp_path / "status"
    proc = subprocess.Popen(["true"])
    proc.wait()
    live = LeaseRequest(profile="flat-max", expires_utc=datetime.now(UTC) + timedelta(seconds=60), owner_pid=proc.pid)
    _atomic_write_text(request_file, live.render())

    _watchdog_step(request_file, status_file, allowlist=frozenset({"flat-max"}), default_profile="curve-default")

    assert parse_status(status_file.read_text())["profile"] == "curve-default"


def test_watchdog_step_unknown_profile_is_ignored(tmp_path: Path) -> None:
    request_file = tmp_path / "request"
    status_file = tmp_path / "status"
    live = LeaseRequest(
        profile="not-a-real-profile", expires_utc=datetime.now(UTC) + timedelta(seconds=60), owner_pid=os.getpid()
    )
    _atomic_write_text(request_file, live.render())

    _watchdog_step(request_file, status_file, allowlist=frozenset({"flat-max"}), default_profile="curve-default")

    assert parse_status(status_file.read_text())["profile"] == "curve-default"


def test_watchdog_step_fault_overrides_a_live_lease(tmp_path: Path) -> None:
    request_file = tmp_path / "request"
    status_file = tmp_path / "status"
    live = LeaseRequest(
        profile="flat-max", expires_utc=datetime.now(UTC) + timedelta(seconds=60), owner_pid=os.getpid()
    )
    _atomic_write_text(request_file, live.render())

    _watchdog_step(
        request_file, status_file, allowlist=frozenset({"flat-max"}), default_profile="curve-default", fault=True
    )

    assert parse_status(status_file.read_text())["profile"] == "full-duty-fault"


# ---- FanLease: acquire / release, driven by the fake watchdog ---------------


def test_fan_lease_acquire_confirms_readback(tmp_path: Path) -> None:
    request_file = tmp_path / "request"
    status_file = tmp_path / "status"
    allowlist = frozenset({"flat-max"})

    def step() -> None:
        _watchdog_step(request_file, status_file, allowlist=allowlist, default_profile="curve-default")

    lease = FanLease(
        "flat-max",
        request_file=request_file,
        status_file=status_file,
        readback_timeout_s=2.0,
        poll_s=0.01,
        sleep=lambda _s: step(),
    )

    result = lease.acquire()

    assert result["requested_profile"] == "flat-max"
    assert result["readback_profile"] == "flat-max"
    assert request_file.read_text().startswith("profile=flat-max")


def test_fan_lease_acquire_writes_the_request_atomically(tmp_path: Path) -> None:
    request_file = tmp_path / "request"
    status_file = tmp_path / "status"
    allowlist = frozenset({"flat-max"})

    def step() -> None:
        _watchdog_step(request_file, status_file, allowlist=allowlist, default_profile="curve-default")

    lease = FanLease(
        "flat-max",
        request_file=request_file,
        status_file=status_file,
        readback_timeout_s=2.0,
        poll_s=0.01,
        sleep=lambda _s: step(),
    )

    lease.acquire()

    # No leftover temp file: the write was `mkstemp` + `os.replace`, not in place.
    assert {p.name for p in tmp_path.iterdir()} == {"request", "status"}


def test_fan_lease_acquire_times_out_when_the_watchdog_ignores_the_request(tmp_path: Path) -> None:
    lease = FanLease(
        "flat-max",
        request_file=tmp_path / "request",
        status_file=tmp_path / "status",
        readback_timeout_s=0.05,
        poll_s=0.01,
        sleep=lambda _s: None,
    )

    with pytest.raises(ControlError, match="not confirmed"):
        lease.acquire()


def test_fan_lease_release_confirms_restore(tmp_path: Path) -> None:
    request_file = tmp_path / "request"
    status_file = tmp_path / "status"
    allowlist = frozenset({"flat-max"})

    def step() -> None:
        _watchdog_step(request_file, status_file, allowlist=allowlist, default_profile="curve-default")

    lease = FanLease(
        "flat-max",
        request_file=request_file,
        status_file=status_file,
        readback_timeout_s=2.0,
        poll_s=0.01,
        sleep=lambda _s: step(),
    )
    lease.acquire()

    result = lease.release()

    assert result["restore_confirmed"] is True
    assert result["restore_profile"] == "curve-default"
    assert not request_file.exists()


def test_fan_lease_release_reports_unconfirmed_when_status_never_changes(tmp_path: Path) -> None:
    request_file = tmp_path / "request"
    status_file = tmp_path / "status"
    status_file.write_text("state=OK target_pwm=255 max_gpu_junction_mC=36000 profile=flat-max\n")
    request_file.write_text(
        LeaseRequest(
            profile="flat-max", expires_utc=datetime.now(UTC) + timedelta(seconds=60), owner_pid=os.getpid()
        ).render()
    )

    lease = FanLease(
        "flat-max",
        request_file=request_file,
        status_file=status_file,
        readback_timeout_s=0.05,
        poll_s=0.01,
        sleep=lambda _s: None,
    )

    result = lease.release()

    assert result["restore_confirmed"] is False
    assert not request_file.exists()


def test_fan_lease_release_of_the_default_profile_confirms_restore(tmp_path: Path) -> None:
    """A curve-default lease has nothing to fall back from (live run 2026-10-09)."""
    request_file = tmp_path / "request"
    status_file = tmp_path / "status"
    allowlist = frozenset({"flat-max", "curve-default"})

    def step() -> None:
        _watchdog_step(request_file, status_file, allowlist=allowlist, default_profile="curve-default")

    lease = FanLease(
        "curve-default",
        request_file=request_file,
        status_file=status_file,
        readback_timeout_s=2.0,
        poll_s=0.01,
        sleep=lambda _s: step(),
    )
    lease.acquire()

    result = lease.release()

    assert result["restore_confirmed"] is True
    assert result["restore_profile"] == "curve-default"
    assert not request_file.exists()


def test_fan_lease_release_of_the_default_profile_needs_an_ok_state(tmp_path: Path) -> None:
    request_file = tmp_path / "request"
    status_file = tmp_path / "status"
    status_file.write_text("state=FAULT target_pwm=255 max_gpu_junction_mC=36000 profile=curve-default\n")

    lease = FanLease(
        "curve-default",
        request_file=request_file,
        status_file=status_file,
        readback_timeout_s=0.05,
        poll_s=0.01,
        sleep=lambda _s: None,
    )

    result = lease.release()

    assert result["restore_confirmed"] is False


def test_fan_lease_renew_if_due_raises_before_acquire(tmp_path: Path) -> None:
    lease = FanLease("flat-max", request_file=tmp_path / "request", status_file=tmp_path / "status")

    with pytest.raises(ControlError, match="before acquire"):
        lease.renew_if_due()


# ---- power profile readback --------------------------------------------------


def test_normalize_power_profile_upper_cases_and_replaces_hyphens() -> None:
    assert normalize_power_profile("power-saving") == "POWER_SAVING"
    assert normalize_power_profile("COMPUTE") == "COMPUTE"


def test_read_power_state_and_verify_profile_match(tmp_path: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    _make_power_card(drm_root, 0, active_profile="POWER_SAVING")
    _make_power_card(drm_root, 1, active_profile="POWER_SAVING")

    state = read_power_state((0, 1), drm_root=drm_root)

    assert state[0] == {"profile": "POWER_SAVING", "performance_level": "manual"}
    verify_power_profile("power-saving", state)  # does not raise


def test_verify_power_profile_refuses_a_mismatch(tmp_path: Path) -> None:
    drm_root = tmp_path / "sys-class-drm"
    _make_power_card(drm_root, 0, active_profile="POWER_SAVING")
    _make_power_card(drm_root, 1, active_profile="COMPUTE")

    state = read_power_state((0, 1), drm_root=drm_root)

    with pytest.raises(ControlError, match="mismatch"):
        verify_power_profile("power-saving", state)


# ---- SharedFanLease: one real lease, many holders ---------------------------


class _CountingLease:
    """A stand-in lease that counts the real calls."""

    def __init__(self, *, fail_first_acquire: bool = False) -> None:
        self.calls: list[str] = []
        self.fail_first_acquire = fail_first_acquire

    def acquire(self) -> dict[str, object]:
        self.calls.append("acquire")
        if self.fail_first_acquire:
            self.fail_first_acquire = False
            raise ControlError("not confirmed")
        return {"requested_profile": "flat-max", "readback_profile": "flat-max"}

    def renew_if_due(self) -> None:
        self.calls.append("renew")

    def release(self) -> dict[str, object]:
        self.calls.append("release")
        return {"released_utc": "t", "restore_confirmed": True, "restore_profile": "curve-default"}


def test_shared_lease_acquires_once_and_releases_on_the_last_holder() -> None:
    real = _CountingLease()
    shared = SharedFanLease(real)

    first = shared.acquire()
    second = shared.acquire()
    deferred = shared.release()
    last = shared.release()

    assert real.calls == ["acquire", "release"]
    assert first == second
    assert deferred["restore_confirmed"] is True
    assert deferred["deferred"] is True
    assert last["restore_profile"] == "curve-default"
    assert "deferred" not in last
    assert shared.holders == 0


def test_shared_lease_renews_under_its_lock_while_held() -> None:
    real = _CountingLease()
    shared = SharedFanLease(real)
    shared.acquire()
    shared.renew_if_due()
    shared.release()

    assert real.calls == ["acquire", "renew", "release"]


def test_shared_lease_refuses_renew_and_release_with_no_holder() -> None:
    shared = SharedFanLease(_CountingLease())

    with pytest.raises(ControlError, match="no holder"):
        shared.renew_if_due()
    with pytest.raises(ControlError, match="no holder"):
        shared.release()


def test_shared_lease_does_not_count_a_failed_acquire() -> None:
    real = _CountingLease(fail_first_acquire=True)
    shared = SharedFanLease(real)

    with pytest.raises(ControlError):
        shared.acquire()
    assert shared.holders == 0
    shared.acquire()
    assert shared.holders == 1
    assert real.calls == ["acquire", "acquire"]


def test_a_member_that_never_joined_does_not_drop_a_peer_hold() -> None:
    real = _CountingLease()
    shared = SharedFanLease(real)
    peer = shared.member()
    failed = shared.member()
    peer.acquire()

    # A run whose acquire failed still releases in its `finally`.
    result = failed.release()

    assert result["deferred"] is True
    assert shared.holders == 1
    assert real.calls == ["acquire"]
    peer.release()
    assert real.calls == ["acquire", "release"]


def test_a_failed_acquire_with_no_holder_still_removes_the_request() -> None:
    real = _CountingLease(fail_first_acquire=True)
    shared = SharedFanLease(real)
    member = shared.member()

    with pytest.raises(ControlError):
        member.acquire()
    # A failed real acquire may leave the request file. The release removes it.
    result = member.release()

    assert real.calls == ["acquire", "release"]
    assert "deferred" not in result
    assert shared.holders == 0


def test_members_release_once_and_the_last_member_restores() -> None:
    real = _CountingLease()
    shared = SharedFanLease(real)
    first, second = shared.member(), shared.member()
    first.acquire()
    second.acquire()

    assert first.release()["deferred"] is True
    assert "deferred" not in second.release()
    assert real.calls == ["acquire", "release"]
    assert shared.holders == 0
