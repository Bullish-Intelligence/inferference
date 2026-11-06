"""The fan and power profile lease: request, verify, hold, restore.

Safety is split from control. The fan watchdog (owned by the host's fan
controller, out of scope here) is the only thing that moves a PWM, a power
profile, or a clock. An experiment never writes an actuator. It writes a
*request* naming the profile it wants, and the watchdog applies it, verifies
readback, holds it, and restores the default profile when the lease ends. This
module is the requester's half of that interface: it never touches the
actuator itself, only the lease files and the read-only power sysfs.

`FanLease` owns the request file's lifecycle: `acquire()` writes the request
and waits for readback, `renew_if_due()` extends the lease before it expires,
and `release()` removes the request and waits for the watchdog to fall back to
its default. `arbitrate()` is a pure, I/O-free reference of the rule the
watchdog itself must run on each of its passes; it lives here so a test can
drive a fake watchdog with the real rule, not so this module can enforce it.
nix-meta is the actual authority over the fan.

`read_power_state` and `verify_power_profile` are readback-only: they read the
starred row of `pp_power_profile_mode` and the performance level, and compare
them with what the spec declared. Neither writes anything.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

DEFAULT_REQUEST_FILE = Path("/run/arctic-fan/request")
DEFAULT_STATUS_FILE = Path("/run/arctic-fan/status")
# The profile the watchdog falls back to when no lease is active
# (nix-meta arctic-fan-controller, `defaultProfile`).
DEFAULT_FAN_PROFILE = "curve-default"
DEFAULT_DRM_ROOT = Path("/sys/class/drm")

# Matches `inferference.telemetry._PP_PROFILE_LINE`. Re-implemented here,
# rather than imported, so this module does not depend on a private name in
# another module; the format is stable sysfs text, not a shared object.
_PP_PROFILE_LINE = re.compile(r"^\s*(\d+)\s+([A-Z0-9_]+)(\*)?\s*:")

_KV_TOKEN = re.compile(r"(\S+)=(\S+)")


class ControlError(RuntimeError):
    """A lease could not be confirmed, or a power profile did not match."""


@dataclass(frozen=True, slots=True)
class LeaseRequest:
    """One fan-lease request: the profile wanted, its expiry, and its owner.

    The request names exactly three keys: `profile`, `expires`,
    `owner_pid`. The watchdog drops the request when `expires` has passed or
    `owner_pid` is gone, so a crashed requester cannot hold a profile forever.
    """

    profile: str
    expires_utc: datetime
    owner_pid: int

    def render(self) -> str:
        """Render this request as the request file's exact text."""
        expires = self.expires_utc.astimezone(UTC).isoformat().replace("+00:00", "Z")
        return f"profile={self.profile}\nexpires={expires}\nowner_pid={self.owner_pid}\n"


def parse_request(text: str) -> LeaseRequest:
    """Parse a lease request file's text back into a `LeaseRequest`.

    Refuses with `ControlError` if any of the three required keys is
    missing or malformed.
    """
    tokens = parse_status(text)
    missing = {"profile", "expires", "owner_pid"} - tokens.keys()
    if missing:
        raise ControlError(f"lease request is missing key(s) {sorted(missing)}: {text!r}")
    try:
        expires_utc = datetime.fromisoformat(tokens["expires"].replace("Z", "+00:00"))
        owner_pid = int(tokens["owner_pid"])
    except ValueError as exc:
        raise ControlError(f"malformed lease request: {text!r}") from exc
    return LeaseRequest(profile=tokens["profile"], expires_utc=expires_utc, owner_pid=owner_pid)


def parse_status(text: str) -> dict[str, str]:
    """Parse every `key=value` token in `text`, on one line or several.

    This reads both the request file (one key per line) and the watchdog's
    status line (several keys on one line), with the same rule: no
    assumption about layout, only `key=value` tokens separated by whitespace.
    """
    return dict(_KV_TOKEN.findall(text))


def arbitrate(
    request: LeaseRequest | None,
    *,
    now: datetime,
    owner_alive: bool,
    allowlist: frozenset[str],
    fault: bool,
    default_profile: str,
) -> tuple[str, str]:
    """The reference rule for the active fan profile, given one lease request.

    This is a pure function: no I/O, no clock read, no file access. It exists
    so a test can drive a fake watchdog with the exact rule, not because this
    module enforces anything on the real fan. **nix-meta's watchdog is the
    actual authority**: it must implement this same rule on every pass over
    the real request and status files. This function never runs against
    hardware.

    Returns `(active_profile, reason)`. A fault always wins, even over a live
    lease: a lease can never weaken a fault response.
    """
    if fault:
        return "full-duty-fault", "fault override"
    if request is None:
        return default_profile, "no request"
    if request.profile not in allowlist:
        return default_profile, "unknown profile ignored"
    if request.expires_utc <= now or not owner_alive:
        return default_profile, "lease dropped"
    return request.profile, "requested"


def _atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: a temp file, then `os.replace`.

    A reader polling `path` on its own schedule (the watchdog, or a test's
    fake watchdog) must never observe a truncated or partial file.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _now_utc() -> datetime:
    return datetime.now(UTC)


class FanLease:
    """Requests a named fan profile, confirms it by readback, and restores it.

    This never writes a PWM or a profile. It writes a request file and reads
    a status file; the watchdog on the other end is the only actuator.
    """

    def __init__(
        self,
        profile: str,
        *,
        ttl_s: float = 120.0,
        default_profile: str = DEFAULT_FAN_PROFILE,
        request_file: Path = DEFAULT_REQUEST_FILE,
        status_file: Path = DEFAULT_STATUS_FILE,
        readback_timeout_s: float = 30.0,
        poll_s: float = 0.5,
        clock: Callable[[], datetime] = _now_utc,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.profile = profile
        self.ttl_s = ttl_s
        self.default_profile = default_profile
        self.request_file = request_file
        self.status_file = status_file
        self.readback_timeout_s = readback_timeout_s
        self.poll_s = poll_s
        self._clock = clock
        self._sleep = sleep
        self._owner_pid = os.getpid()
        self._expires_utc: datetime | None = None

    def _read_status(self) -> dict[str, str]:
        try:
            text = self.status_file.read_text()
        except OSError:
            return {}
        return parse_status(text)

    def _write_request(self, expires_utc: datetime) -> None:
        request = LeaseRequest(profile=self.profile, expires_utc=expires_utc, owner_pid=self._owner_pid)
        try:
            _atomic_write(self.request_file, request.render())
        except OSError as exc:
            raise ControlError(f"could not write fan lease request {self.request_file}: {exc}") from exc

    def acquire(self) -> dict[str, Any]:
        """Write the lease request and wait for the watchdog to confirm it.

        Raises `ControlError` if `status` never reports this profile active
        with `state=OK` within `readback_timeout_s`. It never assumes success.
        """
        started = self._clock()
        self._expires_utc = started + timedelta(seconds=self.ttl_s)
        self._write_request(self._expires_utc)

        deadline = time.monotonic() + self.readback_timeout_s
        status: dict[str, str] = {}
        while True:
            status = self._read_status()
            if status.get("state") == "OK" and status.get("profile") == self.profile:
                return {
                    "requested_profile": self.profile,
                    "request_written_utc": started.isoformat(),
                    "readback_profile": status.get("profile"),
                    "readback_utc": self._clock().isoformat(),
                }
            if time.monotonic() >= deadline:
                raise ControlError(
                    f"fan lease for profile {self.profile!r} was not confirmed within "
                    f"{self.readback_timeout_s}s; last status: {status!r}"
                )
            self._sleep(self.poll_s)

    def renew_if_due(self) -> None:
        """Rewrite the lease's `expires` field once less than half its TTL remains.

        Does nothing if `acquire()` has not run yet, or if the lease still
        has more than half its TTL left.
        """
        if self._expires_utc is None:
            raise ControlError("renew_if_due called before acquire")
        now = self._clock()
        remaining_s = (self._expires_utc - now).total_seconds()
        if remaining_s >= self.ttl_s / 2:
            return
        self._expires_utc = now + timedelta(seconds=self.ttl_s)
        self._write_request(self._expires_utc)

    def release(self) -> dict[str, Any]:
        """Remove the request and wait for the watchdog to fall back from it.

        Always attempts the removal and the readback wait. It reports
        `restore_confirmed: false` instead of raising when the status never
        stops reporting this profile — the caller runs this in `finally` and
        must get a result back regardless. A lease on the default profile has
        nothing to change back to: its restore is confirmed by a fresh
        `state=OK` readback of that profile after the removal. A genuine I/O error removing the
        request file still raises `ControlError`: that is not the expected
        "no readback change" case.
        """
        try:
            self.request_file.unlink(missing_ok=True)
        except OSError as exc:
            raise ControlError(f"could not remove fan lease request {self.request_file}: {exc}") from exc

        released_utc = self._clock().isoformat()
        deadline = time.monotonic() + self.readback_timeout_s
        status: dict[str, str] = {}
        while time.monotonic() < deadline:
            status = self._read_status()
            current = status.get("profile")
            if current is not None and current != self.profile:
                return {"released_utc": released_utc, "restore_confirmed": True, "restore_profile": current}
            if self.profile == self.default_profile and current == self.profile and status.get("state") == "OK":
                return {"released_utc": released_utc, "restore_confirmed": True, "restore_profile": current}
            self._sleep(self.poll_s)
        return {
            "released_utc": released_utc,
            "restore_confirmed": False,
            "restore_profile": status.get("profile"),
        }


class FanLeaseLike(Protocol):
    """What the runner needs from a lease: `FanLease` and `SharedFanLease` both fit."""

    def acquire(self) -> dict[str, Any]: ...

    def renew_if_due(self) -> None: ...

    def release(self) -> dict[str, Any]: ...


class SharedFanLease:
    """One `FanLease` shared by several runs in one process, with a holder count.

    The fan watchdog has one request file. Two parallel runs must not each
    own a lease, because the first release would drop the fan profile under the
    second run. The first `acquire()` makes the real acquire. Later calls join
    the held lease and return its readback. A `release()` that leaves other
    holders returns `restore_confirmed: True` with `deferred: True` and
    changes nothing. The last `release()` makes the real release. Every call
    runs under one lock, so two collector threads can renew safely.

    Each run takes its own `member()`. A member that never joined (its
    `acquire()` failed) must not drop the hold of a peer run. When nobody
    holds the lease, such a member still makes the real release. A failed real
    `acquire()` may leave the request file in place, and the release removes it.
    """

    def __init__(self, lease: FanLeaseLike) -> None:
        self._lease = lease
        self._lock = threading.Lock()
        self._holders = 0
        self._readback: dict[str, Any] = {}

    @property
    def holders(self) -> int:
        with self._lock:
            return self._holders

    def acquire(self) -> dict[str, Any]:
        """Join the lease. Only the first holder writes the request."""
        with self._lock:
            if self._holders == 0:
                self._readback = self._lease.acquire()
            self._holders += 1
            return dict(self._readback)

    def member(self) -> FanLeaseLike:
        """A lease handle for one run. See the class text for why each run needs its own."""
        return _FanLeaseMember(self)

    def renew_if_due(self) -> None:
        """Extend the lease when it is due. Raises `ControlError` when nobody holds it."""
        with self._lock:
            if self._holders == 0:
                raise ControlError("renew_if_due called with no holder")
            self._lease.renew_if_due()

    def release(self) -> dict[str, Any]:
        """Leave the lease. Only the last holder removes the request."""
        with self._lock:
            if self._holders == 0:
                raise ControlError("release called with no holder")
            self._holders -= 1
            if self._holders > 0:
                return {
                    "released_utc": datetime.now(UTC).isoformat(),
                    "restore_confirmed": True,
                    "restore_profile": None,
                    "deferred": True,
                }
            return self._lease.release()

    def _release_unjoined(self) -> dict[str, Any]:
        """Release for a member whose `acquire()` failed."""
        with self._lock:
            if self._holders > 0:
                return {
                    "released_utc": datetime.now(UTC).isoformat(),
                    "restore_confirmed": True,
                    "restore_profile": None,
                    "deferred": True,
                }
            return self._lease.release()


class _FanLeaseMember:
    """One run's handle on a `SharedFanLease`. It joins once and leaves once."""

    def __init__(self, shared: SharedFanLease) -> None:
        self._shared = shared
        self._joined = False

    def acquire(self) -> dict[str, Any]:
        readback = self._shared.acquire()
        self._joined = True
        return readback

    def renew_if_due(self) -> None:
        self._shared.renew_if_due()

    def release(self) -> dict[str, Any]:
        if not self._joined:
            return self._shared._release_unjoined()
        self._joined = False
        return self._shared.release()


def normalize_power_profile(name: str) -> str:
    """Normalize a spec power-profile name to its sysfs row spelling.

    `"power-saving"` becomes `"POWER_SAVING"`: upper case, hyphen to
    underscore. Compare the result with the starred row of
    `pp_power_profile_mode`.
    """
    return name.strip().upper().replace("-", "_")


def _parse_active_power_profile(text: str) -> str:
    """Parse the starred (`*`) row of a `pp_power_profile_mode` dump.

    Reads the same format as `inferference.telemetry._parse_active_power_
    profile`. Re-implemented here rather than imported: that function is
    private to its module, and the format is stable sysfs text, not a shared
    object.
    """
    for line in text.splitlines():
        match = _PP_PROFILE_LINE.match(line)
        if match is not None and match.group(3) == "*":
            return match.group(2)
    raise ControlError(f"no active ('*') row found in pp_power_profile_mode: {text!r}")


def read_power_state(cards: Sequence[int], drm_root: Path = DEFAULT_DRM_ROOT) -> dict[int, dict[str, str]]:
    """Read each card's active power profile and performance level.

    Read-only. This is the readback half of the control surface: the host owns
    the power-profile actuator, and this function only observes what it set.
    """
    state: dict[int, dict[str, str]] = {}
    for card in cards:
        device = drm_root / f"card{card}" / "device"
        try:
            profile_text = (device / "pp_power_profile_mode").read_text()
            level_text = (device / "power_dpm_force_performance_level").read_text()
        except OSError as exc:
            raise ControlError(f"could not read power state for card {card}: {exc}") from exc
        state[card] = {
            "profile": _parse_active_power_profile(profile_text),
            "performance_level": level_text.strip(),
        }
    return state


def verify_power_profile(declared: str, state: Mapping[int, Mapping[str, str]]) -> None:
    """Refuse with `ControlError` unless every card's profile matches `declared`.

    `declared` is normalized first (`normalize_power_profile`), so the spec
    may use either spelling.
    """
    expected = normalize_power_profile(declared)
    mismatched = {
        card: card_state.get("profile") for card, card_state in state.items() if card_state.get("profile") != expected
    }
    if mismatched:
        raise ControlError(f"power profile mismatch (expected {expected!r}): {mismatched}")
