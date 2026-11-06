"""The execution boundary for generated code.

Model output is untrusted. It must never run against the live repo, the
network, or the owner's files. Every candidate runs in a throwaway directory
under ``bwrap`` with no network, a read-only root, a wall-clock timeout and a
working directory that is the only writable path.

**If ``bwrap`` is not available the caller skips the behavioural layer loudly.**
There is no unsandboxed fallback. Running model-written code on the host to
keep a number flowing is exactly the silent pass this repo forbids.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SandboxResult", "SandboxUnavailable", "available", "run"]


class SandboxUnavailable(RuntimeError):
    """No usable sandbox on this machine."""


@dataclass(frozen=True, slots=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def available() -> bool:
    return shutil.which("bwrap") is not None


def run(
    argv: list[str],
    *,
    workdir: Path,
    timeout_s: float = 120.0,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Run ``argv`` in ``workdir`` with no network and a read-only host root."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SandboxUnavailable("bwrap is not on PATH")

    workdir = workdir.resolve()
    wrapped = [
        bwrap,
        "--unshare-net",  # no network, the point of the exercise
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-pid",
        "--die-with-parent",
        "--new-session",  # no terminal to escape onto
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(workdir),
        str(workdir),  # the only writable path
        "--chdir",
        str(workdir),
        "--setenv",
        "HOME",
        str(workdir),
        "--setenv",
        "TMPDIR",
        "/tmp",
        "--",
        *argv,
    ]

    base_env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "NO_NETWORK": "1",
    }
    if env:
        base_env.update(env)

    try:
        proc = subprocess.run(
            wrapped,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=base_env,
        )
    except subprocess.TimeoutExpired as exc:
        return SandboxResult(
            returncode=124,
            stdout=_text(exc.stdout),
            stderr=_text(exc.stderr),
            timed_out=True,
        )
    return SandboxResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        timed_out=False,
    )


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)
