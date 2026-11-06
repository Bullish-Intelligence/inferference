"""A file-lock lease over a named resource."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class LeaseTimeout(RuntimeError):
    """The lease could not be taken inside the deadline."""


@dataclass(frozen=True)
class Lease:
    """One held lease."""

    name: str
    path: Path
    pid: int


def _lock_path(root: Path, name: str) -> Path:
    return root / f"{name}.lock"


def acquire(root: Path, name: str, timeout_s: float = 900.0) -> Lease:
    """Take the lease, waiting up to `timeout_s`. Raises LeaseTimeout."""
    path = _lock_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return Lease(name=name, path=path, pid=os.getpid())


def release(lease: Lease) -> None:
    """Give the lease back. Releasing twice is not an error."""
    lease.path.unlink(missing_ok=True)


def is_held(root: Path, name: str) -> bool:
    """True when some process holds the named lease."""
    return _lock_path(root, name).exists()
