"""Transport errors."""

from __future__ import annotations


class TransientError(RuntimeError):
    """A failure worth retrying."""


class PermanentError(RuntimeError):
    """A failure that will not improve."""
