"""A stop request that a signal handler can set and the runners can read.

A signal handler runs in the main thread, between two bytecodes. It must not
take a lock that the interrupted code may hold. `StopRequest` therefore only
sets a `threading.Event` and stores the first reason. A watcher thread in each
run reads the request and stops that run's server. This keeps the stop path
to one `SIGINT` per server: the runner's own `stop_server` is the only sender.
"""

from __future__ import annotations

import threading


class StopRequest:
    """One stop request for a whole command. The first reason wins."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: str | None = None

    def request(self, reason: str) -> bool:
        """Ask for a stop. Return True when this call was the first one."""
        if self._event.is_set():
            return False
        # Store the reason before the event, so a reader that sees the event sees the reason.
        self._reason = reason
        self._event.set()
        return True

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until a stop is requested or the timeout ends. Return whether it is set."""
        return self._event.wait(timeout)

    @property
    def reason(self) -> str | None:
        """The reason of the first request, or None."""
        return self._reason if self._event.is_set() else None
