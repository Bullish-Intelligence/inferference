"""A tiny transport client with retries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .errors import TransientError

MAX_ATTEMPTS = 4
BASE_DELAY = 0.5


class Client:
    """Wraps a transport callable and retries transient failures."""

    def __init__(self, transport: Callable[..., Any], sleep: Callable[[float], None]):
        self.transport = transport
        self.sleep = sleep
        self.attempts = 0

    def _with_retry(self, *args: Any) -> Any:
        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                self.attempts += 1
                return self.transport(*args)
            except TransientError as exc:
                last = exc
                if attempt < MAX_ATTEMPTS - 1:
                    self.sleep(BASE_DELAY * (2**attempt))
        raise last

    def get(self, path: str) -> Any:
        return self._with_retry("GET", path)

    def put(self, path: str, body: Any) -> Any:
        return self._with_retry("PUT", path, body)

    def delete(self, path: str) -> Any:
        return self._with_retry("DELETE", path)
