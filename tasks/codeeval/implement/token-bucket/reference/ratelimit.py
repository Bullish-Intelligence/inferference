"""A monotonic-clock token bucket."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Refills at `rate` tokens per second, up to `capacity`."""

    capacity: float
    rate: float
    _tokens: float = field(init=False, default=0.0)
    _updated: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        self._tokens = float(self.capacity)
        self._updated = 0.0

    def tokens(self, now: float) -> float:
        elapsed = max(0.0, now - self._updated)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._updated = max(now, self._updated)
        return self._tokens

    def consume(self, now: float, amount: float = 1.0) -> bool:
        if amount <= 0:
            raise ValueError("amount must be positive")
        available = self.tokens(now)
        if available < amount:
            return False
        self._tokens = available - amount
        return True

    def wait_time(self, now: float, amount: float = 1.0) -> float:
        if amount > self.capacity:
            raise ValueError("amount exceeds capacity")
        available = self.tokens(now)
        if available >= amount:
            return 0.0
        return (amount - available) / self.rate
