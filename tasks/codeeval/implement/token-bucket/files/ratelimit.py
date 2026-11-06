"""A monotonic-clock token bucket."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Refills at `rate` tokens per second, up to `capacity`.

    The bucket starts full. Time is supplied by the caller through `now`, so the
    class has no dependency on a clock and is testable without sleeping.
    """

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
        """Tokens available at `now`, after refilling. Never above capacity.

        Going backwards in time must not create tokens: if `now` is earlier than
        the last update, the level is unchanged.
        """
        raise NotImplementedError

    def consume(self, now: float, amount: float = 1.0) -> bool:
        """Take `amount` tokens if they are available at `now`.

        Returns True and deducts them, or returns False and deducts nothing.
        Advances the internal clock to `now` in both cases. A non-positive
        `amount` raises ValueError.
        """
        raise NotImplementedError

    def wait_time(self, now: float, amount: float = 1.0) -> float:
        """Seconds until `amount` tokens are available. 0.0 if available now.

        Raises ValueError if `amount` is above capacity, because that wait would
        never end.
        """
        raise NotImplementedError
