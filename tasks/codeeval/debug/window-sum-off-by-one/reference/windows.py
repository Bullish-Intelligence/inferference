"""Rolling window helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence


def windows(values: Sequence[float], size: int) -> Iterator[Sequence[float]]:
    """Yield every contiguous window of `size` values, left to right."""
    if size <= 0:
        raise ValueError("size must be positive")
    for start in range(len(values) - size + 1):
        yield values[start : start + size]


def rolling_max(values: Sequence[float], size: int) -> list[float]:
    """The maximum of each window."""
    return [max(w) for w in windows(values, size)]
