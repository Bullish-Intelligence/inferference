"""Interval arithmetic."""

from __future__ import annotations

Interval = tuple[int, int]


def normalize(interval: Interval) -> Interval:
    """Order the endpoints of one interval."""
    low, high = interval
    return (low, high) if low <= high else (high, low)


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping and touching intervals, returning them sorted.

    Endpoints are inclusive, so (1, 2) and (3, 4) do NOT touch, while (1, 3) and
    (3, 5) do and merge into (1, 5). Each input interval is normalized first.
    An empty input returns an empty list.
    """
    if not intervals:
        return []
    ordered = sorted(normalize(i) for i in intervals)
    # TODO: fold `ordered` into the merged result and return it.
