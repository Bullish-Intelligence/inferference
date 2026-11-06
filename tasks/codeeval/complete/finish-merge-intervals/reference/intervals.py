"""Interval arithmetic."""

from __future__ import annotations

Interval = tuple[int, int]


def normalize(interval: Interval) -> Interval:
    """Order the endpoints of one interval."""
    low, high = interval
    return (low, high) if low <= high else (high, low)


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping and touching intervals, returning them sorted."""
    if not intervals:
        return []
    ordered = sorted(normalize(i) for i in intervals)
    merged: list[Interval] = [ordered[0]]
    for low, high in ordered[1:]:
        last_low, last_high = merged[-1]
        if low <= last_high:
            merged[-1] = (last_low, max(last_high, high))
        else:
            merged.append((low, high))
    return merged
