"""Small statistics helpers."""

from __future__ import annotations

from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bucket(values: Sequence[float], width: float) -> dict[int, int]:
    counts: dict[int, int] = {}
    for v in values:
        key = int(v // width)
        counts[key] = counts.get(key, 0) + 1
    return counts


def top_n(counts: dict[int, int], n: int) -> list[tuple[int, int]]:
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
