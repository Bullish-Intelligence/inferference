"""Backoff schedules."""

from __future__ import annotations


def exponential(base: float, attempt: int, cap: float) -> float:
    """Delay for a zero-based attempt number, never above `cap`."""
    return min(cap, base * (2**attempt))


def schedule(base: float, max_attempts: int, cap: float) -> list[float]:
    """Every delay between attempts. One shorter than the attempt count."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    return [exponential(base, i, cap) for i in range(max_attempts - 1)]


def total_wait(base: float, max_attempts: int, cap: float) -> float:
    """Sum of the schedule."""
    return sum(schedule(base, max_attempts, cap))
