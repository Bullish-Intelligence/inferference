"""Bounds validation for scheduler settings."""

from __future__ import annotations


def _check_range(name: str, value: int, low: int, high: int) -> int:
    """Validate one integer setting against an inclusive range."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}")
    if value > high:
        raise ValueError(f"{name} must be <= {high}, got {value}")
    return value


def check_workers(value: int) -> int:
    return _check_range("workers", value, 1, 64)


def check_retries(value: int) -> int:
    return _check_range("retries", value, 0, 10)


def check_timeout_s(value: int) -> int:
    return _check_range("timeout_s", value, 1, 3600)


def check_batch_size(value: int) -> int:
    return _check_range("batch_size", value, 1, 4096)
