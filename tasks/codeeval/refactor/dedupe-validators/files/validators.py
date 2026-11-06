"""Bounds validation for scheduler settings."""

from __future__ import annotations


def check_workers(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("workers must be an int")
    if value < 1:
        raise ValueError("workers must be >= 1, got " + str(value))
    if value > 64:
        raise ValueError("workers must be <= 64, got " + str(value))
    return value


def check_retries(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("retries must be an int")
    if value < 0:
        raise ValueError("retries must be >= 0, got " + str(value))
    if value > 10:
        raise ValueError("retries must be <= 10, got " + str(value))
    return value


def check_timeout_s(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("timeout_s must be an int")
    if value < 1:
        raise ValueError("timeout_s must be >= 1, got " + str(value))
    if value > 3600:
        raise ValueError("timeout_s must be <= 3600, got " + str(value))
    return value


def check_batch_size(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("batch_size must be an int")
    if value < 1:
        raise ValueError("batch_size must be >= 1, got " + str(value))
    if value > 4096:
        raise ValueError("batch_size must be <= 4096, got " + str(value))
    return value
