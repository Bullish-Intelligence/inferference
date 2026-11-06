"""Semantic version parsing."""

from __future__ import annotations

_PRERELEASE_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")


def parse_version(text: str) -> tuple[int, int, int, str | None]:
    """Parse a semantic version string into (major, minor, patch, prerelease).

    Grammar::

        <major>.<minor>.<patch>[-<prerelease>]

    `major`, `minor` and `patch` are non-negative decimal integers with no
    leading zeros, except that "0" itself is allowed. `prerelease`, when
    present, is a non-empty string of ASCII letters, digits, dots and hyphens.

    Returns a 4-tuple. `prerelease` is None when absent.

    Raises ValueError, with a message naming the problem, when:
      * the input is empty or not a string
      * there are not exactly three dot-separated numeric parts
      * any numeric part is empty, non-numeric, negative, or has a leading zero
      * the prerelease marker "-" is present but the prerelease is empty
      * the prerelease contains a character outside [A-Za-z0-9.-]
    """
    if not isinstance(text, str) or not text:
        raise ValueError("version must be a non-empty string")

    core, sep, prerelease = text.partition("-")
    if sep and not prerelease:
        raise ValueError("prerelease marker '-' present but prerelease is empty")
    if prerelease and not set(prerelease) <= _PRERELEASE_OK:
        bad = sorted(set(prerelease) - _PRERELEASE_OK)
        raise ValueError(f"prerelease contains illegal character(s): {bad}")

    parts = core.split(".")
    if len(parts) != 3:
        raise ValueError(f"expected 3 dot-separated parts, got {len(parts)}")

    numbers: list[int] = []
    for name, part in zip(("major", "minor", "patch"), parts, strict=True):
        if not part:
            raise ValueError(f"{name} is empty")
        if not part.isdigit():
            raise ValueError(f"{name} is not a non-negative integer: {part!r}")
        if len(part) > 1 and part[0] == "0":
            raise ValueError(f"{name} has a leading zero: {part!r}")
        numbers.append(int(part))

    return numbers[0], numbers[1], numbers[2], (prerelease or None)
