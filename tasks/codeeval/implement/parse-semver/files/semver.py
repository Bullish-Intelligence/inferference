"""Semantic version parsing."""

from __future__ import annotations


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
    raise NotImplementedError
