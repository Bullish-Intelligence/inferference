"""Human duration strings."""

from __future__ import annotations

UNITS = {"h": 3600, "m": 60, "s": 1}


def parse_duration(text: str) -> int:
    """Parse a duration such as "1h30m" or "90s" into whole seconds.

    Grammar: one or more `<integer><unit>` pairs, no separators, units drawn
    from h, m and s. Units may appear in any order and each unit at most once.
    Surrounding whitespace is ignored. The integer part is non-negative.

    Raises ValueError on: an empty string, a trailing or leading character that
    is not part of a pair, an unknown unit, a repeated unit, or a number with
    no unit.
    """
    raise NotImplementedError
