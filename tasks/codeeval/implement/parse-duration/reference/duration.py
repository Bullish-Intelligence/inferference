"""Human duration strings."""

from __future__ import annotations

import re

UNITS = {"h": 3600, "m": 60, "s": 1}

_PAIR = re.compile(r"(\d+)([a-zA-Z])")


def parse_duration(text: str) -> int:
    """Parse a duration such as "1h30m" or "90s" into whole seconds."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty duration")
    total = 0
    seen: set[str] = set()
    position = 0
    for match in _PAIR.finditer(cleaned):
        if match.start() != position:
            raise ValueError(f"unexpected text in {text!r}")
        number, unit = match.group(1), match.group(2)
        if unit not in UNITS:
            raise ValueError(f"unknown unit {unit!r}")
        if unit in seen:
            raise ValueError(f"repeated unit {unit!r}")
        seen.add(unit)
        total += int(number) * UNITS[unit]
        position = match.end()
    if position != len(cleaned):
        raise ValueError(f"trailing text in {text!r}")
    return total
