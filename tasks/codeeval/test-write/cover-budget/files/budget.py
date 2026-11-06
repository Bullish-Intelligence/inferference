"""Refuse a placement that would not fit on a card."""

from __future__ import annotations

from dataclasses import dataclass


class OvercommitError(RuntimeError):
    """The placement does not fit, and nothing is evicted to make it fit."""


@dataclass(frozen=True)
class Card:
    """One device and what is already on it."""

    name: str
    total_mib: int
    used_mib: int

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib


def fits(card: Card, footprint_mib: int, reserve_mib: int = 256) -> bool:
    """True when the footprint fits with the reserve still spare."""
    if footprint_mib <= 0:
        raise ValueError("footprint must be positive")
    if reserve_mib < 0:
        raise ValueError("reserve must not be negative")
    return footprint_mib + reserve_mib <= card.free_mib


def admit(card: Card, footprint_mib: int, reserve_mib: int = 256) -> Card:
    """Return the card with the footprint added, or raise OvercommitError."""
    if not fits(card, footprint_mib, reserve_mib):
        raise OvercommitError(
            f"{card.name}: {footprint_mib} MiB does not fit in "
            f"{card.free_mib} MiB free (reserve {reserve_mib} MiB)"
        )
    return Card(card.name, card.total_mib, card.used_mib + footprint_mib)
