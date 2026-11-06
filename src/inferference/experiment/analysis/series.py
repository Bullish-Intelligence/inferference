"""Turn a loaded run into per-card time series, episodes, and bands.

A `RunRecord` holds one tick per `Observation`. This module reduces that to
one named field on one card (`card_series`), finds the load episodes in a
power series (`episodes`), extracts the samples inside a temperature band
during one episode (`band_samples`), and windows any point series to a time
range (`window`). All four are pure: no file or sensor access.

**Episode detection has no spec field for its power threshold** — it is a
parameter the caller (the CLI, or a test) supplies, not something an
experiment spec declares. The CLI's own default is 150 W; this module
hardcodes nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .load import RunRecord


@dataclass(frozen=True, slots=True)
class Point:
    """One sample: a monotonic-ish time coordinate and a value."""

    mono: float
    value: float


@dataclass(frozen=True, slots=True)
class Episode:
    """A load episode: `end_mono` is `None` while the episode is still open."""

    start_mono: float
    end_mono: float | None


def card_series(record: RunRecord, card: int, field: str) -> tuple[Point, ...]:
    """The named `CardObservation` field for `card`, across every tick that has it.

    A tick where `card` was not sampled, or where the field reads `None`
    (an optional reading the legacy adapter could not supply), is skipped
    rather than filled in — never a derived or guessed value.
    """
    points = []
    for observation in record.observations:
        card_observation = observation.cards.get(card)
        if card_observation is None:
            continue
        value = getattr(card_observation, field)
        if value is None:
            continue
        points.append(Point(mono=observation.mono, value=float(value)))
    return tuple(points)


def episodes(power: Sequence[Point], *, threshold_w: float, k: int = 3) -> tuple[Episode, ...]:
    """Find load episodes in a power series.

    An episode starts at the first of `k` consecutive samples at or above
    `threshold_w`, and ends at the first of `k` consecutive samples below it.
    A run shorter than `k` samples, in either direction, never starts or ends
    an episode — this is why a 2-sample spike at `k=3` is invisible by
    design, not a bug. An episode still open at the end of the series gets
    `end_mono=None`.
    """
    result = []
    in_episode = False
    start_mono = 0.0
    above_run = 0
    below_run = 0
    for index, point in enumerate(power):
        if point.value >= threshold_w:
            above_run += 1
            below_run = 0
        else:
            below_run += 1
            above_run = 0

        if not in_episode and above_run == k:
            start_mono = power[index - k + 1].mono
            in_episode = True
        elif in_episode and below_run == k:
            end_mono = power[index - k + 1].mono
            result.append(Episode(start_mono=start_mono, end_mono=end_mono))
            in_episode = False

    if in_episode:
        result.append(Episode(start_mono=start_mono, end_mono=None))
    return tuple(result)


def band_samples(temp: Sequence[Point], episode: Episode, *, band_c: tuple[float, float]) -> tuple[Point, ...]:
    """The contiguous span of `temp`, inside `episode`, that sits in `band_c`.

    The span runs from the first sample at or above `band_c[0]` up to (not
    including) the first sample above `band_c[1]` — so an excursion past the
    top of the band ends the span there, even if later samples fall back
    inside it. Only samples with `lo <= value <= hi` are kept from that span.
    """
    lo, hi = band_c
    end_mono = episode.end_mono if episode.end_mono is not None else math.inf
    in_episode = [point for point in temp if episode.start_mono <= point.mono <= end_mono]

    span_started = False
    result = []
    for point in in_episode:
        if not span_started:
            if point.value < lo:
                continue
            span_started = True
        if point.value > hi:
            break
        if lo <= point.value <= hi:
            result.append(point)
    return tuple(result)


def window(points: Sequence[Point], start_mono: float, end_mono: float) -> tuple[Point, ...]:
    """Every point with `start_mono <= mono <= end_mono`, inclusive of both ends."""
    return tuple(point for point in points if start_mono <= point.mono <= end_mono)
