"""Tests for `inferference.experiment.analysis.series`: pure, no files."""

from __future__ import annotations

from pathlib import Path

from inferference.experiment.analysis.load import CardObservation, Observation, RunRecord
from inferference.experiment.analysis.series import Episode, Point, band_samples, card_series, episodes, window


def _card(*, junction_c: float, power_w: float, gpu_busy_percent: int | None = None) -> CardObservation:
    return CardObservation(
        junction_c=junction_c,
        edge_c=junction_c - 5.0,
        mem_c=junction_c - 2.0,
        power_w=power_w,
        gpu_busy_percent=gpu_busy_percent,
        throttled=False,
        gtt_used_mib=0.0,
    )


def _record(observations: tuple[Observation, ...]) -> RunRecord:
    return RunRecord(
        path=Path("."),
        layout="030-v1",
        run=None,
        declaration=None,
        stages=None,
        outcome=None,
        observations=observations,
    )


# ---- card_series --------------------------------------------------------


def test_card_series_extracts_the_named_field_for_one_card() -> None:
    observations = (
        Observation(utc="t0", mono=0.0, clock="monotonic", cards={0: _card(junction_c=50.0, power_w=90.0)}),
        Observation(utc="t1", mono=1.0, clock="monotonic", cards={0: _card(junction_c=55.0, power_w=95.0)}),
    )
    record = _record(observations)

    points = card_series(record, 0, "junction_c")

    assert points == (Point(mono=0.0, value=50.0), Point(mono=1.0, value=55.0))


def test_card_series_skips_a_tick_missing_the_card() -> None:
    observations = (
        Observation(utc="t0", mono=0.0, clock="monotonic", cards={0: _card(junction_c=50.0, power_w=90.0)}),
        Observation(utc="t1", mono=1.0, clock="monotonic", cards={1: _card(junction_c=40.0, power_w=20.0)}),
    )
    record = _record(observations)

    points = card_series(record, 0, "junction_c")

    assert points == (Point(mono=0.0, value=50.0),)


def test_card_series_skips_a_none_field_value() -> None:
    card_without_busy = _card(junction_c=50.0, power_w=90.0, gpu_busy_percent=None)
    card_with_busy = _card(junction_c=55.0, power_w=95.0, gpu_busy_percent=80)
    observations = (
        Observation(utc="t0", mono=0.0, clock="monotonic", cards={0: card_without_busy}),
        Observation(utc="t1", mono=1.0, clock="monotonic", cards={0: card_with_busy}),
    )
    record = _record(observations)

    points = card_series(record, 0, "gpu_busy_percent")

    assert points == (Point(mono=1.0, value=80.0),)


# ---- episodes -------------------------------------------------------------


def _points(values: list[float], step: float = 1.0) -> tuple[Point, ...]:
    return tuple(Point(mono=index * step, value=value) for index, value in enumerate(values))


def test_a_two_sample_spike_does_not_start_an_episode() -> None:
    power = _points([10.0, 200.0, 200.0, 10.0, 10.0, 10.0])

    result = episodes(power, threshold_w=150.0, k=3)

    assert result == ()


def test_a_three_sample_run_starts_and_ends_an_episode() -> None:
    power = _points([10.0, 200.0, 200.0, 200.0, 10.0, 10.0, 10.0])

    result = episodes(power, threshold_w=150.0, k=3)

    assert result == (Episode(start_mono=1.0, end_mono=4.0),)


def test_an_episode_still_running_at_the_end_of_the_series_has_no_end() -> None:
    power = _points([10.0, 10.0, 200.0, 200.0, 200.0, 200.0])

    result = episodes(power, threshold_w=150.0, k=3)

    assert result == (Episode(start_mono=2.0, end_mono=None),)


def test_two_separate_episodes_are_both_found() -> None:
    power = _points([200.0, 200.0, 200.0, 10.0, 10.0, 10.0, 200.0, 200.0, 200.0, 10.0, 10.0, 10.0])

    result = episodes(power, threshold_w=150.0, k=3)

    assert result == (
        Episode(start_mono=0.0, end_mono=3.0),
        Episode(start_mono=6.0, end_mono=9.0),
    )


# ---- band_samples -----------------------------------------------------------


def test_band_extraction_with_an_excursion_above_hi_stops_at_the_excursion() -> None:
    episode = Episode(start_mono=0.0, end_mono=None)
    temp = _points([30.0, 61.0, 70.0, 75.0, 90.0, 72.0, 65.0])

    result = band_samples(temp, episode, band_c=(60.0, 80.0))

    # index 4 (value 90.0) is the first excursion above hi=80; the span stops
    # there, so the samples at index 5 and 6 (back inside the band) are never
    # reached even though their values alone would qualify.
    assert result == (
        Point(mono=1.0, value=61.0),
        Point(mono=2.0, value=70.0),
        Point(mono=3.0, value=75.0),
    )


def test_band_extraction_restricts_to_the_episode_window() -> None:
    episode = Episode(start_mono=2.0, end_mono=4.0)
    temp = _points([70.0, 70.0, 70.0, 70.0, 70.0, 70.0])

    result = band_samples(temp, episode, band_c=(60.0, 80.0))

    assert [point.mono for point in result] == [2.0, 3.0, 4.0]


def test_band_extraction_never_starts_before_the_low_bound() -> None:
    episode = Episode(start_mono=0.0, end_mono=None)
    temp = _points([10.0, 20.0, 65.0, 70.0])

    result = band_samples(temp, episode, band_c=(60.0, 80.0))

    assert [point.value for point in result] == [65.0, 70.0]


# ---- window -----------------------------------------------------------------


def test_window_keeps_points_inclusive_of_both_ends() -> None:
    points = _points([0.0, 1.0, 2.0, 3.0, 4.0])

    result = window(points, 1.0, 3.0)

    assert [point.mono for point in result] == [1.0, 2.0, 3.0]
