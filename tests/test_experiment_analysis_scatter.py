"""Tests for `inferference.experiment.analysis.scatter`: pure, no files."""

from __future__ import annotations

import math

import pytest

from inferference.experiment.analysis.scatter import ScatterError, within_condition_scatter


def test_three_replicates_compute_a_mean_and_sample_standard_deviation() -> None:
    scatter = within_condition_scatter("decode_tok_s", [10.0, 12.0, 14.0])

    assert scatter.endpoint == "decode_tok_s"
    assert scatter.n == 3
    assert scatter.mean == 12.0
    assert scatter.sd == pytest.approx(math.sqrt(4.0))  # sample variance = ((2^2+0+2^2))/2 = 4
    assert scatter.values == (10.0, 12.0, 14.0)


def test_two_replicates_refuse_the_stricter_floor_of_three() -> None:
    # The documented floor is 2; this module holds a stricter floor of 3, so
    # n=2 must still refuse.
    with pytest.raises(ScatterError):
        within_condition_scatter("decode_tok_s", [10.0, 12.0])


def test_min_n_is_configurable_and_still_refuses_below_it() -> None:
    with pytest.raises(ScatterError):
        within_condition_scatter("decode_tok_s", [10.0, 12.0, 14.0, 16.0], min_n=5)
