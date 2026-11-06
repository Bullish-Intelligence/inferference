"""Tests for the pure analysis estimators: OLS slope, AR(1) correction, t quantiles."""

from __future__ import annotations

import random

from inferference.experiment.analysis import estimators
from inferference.experiment.analysis.estimators import ols_slope, student_t_quantile


def test_ols_slope_is_exact_on_a_hand_computed_line() -> None:
    # y = 1 + 2t, no noise: slope must come back exactly 2.0.
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [1.0, 3.0, 5.0, 7.0, 9.0]

    estimate = ols_slope(t, y, band_c=(0.0, 10.0))

    assert estimate.slope_c_per_s == 2.0
    assert estimate.n == 5
    assert estimate.band_c == (0.0, 10.0)
    assert estimate.method == "ols+ar1"


def test_alternating_residual_gives_negative_rho_and_leaves_se_ar1_unchanged() -> None:
    # Residuals around the fitted line alternate in sign: +0.8, -1.2, +0.8, -1.2, +0.8.
    # A constant-step alternating pattern autocorrelates negatively at lag 1.
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [1.0, 0.0, 3.0, 2.0, 5.0]

    estimate = ols_slope(t, y, band_c=(0.0, 10.0))

    assert estimate.rho1 < 0.0
    assert estimate.se_ar1 == estimate.se_ols
    assert estimate.n_eff == float(estimate.n)


def test_seeded_ar1_series_inflates_se_ar1_above_se_ols() -> None:
    # An AR(1) noise series with phi 0.8 around a fixed line: positive residual
    # autocorrelation must inflate the standard error over the plain OLS value.
    rng = random.Random(7)
    phi = 0.8
    innovations = [rng.gauss(0.0, 1.0) for _ in range(50)]
    eps = []
    prev = 0.0
    for innovation in innovations:
        prev = phi * prev + innovation
        eps.append(prev)

    t = [float(i) for i in range(50)]
    y = [2.0 + 0.5 * ti + e for ti, e in zip(t, eps, strict=True)]

    estimate = ols_slope(t, y, band_c=(0.0, 100.0))

    assert estimate.rho1 > 0.0
    assert estimate.se_ar1 > estimate.se_ols
    assert estimate.n_eff < estimate.n


def test_student_t_quantile_matches_the_reference_table() -> None:
    cases = [
        (0.975, 1, 12.706),
        (0.975, 10, 2.228),
        (0.95, 10, 1.812),
        (0.975, 1e6, 1.960),
    ]

    for p, dof, expected in cases:
        assert round(student_t_quantile(p, dof), 3) == expected


def test_all_equals_the_pinned_list() -> None:
    assert estimators.__all__ == [
        "SlopeEstimate",
        "ols_slope",
        "student_t_quantile",
        "upper_confidence_bound",
        "TransitTime",
        "transit_time",
        "Peak",
        "peak",
        "mean",
        "EstimatorError",
    ]
