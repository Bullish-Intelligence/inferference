"""Thermal analysis estimators: pure math, no hardware or filesystem access.

This module holds the slope, peak, transit-time and mean estimators the
analysis layer needs. Every function takes plain sequences and
returns a plain record. Nothing here reads a file or a sensor.

A two-point rate (`(peak - onset) / dt`) is unfit: its two endpoints are each
biased by a different amount, the biases do not cancel, and its own repeat
scatter is several times the effect it claims to measure. This module never
offers that estimator, under any name. `ols_slope` fits every sample in the
band with ordinary least squares, corrects the standard error for first-order
residual autocorrelation, and reports an uncertainty next to the number.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
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


class EstimatorError(ValueError):
    """An estimator refused its input: too few samples, or no time spread."""


def _check_equal_length(t: Sequence[float], y: Sequence[float]) -> int:
    """Return the shared length of `t` and `y`, or refuse a mismatch."""
    n = len(t)
    if len(y) != n:
        raise EstimatorError(f"t and y must have the same length, got {n} and {len(y)}")
    return n


@dataclass(frozen=True, slots=True)
class SlopeEstimate:
    """An OLS slope over a time band, with an AR(1)-corrected standard error."""

    slope_c_per_s: float
    se_ols: float
    se_ar1: float
    rho1: float
    n: int
    n_eff: float
    dof: float
    residual_rms: float
    band_c: tuple[float, float]
    method: str
    t_start_mono: float
    t_end_mono: float


def ols_slope(t: Sequence[float], y: Sequence[float], *, band_c: tuple[float, float]) -> SlopeEstimate:
    """Fit `y` on `t` by ordinary least squares, and correct its standard error.

    Refuses fewer than 3 samples and refuses zero time variance (every
    timestamp equal), both with `EstimatorError`: a slope has no meaning in
    either case.

    The residual lag-1 autocorrelation `rho1` inflates the standard error when
    positive: `se_ar1 = se_ols * sqrt((1 + rho) / (1 - rho))`, with `rho`
    clamped to 0.99, and the effective sample size shrinks to
    `n_eff = n * (1 - rho) / (1 + rho)`. A non-positive `rho1` never shrinks
    the error bar: `se_ar1` stays equal to `se_ols` and `n_eff` stays equal to
    `n`. `dof` is `max(n_eff - 2, 1)`.
    """
    n = _check_equal_length(t, y)
    if n < 3:
        raise EstimatorError(f"ols_slope needs at least 3 samples, got {n}")

    t_mean = sum(t) / n
    y_mean = sum(y) / n
    sxx = sum((ti - t_mean) ** 2 for ti in t)
    if sxx == 0.0:
        raise EstimatorError("ols_slope refuses zero time variance: every sample has the same timestamp")

    sxy = sum((ti - t_mean) * (yi - y_mean) for ti, yi in zip(t, y, strict=True))
    slope = sxy / sxx
    intercept = y_mean - slope * t_mean

    residuals = [yi - (intercept + slope * ti) for ti, yi in zip(t, y, strict=True)]
    sum_sq_residuals = sum(e * e for e in residuals)
    residual_rms = math.sqrt(sum_sq_residuals / n)
    residual_variance = sum_sq_residuals / (n - 2)
    se_ols = math.sqrt(residual_variance / sxx)

    numerator = sum(residuals[i] * residuals[i + 1] for i in range(n - 1))
    rho1_raw = numerator / sum_sq_residuals if sum_sq_residuals > 0.0 else 0.0

    if rho1_raw > 0.0:
        rho1 = min(rho1_raw, 0.99)
        se_ar1 = se_ols * math.sqrt((1.0 + rho1) / (1.0 - rho1))
        n_eff = n * (1.0 - rho1) / (1.0 + rho1)
    else:
        rho1 = rho1_raw
        se_ar1 = se_ols
        n_eff = float(n)

    dof = max(n_eff - 2.0, 1.0)

    return SlopeEstimate(
        slope_c_per_s=slope,
        se_ols=se_ols,
        se_ar1=se_ar1,
        rho1=rho1,
        n=n,
        n_eff=n_eff,
        dof=dof,
        residual_rms=residual_rms,
        band_c=band_c,
        method="ols+ar1",
        t_start_mono=t[0],
        t_end_mono=t[-1],
    )


def _regularized_incomplete_beta_cf(x: float, a: float, b: float, *, max_iter: int = 200, eps: float = 1e-14) -> float:
    """Lentz's continued fraction for the regularized incomplete beta function.

    Valid for `0 < x < (a + 1) / (a + b + 2)`; the caller picks the stable
    side of the symmetry relation before calling this.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """The regularized incomplete beta function `I_x(a, b)`, by continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _regularized_incomplete_beta_cf(x, a, b) / a
    return 1.0 - front * _regularized_incomplete_beta_cf(1.0 - x, b, a) / b


def _student_t_cdf(t: float, dof: float) -> float:
    """`P(T <= t)` for a Student's t distribution with `dof` degrees of freedom."""
    x = dof / (dof + t * t)
    incomplete_beta = _regularized_incomplete_beta(x, dof / 2.0, 0.5)
    if t >= 0.0:
        return 1.0 - 0.5 * incomplete_beta
    return 0.5 * incomplete_beta


def student_t_quantile(p: float, dof: float) -> float:
    """The value `t` such that `P(T <= t) == p`, for `dof` degrees of freedom.

    Uses the regularized incomplete beta function (continued fraction) to
    evaluate the CDF, then bisects on `t` to invert it. Pure Python: no
    numpy, no scipy.
    """
    if not (0.0 < p < 1.0):
        raise EstimatorError(f"student_t_quantile needs 0 < p < 1, got {p}")
    if dof <= 0.0:
        raise EstimatorError(f"student_t_quantile needs dof > 0, got {dof}")
    if p < 0.5:
        return -student_t_quantile(1.0 - p, dof)
    if p == 0.5:
        return 0.0

    lo, hi = 0.0, 1.0
    while _student_t_cdf(hi, dof) < p:
        hi *= 2.0
        if hi > 1e15:
            raise EstimatorError("student_t_quantile did not converge")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _student_t_cdf(mid, dof) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def upper_confidence_bound(est: SlopeEstimate, confidence: float = 0.95) -> float:
    """The one-sided upper confidence bound on `est.slope_c_per_s`.

    Uses the AR(1)-corrected standard error `se_ar1` and the Student's t
    quantile at `confidence` (read as one-sided, so `confidence=0.95` uses
    `t_{0.95}`, the TOST convention).
    """
    quantile = student_t_quantile(confidence, est.dof)
    return est.slope_c_per_s + quantile * est.se_ar1


@dataclass(frozen=True, slots=True)
class TransitTime:
    """The time a series spends crossing a band, from the first `lo` sample to the first `hi` sample."""

    band_c: tuple[float, float]
    reached: bool
    seconds: float | None
    t_enter_mono: float | None
    t_exit_mono: float | None


def transit_time(t: Sequence[float], y: Sequence[float], *, band_c: tuple[float, float]) -> TransitTime:
    """The time from the first sample at or above `band_c[0]` to the first at or above `band_c[1]`.

    `reached` is `False` and `seconds` is `None` when the series never
    reaches `band_c[1]` after entering the band. Never returns a number in
    that case.
    """
    _check_equal_length(t, y)
    lo, hi = band_c
    t_enter: float | None = None
    t_exit: float | None = None
    for ti, yi in zip(t, y, strict=True):
        if t_enter is None and yi >= lo:
            t_enter = ti
        if t_enter is not None and yi >= hi:
            t_exit = ti
            break
    if t_enter is None or t_exit is None:
        return TransitTime(band_c=band_c, reached=False, seconds=None, t_enter_mono=t_enter, t_exit_mono=None)
    return TransitTime(band_c=band_c, reached=True, seconds=t_exit - t_enter, t_enter_mono=t_enter, t_exit_mono=t_exit)


@dataclass(frozen=True, slots=True)
class Peak:
    """The largest value in a series, and when it occurred."""

    value_c: float
    t_mono: float
    sensor: str
    n: int


def peak(t: Sequence[float], y: Sequence[float], *, sensor: str) -> Peak:
    """The maximum of `y` and its matching time, named for the sensor it came from."""
    n = _check_equal_length(t, y)
    if n == 0:
        raise EstimatorError("peak refuses an empty series")
    best_index = max(range(n), key=lambda i: y[i])
    return Peak(value_c=y[best_index], t_mono=t[best_index], sensor=sensor, n=n)


def mean(values: Sequence[float]) -> float:
    """The arithmetic mean of `values`. Refuses an empty sequence."""
    if not values:
        raise EstimatorError("mean refuses an empty sequence")
    return sum(values) / len(values)
