"""Within-condition scatter: how much one endpoint varies across replicates.

Two replicates is the floor below which scatter means nothing at all. This
module is stricter: `min_n` defaults to **3**, because a sample standard
deviation from 2 points carries almost no information about the true spread.
`within_condition_scatter` refuses fewer than `min_n` values outright: there is
no degraded-but-passing path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .estimators import mean


class ScatterError(ValueError):
    """Too few replicates were given to compute a within-condition scatter."""


@dataclass(frozen=True, slots=True)
class Scatter:
    """One endpoint's replicate mean and sample standard deviation."""

    endpoint: str
    n: int
    mean: float
    sd: float
    values: tuple[float, ...]


def within_condition_scatter(endpoint: str, values: Sequence[float], *, min_n: int = 3) -> Scatter:
    """The mean and sample standard deviation of `values` for one endpoint.

    Refuses `len(values) < min_n` with `ScatterError`. `min_n` defaults to 3,
    stricter than a floor of 2: two replicates give a
    standard deviation with one degree of freedom, too thin to call a
    scatter.
    """
    n = len(values)
    if n < min_n:
        raise ScatterError(
            f"within_condition_scatter needs at least {min_n} replicates for {endpoint!r} "
            f"(two is the absolute floor; this module requires 3), got {n}"
        )
    endpoint_mean = mean(values)
    variance = sum((value - endpoint_mean) ** 2 for value in values) / (n - 1)
    sd = math.sqrt(variance)
    return Scatter(endpoint=endpoint, n=n, mean=endpoint_mean, sd=sd, values=tuple(values))
