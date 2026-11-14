"""A condition: the fixed block plus one factor value, named by a stable hash.

Two runs of the same condition are recognisably replicates because they hash
to the same id. `replication.n` governs how many replicates a condition gets.
`schedule()` turns that count into the actual run order: `replication.order`
accepts only `counterbalanced`, so replicate r (1-based) runs the declared
condition order when r is odd, and the reversed order when r is even (ABBA:
the start side alternates per replicate).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .spec import ExperimentSpec, Fixed


def hash_fixed(fixed: Fixed) -> str:
    """A stable sha256 of the `fixed` block, independent of key order.

    `Fixed` leaves an absent `placement` or `server` out of its dump. Every
    spec without them keeps the digest it had before they existed. A present
    `server.extra_args` is part of the digest.
    """
    canonical = json.dumps(fixed.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Condition:
    """The fixed block plus one factor value, for one experiment."""

    experiment: str
    factor_name: str
    factor_value: str
    fixed_digest: str

    @property
    def condition_id(self) -> str:
        """A short, filesystem-safe id: stable across two runs of this condition."""
        return f"{self.experiment}-{self.factor_name}-{self.factor_value}-{self.fixed_digest[:12]}"


def conditions_for(spec: ExperimentSpec) -> tuple[Condition, ...]:
    """Every condition this spec declares: one per factor value."""
    digest = hash_fixed(spec.fixed)
    return tuple(
        Condition(
            experiment=spec.experiment,
            factor_name=spec.factor.name,
            factor_value=value,
            fixed_digest=digest,
        )
        for value in spec.factor.values
    )


@dataclass(frozen=True, slots=True)
class ScheduledRun:
    """One run this spec requires: a condition, run as replicate `replicate`."""

    condition: Condition
    replicate: int


def schedule(spec: ExperimentSpec) -> tuple[ScheduledRun, ...]:
    """Every run this spec requires, in counterbalanced (ABBA) order.

    Replicate r (1-based) runs the declared condition order when r is odd,
    and the reversed order when r is even. The schema accepts no other
    `replication.order` value, so this function never has to check it.
    """
    forward = conditions_for(spec)
    backward = tuple(reversed(forward))
    runs: list[ScheduledRun] = []
    for replicate in range(1, spec.replication.n + 1):
        ordered = forward if replicate % 2 == 1 else backward
        runs.extend(ScheduledRun(condition=condition, replicate=replicate) for condition in ordered)
    return tuple(runs)
