"""Compare one endpoint between two groups of runs, with a named control audit.

The rule a comparison must follow before a delta means anything: the two
groups must each be one condition (not a mix), and whatever differs between
their declared fixed blocks must be named on purpose, not discovered by
accident. `compare` enforces both, then reports
whether the delta is resolved at the measured scatter.

**A cap sweep's arms are separate spec files**, one per pp_table variant, so
comparing arm A against arm B compares two different `fixed.control` blocks
by construction — exactly what this module exists to catch. The named
exception is `allowed_differences`: pass the dotted paths the comparison is
allowed to ignore, for example
`("control.pp_table", "safety.machine_limit_c", "safety.arm_bound_c")` to
compare two arms on everything except the table itself and the two safety
bounds that come with it. One exception never covers
`endpoints.band_c`: a differing measurement band makes the two groups'
numbers incomparable outright, so that check runs first, unconditionally,
and ignores `allowed_differences` entirely.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .load import RunRecord
from .scatter import Scatter, within_condition_scatter

_RESOLVED = "resolved"
_NOT_RESOLVED = "not resolved at this n"


class ComparisonError(ValueError):
    """Two groups could not be compared: mixed conditions, or an undeclared control difference."""


@dataclass(frozen=True, slots=True)
class Comparison:
    """One endpoint's comparison between group `a` and group `b`."""

    endpoint: str
    a: Scatter
    b: Scatter
    delta: float
    paired_delta: float | None
    scatter: float
    resolved: bool
    label: str
    allowed_differences: tuple[str, ...]


def _condition_id(records: Sequence[RunRecord], label: str) -> str:
    if not records:
        raise ComparisonError(f"group {label!r} is empty; compare needs at least one run")
    ids: set[str] = set()
    for record in records:
        if record.run is None:
            raise ComparisonError(f"group {label!r} record {record.path} has no run.json; compare needs a condition id")
        ids.add(record.run["condition"])
    if len(ids) != 1:
        raise ComparisonError(f"group {label!r} does not share one condition id: {sorted(ids)}")
    return next(iter(ids))


def _declaration(records: Sequence[RunRecord], label: str) -> Mapping[str, Any]:
    for record in records:
        if record.declaration is None:
            raise ComparisonError(
                f"group {label!r} record {record.path} has no declaration.json; compare needs declaration.fixed"
            )
    declaration = records[0].declaration
    if declaration is None:
        raise ComparisonError(f"group {label!r} record {records[0].path} has no declaration.json")  # unreachable
    return declaration


def _check_band_c(declaration_a: Mapping[str, Any], declaration_b: Mapping[str, Any]) -> None:
    """Refuse a differing `endpoints.band_c`, unconditionally — never an allowed difference."""
    band_a = tuple(declaration_a["endpoints"]["band_c"])
    band_b = tuple(declaration_b["endpoints"]["band_c"])
    if band_a != band_b:
        raise ComparisonError(
            f"endpoints.band_c differs ({band_a} vs {band_b}); a differing measurement band can never be allowed"
        )


def _diff_paths(a: Any, b: Any, path: str, allowed: frozenset[str]) -> tuple[str, ...]:
    """Every dotted path where `a` and `b` differ, excusing `allowed` paths before comparing.

    A path listed in `allowed` is excused at the exact depth it names, even
    when both sides are dicts: the comparison never recurses past an excused
    path, so `allowed_differences=("control.pp_table",)` excuses the whole
    `pp_table` block in one step, whether one side is `None` or a dict.
    """
    if path and path in allowed:
        return ()
    if isinstance(a, dict) and isinstance(b, dict):
        dict_a: dict[str, Any] = a
        dict_b: dict[str, Any] = b
        diffs: list[str] = []
        for key in sorted(set(dict_a) | set(dict_b)):
            child_path = f"{path}.{key}" if path else key
            diffs.extend(_diff_paths(dict_a.get(key), dict_b.get(key), child_path, allowed))
        return tuple(diffs)
    if a != b:
        return (path,)
    return ()


def _check_fixed_fields(
    declaration_a: Mapping[str, Any], declaration_b: Mapping[str, Any], allowed_differences: tuple[str, ...]
) -> None:
    diffs = _diff_paths(declaration_a["fixed"], declaration_b["fixed"], "", frozenset(allowed_differences))
    if diffs:
        raise ComparisonError(
            f"declaration.fixed differs at {', '.join(diffs)}, and none of these is in allowed_differences "
            f"{allowed_differences}"
        )


def _paired_delta(
    a: Sequence[RunRecord], b: Sequence[RunRecord], values_a: Sequence[float], values_b: Sequence[float]
) -> float | None:
    """The mean of per-replicate `(b_r - a_r)`, when both groups' replicate numbers align exactly.

    Judgment call: the interface takes no explicit replicate index, so
    alignment is read from each record's own `run.json["replicate"]`. When
    the two groups do not name the same set of replicate numbers — or either
    group repeats one — there is no well-defined pairing, and this returns
    `None` rather than guessing an order.
    """
    try:
        replicates_a = _replicate_values(a, values_a)
        replicates_b = _replicate_values(b, values_b)
    except ValueError:
        return None
    if replicates_a is None or replicates_b is None:
        return None
    if len(replicates_a) != len(a) or len(replicates_b) != len(b):
        return None  # a duplicate replicate number inside one group: no well-defined pairing
    shared = sorted(set(replicates_a) & set(replicates_b))
    if not shared or set(replicates_a) != set(replicates_b):
        return None
    deltas = [replicates_b[r] - replicates_a[r] for r in shared]
    return sum(deltas) / len(deltas)


def _replicate_values(records: Sequence[RunRecord], values: Sequence[float]) -> dict[int, float] | None:
    """Map each record's declared replicate number to its value, or `None` if any record lacks `run.json`."""
    result: dict[int, float] = {}
    for record, value in zip(records, values, strict=True):
        if record.run is None:
            return None
        result[record.run["replicate"]] = value
    return result


def compare(
    a: Sequence[RunRecord],
    b: Sequence[RunRecord],
    *,
    endpoint: str,
    values_a: Sequence[float],
    values_b: Sequence[float],
    allowed_differences: tuple[str, ...] = (),
) -> Comparison:
    """Compare `endpoint` between group `a` and group `b`, auditing the control state first.

    Refuses (`ComparisonError`) when either group mixes more than one
    condition id, when `endpoints.band_c` differs between the groups'
    declarations, or when `declaration.fixed` differs at a dotted path not
    named in `allowed_differences`. Aggregation is always the mean of
    replicates (`within_condition_scatter`); there is no "best of" path.
    """
    _condition_id(a, "a")
    _condition_id(b, "b")
    declaration_a = _declaration(a, "a")
    declaration_b = _declaration(b, "b")
    _check_band_c(declaration_a, declaration_b)
    _check_fixed_fields(declaration_a, declaration_b, allowed_differences)

    scatter_a = within_condition_scatter(endpoint, values_a)
    scatter_b = within_condition_scatter(endpoint, values_b)

    delta = scatter_b.mean - scatter_a.mean
    scatter = max(scatter_a.sd, scatter_b.sd)
    resolved = abs(delta) >= 2.0 * scatter
    label = _RESOLVED if resolved else _NOT_RESOLVED

    return Comparison(
        endpoint=endpoint,
        a=scatter_a,
        b=scatter_b,
        delta=delta,
        paired_delta=_paired_delta(a, b, values_a, values_b),
        scatter=scatter,
        resolved=resolved,
        label=label,
        allowed_differences=allowed_differences,
    )
