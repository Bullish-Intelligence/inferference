"""Tests for `inferference.experiment.analysis.compare`: synthetic runs, no files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inferference.experiment.analysis.compare import ComparisonError, compare
from inferference.experiment.analysis.load import RunRecord


def _fixed(
    *,
    cache_k: str = "q8_0",
    pp_table: dict[str, Any] | None = None,
    machine_limit_c: int = 95,
    arm_bound_c: int = 90,
) -> dict[str, Any]:
    return {
        "model": {"ref": "resident", "sha256_must_match": True, "sha256": None},
        "build": {"manifest_must_match": True, "build_id": None, "features": None},
        "context": {"slots": 4, "tokens_per_slot": 4096},
        "cache": {"k": cache_k, "v": "f16"},
        "batch": {"size": 2048, "ubatch": 512},
        "control": {"fan_profile": "flat-max", "power_profile": "power-saving", "pp_table": pp_table},
        "safety": {"machine_limit_c": machine_limit_c, "arm_bound_c": arm_bound_c, "gtt_limit_mib": 512},
        "start_gate": {"max_c": 24, "tolerance_c": 1, "hold_s": 30},
        "host": {"loadavg_1m_max": 2.0},
    }


def _declaration(fixed: dict[str, Any], *, band_c: tuple[int, int] = (20, 90)) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "written_utc": "2026-10-07T00:00:00+00:00",
        "experiment": "032-thermal",
        "spec_sha256": "deadbeef",
        "fixed": fixed,
        "factor": {"name": "lanes", "value": "4"},
        "workloads": ["soak_300s"],
        "replication": {"n": 3, "order": "counterbalanced", "cooldown_between": "300s"},
        "endpoints": {"primary": "decode_tok_s", "band_c": list(band_c), "secondary": []},
    }


def _group(condition: str, declaration: dict[str, Any], n: int = 3) -> list[RunRecord]:
    return [
        RunRecord(
            path=Path(f"run-{condition}-{replicate}"),
            layout="030-v1",
            run={
                "schema_version": 1,
                "experiment": "032-thermal",
                "condition": condition,
                "factor_name": "lanes",
                "factor_value": "4",
                "fixed_digest": "x",
                "replicate": replicate,
                "spec_sha256": "deadbeef",
            },
            declaration=declaration,
            stages=None,
            outcome=None,
            observations=(),
        )
        for replicate in range(1, n + 1)
    ]


def test_a_differing_cache_k_refuses() -> None:
    a = _group("cond-a", _declaration(_fixed(cache_k="q8_0")))
    b = _group("cond-b", _declaration(_fixed(cache_k="f16")))

    with pytest.raises(ComparisonError, match="cache.k"):
        compare(a, b, endpoint="decode_tok_s", values_a=[1.0, 2.0, 3.0], values_b=[4.0, 5.0, 6.0])


def test_the_same_difference_passes_when_named_in_allowed_differences() -> None:
    a = _group("cond-a", _declaration(_fixed(cache_k="q8_0")))
    b = _group("cond-b", _declaration(_fixed(cache_k="f16")))

    result = compare(
        a,
        b,
        endpoint="decode_tok_s",
        values_a=[1.0, 2.0, 3.0],
        values_b=[4.0, 5.0, 6.0],
        allowed_differences=("cache.k",),
    )

    assert result.endpoint == "decode_tok_s"
    assert result.allowed_differences == ("cache.k",)


def test_a_band_c_difference_refuses_even_when_listed_in_allowed_differences() -> None:
    a = _group("cond-a", _declaration(_fixed(), band_c=(20, 90)))
    b = _group("cond-b", _declaration(_fixed(), band_c=(20, 95)))

    with pytest.raises(ComparisonError, match="band_c"):
        compare(
            a,
            b,
            endpoint="decode_tok_s",
            values_a=[1.0, 2.0, 3.0],
            values_b=[4.0, 5.0, 6.0],
            allowed_differences=("endpoints.band_c",),
        )


def test_a_delta_of_1_5x_scatter_is_labelled_not_resolved_at_this_n() -> None:
    a = _group("cond-a", _declaration(_fixed()))
    b = _group("cond-b", _declaration(_fixed()))

    # values_a is constant (sd=0); values_b has sd=2.0 and mean 3.0 above
    # values_a's mean, so scatter = max(0, 2.0) = 2.0 and delta = 3.0 = 1.5x.
    result = compare(a, b, endpoint="decode_tok_s", values_a=[9.0, 9.0, 9.0], values_b=[10.0, 12.0, 14.0])

    assert result.delta == pytest.approx(3.0)
    assert result.scatter == pytest.approx(2.0)
    assert result.resolved is False
    assert result.label == "not resolved at this n"


def test_a_delta_at_least_twice_the_scatter_is_labelled_resolved() -> None:
    a = _group("cond-a", _declaration(_fixed()))
    b = _group("cond-b", _declaration(_fixed()))

    # Same scatter as above (2.0), but delta is now exactly 2x it.
    result = compare(a, b, endpoint="decode_tok_s", values_a=[8.0, 8.0, 8.0], values_b=[10.0, 12.0, 14.0])

    assert result.delta == pytest.approx(4.0)
    assert result.resolved is True
    assert result.label == "resolved"


def test_mixed_condition_ids_in_one_group_refuse() -> None:
    a = _group("cond-a", _declaration(_fixed()))
    odd_run = {
        "schema_version": 1,
        "experiment": "032-thermal",
        "condition": "cond-a-other",
        "factor_name": "lanes",
        "factor_value": "4",
        "fixed_digest": "x",
        "replicate": 2,
        "spec_sha256": "deadbeef",
    }
    a[1] = RunRecord(
        path=a[1].path,
        layout=a[1].layout,
        run=odd_run,
        declaration=a[1].declaration,
        stages=a[1].stages,
        outcome=a[1].outcome,
        observations=a[1].observations,
    )
    b = _group("cond-b", _declaration(_fixed()))

    with pytest.raises(ComparisonError, match="does not share one condition id"):
        compare(a, b, endpoint="decode_tok_s", values_a=[1.0, 2.0, 3.0], values_b=[4.0, 5.0, 6.0])


def test_the_project_032_two_axis_arm_comparison_call_shape_is_supported() -> None:
    # The arms are separate spec files: arm A carries no pp_table
    # and the stock safety bounds; arm B carries a named pp_table variant and
    # its own bounds. Comparing them needs exactly these three paths allowed.
    fixed_a = _fixed(pp_table=None, machine_limit_c=95, arm_bound_c=90)
    fixed_b = _fixed(
        pp_table={"name": "arm-b", "sha256": "abc123", "deltas_from_stock": []},
        machine_limit_c=99,
        arm_bound_c=92,
    )
    a = _group("arm-a", _declaration(fixed_a))
    b = _group("arm-b", _declaration(fixed_b))

    result = compare(
        a,
        b,
        endpoint="decode_tok_s",
        values_a=[1.0, 2.0, 3.0],
        values_b=[4.0, 5.0, 6.0],
        allowed_differences=("control.pp_table", "safety.machine_limit_c", "safety.arm_bound_c"),
    )

    assert result.resolved is True
