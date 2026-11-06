"""CPU contract tests for the benchkit outcome aggregation.

Ported from the reference's ``tests/test_grammar_soak.py`` (deleted with the
project-24 bench removal): the soak accounting machinery lives in
``src/inferference/benchkit/gates.py`` (``aggregate_outcomes``), so its assertions follow
the code. The reference-only ``compare_decode_overhead`` helper was not ported
— the runners derive per-token comparisons from ``end_to_end`` directly.
"""

from __future__ import annotations

from inferference.benchkit.gates import aggregate_outcomes


def _outcome(
    status: str,
    *,
    completion_tokens: int,
    timings: dict[str, int] | None = None,
    wall: dict[str, int] | None = None,
    prompt_tokens: int = 5,
) -> dict[str, object]:
    return {
        "status": status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "timings_ns": timings or {},
        "wall_timings_ns": wall or {},
    }


def test_aggregate_outcomes_counts_failures_cutoffs_and_tokens() -> None:
    summary = aggregate_outcomes(
        [
            _outcome("valid", completion_tokens=2, timings={"mask_creation": 2, "mask_application": 3}),
            _outcome("malformed", completion_tokens=4, timings={"mask_creation": 4, "mask_application": 6}),
            _outcome("rejected", completion_tokens=1),
            _outcome("cutoff", completion_tokens=5),
        ]
    )

    assert summary["request_count"] == 4
    assert summary["valid_count"] == 1
    assert summary["invalid_count"] == 2  # malformed + rejected (cutoff excluded)
    assert summary["cutoff_count"] == 1
    assert summary["hard_failure_count"] == 2
    assert summary["failure_count"] == 3
    assert summary["performance_result"] == "completed"
    assert summary["structured_correctness_result"] == "incomplete"
    assert summary["status_counts"] == {"valid": 1, "malformed": 1, "rejected": 1, "cutoff": 1}
    assert summary["token_counts"] == {"prompt": 20, "completion": 12, "total": 32}
    assert summary["phase_timings_ns"]["mask_creation"]["total_ns"] == 6.0
    assert summary["phase_timings_ns"]["mask_application"]["total_ns"] == 9.0
    assert summary["wall_timings_ns"]["generation"]["total_ns"] == 0.0  # no wall recorded
    assert summary["end_to_end"]["generation_tokens_per_second"] is None  # no wall -> no rate


def test_aggregate_outcomes_rates_and_per_token_overhead() -> None:
    outcomes = [
        _outcome("valid", completion_tokens=2, wall={"generation": 40, "request": 50}),
        _outcome("valid", completion_tokens=4, wall={"generation": 40, "request": 50}),
    ]
    summary = aggregate_outcomes(outcomes)

    assert summary["structured_correctness_result"] == "pass"
    assert summary["wall_timings_ns"]["generation"]["p50_ns"] == 40.0
    assert summary["end_to_end"]["generation_ns_per_completion_token"] == 80.0 / 6.0
    assert summary["end_to_end"]["generation_tokens_per_second"] == 6 * 1_000_000_000 / 80.0
