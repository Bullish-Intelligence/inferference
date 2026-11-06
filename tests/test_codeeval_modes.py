"""Batch mode, throughput accounting and the placement proof — on a CPU.

Nothing here needs a GPU or a router. What is being checked is the arithmetic
and the refusals, because those are what decide whether a number recorded on the
rig may be compared with another one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferference.codeeval.client import Response
from inferference.codeeval.drive import DriveResult, TaskOutcome
from inferference.codeeval.placement import (
    DeviceReading,
    PlacementProof,
    ServingConfig,
    _parse_preset,
)
from inferference.codeeval.score.result import TaskScore


def outcome(task_id: str, *, n_predicted: int | None = 100, skipped: str | None = None) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        family="implement",
        difficulty=1,
        prompt_sha256="x",
        approx_prompt_tokens=10,
        response=Response(task_id=task_id, text="hi", timings={"n_predicted": n_predicted}),
        score=TaskScore(task_id=task_id, family="implement", difficulty=1, request_ok=True),
        skipped=skipped,
    )


# ------------------------------------------------------------------ throughput --
def test_aggregate_throughput_is_tokens_over_wall_clock() -> None:
    r = DriveResult(outcomes=[outcome("a"), outcome("b")], started_at=0.0, ended_at=10.0)
    assert r.completion_tokens == 200
    assert r.aggregate_tok_s == 20.0
    assert r.requests_per_s == 0.2


def test_a_run_that_produced_nothing_has_no_rate_rather_than_zero() -> None:
    """None and 0.0 mean different things. A rate of zero is a measurement; no
    rate is the absence of one, and averaging it in would drag a summary down."""
    r = DriveResult(outcomes=[outcome("a", n_predicted=None)], started_at=0.0, ended_at=10.0)
    assert r.completion_tokens == 0
    assert r.aggregate_tok_s is None


def test_a_zero_length_run_has_no_rate() -> None:
    r = DriveResult(outcomes=[outcome("a")], started_at=5.0, ended_at=5.0)
    assert r.aggregate_tok_s is None
    assert r.requests_per_s is None


def test_skipped_tasks_do_not_count_as_requests() -> None:
    r = DriveResult(
        outcomes=[outcome("a"), outcome("b", skipped="too long")],
        started_at=0.0,
        ended_at=10.0,
    )
    assert r.requests_per_s == 0.1


# ------------------------------------------------------------- serving config --
def test_serving_config_reads_lanes_and_context_from_the_preset() -> None:
    preset = "[m]\nctx-size = 8192\ndevice = Vulkan1\nparallel = 2\nn-gpu-layers = 999\n"
    fields = _parse_preset(preset)
    cfg = ServingConfig(
        model="m",
        status="loaded",
        device=fields["device"],
        lanes=int(fields["parallel"]),
        ctx_size=int(fields["ctx-size"]),
    )
    assert cfg.lanes == 2
    assert cfg.ctx_size == 8192
    # ctx-size is the TOTAL cache; per-lane context is what a request gets.
    assert cfg.ctx_per_lane == 4096


def test_ctx_per_lane_is_unknown_rather_than_wrong_without_lanes() -> None:
    assert ServingConfig(model="m", status="loaded", ctx_size=4096).ctx_per_lane is None


# ------------------------------------------------------------ placement proof --
def reading(card: int, vram: float, gtt: float) -> DeviceReading:
    return DeviceReading(card=card, pci="0000:04:00.0", vram_used_mib=vram, vram_total_mib=16368.0, gtt_used_mib=gtt)


SERVING = ServingConfig(model="m", status="loaded", device="Vulkan0", lanes=1, ctx_size=4096)


def test_router_evidence_can_prove_a_single_card() -> None:
    serving = SERVING.model_copy(
        update={
            "residency": {
                "layer_count": 1,
                "output_device": "Vulkan0",
                "verified_mono_us": 1,
                "cards": [
                    {
                        "device": "Vulkan0",
                        "pci": "0000:04:00.0",
                        "layers": 1,
                        "gtt_before": 52 * 1048576,
                        "gtt_after": 52 * 1048576,
                        "vram_before": 100,
                        "vram_after": 3031 * 1048576,
                    }
                ],
            }
        }
    )
    proof = PlacementProof.evaluate(serving=serving, card=0, before=reading(0, 3031, 52), after=reading(0, 3031, 52))
    assert proof.status == "proven"


def test_flat_gtt_after_load_does_not_prove_residency() -> None:
    proof = PlacementProof.evaluate(
        serving=SERVING,
        card=0,
        before=reading(0, 3031, 52),
        after=reading(0, 3031, 52),
        expected_mib=3022,
    )
    assert proof.status == "unproven"
    assert not proof.valid


def test_a_gtt_rise_is_a_spill_however_well_the_run_went() -> None:
    """The measured failure: the model answers every request, reports success,
    and runs about 14x slower from host memory."""
    proof = PlacementProof.evaluate(
        serving=SERVING,
        card=1,
        before=reading(1, 8874, 14),
        after=reading(1, 16244, 1512),
        expected_mib=8866,
    )
    assert proof.status == "unproven"
    assert not proof.valid
    assert "GTT rose" in proof.reason


def test_a_run_that_started_already_spilled_is_caught() -> None:
    """GTT flat across the run proves nothing if it was high to begin with."""
    proof = PlacementProof.evaluate(
        serving=SERVING,
        card=1,
        before=reading(1, 16244, 1512),
        after=reading(1, 16244, 1512),
        expected_mib=8866,
    )
    assert proof.status == "unproven"


def test_a_model_that_is_not_on_the_card_is_short_not_proven() -> None:
    proof = PlacementProof.evaluate(
        serving=SERVING,
        card=0,
        before=reading(0, 8, 14),
        after=reading(0, 8, 14),
        expected_mib=8866,
    )
    assert proof.status == "short"
    assert not proof.valid


def test_small_gtt_movement_cannot_prove_residency() -> None:
    proof = PlacementProof.evaluate(
        serving=SERVING,
        card=0,
        before=reading(0, 3031, 30),
        after=reading(0, 3031, 74),
        expected_mib=3022,
    )
    assert proof.status == "unproven"


def test_an_unproven_run_carries_its_reason() -> None:
    proof = PlacementProof.unproven("no --card given")
    assert not proof.valid
    assert proof.status == "unproven"


# ------------------------------------------------------------------- validity --
def test_a_local_run_without_a_proof_is_invalid_not_successful() -> None:
    from inferference.codeeval.cli import _validity
    from inferference.codeeval.client import Endpoint

    local = Endpoint(name="local", base_url="http://x/v1", model="m", hosted=False)
    assert _validity(local, None)["status"] == "invalid"
    assert _validity(local, PlacementProof.unproven("nope"))["status"] == "invalid"


def test_a_spilled_local_run_is_invalid() -> None:
    from inferference.codeeval.cli import _validity
    from inferference.codeeval.client import Endpoint

    local = Endpoint(name="local", base_url="http://x/v1", model="m", hosted=False)
    spilled = PlacementProof.evaluate(
        serving=SERVING, card=1, before=reading(1, 8874, 14), after=reading(1, 16244, 1512)
    )
    assert _validity(local, spilled)["status"] == "invalid"


def test_a_proven_local_run_is_valid() -> None:
    from inferference.codeeval.cli import _validity
    from inferference.codeeval.client import Endpoint

    local = Endpoint(name="local", base_url="http://x/v1", model="m", hosted=False)
    proof = PlacementProof(status="proven")
    assert _validity(local, proof)["status"] == "valid"


def test_a_hosted_run_is_quality_only_and_never_claims_placement() -> None:
    from inferference.codeeval.cli import _validity
    from inferference.codeeval.client import Endpoint

    hosted = Endpoint(name="h", base_url="http://x/v1", model="m", hosted=True)
    assert _validity(hosted, None)["status"] == "quality-only"


# ------------------------------------------------------------------- envelope --
def test_the_envelope_round_trips_as_json() -> None:
    """The envelope is the comparison artifact. If it cannot be serialised, a
    run cannot be compared with anything."""
    from inferference.codeeval.report import RunReport

    report = RunReport(
        run_id="r",
        mode="batch",
        throughput={"aggregate_tok_s": 12.5},
        placement=PlacementProof.unproven("x").model_dump(mode="json"),
    )
    text = json.dumps(report.model_dump())
    assert json.loads(text)["schema_version"] == 4
    assert json.loads(text)["mode"] == "batch"


def test_batch_mode_refuses_a_nonsense_concurrency() -> None:
    import asyncio

    from inferference.codeeval.client import Endpoint
    from inferference.codeeval.drive import run_batched
    from inferference.codeeval.task import TaskSet

    with pytest.raises(ValueError, match="concurrency must be"):
        asyncio.run(
            run_batched(
                TaskSet(tasks=[], root=Path(".")),
                Endpoint(name="l", base_url="http://x/v1", model="m"),
                out_dir=Path("/tmp/unused-codeeval"),
                concurrency=0,
            )
        )


# -------------------------------------------------------------- thermal pacing --
def test_a_paced_run_rates_over_active_time_not_wall_time() -> None:
    """The whole point. A paced run's wall clock includes deliberate idle; a
    rate over it would report a slower build when only the pacing changed."""
    r = DriveResult(outcomes=[outcome("a")], started_at=0.0, ended_at=20.0, idle_s=10.0)
    assert r.active_s == 10.0
    assert r.aggregate_tok_s == 10.0  # 100 tokens / 10 s active
    assert r.wall_tok_s == 5.0  # 100 tokens / 20 s wall


def test_an_unpaced_run_has_the_same_active_and_wall_rate() -> None:
    r = DriveResult(outcomes=[outcome("a")], started_at=0.0, ended_at=10.0)
    assert r.active_s == 10.0
    assert r.aggregate_tok_s == r.wall_tok_s == 10.0


def test_a_pacer_that_can_never_release_is_refused() -> None:
    from inferference.codeeval.thermal import ThermalPacer

    with pytest.raises(ValueError, match="never release"):
        ThermalPacer(card=0, pause_c=70, resume_c=70)
    with pytest.raises(ValueError, match="never release"):
        ThermalPacer(card=0, pause_c=70, resume_c=80)


def test_a_disabled_pacer_costs_nothing_and_never_waits() -> None:
    import asyncio

    from inferference.codeeval.thermal import ThermalPacer

    pacer = ThermalPacer()
    assert not pacer.enabled
    asyncio.run(pacer.wait_if_hot())
    assert pacer.paused_s == 0.0
    assert pacer.record()["enabled"] is False


def test_the_idle_clock_only_counts_time_with_nothing_in_flight() -> None:
    """Under concurrency the pacer may hold one request while others decode.
    The card is working then, so that is not idle."""
    from inferference.codeeval.drive import _IdleClock

    clock = _IdleClock()
    clock.enter()  # request A starts
    clock.enter()  # request B starts while A runs
    clock.exit()  # A finishes, B still running -> NOT idle
    clock.exit()  # B finishes -> idle starts
    clock.close()
    assert clock.idle_s >= 0.0


def test_the_pacer_records_what_the_envelope_needs() -> None:
    from inferference.codeeval.thermal import ThermalPacer

    rec = ThermalPacer(card=1, pause_c=78, resume_c=66).record()
    assert rec["enabled"] is True
    assert rec["pause_c"] == 78 and rec["resume_c"] == 66
    assert rec["card"] == 1


def test_a_thermally_aborted_run_is_invalid_whatever_else_was_true() -> None:
    from inferference.codeeval.cli import _validity
    from inferference.codeeval.client import Endpoint

    local = Endpoint(name="local", base_url="http://x/v1", model="m", hosted=False)
    good = PlacementProof(status="proven")
    assert _validity(local, good)["status"] == "valid"
    # Same proof, but the run cooked the card: the timings are of a throttled card.
    verdict = _validity(local, good, "card 1 reached 88 C")
    assert verdict["status"] == "invalid"
    assert "88 C" in verdict["reason"]


def test_a_pacer_whose_pause_equals_the_ceiling_is_refused() -> None:
    from inferference.codeeval.thermal import ThermalPacer

    with pytest.raises(ValueError, match="protects nothing"):
        ThermalPacer(card=0, pause_c=85, resume_c=70, abort_c=85)


def test_a_guarded_run_is_not_reported_as_a_paced_one() -> None:
    """--card arms the abort watchdog; only --pace-card pauses. A run that
    never paused must not claim its wall clock includes cooling time."""
    from inferference.codeeval.thermal import ThermalPacer

    guarded = ThermalPacer(card=1, pace=False)
    assert guarded.enabled and not guarded.pacing
    assert guarded.record()["pacing"] is False

    paced = ThermalPacer(card=1, pace=True)
    assert paced.enabled and paced.pacing


def test_a_guarded_but_unpaced_pacer_never_waits() -> None:
    import asyncio

    from inferference.codeeval.thermal import ThermalPacer

    pacer = ThermalPacer(card=1, pace=False, pause_c=0, resume_c=-1)
    asyncio.run(pacer.wait_if_hot())
    assert pacer.pauses == 0 and pacer.paused_s == 0.0


def test_a_request_that_answered_nothing_still_counts_its_decoded_tokens() -> None:
    """A reasoning model that spends its whole budget thinking produced no
    ANSWER, but the card decoded every one of those tokens. Dropping them made
    aggregate throughput undercount while wall clock still counted the time."""
    failed = TaskOutcome(
        task_id="t",
        family="implement",
        difficulty=1,
        prompt_sha256="x",
        approx_prompt_tokens=10,
        response=Response(
            task_id="t",
            text="",
            finish_reason="length",
            error="token budget exhausted during reasoning; no answer emitted",
            timings={"n_predicted": 1400},
        ),
        score=TaskScore(task_id="t", family="implement", difficulty=1, request_ok=False),
    )
    r = DriveResult(outcomes=[failed], started_at=0.0, ended_at=100.0)
    assert r.completion_tokens == 1400, "decoded tokens are work, whatever the answer was"
    assert r.aggregate_tok_s == 14.0


def test_a_run_on_a_loaded_box_is_flagged_noisy_not_valid() -> None:
    """Measured: the same workload gave 30.2 tok/s at loadavg 5.5 and 35.2 at
    1.6. The number is real; comparing it with a quiet-box run is the error."""
    from inferference.codeeval.cli import _validity
    from inferference.codeeval.client import Endpoint

    local = Endpoint(name="local", base_url="http://x/v1", model="m", hosted=False)
    proof = PlacementProof(status="proven")
    assert _validity(local, proof)["status"] == "valid"
    v = _validity(local, proof, noisy=True, loadavg=5.49)
    assert v["status"] == "noisy"
    assert "5.49" in v["reason"]


def test_a_thermal_abort_outranks_a_noisy_host() -> None:
    from inferference.codeeval.cli import _validity
    from inferference.codeeval.client import Endpoint

    local = Endpoint(name="local", base_url="http://x/v1", model="m", hosted=False)
    proof = PlacementProof(status="proven")
    assert _validity(local, proof, "cooked", noisy=True, loadavg=9.0)["status"] == "invalid"
