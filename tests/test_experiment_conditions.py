"""Tests for condition identity: the fixed block plus one factor value."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inferference.experiment.conditions import Condition, conditions_for, hash_fixed, schedule
from inferference.experiment.spec import ExperimentSpec, load_spec

from .test_experiment_spec import _spec_dict


def test_hash_fixed_is_stable_regardless_of_key_order() -> None:
    spec_a = ExperimentSpec.model_validate(_spec_dict())
    raw_b = _spec_dict()
    raw_b["fixed"] = dict(reversed(list(raw_b["fixed"].items())))
    spec_b = ExperimentSpec.model_validate(raw_b)

    assert hash_fixed(spec_a.fixed) == hash_fixed(spec_b.fixed)


def test_hash_fixed_changes_when_a_fixed_field_changes() -> None:
    spec_a = ExperimentSpec.model_validate(_spec_dict())
    raw_b = _spec_dict()
    raw_b["fixed"]["safety"]["gtt_limit_mib"] = 256
    spec_b = ExperimentSpec.model_validate(raw_b)

    assert hash_fixed(spec_a.fixed) != hash_fixed(spec_b.fixed)


def test_conditions_for_yields_one_condition_per_factor_value() -> None:
    spec = ExperimentSpec.model_validate(_spec_dict())

    conditions = conditions_for(spec)

    assert [condition.factor_value for condition in conditions] == ["2", "4", "8"]
    assert all(condition.factor_name == "lanes" for condition in conditions)
    assert all(condition.experiment == spec.experiment for condition in conditions)


def test_conditions_for_the_same_spec_share_one_fixed_digest() -> None:
    spec = ExperimentSpec.model_validate(_spec_dict())

    conditions = conditions_for(spec)

    assert len({condition.fixed_digest for condition in conditions}) == 1


def test_condition_id_is_stable_and_distinguishes_factor_values() -> None:
    spec = ExperimentSpec.model_validate(_spec_dict())
    two_lanes, four_lanes, _ = conditions_for(spec)

    assert two_lanes.condition_id != four_lanes.condition_id
    assert (
        two_lanes.condition_id
        == Condition(
            experiment=two_lanes.experiment,
            factor_name=two_lanes.factor_name,
            factor_value=two_lanes.factor_value,
            fixed_digest=two_lanes.fixed_digest,
        ).condition_id
    )


# ---- schedule(): counterbalanced (ABBA) run order ---------------------------


def test_schedule_counterbalances_three_replicates_over_three_values() -> None:
    # The base spec already declares n=3 over values 2, 4, 8.
    spec = ExperimentSpec.model_validate(_spec_dict())

    runs = schedule(spec)

    assert [run.condition.factor_value for run in runs] == ["2", "4", "8", "8", "4", "2", "2", "4", "8"]
    assert [run.replicate for run in runs] == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_schedule_yields_n_times_the_number_of_conditions() -> None:
    spec = ExperimentSpec.model_validate(_spec_dict())

    runs = schedule(spec)

    assert len(runs) == spec.replication.n * len(spec.factor.values)


def test_schedule_every_run_shares_the_spec_s_conditions() -> None:
    spec = ExperimentSpec.model_validate(_spec_dict())

    conditions = set(conditions_for(spec))
    runs = schedule(spec)

    assert all(run.condition in conditions for run in runs)


_ROOT = Path(__file__).resolve().parents[1]
_SPECS = Path(
    os.environ.get("INFERFERENCE_EXPERIMENT_SPECS", "~/.local/share/inferference/experiment-specs")
).expanduser()


def _spec(path: Path):
    """Load one of the maintainers' spec files, or skip when they are absent.

    The experiment specs are machine-specific and are NOT shipped in this
    repository. They are supplied with INFERFERENCE_EXPERIMENT_SPECS.
    """
    if not path.is_file():
        pytest.skip(
            f"experiment spec not present at {path}: the specs are not redistributed with this "
            "repository; set INFERFERENCE_EXPERIMENT_SPECS to the directory that holds them"
        )
    return load_spec(path)


# Computed from `032-v620-board-power-cap/arm-<W>w.yaml` before
# `fixed.placement` existed. A change to one of these digests means a change to
# the identity of every run already on disk.
_GOLDEN_032_FIXED_DIGESTS = {
    160: "eb321ea886c153b00523d17e3a125f8d914e652815c1df27e1c18c7701027952",
    180: "0a56f4d279fe36702620dd7d584950ec2810c3d160b7aafaf5c02065b89d6f16",
    250: "328317ee3bc06d598e3c7147898affc13bd92deae0e309522cecae064a4fb162",
}


def test_hash_fixed_keeps_the_digest_of_every_032_arm() -> None:
    for watts, digest in _GOLDEN_032_FIXED_DIGESTS.items():
        spec = _spec(_SPECS / "032-v620-board-power-cap" / f"arm-{watts}w.yaml")
        assert hash_fixed(spec.fixed) == digest, f"arm {watts} W changed its fixed digest"
        assert "placement" not in spec.fixed.model_dump(mode="json")


# Computed from the single-card arms on `main` at 7220e5cb, before
# `fixed.placement` existed. The 160, 180, and 250 W arms share the 032 digests.
_GOLDEN_033_FIXED_DIGESTS = {
    **_GOLDEN_032_FIXED_DIGESTS,
    200: "87c87996458eb2f3ea2e314842cf9bbcf5799039dcf30f5c7356e74b7f040ec0",
    220: "c72146969110d1c68cd32e4c55f0b7004d8bb9e329ee506dc8c14cae1c9c9d1e",
}


def test_hash_fixed_keeps_the_digest_of_every_033_single_card_arm() -> None:
    for watts, digest in _GOLDEN_033_FIXED_DIGESTS.items():
        spec = _spec(_SPECS / "033-v620-cooling-v2-cap" / f"arm-{watts}w.yaml")
        assert hash_fixed(spec.fixed) == digest, f"arm {watts} W changed its fixed digest"
        assert "placement" not in spec.fixed.model_dump(mode="json")


def test_hash_fixed_changes_when_a_placement_is_declared() -> None:
    plain = ExperimentSpec.model_validate(_spec_dict())
    raw = _spec_dict()
    raw["fixed"]["control"] = {"fan_profile": "flat-max", "power_profile": "power-saving", "power_cap_w": 180}
    raw["fixed"]["placement"] = {"mode": "per-card", "cards": [0, 1], "on_arm_bound": "continue_peer"}
    paired = ExperimentSpec.model_validate(raw)
    raw["fixed"]["placement"]["cards"] = [1, 0]
    swapped = ExperimentSpec.model_validate(raw)

    assert hash_fixed(plain.fixed) != hash_fixed(paired.fixed)
    assert hash_fixed(paired.fixed) != hash_fixed(swapped.fixed)
    assert paired.fixed.model_dump(mode="json")["placement"]["cards"] == [0, 1]
