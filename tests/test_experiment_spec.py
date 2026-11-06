"""Tests for the declarative experiment spec schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from inferference.experiment.spec import (
    ExperimentSpec,
    PPTableArm,
    PPTableFieldDelta,
    SpecError,
    load_spec,
)


def _spec_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "experiment": "v620-thermal-self-regulation-arm-b",
        "schema_version": 1,
        "fixed": {
            "model": {"ref": "resident", "sha256_must_match": True},
            "build": {"manifest_must_match": True},
            "context": {"slots": 2, "tokens_per_slot": 40960},
            "cache": {"k": "q8_0", "v": "q8_0"},
            "batch": {"size": 1024, "ubatch": 1024},
            "control": {
                "fan_profile": "flat-max",
                "power_profile": "power-saving",
                "pp_table": {
                    "name": "arm-b",
                    "sha256": None,
                    "deltas_from_stock": [
                        {"field": "temperature_limit", "old": [100, 100, 98], "new": [85, 85, 98]},
                    ],
                },
            },
            "safety": {"machine_limit_c": 95, "arm_bound_c": 85, "gtt_limit_mib": 512},
            "start_gate": {"max_c": 24, "tolerance_c": 1, "hold_s": 30},
            "host": {"loadavg_1m_max": 2.0},
        },
        "factor": {"name": "lanes", "values": ["2", "4", "8"]},
        "workloads": ["prefill", "decode", "lane_sweep", "soak"],
        "replication": {"n": 3, "order": "counterbalanced", "cooldown_between": "start_gate"},
        "endpoints": {
            "primary": "band_slope",
            "band_c": [52, 62],
            "secondary": ["transit_45_65", "ttft", "decode_rate", "band_power"],
        },
    }
    base.update(overrides)
    return base


def test_a_well_formed_spec_round_trips_through_validation() -> None:
    spec = ExperimentSpec.model_validate(_spec_dict())

    assert spec.experiment == "v620-thermal-self-regulation-arm-b"
    assert spec.fixed.safety.machine_limit_c == 95
    assert spec.factor.values == ("2", "4", "8")
    assert spec.fixed.control.pp_table is not None
    assert spec.fixed.control.pp_table.name == "arm-b"


def test_pp_table_control_is_optional() -> None:
    raw = _spec_dict()
    raw["fixed"]["control"] = {"fan_profile": "flat-max", "power_profile": "power-saving"}

    spec = ExperimentSpec.model_validate(raw)

    assert spec.fixed.control.pp_table is None


def test_an_unknown_key_is_refused_not_ignored() -> None:
    raw = _spec_dict()
    raw["fixed"]["control"]["extra_unplanned_key"] = True

    with pytest.raises(Exception, match="extra_unplanned_key|extra fields"):
        ExperimentSpec.model_validate(raw)


def test_wrong_schema_version_is_refused() -> None:
    raw = _spec_dict(schema_version=2)

    with pytest.raises(Exception, match="schema_version"):
        ExperimentSpec.model_validate(raw)


def test_arm_bound_must_not_exceed_machine_limit() -> None:
    raw = _spec_dict()
    raw["fixed"]["safety"] = {"machine_limit_c": 80, "arm_bound_c": 80, "gtt_limit_mib": 512}

    assert ExperimentSpec.model_validate(raw).fixed.safety.arm_bound_c == 80

    raw["fixed"]["safety"] = {"machine_limit_c": 80, "arm_bound_c": 81, "gtt_limit_mib": 512}
    with pytest.raises(Exception, match="arm_bound_c"):
        ExperimentSpec.model_validate(raw)


def test_pp_table_field_delta_accepts_scalars_and_tuples() -> None:
    PPTableFieldDelta(field="socket_power_limit_ac", old=250, new=220)
    PPTableFieldDelta(field="temperature_limit", old=(100, 100, 98), new=(85, 85, 98))


def test_pp_table_arm_defaults_to_no_sha256_and_no_deltas() -> None:
    arm = PPTableArm(name="arm-a")

    assert arm.sha256 is None
    assert arm.deltas_from_stock == ()


# ---- identity fields: sha256, build_id, features ----------------------------


def test_model_sha256_round_trips_and_defaults_to_none() -> None:
    spec = ExperimentSpec.model_validate(_spec_dict())
    assert spec.fixed.model.sha256 is None

    raw = _spec_dict()
    raw["fixed"]["model"]["sha256"] = "a" * 64
    spec = ExperimentSpec.model_validate(raw)

    assert spec.fixed.model.sha256 == "a" * 64


def test_build_id_and_features_round_trip_and_default_to_none() -> None:
    spec = ExperimentSpec.model_validate(_spec_dict())
    assert spec.fixed.build.build_id is None
    assert spec.fixed.build.features is None

    raw = _spec_dict()
    raw["fixed"]["build"]["build_id"] = "vulkan-mi25-p2fork-server-ecc0cb0d3"
    raw["fixed"]["build"]["features"] = ["flash-attn", "lora-seq-routing"]
    spec = ExperimentSpec.model_validate(raw)

    assert spec.fixed.build.build_id == "vulkan-mi25-p2fork-server-ecc0cb0d3"
    assert spec.fixed.build.features == ("flash-attn", "lora-seq-routing")


# ---- control.power_cap_w: the hwmon power1_cap actuator ----------------------


def _power_cap_dict(power_cap_w: Any) -> dict[str, Any]:
    raw = _spec_dict()
    raw["fixed"]["control"] = {"fan_profile": "flat-max", "power_profile": "power-saving", "power_cap_w": power_cap_w}
    return raw


def test_power_cap_w_defaults_to_none() -> None:
    assert ExperimentSpec.model_validate(_spec_dict()).fixed.control.power_cap_w is None


@pytest.mark.parametrize("watts", [1, 119, 120, 180, 250, 251, 500])
def test_power_cap_w_accepts_positive_integers(watts: int) -> None:
    spec = ExperimentSpec.model_validate(_power_cap_dict(watts))

    assert spec.fixed.control.power_cap_w == watts
    assert spec.fixed.control.pp_table is None


@pytest.mark.parametrize("watts", [0, -5])
def test_power_cap_w_rejects_nonpositive_values(watts: int) -> None:
    with pytest.raises(Exception, match="power_cap_w.*positive"):
        ExperimentSpec.model_validate(_power_cap_dict(watts))


@pytest.mark.parametrize("watts", [True, 220.5, "220", 220.0])
def test_power_cap_w_must_be_an_integer(watts: Any) -> None:
    with pytest.raises(Exception, match="power_cap_w"):
        ExperimentSpec.model_validate(_power_cap_dict(watts))


def test_power_cap_w_and_pp_table_are_mutually_exclusive(tmp_path: Path) -> None:
    raw = _spec_dict()
    raw["fixed"]["control"]["power_cap_w"] = 220
    path = tmp_path / "both.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(SpecError, match="both pp_table and power_cap_w"):
        load_spec(path)


# ---- replication.order: counterbalanced only --------------------------------


def test_an_unknown_replication_order_fails_with_spec_error(tmp_path: Path) -> None:
    raw = _spec_dict()
    raw["replication"]["order"] = "randomized"
    path = tmp_path / "bad-order.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(SpecError, match="order"):
        load_spec(path)


# ---- load_spec: YAML file reading -------------------------------------------


def test_load_spec_reads_a_valid_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "arm-b.yaml"
    path.write_text(yaml.safe_dump(_spec_dict()))

    spec = load_spec(path)

    assert spec.experiment == "v620-thermal-self-regulation-arm-b"


def test_load_spec_refuses_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("fixed: [unterminated\n")

    with pytest.raises(SpecError, match="not valid YAML"):
        load_spec(path)


def test_load_spec_refuses_a_non_mapping_document(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- just\n- a\n- list\n")

    with pytest.raises(SpecError, match="mapping"):
        load_spec(path)


def test_load_spec_wraps_a_validation_failure(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(_spec_dict(schema_version=99)))

    with pytest.raises(SpecError):
        load_spec(path)


# ---- fixed.placement: the dual-card declaration ------------------------------


def _placement_dict(**placement: Any) -> dict[str, Any]:
    raw = _power_cap_dict(180)
    raw["fixed"]["placement"] = {
        "mode": "per-card",
        "cards": [0, 1],
        "on_arm_bound": "continue_peer",
        **placement,
    }
    return raw


def test_placement_defaults_to_none_and_stays_out_of_the_dump() -> None:
    spec = ExperimentSpec.model_validate(_spec_dict())

    assert spec.fixed.placement is None
    assert "placement" not in spec.fixed.model_dump(mode="json")


def test_a_valid_placement_loads() -> None:
    spec = ExperimentSpec.model_validate(_placement_dict())

    assert spec.fixed.placement is not None
    assert spec.fixed.placement.cards == (0, 1)
    assert spec.schema_version == 1


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"cards": [1, 1]}, "two distinct cards"),
        ({"cards": [0]}, "cards"),
        ({"cards": [0, 1, 2]}, "cards"),
        ({"cards": [-1, 1]}, "zero or greater"),
        ({"mode": "tensor-split"}, "mode"),
        ({"on_arm_bound": "stop_both"}, "on_arm_bound"),
        ({"extra": 1}, "extra"),
    ],
)
def test_a_bad_placement_is_refused(override: dict[str, Any], message: str) -> None:
    with pytest.raises(Exception, match=message):
        ExperimentSpec.model_validate(_placement_dict(**override))


def test_placement_refuses_a_pp_table_arm_that_changes_the_table() -> None:
    raw = _spec_dict()
    raw["fixed"]["placement"] = {"mode": "per-card", "cards": [0, 1], "on_arm_bound": "continue_peer"}

    with pytest.raises(Exception, match="stock pp_table"):
        ExperimentSpec.model_validate(raw)


def test_placement_accepts_the_stock_table_arm_of_the_250_w_spec() -> None:
    raw = _spec_dict()
    raw["fixed"]["control"]["pp_table"] = {"name": "cap-250w", "sha256": "a" * 64, "deltas_from_stock": []}
    raw["fixed"]["placement"] = {"mode": "per-card", "cards": [0, 1], "on_arm_bound": "continue_peer"}

    assert ExperimentSpec.model_validate(raw).fixed.placement is not None
