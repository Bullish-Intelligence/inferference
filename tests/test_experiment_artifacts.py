"""Tests for the write-once run directory."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from inferference.experiment.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    RunDirectoryError,
    TelemetryWriter,
    create_run_directory,
    run_directory_name,
    write_blob,
    write_control,
    write_declaration,
    write_identity,
    write_json_artifact,
    write_outcome,
    write_preflight,
    write_request_body,
    write_residency,
    write_resolved_command,
    write_response_body,
    write_run_manifest,
    write_stages,
    write_text_artifact,
)
from inferference.experiment.conditions import Condition
from inferference.experiment.spec import ExperimentSpec
from inferference.telemetry import TelemetrySample

CONDITION = Condition(
    experiment="v620-thermal-self-regulation-arm-b",
    factor_name="lanes",
    factor_value="4",
    fixed_digest="a" * 64,
)
UTC_TIME = datetime(2026, 10, 7, 19, 0, 0, tzinfo=UTC)


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
            "control": {"fan_profile": "flat-max", "power_profile": "power-saving"},
            "safety": {"machine_limit_c": 95, "arm_bound_c": 85, "gtt_limit_mib": 512},
            "start_gate": {"max_c": 24, "tolerance_c": 1, "hold_s": 30},
            "host": {"loadavg_1m_max": 2.0},
        },
        "factor": {"name": "lanes", "values": ["2", "4", "8"]},
        "workloads": ["prefill_2048", "decode_single_stream"],
        "replication": {"n": 1, "order": "counterbalanced", "cooldown_between": "start_gate"},
        "endpoints": {"primary": "band_slope", "band_c": [52, 62]},
    }
    base.update(overrides)
    return base


SPEC = ExperimentSpec.model_validate(_spec_dict())


def _sample(monotonic: float = 0.0) -> TelemetrySample:
    return TelemetrySample(utc="2026-10-07T19:00:00+00:00", monotonic=monotonic, cards=(), fan_ducts=(), watchdog=None)


def test_run_directory_name_carries_timestamp_experiment_condition_and_replicate() -> None:
    name = run_directory_name(UTC_TIME, CONDITION.experiment, CONDITION, replicate=2)

    assert name == f"20261007T190000Z-v620-thermal-self-regulation-arm-b-{CONDITION.fixed_digest[:12]}-r2"


def test_create_run_directory_makes_a_fresh_empty_tree(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)

    assert run_dir.is_dir()
    assert (run_dir / "requests").is_dir()


def test_create_run_directory_refuses_to_reuse_an_existing_one(tmp_path: Path) -> None:
    create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)

    with pytest.raises(RunDirectoryError, match="write-once"):
        create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)


def test_write_json_artifact_writes_valid_json(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)

    path = write_json_artifact(run_dir, "outcome.json", {"state": "completed"})

    assert json.loads(path.read_text()) == {"state": "completed"}


def test_write_json_artifact_refuses_to_overwrite(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    write_json_artifact(run_dir, "outcome.json", {"state": "completed"})

    with pytest.raises(RunDirectoryError, match="write-once"):
        write_json_artifact(run_dir, "outcome.json", {"state": "aborted"})


def test_write_run_manifest_names_the_condition(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)

    path = write_run_manifest(
        run_dir,
        experiment=CONDITION.experiment,
        condition=CONDITION,
        replicate=1,
        spec_sha256="b" * 64,
    )

    manifest = json.loads(path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["condition"] == CONDITION.condition_id
    assert manifest["factor_value"] == "4"
    assert manifest["spec_sha256"] == "b" * 64


def test_write_declaration_carries_the_arm_bound_and_machine_limit(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)

    path = write_declaration(run_dir, spec=SPEC, condition=CONDITION, spec_sha256="c" * 64)

    declaration = json.loads(path.read_text())
    assert declaration["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert declaration["experiment"] == SPEC.experiment
    assert declaration["fixed"]["safety"]["arm_bound_c"] == 85
    assert declaration["fixed"]["safety"]["machine_limit_c"] == 95
    assert declaration["factor"] == {"name": "lanes", "value": "4"}
    assert declaration["workloads"] == ["prefill_2048", "decode_single_stream"]


def test_write_declaration_refuses_a_second_write(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    write_declaration(run_dir, spec=SPEC, condition=CONDITION, spec_sha256="c" * 64)

    with pytest.raises(RunDirectoryError, match="write-once"):
        write_declaration(run_dir, spec=SPEC, condition=CONDITION, spec_sha256="c" * 64)


def test_write_preflight_refuses_a_second_write(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    write_preflight(run_dir, {"checks": []})

    with pytest.raises(RunDirectoryError, match="write-once"):
        write_preflight(run_dir, {"checks": []})


def test_write_resolved_command_writes_argv_env_and_cwd(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)

    path = write_resolved_command(
        run_dir,
        argv=["llama-server", "--port", "8080"],
        env={"GGML_VK_VISIBLE_DEVICES": "0"},
        cwd="/tmp",
    )

    resolved = json.loads(path.read_text())
    assert resolved["argv"] == ["llama-server", "--port", "8080"]
    assert resolved["env"] == {"GGML_VK_VISIBLE_DEVICES": "0"}
    assert resolved["cwd"] == "/tmp"


def test_write_resolved_command_refuses_a_second_write(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    write_resolved_command(run_dir, argv=["x"], env={}, cwd="/tmp")

    with pytest.raises(RunDirectoryError, match="write-once"):
        write_resolved_command(run_dir, argv=["x"], env={}, cwd="/tmp")


def test_write_identity_refuses_a_second_write(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    write_identity(run_dir, {"repo_revision": "abc"})

    with pytest.raises(RunDirectoryError, match="write-once"):
        write_identity(run_dir, {"repo_revision": "abc"})


def test_write_residency_refuses_a_second_write(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    write_residency(run_dir, {"checks": []})

    with pytest.raises(RunDirectoryError, match="write-once"):
        write_residency(run_dir, {"checks": []})


def test_write_stages_refuses_a_second_write(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    write_stages(run_dir, {"stages": []})

    with pytest.raises(RunDirectoryError, match="write-once"):
        write_stages(run_dir, {"stages": []})


def test_write_control_refuses_a_second_write(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    write_control(run_dir, {"fan": {}})

    with pytest.raises(RunDirectoryError, match="write-once"):
        write_control(run_dir, {"fan": {}})


def test_write_outcome_writes_the_state_and_boundary(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)

    path = write_outcome(
        run_dir,
        state="completed",
        failure_boundary=None,
        detail=None,
        started_utc="2026-10-07T19:00:00Z",
        ended_utc="2026-10-07T19:05:00Z",
    )

    outcome = json.loads(path.read_text())
    assert outcome["state"] == "completed"
    assert outcome["failure_boundary"] is None


def test_write_outcome_refuses_a_second_write(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    write_outcome(run_dir, state="completed", failure_boundary=None, detail=None, started_utc="s", ended_utc="e")

    with pytest.raises(RunDirectoryError, match="write-once"):
        write_outcome(run_dir, state="completed", failure_boundary=None, detail=None, started_utc="s", ended_utc="e")


def test_write_outcome_refuses_an_unknown_state(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)

    with pytest.raises(RunDirectoryError, match="not one of"):
        write_outcome(
            run_dir,
            state="cancelled",  # type: ignore
            failure_boundary=None,
            detail=None,
            started_utc="s",
            ended_utc="e",
        )


def test_write_outcome_refuses_a_null_boundary_paired_with_aborted(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)

    with pytest.raises(RunDirectoryError, match="failure_boundary"):
        write_outcome(run_dir, state="aborted", failure_boundary=None, detail=None, started_utc="s", ended_utc="e")


def test_write_text_artifact_refuses_a_second_write(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    path = write_text_artifact(run_dir, "server.log", "line one\n")

    assert path.read_text() == "line one\n"
    with pytest.raises(RunDirectoryError, match="write-once"):
        write_text_artifact(run_dir, "server.log", "line two\n")


def test_write_blob_returns_sha256_and_refuses_a_second_write(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    data = b"\x01\x02\x03pp_table bytes"

    digest = write_blob(run_dir, "pp_table.bin", data)

    assert digest == hashlib.sha256(data).hexdigest()
    assert (run_dir / "pp_table.bin").read_bytes() == data
    with pytest.raises(RunDirectoryError, match="write-once"):
        write_blob(run_dir, "pp_table.bin", data)


def test_write_request_body_is_idempotent_for_equal_bytes(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    payload = {"model": "resident", "seed": 1}

    first = write_request_body(run_dir, payload)
    second = write_request_body(run_dir, payload)

    assert first == second
    assert (run_dir / "requests" / f"{first}.request.json").is_file()


def test_write_request_body_refuses_unequal_bytes_under_one_name(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    payload = {"model": "resident", "seed": 1}
    digest = write_request_body(run_dir, payload)
    target = run_dir / "requests" / f"{digest}.request.json"
    target.write_bytes(b'{"tampered": true}')

    with pytest.raises(RunDirectoryError, match="content"):
        write_request_body(run_dir, payload)


def test_write_response_body_is_idempotent_for_equal_bytes(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    raw = b'data: {"ok": true}\n\ndata: [DONE]\n\n'

    first = write_response_body(run_dir, raw)
    second = write_response_body(run_dir, raw)

    assert first == second
    assert (run_dir / "requests" / f"{first}.response.sse").read_bytes() == raw


def test_write_response_body_refuses_unequal_bytes_under_one_name(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    raw = b'data: {"ok": true}\n\n'
    digest = write_response_body(run_dir, raw)
    target = run_dir / "requests" / f"{digest}.response.sse"
    target.write_bytes(b"tampered bytes")

    with pytest.raises(RunDirectoryError, match="content"):
        write_response_body(run_dir, raw)


def test_concurrent_equal_response_writes_both_succeed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two lanes can write equal bytes at once; the lost link race is agreement, not an error."""
    from inferference.experiment import artifacts

    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    raw = b'data: {"ok": true}\n\ndata: [DONE]\n\n'
    real_write_once = artifacts._write_once_bytes

    def lose_the_race(path: Path, data: bytes) -> None:
        real_write_once(path, data)  # the other lane links first ...
        real_write_once(path, data)  # ... so this lane's link finds the file

    monkeypatch.setattr(artifacts, "_write_once_bytes", lose_the_race)

    digest = write_response_body(run_dir, raw)

    assert (run_dir / "requests" / f"{digest}.response.sse").read_bytes() == raw


def test_concurrent_unequal_writes_under_one_name_still_refuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferference.experiment import artifacts

    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    raw = b'data: {"ok": true}\n\n'
    digest = hashlib.sha256(raw).hexdigest()
    target = run_dir / "requests" / f"{digest}.response.sse"
    real_write_once = artifacts._write_once_bytes

    def lose_to_different_bytes(path: Path, data: bytes) -> None:
        real_write_once(path, b"another lane wrote other bytes")
        real_write_once(path, data)

    monkeypatch.setattr(artifacts, "_write_once_bytes", lose_to_different_bytes)

    with pytest.raises(RunDirectoryError):
        write_response_body(run_dir, raw)
    assert target.read_bytes() == b"another lane wrote other bytes"


def test_telemetry_writer_writes_one_line_per_sample(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)

    with TelemetryWriter(run_dir) as writer:
        writer.write(_sample(monotonic=0.0))
        writer.write(_sample(monotonic=0.25))

    lines = (run_dir / "telemetry.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["monotonic"] == 0.0
    assert json.loads(lines[1])["monotonic"] == 0.25


def test_telemetry_writer_refuses_an_existing_file(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, CONDITION.experiment, CONDITION, replicate=1, utc=UTC_TIME)
    (run_dir / "telemetry.jsonl").write_text("")

    with pytest.raises(RunDirectoryError, match="write-once"):
        TelemetryWriter(run_dir)
