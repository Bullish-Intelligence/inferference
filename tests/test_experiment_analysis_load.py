"""Tests for `inferference.experiment.analysis.load`, against the two fixture layouts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from inferference.experiment.analysis.load import LoadError, load_run

FIXTURES = Path(__file__).parent / "fixtures" / "experiment"


def test_loads_030_v1_layout() -> None:
    record = load_run(FIXTURES / "run-030-v1")

    assert record.layout == "030-v1"
    assert record.run is not None
    assert record.run["schema_version"] == 1
    assert record.declaration is not None
    assert record.stages is not None
    assert record.outcome is not None
    assert len(record.observations) == 3

    first = record.observations[0]
    assert first.clock == "monotonic"
    assert first.mono == 2.0
    card0 = first.cards[0]
    assert card0.junction_c == 52.0
    assert card0.edge_c == 45.0
    assert card0.mem_c == 48.0
    assert card0.power_w == 95.0
    assert card0.gpu_busy_percent == 80
    assert card0.throttled is False
    assert card0.gtt_used_mib == 0.0
    card1 = first.cards[1]
    assert card1.power_w == 30.0


def test_loads_legacy_layout() -> None:
    record = load_run(FIXTURES / "legacy-pre030")

    assert record.layout == "pre030-v1"
    assert record.run is None
    assert record.declaration is None
    assert record.stages is None
    assert record.outcome is None
    assert len(record.observations) == 3

    first, second, third = record.observations
    assert first.clock == "utc-derived"
    assert first.mono == 0.0
    assert second.mono == pytest.approx(5.0)
    assert third.mono == pytest.approx(10.0)

    card0 = first.cards[0]
    assert card0.junction_c == 52.0
    assert card0.edge_c == 45.0
    assert card0.mem_c == 48.0
    assert card0.power_w == 95.0
    assert card0.gpu_busy_percent is None
    assert card0.throttled is None
    assert card0.gtt_used_mib == 0.0

    second_card0 = second.cards[0]
    assert second_card0.gtt_used_mib == pytest.approx(1.0)


def _copy_fixture(src_name: str, dest: Path) -> Path:
    shutil.copytree(FIXTURES / src_name, dest)
    return dest


def test_unknown_schema_version_on_run_json_refuses(tmp_path: Path) -> None:
    run_dir = _copy_fixture("run-030-v1", tmp_path / "run")
    run_json = run_dir / "run.json"
    payload = json.loads(run_json.read_text())
    payload["schema_version"] = 2
    run_json.write_text(json.dumps(payload))

    with pytest.raises(LoadError, match="schema_version"):
        load_run(run_dir)


def test_unknown_schema_version_on_telemetry_line_refuses(tmp_path: Path) -> None:
    run_dir = _copy_fixture("run-030-v1", tmp_path / "run")
    telemetry = run_dir / "telemetry.jsonl"
    lines = telemetry.read_text().splitlines()
    first = json.loads(lines[0])
    first["schema_version"] = 2
    lines[0] = json.dumps(first)
    telemetry.write_text("\n".join(lines) + "\n")

    with pytest.raises(LoadError, match="schema_version"):
        load_run(run_dir)


def test_mixed_directory_refuses(tmp_path: Path) -> None:
    run_dir = _copy_fixture("run-030-v1", tmp_path / "run")
    (run_dir / "summary.json").write_text("{}")

    with pytest.raises(LoadError, match="mixed"):
        load_run(run_dir)


def test_neither_shape_refuses(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(LoadError, match="run.json"):
        load_run(empty_dir)


def test_legacy_without_telemetry_refuses_naming_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "partial"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}")

    with pytest.raises(LoadError, match="telemetry.jsonl"):
        load_run(run_dir)
