"""Test the experiment command interface with fixtures and fake hosts."""

from __future__ import annotations

import json
import os
import runpy
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from inferference.experiment import cli
from inferference.experiment.analysis.load import load_run
from tests.test_experiment_preflight import _build_host as _build_preflight_host
from tests.test_experiment_runner import _FakeResidentLoader, _make_host

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "experiment"
_RUN_FIXTURE = _FIXTURES / "run-030-v1"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_atomic(path, json.dumps(value, sort_keys=True) + "\n")


def _target_args(target: Any, spec_path: Path, artifact_root: Path | None = None) -> list[str]:
    args = [
        "--card",
        str(target.card),
        "--port",
        str(target.port),
        "--server-bin",
        str(target.server_bin),
        "--config-root",
        str(target.config_root),
        "--stock-pp-table",
        str(target.stock_pp_table),
        "--thermal-guard",
        str(target.thermal_guard),
        "--gpu-lease",
        str(target.gpu_lease),
        "--expect-pci",
        str(target.expect_pci),
        "--stall-floor",
        "2=1000",
    ]
    if artifact_root is not None:
        args.extend(("--artifact-root", str(artifact_root)))
    return [str(spec_path), *args]


@pytest.mark.parametrize("command", ("preflight", "run", "analyze", "compare"))
def test_each_subcommand_help_returns_zero(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([command, "--help"]) == 0
    assert "usage:" in capsys.readouterr().out


def test_unknown_subcommand_returns_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["not-a-command"]) == 2
    assert "usage:" in capsys.readouterr().err


def test_argument_errors_return_usage_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["preflight", "spec.yaml"]) == 2
    assert "required" in capsys.readouterr().err
    assert cli.main(["analyze", str(_RUN_FIXTURE), "--load-threshold-w", "invalid"]) == 2
    assert "load threshold" in capsys.readouterr().err
    assert cli.main(["preflight", "spec.yaml", "--stall-floor", "bad"]) == 2
    assert "CH=RPM" in capsys.readouterr().err


def test_unit_active_uses_the_watchdog_systemd_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], bool]] = []

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        calls.append((command, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.unit_active() is True
    assert calls == [(["systemctl", "is-active", "--quiet", "arctic-fan-watchdog.service"], False)]


def test_preflight_prints_pass_and_refusal_from_the_wp11_fake_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths, target, spec = _build_preflight_host(tmp_path)
    spec_path = tmp_path / "experiment.yaml"
    spec_path.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    monkeypatch.setattr(cli, "_host_paths", lambda: paths)
    monkeypatch.setattr(cli, "unit_active", lambda: True)
    args = _target_args(target, spec_path)

    assert cli.main(["preflight", *args]) == 0
    passed = json.loads(capsys.readouterr().out)
    assert passed["passed"] is True
    assert passed["checks"][-1]["name"] == "vram_fit"

    refusal_args = list(args)
    refusal_args[refusal_args.index("--expect-pci") + 1] = "0000:00:00.0"
    assert cli.main(["preflight", *refusal_args]) == 1
    refused = json.loads(capsys.readouterr().out)
    assert refused["passed"] is False
    assert refused["check"] == "card_identity"


def test_run_returns_zero_only_for_completed_wp12_fake_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with _make_host(tmp_path / "completed", workloads=("lane_sweep",)) as host:
        monkeypatch.setattr(cli, "_host_paths", lambda: host.paths)
        monkeypatch.setattr(cli, "unit_active", lambda: True)
        monkeypatch.setattr(cli, "_fan_lease_factory", host.lease_factory)
        monkeypatch.setenv("FAKE_SERVER_SIGINT_COUNT_FILE", str(host.sigint_file))
        monkeypatch.setenv("FAKE_SERVER_DRM_ROOT", str(host.paths.drm_root))
        monkeypatch.setenv("GPU_LEASE_BUDGET_FILE", str(host.lease_budget_file))
        with _FakeResidentLoader(host):
            code = cli.main(["run", *_target_args(host.target, host.spec_path, host.artifact_root)])
        assert code == 0
        result = json.loads(capsys.readouterr().out)
        assert result["passed"] is True
        assert result["runs"][0]["state"] == "completed"
        assert host.sigint_file.read_text() == "1"


def test_run_returns_one_for_a_non_completed_runner_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with _make_host(tmp_path / "refused", workloads=("lane_sweep",)) as host:
        _write_atomic(host.paths.loadavg, "99.0 99.0 99.0 1/1 1\n")
        monkeypatch.setattr(cli, "_host_paths", lambda: host.paths)
        monkeypatch.setattr(cli, "unit_active", lambda: True)
        monkeypatch.setattr(cli, "_fan_lease_factory", host.lease_factory)

        code = cli.main(["run", *_target_args(host.target, host.spec_path, host.artifact_root)])

        assert code == 1
        result = json.loads(capsys.readouterr().out)
        assert result["passed"] is False
        assert result["runs"][0]["state"] == "aborted"
        assert result["runs"][0]["failure_boundary"] == "preflight:host_load"


def test_analyze_prints_json_and_optional_markdown(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["analyze", str(_RUN_FIXTURE), "--load-threshold-w", "90"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["episode"]["start_mono"] == 2.0
    assert result["band_slope"] is None
    assert result["band_fit_error"] == "ols_slope needs at least 3 samples, got 2"
    assert result["transit_time"]["reached"] is False
    assert result["peak"]["value_c"] == 65.0
    assert result["band_power_w"] == 156.5
    assert "soak_300s" in result["plateau"]
    assert result["rates"]["decode"]["tokens_per_second"] == pytest.approx(20 / 300)
    assert result["rates"]["prefill"]["tokens_per_second"] is None
    assert "decode" in result and "prefill" in result

    assert cli.main(["analyze", str(_RUN_FIXTURE), "--load-threshold-w", "90", "--report"]) == 0
    report = capsys.readouterr().out
    assert report.startswith("# Analysis\n")
    assert "## Decode" in report
    assert "## Prefill" in report


def _copy_run_fixture(dest: Path) -> Path:
    shutil.copytree(_RUN_FIXTURE, dest)
    return dest


def test_analyze_reports_the_soak_rate_and_the_rate_per_band_watt() -> None:
    record = load_run(_RUN_FIXTURE)

    endpoints = cli._analyze_record(record, card=None, load_threshold_w=90.0)["endpoints"]

    assert endpoints["soak_tok_s"] == pytest.approx(20 / 300)
    assert endpoints["lane_sweep_tok_s"] is None
    assert endpoints["tok_s_per_w"] == pytest.approx((20 / 300) / endpoints["band_power"])
    assert endpoints["band_power"] == 156.5


def test_analyze_reports_the_lane_sweep_rate(tmp_path: Path) -> None:
    run_dir = _copy_run_fixture(tmp_path / "run")
    stages = json.loads((run_dir / "stages.json").read_text())
    request = dict(stages["stages"][0]["requests"][0])
    stages["stages"].append(
        {
            "name": "lane_sweep",
            "workload": "lane_sweep",
            "concurrency": 2,
            "started_mono": 400.0,
            "ended_mono": 440.0,
            "completed": True,
            "stop_reason": None,
            "requests": [{**request, "completion_tokens": 30}, {**request, "completion_tokens": 50}],
        }
    )
    (run_dir / "stages.json").write_text(json.dumps(stages))

    endpoints = cli._analyze_record(load_run(run_dir), card=None, load_threshold_w=90.0)["endpoints"]

    assert endpoints["lane_sweep_tok_s"] == pytest.approx(2.0)
    assert endpoints["soak_tok_s"] == pytest.approx(20 / 300)


def test_the_rate_per_watt_is_unavailable_without_a_band_power(tmp_path: Path) -> None:
    run_dir = _copy_run_fixture(tmp_path / "run")
    declaration = json.loads((run_dir / "declaration.json").read_text())
    del declaration["endpoints"]["band_c"]
    (run_dir / "declaration.json").write_text(json.dumps(declaration))
    analysis = cli._analyze_record(load_run(run_dir), card=None, load_threshold_w=90.0)

    assert analysis["endpoints"]["soak_tok_s"] == pytest.approx(20 / 300)
    assert analysis["endpoints"]["tok_s_per_w"] is None


def test_the_rate_endpoints_are_unavailable_for_a_run_without_an_outcome(tmp_path: Path) -> None:
    run_dir = _copy_run_fixture(tmp_path / "run")
    (run_dir / "outcome.json").unlink()

    endpoints = cli._analyze_record(load_run(run_dir), card=None, load_threshold_w=90.0)["endpoints"]

    assert endpoints["soak_tok_s"] is None
    assert endpoints["tok_s_per_w"] is None
    with pytest.raises(cli.ExperimentCLIError, match="unavailable"):
        cli._endpoint_value("soak_tok_s", {"endpoints": endpoints})


@pytest.mark.parametrize("name", ["soak_tok_s", "lane_sweep_tok_s", "tok_s_per_w"])
def test_compare_accepts_the_rate_endpoints(name: str) -> None:
    assert cli._valid_endpoint_name(name)


def _comparison_run(root: Path, condition: str, replicate: int, slope: float) -> Path:
    run_dir = root / f"{condition}-{replicate}"
    run_dir.mkdir(parents=True)
    _write_json_atomic(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "experiment": "fixture-experiment",
            "condition": condition,
            "replicate": replicate,
        },
    )
    _write_json_atomic(
        run_dir / "declaration.json",
        {
            "schema_version": 1,
            "fixed": {"cache": {"k": "q8_0", "v": "q8_0"}},
            "endpoints": {"band_c": [50.0, 70.0], "primary": "band_slope", "secondary": []},
        },
    )
    samples = []
    for index in range(4):
        samples.append(
            {
                "schema_version": 1,
                "utc": f"2026-10-07T15:00:0{index}+00:00",
                "monotonic": float(index),
                "cards": [
                    {
                        "card": 0,
                        "temperature_hotspot_c": 55.0 + slope * index,
                        "temperature_edge_c": 45.0 + slope * index,
                        "temperature_mem_c": 48.0 + slope * index,
                        "power_w": 160.0,
                    }
                ],
            }
        )
    _write_atomic(run_dir / "telemetry.jsonl", "".join(json.dumps(sample) + "\n" for sample in samples))
    return run_dir


def test_compare_requires_explicit_condition_for_mixed_groups_and_renders_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    group_a = tmp_path / "group-a"
    group_b = tmp_path / "group-b"
    for replicate, slope in enumerate((0.10, 0.11, 0.09), start=1):
        _comparison_run(group_a, "condition-a", replicate, slope)
    _comparison_run(group_a, "condition-extra", 1, 0.12)
    for replicate, slope in enumerate((0.30, 0.31, 0.29), start=1):
        _comparison_run(group_b, "condition-b", replicate, slope)

    assert cli.main(["compare", str(group_a), str(group_b), "--endpoint", "band_slope"]) == 1
    assert "mixed condition IDs" in capsys.readouterr().err

    assert (
        cli.main(
            [
                "compare",
                str(group_a),
                str(group_b),
                "--endpoint",
                "band_slope",
                "--condition-a",
                "condition-a",
                "--condition-b",
                "condition-b",
                "--allow",
                "cache.k",
                "--report",
            ]
        )
        == 0
    )
    report = capsys.readouterr().out
    assert report.startswith("# Comparison\n")
    assert "Endpoint: `band_slope`" in report
    assert "Allowed differences: cache.k" in report


def test_compare_unknown_endpoint_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["compare", "a", "b", "--endpoint", "unknown_metric"]) == 2
    assert "unsupported endpoint" in capsys.readouterr().err


def test_module_entry_point_exits_with_main_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "main", lambda argv=None: 7)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("inferference.experiment.__main__", run_name="__main__")

    assert excinfo.value.code == 7
