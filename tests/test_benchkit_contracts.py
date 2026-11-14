"""CPU-only contract tests for the bench package.

No GPU, no llama-cpp-python at runtime: the workload registry (sha256-verified
corpus), envelope v2, the baseline store, and the GPU-isolation classifier are
all exercised with fakes — exactly like the library's own CPU tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from inferference.benchkit import env as benchkit_env
from inferference.benchkit.baselines import BaselineStore
from inferference.benchkit.env import require_gpu0_only
from inferference.benchkit.envelope import RunEnvelope
from inferference.benchkit.workload import list_workloads, load


def test_workload_registry_loads_and_verifies_hashes() -> None:
    workload = load("json_workload")
    assert workload.name == "json_workload"
    assert workload.version == "project17-json-workload-v1"
    assert len(workload.entries) == 1000
    # The manifest's sha256 contract, re-verified through the registry.
    expected = workload.manifest["files"]["json_workload_1000.jsonl"]["sha256"]
    assert workload.corpus_sha256("json_workload_1000.jsonl") == expected
    # Deterministic stable prefix.
    assert [item["id"] for item in workload.select(10)] == [f"p17-{value:04d}" for value in range(1, 11)]


def test_workload_registry_rejects_corrupted_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A tampered corpus must fail closed at load, never silently pass. Point the
    # registry at a tampered copy (the real corpus is never touched).
    import shutil

    from inferference.benchkit import workload as registry

    registry._REGISTRY.clear()
    source = Path(__file__).parents[1] / "bench" / "workloads" / "json_workload"
    monkeypatch.setattr(registry, "_WORKLOADS_DIR", tmp_path)
    shutil.copytree(source, tmp_path / "json_workload")
    (tmp_path / "json_workload" / "json_workload_100.jsonl").write_bytes(b"tampered")
    try:
        with pytest.raises(ValueError, match="sha256 mismatch"):
            registry.load("json_workload")
    finally:
        registry._REGISTRY.clear()


def test_workload_registry_auto_discovers() -> None:
    assert "json_workload" in list_workloads()


def test_envelope_v2_roundtrip() -> None:
    envelope = RunEnvelope(
        run_id="run1",
        workload={"name": "json_workload", "version": "v1", "requests": 10},
        environment={"fingerprint": "abc", "nvidia_smi": {"driver_version": "595.84"}},
        gate={"kind": "perf", "verdict": "pass"},
        summary={"generation_tokens_per_second": 19.0},
        command=["bench", "run", "json_workload@v1"],
    )
    payload = json.loads(envelope.model_dump_json())
    assert payload["schema_version"] == 2
    assert payload["gate"]["verdict"] == "pass"
    # Extra fields are forbidden (the contract is exact): model_validate with a
    # bogus key must raise, without tripping a static checker on the constructor.
    with pytest.raises(ValidationError):
        RunEnvelope.model_validate({"run_id": "x", "workload": {}, "environment": {}, "gate": {}, "bogus": 1})


def test_baselines_save_load_and_regression(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path)
    key = "json_workload@b10103-p2"
    summary = {"end_to_end": {"generation_tokens_per_second": 20.0}}
    store.save(key, summary)
    assert store.load(key) == summary

    # No baseline -> no regression (a new baseline is the caller's decision).
    ok, detail = store.check_regression("missing@x", summary)
    assert not ok and detail["reason"] == "no_baseline"

    # A >5% regression is detected; a small delta is not.
    slow = {"end_to_end": {"generation_tokens_per_second": 18.0}}
    ok, detail = store.check_regression(key, slow)
    assert ok and detail["delta_percent"] <= -10.0
    fine = {"end_to_end": {"generation_tokens_per_second": 19.8}}
    ok, _ = store.check_regression(key, fine)
    assert not ok

    # A missing metric is fail-closed (cannot verify == regression).
    ok, detail = store.check_regression(key, {"end_to_end": {}})
    assert ok and detail["reason"] == "missing_metric"


def _gpu_snapshot(gpu0_used: int, gpu1_used: int) -> dict[str, str]:
    return {
        "stdout": (
            "0, NVIDIA Example GPU, uuid0, 595.84, 12288 MiB, "
            f"{gpu0_used} MiB\n1, NVIDIA Example GPU, uuid1, 595.84, 12288 MiB, {gpu1_used} MiB\n"
        )
    }


def test_require_gpu0_only_classifies_isolation() -> None:
    require_gpu0_only(_gpu_snapshot(5000, 100))  # GPU0 holds the model, GPU1 idle -> ok

    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        require_gpu0_only(_gpu_snapshot(5000, 8000))  # GPU1 busy (vLLM runner)

    with pytest.raises(RuntimeError, match="GPU isolation rejected"):
        require_gpu0_only(_gpu_snapshot(100, 100))  # GPU0 cold -> reject


def _no_nvidia_smi(*args: object, **kwargs: object) -> None:
    raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")


def test_missing_nvidia_smi_reads_as_no_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A host with neither nvidia-smi nor an amdgpu card reports no GPU; it does not crash."""
    monkeypatch.setattr(benchkit_env.subprocess, "run", _no_nvidia_smi)
    monkeypatch.setattr(benchkit_env, "DRM_ROOT", tmp_path)  # an empty tree: no amdgpu card
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert benchkit_env.visible_gpu_uuid() is None
    snapshot = benchkit_env.gpu_snapshot()
    assert snapshot["returncode"] == 127
    assert snapshot["stdout"] == ""
    assert benchkit_env.driver_version(snapshot) is None
    with pytest.raises(RuntimeError):  # the isolation check still fails closed
        require_gpu0_only(snapshot)
