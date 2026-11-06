"""CPU contract tests for the Task-5 bench additions.

The port is a refactor with contract tests, not a copy. The
new runners' GPU bodies are exercised on the rig; their pure logic is tested
here, CPU-only:

  * the grammar ``validate()`` classifier (router_grammar runner);
  * the baseline key derivation (never a date) + the regression gate wiring;
  * the envelope migration stamp (idempotent, reconstructs from legacy files);
  * the runner result-sorting helper (deterministic request ordering).
"""

from __future__ import annotations

import json

import pytest

from inferference.benchkit.baseline_gate import baseline_key, cross_device_error
from inferference.benchkit.env import require_ci_device
from inferference.benchkit.runners.router_grammar import validate as grammar_validate

_SAMPLE_MANIFEST: dict[str, object] = {
    "llama_cpp_commit": "97ad953ef",
    "features": ["feature-a", "p2-mixed-batch-lora"],
}


def test_grammar_validate_classifies_json_and_schema() -> None:
    good = grammar_validate('{"tool": "search", "confidence": "high"}')
    assert good["json_ok"] is True and good["schema_ok"] is True
    assert good["value"]["tool"] == "search"

    bad_json = grammar_validate("not json")
    assert bad_json["json_ok"] is False and bad_json["schema_ok"] is False

    bad_schema = grammar_validate('{"tool": "nope", "confidence": "high"}')
    assert bad_schema["json_ok"] is True and bad_schema["schema_ok"] is False


def test_baseline_key_is_deterministic_and_never_a_date() -> None:
    key = baseline_key(_SAMPLE_MANIFEST, "json_workload")
    assert key == "json_workload@97ad953ef-feature-a,p2-mixed-batch-lora"
    assert baseline_key(_SAMPLE_MANIFEST, "json_workload") == key
    # stock (features []) gets the "stock" featureset marker
    stock_key = baseline_key({"llama_cpp_commit": "0ab9d6fed", "features": []}, "json_workload")
    assert stock_key == "json_workload@0ab9d6fed-stock"


def test_baseline_gate_compare_only_passes_without_baseline(tmp_path) -> None:
    from inferference.benchkit.baselines import BaselineStore

    store = BaselineStore(tmp_path)
    summary = {"end_to_end": {"generation_tokens_per_second": 30.0}}
    regression, info = store.check_regression("k", summary)
    assert regression is False and info["reason"] == "no_baseline"


def test_baseline_gate_detects_regression(tmp_path) -> None:
    from inferference.benchkit.baselines import BaselineStore

    store = BaselineStore(tmp_path)
    store.save("k", {"end_to_end": {"generation_tokens_per_second": 30.0}})
    regression, info = store.check_regression(
        "k", {"end_to_end": {"generation_tokens_per_second": 28.0}}, max_regression_percent=5.0
    )
    assert regression is True and info["delta_percent"] == -6.666666666666667
    # a within-threshold delta is NOT a regression (the FP rule: report, don't fail)
    regression2, info2 = store.check_regression(
        "k", {"end_to_end": {"generation_tokens_per_second": 29.0}}, max_regression_percent=5.0
    )
    assert regression2 is False and abs(info2["delta_percent"]) < 5.0


def test_migration_stamp_is_idempotent_and_reconstructs_env(tmp_path) -> None:
    from inferference.benchkit.migrate_envelopes import stamp_one

    d = tmp_path / "project17-json-baseline-20260724T174322Z"
    d.mkdir()
    (d / "runtime-environment.txt").write_text(
        "CUDA_VISIBLE_DEVICES=0\nLLAMA_CI_GPU=0\nLLAMA_SERVER_GPU=1\nLLAMA_CPP_LIB_PATH=/some/lib\n"
    )
    (d / "git-commit.txt").write_text("abc123\n")
    (d / "summary.json").write_text(json.dumps({"end_to_end": {"generation_tokens_per_second": 12.3}}))

    first = stamp_one(d, dry=False)
    assert first is not None and first.is_file()
    envelope = json.loads(first.read_text())
    assert envelope["schema_version"] == 2
    assert envelope["environment"]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert envelope["environment"]["env"]["LLAMA_CI_GPU"] == "0"
    assert envelope["environment"]["env"]["LLAMA_SERVER_GPU"] == "1"
    assert envelope["environment"]["fingerprint"]["git_head"] == "abc123"
    assert envelope["summary"]["end_to_end"]["generation_tokens_per_second"] == 12.3
    # idempotent
    assert stamp_one(d, dry=False) is None


@pytest.mark.runtime
def test_runner_token_sorting_is_deterministic() -> None:
    class _G:
        def __init__(self, rid: str, toks: list[int]) -> None:
            self.request_id = rid
            self.token_ids = toks

    from inferference.benchkit.runners.library_throughput import _toks

    gens = [_G("r2", [2]), _G("r1", [1]), _G("r10", [10])]
    assert _toks(gens) == [(1,), (10,), (2,)]


def test_require_ci_device_refuses_wrong_device(monkeypatch) -> None:
    """Gate 4: a run on the wrong device is refused loudly, by name."""
    monkeypatch.setenv("LLAMA_CI_GPU", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    try:
        require_ci_device()
    except RuntimeError as exc:
        msg = str(exc)
        assert "CUDA_VISIBLE_DEVICES='1'" in msg
        assert "'0'" in msg
        assert "LLAMA_CI_GPU" in msg
    else:
        raise AssertionError("require_ci_device did not refuse the wrong device")


def test_require_ci_device_accepts_ci_device(monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_CI_GPU", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    require_ci_device()  # must not raise


def test_cross_device_error_refuses_mismatch() -> None:
    """Gate 5: the baseline gate refuses a cross-device comparison."""
    error = cross_device_error("GPU-aaa", "GPU-bbb")
    assert error is not None
    assert "GPU-aaa" in error and "GPU-bbb" in error
    # same device -> no error; no stored UUID -> no error (first baseline)
    assert cross_device_error("GPU-aaa", "GPU-aaa") is None
    assert cross_device_error("GPU-aaa", None) is None
    # a run whose UUID cannot be read must fail closed, not pass silently
    assert cross_device_error(None, "GPU-bbb") is not None
