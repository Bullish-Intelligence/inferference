"""Tests for `inferference.experiment.workload`."""

from __future__ import annotations

import pytest

from inferference.experiment.workload import (
    WorkloadError,
    calibrate_prefill,
    chat_payload,
    decode_payload,
    parse_workload,
    payload_sha256,
    records_prompt,
    validate_response,
)

# ---------------------------------------------------------------------------
# parse_workload
# ---------------------------------------------------------------------------


def test_parse_workload_prefill() -> None:
    plan = parse_workload("prefill_7799")
    assert plan.kind == "prefill"
    assert plan.name == "prefill_7799"
    assert plan.target_prompt_tokens == 7799
    assert plan.concurrency_from_factor is False
    assert plan.duration_s is None


def test_parse_workload_decode_single_stream() -> None:
    plan = parse_workload("decode_single_stream")
    assert plan.kind == "decode"
    assert plan.concurrency_from_factor is False
    assert plan.target_prompt_tokens is None
    assert plan.duration_s is None


def test_parse_workload_lane_sweep() -> None:
    plan = parse_workload("lane_sweep")
    assert plan.kind == "lane_sweep"
    assert plan.concurrency_from_factor is True
    assert plan.target_prompt_tokens is None
    assert plan.duration_s is None


@pytest.mark.parametrize(("name", "duration_s"), [("soak_300s", 300.0), ("soak_900s", 900.0)])
def test_parse_workload_soak(name: str, duration_s: float) -> None:
    plan = parse_workload(name)
    assert plan.kind == "soak"
    assert plan.concurrency_from_factor is True
    assert plan.duration_s == duration_s
    assert plan.target_prompt_tokens is None


@pytest.mark.parametrize("name", ["decode", "prefill", "garbage", "", "soak_300", "prefill_"])
def test_parse_workload_rejects_unknown_and_bare_names(name: str) -> None:
    with pytest.raises(WorkloadError):
        parse_workload(name)


# ---------------------------------------------------------------------------
# payload_sha256
# ---------------------------------------------------------------------------


def test_payload_sha256_is_stable_across_key_order() -> None:
    payload_a = {
        "model": "test-alias",
        "temperature": 0,
        "seed": 1,
        "stream": True,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload_b = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32,
        "stream": True,
        "seed": 1,
        "temperature": 0,
        "model": "test-alias",
    }
    assert payload_sha256(payload_a) == payload_sha256(payload_b)


def test_payload_sha256_differs_for_different_payloads() -> None:
    payload_a = chat_payload([{"role": "user", "content": "a"}], seed=1, max_tokens=8, model_alias="x")
    payload_b = chat_payload([{"role": "user", "content": "b"}], seed=1, max_tokens=8, model_alias="x")
    assert payload_sha256(payload_a) != payload_sha256(payload_b)


# ---------------------------------------------------------------------------
# chat_payload / decode_payload shape
# ---------------------------------------------------------------------------


def test_chat_payload_shape() -> None:
    payload = chat_payload(
        [{"role": "user", "content": "hi"}],
        seed=7,
        max_tokens=64,
        model_alias="test-alias",
    )
    assert payload["model"] == "test-alias"
    assert payload["temperature"] == 0
    assert payload["seed"] == 7
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["max_tokens"] == 64
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_decode_payload_uses_model_alias_and_seed() -> None:
    payload = decode_payload(seed=11, model_alias="test-alias")
    assert payload["model"] == "test-alias"
    assert payload["seed"] == 11
    assert payload["stream"] is True
    assert "model-name" not in str(payload).lower()


def test_distinct_lanes_get_distinct_seeds_and_hashes() -> None:
    base_seed = 100
    payloads = [decode_payload(seed=base_seed + lane, model_alias="test-alias") for lane in range(3)]
    hashes = {payload_sha256(payload) for payload in payloads}
    assert len(hashes) == 3


# ---------------------------------------------------------------------------
# records_prompt
# ---------------------------------------------------------------------------


def test_records_prompt_expected_facts() -> None:
    content, expected = records_prompt(10)
    assert expected["expected_id"] == "S-0009"
    assert expected["expected_status"] == "GREEN"
    assert expected["expected_value"] == str(9 % 997)
    assert "S-0009" in content


def test_records_prompt_rejects_nonpositive_count() -> None:
    with pytest.raises(WorkloadError):
        records_prompt(0)


# ---------------------------------------------------------------------------
# calibrate_prefill
# ---------------------------------------------------------------------------


def _fake_tokenize(content: str) -> int:
    # Mirrors the plan's example: each record costs about 36 tokens, plus a
    # fixed wrapper cost of 40. The record count is recovered from the
    # generated content so this fake never has to parse real text.
    record_count = content.count("\nS-")
    return 36 * record_count + 40


def test_calibrate_prefill_hits_target_within_tolerance() -> None:
    target = 36 * 220 + 40
    content, info = calibrate_prefill(target, _fake_tokenize)
    assert abs(info["content_tokens"] - target) <= 0.02 * target
    assert info["record_count"] > 0
    assert info["attempts"]
    assert info["expected_id"] in content


def test_calibrate_prefill_raises_for_target_too_small() -> None:
    with pytest.raises(WorkloadError):
        calibrate_prefill(5, _fake_tokenize)


def test_calibrate_prefill_raises_for_target_too_large() -> None:
    with pytest.raises(WorkloadError):
        calibrate_prefill(50_000_000, _fake_tokenize)


def test_calibrate_prefill_rejects_nonpositive_target() -> None:
    with pytest.raises(WorkloadError):
        calibrate_prefill(0, _fake_tokenize)


# ---------------------------------------------------------------------------
# validate_response
# ---------------------------------------------------------------------------


def _ok_result(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "http_status": 200,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "content": "the answer",
        "reasoning_content": "",
        "finish_reason": "stop",
    }
    base.update(overrides)
    return base


def test_validate_response_valid_decode() -> None:
    assert validate_response("decode", _ok_result(), None) is None


def test_validate_response_rejects_non_200() -> None:
    reason = validate_response("decode", _ok_result(http_status=500), None)
    assert reason is not None
    assert "500" in reason


def test_validate_response_rejects_zero_completion_tokens() -> None:
    reason = validate_response("decode", _ok_result(completion_tokens=0), None)
    assert reason is not None
    assert "completion_tokens" in reason


def test_validate_response_rejects_empty_content() -> None:
    reason = validate_response("decode", _ok_result(content="", reasoning_content=""), None)
    assert reason is not None
    assert "content" in reason


def test_validate_response_accepts_reasoning_only_content() -> None:
    # Content is empty but reasoning_content carries text: still counts.
    assert validate_response("decode", _ok_result(content="", reasoning_content="thinking..."), None) is None


def test_validate_response_rejects_missing_finish_reason() -> None:
    reason = validate_response("decode", _ok_result(finish_reason=None), None)
    assert reason is not None
    assert "finish_reason" in reason


def test_validate_response_prefill_missing_expected_fact() -> None:
    expected = {"expected_id": "S-0009", "expected_status": "GREEN", "expected_value": "9"}
    reason = validate_response("prefill", _ok_result(content="no facts here"), expected)
    assert reason is not None
    assert "expected" in reason


def test_validate_response_prefill_valid_with_all_facts() -> None:
    expected = {"expected_id": "S-0009", "expected_status": "GREEN", "expected_value": "9"}
    content = "The final record is S-0009 with status GREEN and value 9."
    assert validate_response("prefill", _ok_result(content=content), expected) is None
