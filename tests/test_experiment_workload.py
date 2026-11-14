"""Tests for `inferference.experiment.workload`."""

from __future__ import annotations

import pytest

from inferference.experiment.workload import (
    AgentSession,
    CorpusLane,
    WorkloadError,
    agent_turns_context_tokens,
    assign_corpus_prompts,
    calibrate_prefill,
    chat_payload,
    corpus_prompts,
    decode_payload,
    follow_up_message,
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


# ---------------------------------------------------------------------------
# corpus_<set>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["corpus_all", "corpus_explain", "corpus_cli-shape", "corpus_test-write"])
def test_parse_workload_corpus(name: str) -> None:
    plan = parse_workload(name)

    assert plan.kind == "corpus"
    assert plan.corpus_set == name.removeprefix("corpus_")
    assert plan.concurrency_from_factor is True
    assert plan.target_prompt_tokens is None
    assert plan.duration_s is None


@pytest.mark.parametrize("name", ["corpus", "corpus_", "corpus_All", "corpus_a_b", "corpus_-x", "corpus_x-"])
def test_parse_workload_rejects_a_bad_corpus_name(name: str) -> None:
    with pytest.raises(WorkloadError):
        parse_workload(name)


def test_corpus_prompts_cover_the_whole_corpus_in_task_order() -> None:
    prompts = corpus_prompts("all")

    assert len(prompts) == 12
    assert [item.task_id for item in prompts] == sorted(item.task_id for item in prompts)
    assert [item.index for item in prompts] == list(range(12))
    for item in prompts:
        assert [message["role"] for message in item.messages] == ["system", "user"]
        assert item.max_tokens > 0
        assert len(item.prompt_sha256) == 64


def test_corpus_prompts_are_stable_between_calls() -> None:
    assert corpus_prompts("all") == corpus_prompts("all")


def test_corpus_prompts_select_one_family() -> None:
    prompts = corpus_prompts("implement")

    assert len(prompts) == 3
    assert all(item.task_id.startswith("implement/") or "implement" in item.task_id for item in prompts)
    assert [item.index for item in prompts] == [0, 1, 2]


def test_corpus_prompts_refuse_an_unknown_set() -> None:
    with pytest.raises(WorkloadError, match="unknown corpus set"):
        corpus_prompts("no-such-family")


def test_corpus_prompts_refuse_a_missing_corpus(tmp_path) -> None:
    with pytest.raises(WorkloadError, match="does not load"):
        corpus_prompts("all", root=tmp_path / "absent")


def test_assign_corpus_prompts_keeps_the_corpus_order_for_one_lane() -> None:
    prompts = corpus_prompts("all")

    assert assign_corpus_prompts(prompts, 1) == [list(prompts)]


def test_assign_corpus_prompts_deals_every_prompt_once_round_robin() -> None:
    prompts = corpus_prompts("all")

    lanes = assign_corpus_prompts(prompts, 8)

    assert len(lanes) == 8
    assert [item.index for item in lanes[0]] == [0, 8]
    assert [item.index for item in lanes[3]] == [3, 11]
    assert sorted(item.index for lane in lanes for item in lane) == list(range(12))


def test_assign_corpus_prompts_gives_a_spare_lane_nothing() -> None:
    lanes = assign_corpus_prompts(corpus_prompts("explain"), 3)

    assert [len(lane) for lane in lanes] == [1, 0, 0]


def test_assign_corpus_prompts_refuses_zero_lanes() -> None:
    with pytest.raises(WorkloadError):
        assign_corpus_prompts(corpus_prompts("all"), 0)


def test_a_corpus_lane_sends_each_prompt_once_at_temperature_zero() -> None:
    prompts = corpus_prompts("implement")
    lane = CorpusLane(prompts, model_alias="alias")

    steps = []
    while (step := lane.next_step(len(steps))) is not None:
        steps.append(step)

    assert len(steps) == 3
    for step, prompt in zip(steps, prompts, strict=True):
        assert step.payload["temperature"] == 0
        assert step.payload["stream"] is True
        assert step.payload["model"] == "alias"
        assert step.payload["max_tokens"] == prompt.max_tokens
        assert step.payload["messages"] == [dict(message) for message in prompt.messages]
        assert step.extra["task_id"] == prompt.task_id
        assert step.extra["prompt_sha256"] == prompt.prompt_sha256


def test_a_corpus_prompt_has_the_same_payload_in_every_mode() -> None:
    prompts = corpus_prompts("all")
    serial = CorpusLane(assign_corpus_prompts(prompts, 1)[0], model_alias="alias")
    wide = [CorpusLane(items, model_alias="alias") for items in assign_corpus_prompts(prompts, 4)]

    serial_hashes = {}
    for index in range(12):
        step = serial.next_step(index)
        assert step is not None
        serial_hashes[step.extra["task_id"]] = payload_sha256(step.payload)
    wide_hashes = {}
    for lane in wide:
        sequence = 0
        while (step := lane.next_step(sequence)) is not None:
            wide_hashes[step.extra["task_id"]] = payload_sha256(step.payload)
            sequence += 1

    assert wide_hashes == serial_hashes


# ---------------------------------------------------------------------------
# agent_turns_<S>x<T>
# ---------------------------------------------------------------------------


def test_parse_workload_agent_turns() -> None:
    plan = parse_workload("agent_turns_6x8")

    assert plan.kind == "agent_turns"
    assert plan.sessions == 6
    assert plan.turns == 8
    assert plan.concurrency_from_factor is False
    assert plan.target_prompt_tokens == 7800
    assert plan.duration_s is None


@pytest.mark.parametrize(
    "name", ["agent_turns", "agent_turns_6", "agent_turns_6x", "agent_turns_x8", "agent_turns_0x8", "agent_turns_6x0"]
)
def test_parse_workload_rejects_a_bad_agent_turns_name(name: str) -> None:
    with pytest.raises(WorkloadError):
        parse_workload(name)


def test_records_prompt_keeps_the_default_set_when_the_salt_is_zero() -> None:
    assert records_prompt(40, salt=0) == records_prompt(40)


def test_records_prompt_salt_changes_the_records_and_keeps_the_identifiers() -> None:
    plain, _ = records_prompt(40)
    salted, expected = records_prompt(40, salt=3)

    assert plain != salted
    assert salted != records_prompt(40, salt=4)[0]
    assert "S-0039" in salted
    assert expected["expected_id"] == "S-0039"


def test_follow_up_message_depends_on_the_turn_only() -> None:
    assert follow_up_message(1) == follow_up_message(1)
    assert follow_up_message(1) != follow_up_message(2)
    assert "Follow-up 5." in follow_up_message(5)


def _answer(session: AgentSession, sequence: int, text: str):
    step = session.next_step(sequence)
    assert step is not None
    session.observe(text)
    return step


def test_an_agent_session_extends_the_conversation_with_each_answer() -> None:
    session = AgentSession(2, "the task", turns=3, model_alias="alias")

    first = _answer(session, 0, "answer one")
    second = _answer(session, 1, "answer two")
    third = session.next_step(2)
    assert third is not None

    assert [m["role"] for m in first.payload["messages"]] == ["system", "user"]
    assert [m["role"] for m in second.payload["messages"]] == ["system", "user", "assistant", "user"]
    assert second.payload["messages"][2]["content"] == "answer one"
    assert second.payload["messages"][3]["content"] == follow_up_message(1)
    assert third.payload["messages"][4]["content"] == "answer two"
    assert third.payload["messages"][5]["content"] == follow_up_message(2)
    assert session.next_step(3) is None


def test_an_agent_turn_keeps_the_whole_previous_turn_as_its_prefix() -> None:
    session = AgentSession(0, "the task", turns=3, model_alias="alias")
    first = _answer(session, 0, "answer one")
    second = _answer(session, 1, "answer two")

    previous = first.payload["messages"] + [{"role": "assistant", "content": "answer one"}]
    assert second.payload["messages"][: len(previous)] == previous
    assert first.extra["prefix_sha256"] is None
    assert first.extra["prefix_matches_previous"] is None
    assert second.extra["prefix_matches_previous"] is True
    assert second.extra["prefix_sha256"] == second.extra["previous_finished_sha256"]
    assert second.extra["messages_sha256"] != first.extra["messages_sha256"]
    assert (second.extra["session"], second.extra["turn"]) == (0, 1)


def test_an_agent_session_sends_temperature_zero_with_a_fixed_seed_and_answer_cap() -> None:
    session = AgentSession(4, "the task", turns=2, model_alias="alias")

    first = _answer(session, 0, "a")
    second = session.next_step(1)
    assert second is not None

    for step in (first, second):
        assert step.payload["temperature"] == 0
        assert step.payload["max_tokens"] == 256
        assert step.payload["seed"] == first.payload["seed"]
    other = AgentSession(5, "the task", turns=2, model_alias="alias").next_step(0)
    assert other is not None
    assert other.payload["seed"] != first.payload["seed"]


def test_an_agent_session_annotates_the_turn_numbers() -> None:
    session = AgentSession(0, "the task", turns=2, model_alias="alias")
    record = {
        "dispatch_mono": 10.0,
        "first_token_mono": 10.5,
        "prompt_tokens": 1000,
        "server_timings": {"prompt_n": 100, "cache_n": 900, "predicted_n": 50, "predicted_ms": 1000.0},
    }

    session.annotate(record)

    assert record["ttft_s"] == pytest.approx(0.5)
    assert record["decode_tok_s"] == pytest.approx(50.0)
    assert record["cache_n"] == 900
    assert record["prompt_n"] == 100
    assert record["reuse_fraction"] == pytest.approx(0.9)


def test_an_agent_session_annotates_none_for_missing_numbers() -> None:
    session = AgentSession(0, "the task", turns=2, model_alias="alias")
    record = {"dispatch_mono": 10.0, "first_token_mono": None, "prompt_tokens": 10, "server_timings": {}}

    session.annotate(record)

    assert record["ttft_s"] is None
    assert record["decode_tok_s"] is None
    assert record["reuse_fraction"] is None


def test_agent_turns_context_tokens_grows_with_each_turn_and_covers_the_task() -> None:
    assert agent_turns_context_tokens(1) > 7800 + 256
    assert agent_turns_context_tokens(8) - agent_turns_context_tokens(7) >= 256


def _agent_result(**changes: object) -> dict[str, object]:
    result = {
        "http_status": 200,
        "prompt_tokens": 10,
        "completion_tokens": 256,
        "finish_reason": "length",
        "content": "an answer",
        "reasoning_content": "",
    }
    result.update(changes)
    return result


def test_validate_response_accepts_an_agent_turns_answer() -> None:
    assert validate_response("agent_turns", _agent_result(), None) is None


def test_validate_response_refuses_a_reasoning_only_agent_turns_reply() -> None:
    reply = _agent_result(content="", reasoning_content="thinking about the records")

    reason = validate_response("agent_turns", reply, None)

    assert reason is not None
    assert "reasoning only" in reason


def test_validate_response_still_accepts_a_reasoning_only_reply_for_decode() -> None:
    reply = _agent_result(content="", reasoning_content="thinking about the records")

    assert validate_response("decode", reply, None) is None
