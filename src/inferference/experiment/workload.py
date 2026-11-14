"""Request generators for the experiment runner: payloads in, verdicts out.

This module builds the HTTP request bodies a workload sends and checks the
plain-data results the caller passes back. It never opens a socket. The
runner owns the HTTP call; this module only owns the shape of the request and
the rule for calling a response valid.

A workload name is one of six spec strings: `prefill_<N>` (prefill `N`
content tokens), `decode_single_stream` (one long, unassisted decode),
`lane_sweep` (concurrency drawn from the experiment's `lanes` factor),
`soak_<N>s` (a fixed-duration run, concurrency also from `lanes`), or
`corpus_<set>` (the codeeval prompts, concurrency also from `lanes`), or
`agent_turns_<S>x<T>` (multi-turn sessions, see below). A bare
`"decode"` or `"prefill"` is refused, because neither names a size.

`agent_turns_<S>x<T>` runs `S` sessions at the same time, `T` turns each. Each
session starts from a long task description. Each turn appends the answer that
the server gave and a fixed follow-up message to the same conversation, so turn
k has turn k-1 as its prefix. The lane count is `S`, not the `lanes` factor.

`corpus_<set>` sends each prompt of the set once, at temperature 0. `<set>` is
`all` or one family name of `tasks/codeeval`. One lane sends the prompts in
corpus order (serial mode). Many lanes share the prompts round robin
(concurrent mode). A prompt has the same seed in every mode.

`chat_payload` and `decode_payload` build one request each. `records_prompt`
and `calibrate_prefill` build the "read every record" prompt a prefill
workload needs, searching for the record count that tokenizes to the target
content-token count. The runner calls `chat_payload` directly for
`lane_sweep` and `soak_<N>s`, with its own messages, so distinct lanes pass
distinct seeds (`base_seed + lane`) and so get distinct payload hashes.

`validate_response` is deliberately thin: it checks that a response is well
formed and, for prefill, that the answer proves the whole prompt was read.
It never checks the model's prose for a specific format or count — that
tests the model, not the rig.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

# A record count never goes below 1, and this ceiling keeps a bisection over
# an unreachable target bounded instead of searching forever.
_MAX_RECORD_COUNT = 100_000

# `decode_single_stream` has no target token count: it is a fixed prompt
# sized to run a long, uninterrupted decode.
_DECODE_MAX_TOKENS = 900

_PREFILL_NAME_RE = re.compile(r"^prefill_(\d+)$")
_SOAK_NAME_RE = re.compile(r"^soak_(\d+)s$")
_CORPUS_NAME_RE = re.compile(r"^corpus_([a-z0-9]+(?:-[a-z0-9]+)*)$")
_AGENT_NAME_RE = re.compile(r"^agent_turns_(\d+)x(\d+)$")

# The task description of an agent session has about this many content tokens,
# the size of the prefill workload. A turn answer is capped at the answer limit.
_AGENT_PROMPT_TOKENS = 7800
_AGENT_ANSWER_TOKENS = 256
_AGENT_SEED_BASE = 5000
# Allowances on top of the content tokens, as upper bounds. The system message
# and the chat template cost `_AGENT_TEMPLATE_TOKENS` once. Each follow-up
# message and its template cost `_AGENT_FOLLOW_UP_TOKENS`.
_AGENT_TEMPLATE_TOKENS = 100
_AGENT_FOLLOW_UP_TOKENS = 100

# A corpus prompt keeps its seed in every mode: the seed follows the position
# of the prompt in the corpus, not the lane that sends it.
_CORPUS_SEED_BASE = 1000

WorkloadKind = Literal["prefill", "decode", "lane_sweep", "soak", "corpus", "agent_turns"]


class WorkloadError(ValueError):
    """A workload name does not parse, or a calibration cannot hit its target."""


@dataclass(frozen=True, slots=True)
class WorkloadPlan:
    """One parsed workload name, with the facts the runner needs to dispatch it."""

    name: str
    kind: WorkloadKind
    concurrency_from_factor: bool
    target_prompt_tokens: int | None
    duration_s: float | None
    corpus_set: str | None = None
    sessions: int | None = None
    turns: int | None = None


def parse_workload(name: str) -> WorkloadPlan:
    """Parse a workload name into a `WorkloadPlan`.

    Only six shapes are accepted: `prefill_<N>`, `decode_single_stream`,
    `lane_sweep`, `soak_<N>s`, `corpus_<set>`, and `agent_turns_<S>x<T>`. A bare `decode` or `prefill`, with no
    size, raises `WorkloadError`: a workload with no size is never accepted.
    """
    prefill_match = _PREFILL_NAME_RE.match(name)
    if prefill_match:
        return WorkloadPlan(
            name=name,
            kind="prefill",
            concurrency_from_factor=False,
            target_prompt_tokens=int(prefill_match.group(1)),
            duration_s=None,
        )
    if name == "decode_single_stream":
        return WorkloadPlan(
            name=name,
            kind="decode",
            concurrency_from_factor=False,
            target_prompt_tokens=None,
            duration_s=None,
        )
    if name == "lane_sweep":
        return WorkloadPlan(
            name=name,
            kind="lane_sweep",
            concurrency_from_factor=True,
            target_prompt_tokens=None,
            duration_s=None,
        )
    soak_match = _SOAK_NAME_RE.match(name)
    if soak_match:
        return WorkloadPlan(
            name=name,
            kind="soak",
            concurrency_from_factor=True,
            target_prompt_tokens=None,
            duration_s=float(soak_match.group(1)),
        )
    corpus_match = _CORPUS_NAME_RE.match(name)
    if corpus_match:
        return WorkloadPlan(
            name=name,
            kind="corpus",
            concurrency_from_factor=True,
            target_prompt_tokens=None,
            duration_s=None,
            corpus_set=corpus_match.group(1),
        )
    agent_match = _AGENT_NAME_RE.match(name)
    if agent_match:
        sessions, turns = int(agent_match.group(1)), int(agent_match.group(2))
        if sessions < 1 or turns < 1:
            raise WorkloadError(f"{name!r} needs at least 1 session and 1 turn")
        return WorkloadPlan(
            name=name,
            kind="agent_turns",
            concurrency_from_factor=False,
            target_prompt_tokens=_AGENT_PROMPT_TOKENS,
            duration_s=None,
            sessions=sessions,
            turns=turns,
        )
    raise WorkloadError(f"unknown workload name: {name!r}")


def agent_turns_context_tokens(turns: int) -> int:
    """Return the most context tokens one `agent_turns` session needs for `turns` turns.

    The last turn holds the task description, `turns - 1` answers with their
    follow-up messages, and room for its own answer. The result is an upper
    bound, so a spec that passes it never overflows a slot. The runner checks
    it before the model loads.
    """
    return (
        _AGENT_PROMPT_TOKENS
        + _AGENT_TEMPLATE_TOKENS
        + (turns - 1) * (_AGENT_ANSWER_TOKENS + _AGENT_FOLLOW_UP_TOKENS)
        + _AGENT_ANSWER_TOKENS
    )


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """A sha256 of `payload`'s canonical JSON, independent of key order."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def chat_payload(
    messages: Sequence[Mapping[str, str]],
    *,
    seed: int,
    max_tokens: int,
    model_alias: str,
) -> dict[str, Any]:
    """Build one chat-completions request body.

    `model_alias` names the resident model (`ModelSpec.name`); this module
    never hardcodes a real model name. `chat_template_kwargs.enable_thinking`
    is a generic template variable that a template without a thinking mode
    ignores; it stays in the payload for comparability with historical runs.
    """
    return {
        "model": model_alias,
        "temperature": 0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [dict(message) for message in messages],
    }


def decode_payload(*, seed: int, model_alias: str) -> dict[str, Any]:
    """Build the `decode_single_stream` request: one long, unassisted decode.

    The prompt asks for open-ended prose on the rig's own domain (serving a
    language model), not on any model's own identity, so the request stays
    model-agnostic. Validation asks only that the response is well formed
    (`validate_response`): no specific count or format is required.
    """
    messages = [
        {"role": "system", "content": "Write clear, direct prose. Do not include reasoning."},
        {
            "role": "user",
            "content": (
                "Explain how an inference server schedules concurrent requests across "
                "GPU memory, batching, and the decode loop. Cover device identity, "
                "memory residency, prompt processing, decoding, and output checks."
            ),
        },
    ]
    return chat_payload(messages, seed=seed, max_tokens=_DECODE_MAX_TOKENS, model_alias=model_alias)


@dataclass(frozen=True, slots=True)
class CorpusPrompt:
    """One corpus task as a request: its messages, its token budget, and its seed."""

    task_id: str
    index: int
    messages: tuple[dict[str, str], ...]
    max_tokens: int
    prompt_sha256: str


@dataclass(frozen=True, slots=True)
class LaneStep:
    """One request that a scripted lane sends, with the fields to add to its record."""

    payload: dict[str, Any]
    extra: dict[str, Any]


class LaneScript(Protocol):
    """The requests one lane sends. The runner asks for a step, sends it, then reports back."""

    def next_step(self, sequence: int) -> LaneStep | None: ...

    def observe(self, content: str) -> None: ...

    def annotate(self, record: dict[str, Any]) -> None: ...


class CorpusLane:
    """The prompts one lane sends, in order. It sends each prompt once."""

    def __init__(self, items: Sequence[CorpusPrompt], *, model_alias: str) -> None:
        self._items = tuple(items)
        self._alias = model_alias

    def next_step(self, sequence: int) -> LaneStep | None:
        if sequence >= len(self._items):
            return None
        item = self._items[sequence]
        payload = chat_payload(
            item.messages,
            seed=_CORPUS_SEED_BASE + item.index,
            max_tokens=item.max_tokens,
            model_alias=self._alias,
        )
        return LaneStep(
            payload=payload,
            extra={"task_id": item.task_id, "corpus_index": item.index, "prompt_sha256": item.prompt_sha256},
        )

    def observe(self, content: str) -> None:
        """A corpus prompt does not depend on an earlier answer."""

    def annotate(self, record: dict[str, Any]) -> None:
        """A corpus record needs no field beyond the ones the runner writes."""


_AGENT_SYSTEM = "You are a careful operations assistant. Answer from the records only, and keep answers short."

_AGENT_FOCUS = (
    "list every record whose status is WARN",
    "name the record with the highest latency_ms and give its queue value",
    "name the record with the lowest checksum and give its region",
    "count the records whose queue is above 30 and name the first one",
)


def follow_up_message(turn: int) -> str:
    """The fixed user message that opens turn `turn` (1 and up). The text depends on `turn` only."""
    focus = _AGENT_FOCUS[turn % len(_AGENT_FOCUS)]
    return (
        f"Follow-up {turn}. Using only the records above, {focus}. "
        "Answer in at most four short sentences. Do not repeat the records. "
        "If the records do not show the answer, say that plainly instead of guessing."
    )


def _messages_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    canonical = json.dumps([dict(message) for message in messages], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class AgentSession:
    """One multi-turn conversation: a long task description, then `turns` turns.

    Turn 0 sends the system prompt and the task description. After each answer,
    `observe` appends the answer as an assistant message and the next fixed
    follow-up as a user message. So turn k sends all of turn k-1, its answer, and
    one new message. The record of turn k carries the hash of that prefix and the
    hash that turn k-1 finished with. They must be equal.
    """

    def __init__(self, session: int, task_description: str, *, turns: int, model_alias: str) -> None:
        self._session = session
        self._turns = turns
        self._alias = model_alias
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": _AGENT_SYSTEM},
            {"role": "user", "content": task_description},
        ]
        self._finished_len: int | None = None
        self._finished_sha256: str | None = None
        self._turn = 0

    def next_step(self, sequence: int) -> LaneStep | None:
        if sequence >= self._turns:
            return None
        self._turn = sequence
        prefix_sha256 = None
        prefix_matches = None
        if self._finished_len is not None:
            prefix_sha256 = _messages_sha256(self._messages[: self._finished_len])
            prefix_matches = prefix_sha256 == self._finished_sha256
        payload = chat_payload(
            self._messages,
            seed=_AGENT_SEED_BASE + self._session,
            max_tokens=_AGENT_ANSWER_TOKENS,
            model_alias=self._alias,
        )
        return LaneStep(
            payload=payload,
            extra={
                "session": self._session,
                "turn": sequence,
                "messages_sha256": _messages_sha256(self._messages),
                "prefix_sha256": prefix_sha256,
                "previous_finished_sha256": self._finished_sha256,
                "prefix_matches_previous": prefix_matches,
            },
        )

    def observe(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})
        self._finished_len = len(self._messages)
        self._finished_sha256 = _messages_sha256(self._messages)
        self._messages.append({"role": "user", "content": follow_up_message(self._turn + 1)})

    def annotate(self, record: dict[str, Any]) -> None:
        """Add the per-turn numbers: time to first token, decode rate, and cache reuse."""
        timings = record.get("server_timings") or {}
        first_token = record.get("first_token_mono")
        record["ttft_s"] = None if first_token is None else first_token - record["dispatch_mono"]
        predicted_n, predicted_ms = timings.get("predicted_n"), timings.get("predicted_ms")
        record["decode_tok_s"] = predicted_n * 1000.0 / predicted_ms if predicted_n and predicted_ms else None
        record["cache_n"] = timings.get("cache_n")
        record["prompt_n"] = timings.get("prompt_n")
        prompt_tokens = record.get("prompt_tokens")
        cache_n = record["cache_n"]
        record["reuse_fraction"] = cache_n / prompt_tokens if prompt_tokens and cache_n is not None else None


def default_corpus_root() -> Path:
    """The codeeval corpus: `INFERFERENCE_CODEEVAL_CORPUS`, else `tasks/codeeval` of this checkout."""
    override = os.environ.get("INFERFERENCE_CODEEVAL_CORPUS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "tasks" / "codeeval"


def corpus_prompts(set_name: str, *, root: Path | None = None) -> tuple[CorpusPrompt, ...]:
    """Render the prompts of one corpus set, in corpus order.

    `set_name` is `all`, or one family of the corpus. An unknown family, or a
    corpus that cannot load, raises `WorkloadError`: an empty set never runs.
    """
    from ..codeeval.render import render
    from ..codeeval.task import FAMILIES, CorpusError, load_task_set

    if set_name != "all" and set_name not in FAMILIES:
        raise WorkloadError(f"unknown corpus set {set_name!r}; known: all, {', '.join(FAMILIES)}")
    corpus_root = default_corpus_root() if root is None else root
    try:
        task_set = load_task_set(corpus_root)
    except CorpusError as exc:
        raise WorkloadError(f"corpus {corpus_root} does not load: {exc}") from exc
    selected = task_set if set_name == "all" else task_set.slice(families=[set_name])
    if not selected.tasks:
        raise WorkloadError(f"corpus set {set_name!r} holds no tasks in {corpus_root}")
    prompts: list[CorpusPrompt] = []
    for index, task in enumerate(selected.tasks):
        rendered = render(task)
        prompts.append(
            CorpusPrompt(
                task_id=task.id,
                index=index,
                messages=tuple(rendered.messages()),
                max_tokens=task.max_output_tokens,
                prompt_sha256=rendered.prompt_sha256,
            )
        )
    return tuple(prompts)


def assign_corpus_prompts(prompts: Sequence[CorpusPrompt], lanes: int) -> list[list[CorpusPrompt]]:
    """Deal `prompts` to `lanes` lanes round robin. One lane keeps the corpus order."""
    if lanes < 1:
        raise WorkloadError(f"lanes must be at least 1, got {lanes}")
    return [list(prompts[lane::lanes]) for lane in range(lanes)]


def records_prompt(record_count: int, *, salt: int = 0) -> tuple[str, dict[str, str]]:
    """Build a "read every record" prompt with `record_count` rows.

    Returns `(content, expected)`. `expected` names the final record's id,
    status, and value — the facts a prefill validator checks for, as proof
    the whole prompt was read, not merely the end of it.

    `salt` picks another record set of the same shape: the identifiers stay,
    and the other fields change. Salt 0 is the set that prefill workloads use.
    """
    if record_count < 1:
        raise WorkloadError(f"record_count must be at least 1, got {record_count}")
    last = record_count - 1
    offset = salt * 104_729

    def field_seed(index: int) -> int:
        return index + offset

    rows = [
        f"S-{index:04d} region={(field_seed(index) * 7) % 13:02d} status="
        f"{'GREEN' if index == last else ('WARN' if field_seed(index) % 17 == 0 else 'OK')} "
        f"latency_ms={31 + (field_seed(index) * 29) % 970} queue={(field_seed(index) * 11) % 41:02d} "
        f"checksum={(field_seed(index) * 811) % 65521:05d} value={field_seed(index) % 997}"
        for index in range(record_count)
    ]
    content = (
        "Review every record below. Report the final record identifier, status, and "
        "value. Also give a short summary of any warning records. Do not invent "
        "missing facts.\n\n"
        + "\n".join(rows)
        + f"\nThe final record has value={field_seed(last) % 997}. State its exact identifier and status."
    )
    expected = {
        "expected_id": f"S-{last:04d}",
        "expected_status": "GREEN",
        "expected_value": str(field_seed(last) % 997),
    }
    return content, expected


def calibrate_prefill(
    target_tokens: int,
    tokenize: Callable[[str], int],
    *,
    tolerance: float = 0.02,
) -> tuple[str, dict[str, Any]]:
    """Bisect `records_prompt`'s record count to hit `target_tokens` content tokens.

    `tokenize` counts content tokens for one prompt (as `/tokenize` does for
    the full user-message content, not the whole request). The search
    brackets `target_tokens` within `tolerance` (a fraction, default 2%) and
    returns the winning prompt plus calibration data: `record_count`,
    `content_tokens`, the three expected facts, and every attempt tried.

    Raises `WorkloadError` if no record count in range comes within
    tolerance — a target too small to reach (below the floor at
    `record_count=1`) or too large to reach within `_MAX_RECORD_COUNT`
    records never returns a best guess silently.
    """
    if target_tokens <= 0:
        raise WorkloadError(f"target_tokens must be positive, got {target_tokens}")

    lo, hi = 1, _MAX_RECORD_COUNT
    attempts: list[dict[str, Any]] = []
    best: tuple[int, int, str, dict[str, str]] | None = None

    while lo <= hi:
        count = (lo + hi) // 2
        content, expected = records_prompt(count)
        tokens = tokenize(content)
        attempts.append({"record_count": count, "content_tokens": tokens})
        if best is None or abs(tokens - target_tokens) < abs(best[1] - target_tokens):
            best = (count, tokens, content, expected)
        if abs(tokens - target_tokens) <= tolerance * target_tokens:
            break
        if tokens < target_tokens:
            lo = count + 1
        else:
            hi = count - 1

    assert best is not None  # the loop always runs at least once (lo=1 <= hi)
    record_count, content_tokens, content, expected = best
    if abs(content_tokens - target_tokens) > tolerance * target_tokens:
        raise WorkloadError(
            f"could not reach {target_tokens} content tokens within {tolerance:.0%}: "
            f"closest attempt was {content_tokens} tokens at record_count={record_count}"
        )
    return content, {
        "record_count": record_count,
        "content_tokens": content_tokens,
        "expected_id": expected["expected_id"],
        "expected_status": expected["expected_status"],
        "expected_value": expected["expected_value"],
        "attempts": attempts,
    }


def validate_response(
    kind: str,
    result: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
) -> str | None:
    """Check one decoded response. Returns `None` when valid, else the reason.

    `result` is plain data the caller already decoded from the raw SSE
    stream: `http_status`, `prompt_tokens`, `completion_tokens`,
    `finish_reason`, `content`, and `reasoning_content` (the last two may be
    absent or empty). This module never reads the stream itself.

    Validation is minimal and model-robust: HTTP 200; both token counts
    above zero; non-empty content, counting `reasoning_content` too;
    `finish_reason` present. A `kind == "prefill"` response additionally
    needs `expected`'s record id, status, and value to appear in the
    content — proof the whole prompt was read. An `agent_turns` response
    needs answer content: a reply of reasoning only would add an empty
    assistant message to the conversation, so it is refused. No other kind
    checks the prose's wording or format.
    """
    http_status = result.get("http_status")
    if http_status != 200:
        return f"HTTP {http_status}"

    prompt_tokens = result.get("prompt_tokens") or 0
    if prompt_tokens <= 0:
        return "prompt_tokens was not above zero"

    completion_tokens = result.get("completion_tokens") or 0
    if completion_tokens <= 0:
        return "completion_tokens was not above zero"

    content = result.get("content") or ""
    reasoning_content = result.get("reasoning_content") or ""
    if not (content + reasoning_content).strip():
        return "response content was empty"

    if not result.get("finish_reason"):
        return "finish_reason was missing"

    if kind == "agent_turns" and not content.strip():
        return "agent_turns reply held reasoning only and no answer content"

    if kind == "prefill":
        if expected is None:
            return "prefill validation needs the expected record facts"
        for key in ("expected_id", "expected_status", "expected_value"):
            fact = expected.get(key)
            if fact is None or str(fact) not in content:
                return f"answer omitted the expected {key}: {fact!r}"

    return None
