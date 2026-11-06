"""Request generators for the experiment runner: payloads in, verdicts out.

This module builds the HTTP request bodies a workload sends and checks the
plain-data results the caller passes back. It never opens a socket. The
runner owns the HTTP call; this module only owns the shape of the request and
the rule for calling a response valid.

A workload name is one of four spec strings: `prefill_<N>` (prefill `N`
content tokens), `decode_single_stream` (one long, unassisted decode),
`lane_sweep` (concurrency drawn from the experiment's `lanes` factor), or
`soak_<N>s` (a fixed-duration run, concurrency also from `lanes`). A bare
`"decode"` or `"prefill"` is refused, because neither names a size.

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
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

# A record count never goes below 1, and this ceiling keeps a bisection over
# an unreachable target bounded instead of searching forever.
_MAX_RECORD_COUNT = 100_000

# `decode_single_stream` has no target token count: it is a fixed prompt
# sized to run a long, uninterrupted decode.
_DECODE_MAX_TOKENS = 900

_PREFILL_NAME_RE = re.compile(r"^prefill_(\d+)$")
_SOAK_NAME_RE = re.compile(r"^soak_(\d+)s$")

WorkloadKind = Literal["prefill", "decode", "lane_sweep", "soak"]


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


def parse_workload(name: str) -> WorkloadPlan:
    """Parse a workload name into a `WorkloadPlan`.

    Only four shapes are accepted: `prefill_<N>`, `decode_single_stream`,
    `lane_sweep`, and `soak_<N>s`. A bare `decode` or `prefill`, with no
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
    raise WorkloadError(f"unknown workload name: {name!r}")


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


def records_prompt(record_count: int) -> tuple[str, dict[str, str]]:
    """Build a "read every record" prompt with `record_count` rows.

    Returns `(content, expected)`. `expected` names the final record's id,
    status, and value — the facts a prefill validator checks for, as proof
    the whole prompt was read, not merely the end of it.
    """
    if record_count < 1:
        raise WorkloadError(f"record_count must be at least 1, got {record_count}")
    last = record_count - 1
    rows = [
        f"S-{index:04d} region={(index * 7) % 13:02d} status="
        f"{'GREEN' if index == last else ('WARN' if index % 17 == 0 else 'OK')} "
        f"latency_ms={31 + (index * 29) % 970} queue={(index * 11) % 41:02d} "
        f"checksum={(index * 811) % 65521:05d} value={index % 997}"
        for index in range(record_count)
    ]
    content = (
        "Review every record below. Report the final record identifier, status, and "
        "value. Also give a short summary of any warning records. Do not invent "
        "missing facts.\n\n"
        + "\n".join(rows)
        + f"\nThe final record has value={last % 997}. State its exact identifier and status."
    )
    expected = {
        "expected_id": f"S-{last:04d}",
        "expected_status": "GREEN",
        "expected_value": str(last % 997),
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
    content — proof the whole prompt was read. No other kind checks the
    prose's wording or format.
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

    if kind == "prefill":
        if expected is None:
            return "prefill validation needs the expected record facts"
        for key in ("expected_id", "expected_status", "expected_value"):
            fact = expected.get(key)
            if fact is None or str(fact) not in content:
                return f"answer omitted the expected {key}: {fact!r}"

    return None
