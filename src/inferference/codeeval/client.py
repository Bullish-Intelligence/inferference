"""The HTTP client, and the timings taken from one request.

OpenAI-compatible chat completions, always streamed. Streaming is not optional:
time-to-first-token cannot be recovered from a buffered response, and TTFT is
the felt latency of an editor completion.

The same client drives the local router and a hosted baseline. What differs is
an :class:`Endpoint` — a base URL, an API key variable and a model string.
Nothing else in the harness knows which is which.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ChatClient",
    "Endpoint",
    "RequestError",
    "RequestTimings",
    "Response",
    "load_dotenv",
]


class RequestError(RuntimeError):
    """The endpoint refused, failed, or returned something unusable."""


def load_dotenv(path: Path | str = ".env") -> dict[str, str]:
    """Read ``KEY=value`` lines into the process environment, without overwrite.

    The repo keeps secrets in a gitignored ``.env``. Existing environment values
    win, so a run can override the file from the command line.
    """
    found: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return found
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        found[key] = value
        os.environ.setdefault(key, value)
    return found


class Endpoint(BaseModel):
    """Where to send requests, and as what model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    hosted: bool = False
    timeout_s: float = 600.0

    def auth_header(self) -> dict[str, str]:
        if not self.api_key_env:
            return {}
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RequestError(f"{self.api_key_env} is not set; the harness will not run {self.name!r} without it")
        return {"Authorization": f"Bearer {key}"}

    def redacted(self) -> dict[str, Any]:
        """Envelope-safe identity. Never carries the key itself."""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "hosted": self.hosted,
            "api_key_env": self.api_key_env,
        }


@dataclass(frozen=True, slots=True)
class RequestTimings:
    """Client-side and server-side timings for one request.

    ``prefill_tok_s`` and ``decode_tok_s`` come from the server when it reports
    them. A hosted endpoint usually does not, so they stay ``None`` rather than
    being derived from client-side wall time, which would silently include the
    network.
    """

    ttft_ms: float | None
    """Time to the first ANSWER token. On a reasoning model this is far later
    than the first token of any kind, because reasoning streams first."""

    ttft_any_ms: float | None = None
    """Time to the first token of any kind, reasoning included."""

    e2e_ms: float = 0.0
    decode_wall_ms: float | None = None
    """Wall time from the first token of any kind to the last."""

    n_prompt: int | None = None
    n_predicted: int | None = None
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    queue_ms: float | None = None
    n_reasoning_tokens: int | None = None

    @property
    def client_decode_tok_s(self) -> float | None:
        """Decode rate measured at the client. Includes the network; say so."""
        if not self.n_predicted or not self.decode_wall_ms or self.n_predicted < 2:
            return None
        return (self.n_predicted - 1) / (self.decode_wall_ms / 1000.0)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "ttft_ms": self.ttft_ms,
            "e2e_ms": self.e2e_ms,
            "decode_wall_ms": self.decode_wall_ms,
            "n_prompt": self.n_prompt,
            "n_predicted": self.n_predicted,
            "prefill_tok_s": self.prefill_tok_s,
            "decode_tok_s": self.decode_tok_s,
            "queue_ms": self.queue_ms,
            "ttft_any_ms": self.ttft_any_ms,
            "n_reasoning_tokens": self.n_reasoning_tokens,
            "client_decode_tok_s": self.client_decode_tok_s,
        }
        return d


class Response(BaseModel):
    """One completed generation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_id: str
    text: str
    reasoning: str = ""
    """A reasoning model's thinking, kept out of `text` so grading never sees it."""

    finish_reason: str | None = None
    timings: dict[str, Any] = Field(default_factory=dict)
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ChatClient:
    """Streamed chat completions with per-request timing."""

    endpoint: Endpoint
    temperature: float = 0.0
    extra_body: dict[str, Any] = field(default_factory=dict)

    async def complete(
        self,
        *,
        task_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        client: httpx.AsyncClient,
        response_format: dict[str, Any] | None = None,
    ) -> Response:
        body: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if response_format is not None:
            body["response_format"] = response_format
        body.update(self.extra_body)

        headers = {"Content-Type": "application/json", **self.endpoint.auth_header()}
        url = self.endpoint.base_url.rstrip("/") + "/chat/completions"

        chunks: list[str] = []
        reasoning: list[str] = []
        started = time.perf_counter()
        first_at: float | None = None
        first_any_at: float | None = None
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        server_timings: dict[str, Any] = {}

        try:
            async with client.stream(
                "POST",
                url,
                json=body,
                headers=headers,
                timeout=self.endpoint.timeout_s,
            ) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode(errors="replace")[:600]
                    raise RequestError(f"HTTP {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    if event.get("timings"):
                        server_timings = event["timings"]
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        # A reasoning model streams `reasoning_content` before
                        # any answer. Keep the two apart: mixing them would put
                        # thinking into the graded artifact and would make the
                        # answer look far slower to start than it is.
                        think = delta.get("reasoning_content")
                        if think:
                            if first_any_at is None:
                                first_any_at = time.perf_counter()
                            reasoning.append(think)
                        piece = delta.get("content")
                        if piece:
                            now = time.perf_counter()
                            if first_any_at is None:
                                first_any_at = now
                            if first_at is None:
                                first_at = now
                            chunks.append(piece)
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
        except RequestError as exc:
            return Response(task_id=task_id, text="", error=str(exc))
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            return Response(task_id=task_id, text="", error=f"{type(exc).__name__}: {exc}")

        ended = time.perf_counter()
        text = "".join(chunks)
        think_text = "".join(reasoning)
        reasoning_tokens = _pick(usage, "reasoning_tokens") or (
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        )
        timings = RequestTimings(
            ttft_ms=None if first_at is None else (first_at - started) * 1000.0,
            ttft_any_ms=(None if first_any_at is None else (first_any_at - started) * 1000.0),
            n_reasoning_tokens=(reasoning_tokens if isinstance(reasoning_tokens, int) else None),
            e2e_ms=(ended - started) * 1000.0,
            # The generation window opens at the first token of ANY kind.
            # `n_predicted` counts reasoning tokens too, so dividing it by the
            # answer-only window would overstate the rate several times over.
            decode_wall_ms=(None if first_any_at is None else (ended - first_any_at) * 1000.0),
            n_prompt=_pick(usage, "prompt_tokens") or server_timings.get("prompt_n"),
            n_predicted=(_pick(usage, "completion_tokens") or server_timings.get("predicted_n")),
            prefill_tok_s=server_timings.get("prompt_per_second"),
            decode_tok_s=server_timings.get("predicted_per_second"),
        )
        if not text:
            # No ANSWER, but the card still decoded every reasoning token. Keep
            # the timings: a request that spent its whole budget thinking is a
            # quality failure, not an absence of work, and dropping its tokens
            # made aggregate throughput undercount badly while wall clock still
            # counted the time it took.
            #
            # Say WHY it is empty. "Ran out of budget while reasoning" is a
            # different defect from "the model answered with nothing", and the
            # fix for it is a bigger budget, not a better prompt.
            reason = (
                "token budget exhausted during reasoning; no answer emitted"
                if finish_reason == "length" and think_text
                else "empty response"
            )
            return Response(
                task_id=task_id,
                text="",
                reasoning=think_text,
                finish_reason=finish_reason,
                timings=timings.as_dict(),
                raw_usage=usage,
                error=reason,
            )

        return Response(
            task_id=task_id,
            text=text,
            reasoning=think_text,
            finish_reason=finish_reason,
            timings=timings.as_dict(),
            raw_usage=usage,
        )


def _pick(d: dict[str, Any], key: str) -> Any:
    value = d.get(key)
    return value if isinstance(value, int) else None
