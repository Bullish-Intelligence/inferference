"""The drive loop: one request at a time, or many in flight.

``run_single`` sends one request at a time. It answers best-case latency: no
queueing, no sharing of the decode batch.

``run_batched`` puts ``concurrency`` requests in flight at once. It answers what
the rig does under load — aggregate throughput, and what each request pays for
it.

CONCURRENCY IS NOT BATCHING, AND THE ARTIFACT SAYS WHICH IS WHICH. This harness
controls how many requests it has OUTSTANDING. The server decides how many it
decodes together, and that is its ``--parallel`` lane count, fixed when the
model was loaded. Ask for more concurrency than the model has lanes and the
surplus does not batch — it QUEUES, and the extra time lands in time-to-first-
token while decode tok/s stays flat. The two numbers are recorded separately in
the envelope for exactly this reason.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .client import ChatClient, Endpoint, Response
from .render import render
from .score import extract_artifact, grade_parse, grade_pytest, grade_ruff
from .score.result import LayerResult, TaskScore
from .task import Task, TaskSet
from .thermal import ThermalAbort, ThermalPacer

__all__ = ["DriveResult", "TaskOutcome", "grade", "run_batched", "run_single"]


@dataclass
class TaskOutcome:
    """One task: what was sent, what came back, and how it scored."""

    task_id: str
    family: str
    difficulty: int
    prompt_sha256: str
    approx_prompt_tokens: int
    response: Response
    score: TaskScore
    artifact: str | None = None
    skipped: str | None = None
    repeat: int = 0
    submitted_at: float = 0.0
    """perf_counter when the request went out. Under concurrency the ORDER
    requests were issued in is not the order they finished, and a queued request
    is only visible by comparing the two."""

    def record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "difficulty": self.difficulty,
            "repeat": self.repeat,
            "prompt_sha256": self.prompt_sha256,
            "approx_prompt_tokens": self.approx_prompt_tokens,
            "skipped": self.skipped,
            "response": {
                "text": self.response.text,
                "reasoning": self.response.reasoning,
                "finish_reason": self.response.finish_reason,
                "error": self.response.error,
                "timings": self.response.timings,
                "usage": self.response.raw_usage,
            },
            "score": self.score.model_dump(),
        }


@dataclass
class DriveResult:
    """Every outcome from one run, plus the run's own bookkeeping."""

    outcomes: list[TaskOutcome] = field(default_factory=list)
    started_at: float = 0.0
    ended_at: float = 0.0
    notes: list[str] = field(default_factory=list)
    mode: str = "single"
    concurrency: int = 1
    idle_s: float = 0.0
    aborted: str | None = None
    """Why the run was cut short, if it was. A run with this set is VOID: its
    timings describe a card that was throttling."""
    """Wall time with NO request in flight, because the pacer was holding the
    card back. It is not work, and a rate computed over it would be measuring
    the cooling system rather than the build."""
    thermal: dict[str, Any] = field(default_factory=dict)

    @property
    def wall_s(self) -> float:
        return self.ended_at - self.started_at

    @property
    def active_s(self) -> float:
        """Wall clock minus the deliberate idle. The denominator for any rate."""
        return max(0.0, self.wall_s - self.idle_s)

    @property
    def completion_tokens(self) -> int:
        """Answer tokens produced across the run.

        Counts what the SERVER says it predicted, which on a reasoning model
        includes its thinking. That is the right denominator for a throughput
        figure: the hardware decoded those tokens whether or not they were the
        answer.
        """
        total = 0
        for o in self.outcomes:
            n = (o.response.timings or {}).get("n_predicted")
            if isinstance(n, (int, float)):
                total += int(n)
        return total

    @property
    def aggregate_tok_s(self) -> float | None:
        """Answer tokens per second of ACTIVE time.

        This is the number that should improve with concurrency. Per-request
        decode rate is expected to get worse at the same time; both are kept.

        Active, not wall: on a thermally paced run the wall clock includes time
        the card was deliberately idle. Dividing by it would report a slower
        build when all that changed was the pacing.
        """
        if self.active_s <= 0 or not self.completion_tokens:
            return None
        return self.completion_tokens / self.active_s

    @property
    def wall_tok_s(self) -> float | None:
        """Tokens per second of WALL time, pauses included.

        What the rig delivers end to end today, cooling and all. Kept beside the
        active rate because they answer different questions: this one sizes a
        job, the active one compares two builds.
        """
        if self.wall_s <= 0 or not self.completion_tokens:
            return None
        return self.completion_tokens / self.wall_s

    @property
    def requests_per_s(self) -> float | None:
        done = [o for o in self.outcomes if o.skipped is None]
        if self.active_s <= 0 or not done:
            return None
        return len(done) / self.active_s


def grade(task: Task, response: Response, workdir: Path) -> tuple[str | None, TaskScore]:
    """Run every declared layer, cheapest first. A skip is recorded as a skip."""
    score = TaskScore(
        task_id=task.id,
        family=task.family,
        difficulty=task.difficulty,
        request_ok=response.ok,
    )
    if not response.ok:
        score.layers.append(
            LayerResult(
                layer="extract",
                status="fail",
                reason=f"request failed: {response.error}",
            )
        )
        return None, score

    artifact, extracted = extract_artifact(task, response.text)
    if extracted.status == "fail" and response.finish_reason == "length":
        # Name the cause. A truncated answer is a budget defect, not a
        # capability defect, and the two must not be summed together.
        extracted = extracted.model_copy(update={"reason": f"{extracted.reason} (finish_reason=length)"})
    score.layers.append(extracted)
    if artifact is None:
        return None, score

    workdir.mkdir(parents=True, exist_ok=True)
    artifact_name = Path(task.output.path).name if task.output.path else "artifact.txt"
    artifact_path = workdir / artifact_name
    artifact_path.write_text(artifact)

    if task.grading.parse:
        score.layers.append(grade_parse(task, artifact))
    if task.grading.ruff:
        score.layers.append(grade_ruff(task, artifact_path))
    if task.grading.pytest:
        score.layers.append(grade_pytest(task, artifact, workdir / "run", timeout_s=120.0))
    if task.grading.judge:
        # Step 8. Declared but not wired: a loud skip, never an assumed pass.
        score.layers.append(LayerResult(layer="judge", status="skip", reason="judge layer not implemented yet"))
    return artifact, score


@dataclass
class _Runner:
    """Everything one task needs, so single and batched share a code path."""

    chat: ChatClient
    records_dir: Path
    artifacts_dir: Path
    max_prompt_tokens: int | None
    structured: bool
    token_budget_scale: float
    reasoning_headroom: int
    repeat: int = 0

    async def one(self, task: Task, http: httpx.AsyncClient) -> TaskOutcome:
        prompt = render(task)

        # No truncation, ever. A prompt that does not fit is skipped and said to
        # be skipped.
        if self.max_prompt_tokens and prompt.approx_tokens > self.max_prompt_tokens:
            outcome = TaskOutcome(
                task_id=task.id,
                family=task.family,
                difficulty=task.difficulty,
                prompt_sha256=prompt.prompt_sha256,
                approx_prompt_tokens=prompt.approx_tokens,
                response=Response(task_id=task.id, text="", error="skipped"),
                score=TaskScore(
                    task_id=task.id,
                    family=task.family,
                    difficulty=task.difficulty,
                    request_ok=False,
                ),
                skipped=(
                    f"prompt ~{prompt.approx_tokens} tokens exceeds the {self.max_prompt_tokens} limit; not truncated"
                ),
            )
            self._write(outcome)
            return outcome

        response_format = None
        if self.structured and task.output.kind == "json":
            response_format = {"type": "json_object"}

        submitted = time.perf_counter()
        response = await self.chat.complete(
            task_id=task.id,
            messages=prompt.messages(),
            max_tokens=(int(task.max_output_tokens * self.token_budget_scale) + self.reasoning_headroom),
            client=http,
            response_format=response_format,
        )
        # Grading is CPU work. Under concurrency it must not sit in the event
        # loop between a response arriving and the next request being issued, so
        # it is pushed to a thread.
        workdir = self.artifacts_dir / f"{task.id.replace('/', '__')}"
        if self.repeat:
            workdir = workdir.with_name(f"{workdir.name}__r{self.repeat}")
        artifact, score = await asyncio.to_thread(grade, task, response, workdir)
        outcome = TaskOutcome(
            task_id=task.id,
            family=task.family,
            difficulty=task.difficulty,
            prompt_sha256=prompt.prompt_sha256,
            approx_prompt_tokens=prompt.approx_tokens,
            response=response,
            score=score,
            artifact=artifact,
            repeat=self.repeat,
            submitted_at=submitted,
        )
        self._write(outcome)
        return outcome

    def _write(self, outcome: TaskOutcome) -> None:
        name = outcome.task_id.replace("/", "__")
        if self.repeat:
            name = f"{name}__r{self.repeat}"
        (self.records_dir / f"{name}.json").write_text(json.dumps(outcome.record(), indent=2, sort_keys=True))


class _IdleClock:
    """Accumulates wall time during which NOTHING was in flight.

    Under concurrency the pacer may hold one request back while others are still
    decoding — the card is working, so that is not idle. Only the stretches with
    zero requests outstanding count against active time.
    """

    def __init__(self) -> None:
        self.idle_s = 0.0
        self._in_flight = 0
        self._since: float | None = time.perf_counter()

    def enter(self) -> None:
        if self._in_flight == 0 and self._since is not None:
            self.idle_s += time.perf_counter() - self._since
            self._since = None
        self._in_flight += 1

    def exit(self) -> None:
        self._in_flight -= 1
        if self._in_flight == 0:
            self._since = time.perf_counter()

    def close(self) -> None:
        if self._in_flight == 0 and self._since is not None:
            self.idle_s += time.perf_counter() - self._since
            self._since = None


async def _with_thermal_guard(work: Any, pacer: ThermalPacer) -> str | None:
    """Run ``work`` while the pacer watches the card. Returns an abort reason.

    The watchdog is the only thing standing between a long request and a cooked
    card, because the pacer itself cannot act until the request returns.
    """
    if not pacer.enabled:
        await work
        return None
    task = asyncio.ensure_future(work)
    guard = asyncio.ensure_future(pacer.guard())
    done, _ = await asyncio.wait({task, guard}, return_when=asyncio.FIRST_COMPLETED)
    if guard in done:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - the run is void either way
            pass
        try:
            guard.result()
        except ThermalAbort as exc:
            return str(exc)
        return "thermal guard stopped without a reason"
    guard.cancel()
    try:
        await guard
    except asyncio.CancelledError:
        pass
    await task
    return None


def _prepare(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records_dir = out_dir / "records"
    artifacts_dir = out_dir / "artifacts"
    records_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    return records_dir, artifacts_dir


async def run_single(
    task_set: TaskSet,
    endpoint: Endpoint,
    *,
    out_dir: Path,
    temperature: float = 0.0,
    max_prompt_tokens: int | None = None,
    structured: bool = False,
    token_budget_scale: float = 1.0,
    reasoning_headroom: int = 0,
    repeat: int = 0,
    pacer: ThermalPacer | None = None,
    progress: Any = None,
) -> DriveResult:
    """Send every task once, in corpus order, one at a time."""
    records_dir, artifacts_dir = _prepare(out_dir)
    runner = _Runner(
        chat=ChatClient(endpoint=endpoint, temperature=temperature),
        records_dir=records_dir,
        artifacts_dir=artifacts_dir,
        max_prompt_tokens=max_prompt_tokens,
        structured=structured,
        token_budget_scale=token_budget_scale,
        reasoning_headroom=reasoning_headroom,
        repeat=repeat,
    )
    pacer = pacer or ThermalPacer()
    clock = _IdleClock()
    result = DriveResult(started_at=time.time(), mode="single", concurrency=1)
    async with httpx.AsyncClient(timeout=endpoint.timeout_s) as http:

        async def sequence() -> None:
            for index, task in enumerate(task_set.tasks, start=1):
                # Between requests only. Pausing inside one would land in its
                # own decode timing.
                await pacer.wait_if_hot()
                clock.enter()
                try:
                    outcome = await runner.one(task, http)
                finally:
                    clock.exit()
                result.outcomes.append(outcome)
                if progress:
                    progress(index, len(task_set), outcome)

        result.aborted = await _with_thermal_guard(sequence(), pacer)
    clock.close()
    result.ended_at = time.time()
    result.idle_s = clock.idle_s
    result.thermal = pacer.record()
    return result


async def run_batched(
    task_set: TaskSet,
    endpoint: Endpoint,
    *,
    out_dir: Path,
    concurrency: int,
    temperature: float = 0.0,
    max_prompt_tokens: int | None = None,
    structured: bool = False,
    token_budget_scale: float = 1.0,
    reasoning_headroom: int = 0,
    repeat: int = 0,
    pacer: ThermalPacer | None = None,
    progress: Any = None,
) -> DriveResult:
    """Keep ``concurrency`` requests in flight until every task is done.

    The tasks are independent, so this is a throughput measurement, not a
    conversation. Order of completion is not corpus order and is not meant to be.

    What this does NOT control is whether the server decodes those requests
    together. That is its ``--parallel`` lane count. Above it, requests queue —
    which shows up as time-to-first-token growing while decode tok/s holds
    steady. The envelope records both numbers so the two cases can be told apart.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    records_dir, artifacts_dir = _prepare(out_dir)
    runner = _Runner(
        chat=ChatClient(endpoint=endpoint, temperature=temperature),
        records_dir=records_dir,
        artifacts_dir=artifacts_dir,
        max_prompt_tokens=max_prompt_tokens,
        structured=structured,
        token_budget_scale=token_budget_scale,
        reasoning_headroom=reasoning_headroom,
        repeat=repeat,
    )
    pacer = pacer or ThermalPacer()
    clock = _IdleClock()
    result = DriveResult(started_at=time.time(), mode="batch", concurrency=concurrency)
    gate = asyncio.Semaphore(concurrency)
    done = 0

    async with httpx.AsyncClient(
        timeout=endpoint.timeout_s,
        # The default pool caps at 10 keep-alive connections. Above that,
        # requests would serialise on the CLIENT and the run would report the
        # server as saturated when it was not.
        limits=httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4),
    ) as http:

        async def worker(task: Task) -> TaskOutcome:
            nonlocal done
            async with gate:
                await pacer.wait_if_hot()
                clock.enter()
                try:
                    outcome = await runner.one(task, http)
                finally:
                    clock.exit()
            done += 1
            if progress:
                progress(done, len(task_set), outcome)
            return outcome

        gathered: list[TaskOutcome] = []

        async def fan_out() -> None:
            gathered.extend(await asyncio.gather(*(worker(t) for t in task_set.tasks)))

        result.aborted = await _with_thermal_guard(fan_out(), pacer)

    clock.close()
    result.ended_at = time.time()
    result.idle_s = clock.idle_s
    result.thermal = pacer.record()
    # Report in corpus order however they finished, so two runs diff cleanly.
    order = {t.id: i for i, t in enumerate(task_set.tasks)}
    result.outcomes = sorted(gathered, key=lambda o: order.get(o.task_id, 0))
    return result
