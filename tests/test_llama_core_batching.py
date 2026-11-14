"""Tests for the continuous / dynamic batching scheduler (Pillar 4).

The unit tests drive :class:`ContinuousBatchScheduler` against an in-memory
:class:`FakeBackend` (KV modeled as a per-slot token list, deterministic scripted
sampling) so admission, slot reuse, finish-reason accounting, and the
observability counters are exercised with no GPU.  The final test is a GPU-gated
integration proving the scheduler produces correct, non-empty results for a set
of prompts and that a batch fitting in one wave matches the router's ``run``.
"""

from __future__ import annotations

import pytest

from inferference.batching import (
    BatchConfig,
    BatchDecodeBackend,
    BatchRequest,
    ContinuousBatchScheduler,
    LlamaContinuousBatchEngine,
)
from inferference.decode import FINISH_LENGTH, FINISH_STOP
from inferference.gpu_env import gpu_env_ready
from tests.asset_paths import asset_path

EOS = 0


class FakeBackend:
    """Deterministic, GPU-free backend: KV is a per-slot scripted token list.

    ``plans`` maps a prompt to the content tokens it will emit; include ``EOS`` in
    a plan to force a clean stop, or leave it out (and cap with ``max_tokens``) to
    force a length cut.  Slot state is keyed by ``seq_id`` so reuse after a
    ``free_slot`` is observable.
    """

    def __init__(self, n_seq_max: int, plans: dict[str, list[int]]) -> None:
        self._n = n_seq_max
        self.plans = plans
        self._pid: dict[str, int] = {}
        self._by_tokens: dict[tuple[int, ...], str] = {}
        self._slot: dict[int, list] = {}
        self.freed: list[int] = []
        self.step_widths: list[int] = []

    @property
    def n_seq_max(self) -> int:
        return self._n

    @property
    def eos(self) -> int:
        return EOS

    def tokenize(self, prompt: str) -> list[int]:
        pid = self._pid.setdefault(prompt, len(self._pid) + 1)
        toks = [pid] * max(len(prompt), 1)
        self._by_tokens[tuple(toks)] = prompt
        return toks

    def detokenize(self, tokens) -> str:
        return " ".join(str(t) for t in tokens)

    def free_slot(self, seq_id: int) -> None:
        self.freed.append(seq_id)
        self._slot.pop(seq_id, None)

    def _next(self, seq_id: int) -> int:
        plan, idx = self._slot[seq_id]
        if idx < len(plan):
            self._slot[seq_id][1] = idx + 1
            return plan[idx]
        return EOS  # plan exhausted -> clean stop

    def prefill(self, tokens, seq_id: int) -> int:
        prompt = self._by_tokens[tuple(tokens)]
        self._slot[seq_id] = [list(self.plans[prompt]), 0]
        return self._next(seq_id)

    def step(self, active):
        self.step_widths.append(len(active))
        return [self._next(seq_id) for (seq_id, _tok, _pos) in active]


def test_backend_satisfies_protocol() -> None:
    assert isinstance(FakeBackend(2, {}), BatchDecodeBackend)


def test_single_request_stops_on_eos() -> None:
    backend = FakeBackend(2, {"hi": [5, 6, EOS]})
    sched = ContinuousBatchScheduler(backend, BatchConfig(n_seq_max=2, default_max_tokens=10))
    run = sched.run([BatchRequest(prompt="hi", request_id="r0")])

    (res,) = run.results
    assert res.token_ids == (5, 6)  # EOS is the boundary, not content
    assert res.finish_reason == FINISH_STOP
    assert res.completion_token_count == 2
    assert res.request_id == "r0"
    assert run.stats.prefills == 1
    assert run.stats.steps == 2  # first token from prefill, then 6, then EOS
    assert run.stats.tokens_generated == 2
    assert run.stats.max_concurrency == 1
    assert run.stats.admission_waits == 0
    assert run.stats.requests_completed == 1


def test_length_cut_when_plan_exceeds_budget() -> None:
    backend = FakeBackend(2, {"p": [1, 2, 3, 4, 5]})
    sched = ContinuousBatchScheduler(backend)
    run = sched.run([BatchRequest(prompt="p", max_tokens=3)])

    (res,) = run.results
    assert res.token_ids == (1, 2, 3)
    assert res.finish_reason == FINISH_LENGTH
    assert run.stats.tokens_generated == 3


def test_max_tokens_one_finishes_at_prefill() -> None:
    backend = FakeBackend(2, {"p": [7, 8, 9]})
    sched = ContinuousBatchScheduler(backend)
    run = sched.run([BatchRequest(prompt="p", max_tokens=1)])

    (res,) = run.results
    assert res.token_ids == (7,)
    assert res.finish_reason == FINISH_LENGTH
    assert run.stats.steps == 0  # completed entirely at prefill, no batched step
    assert run.stats.tokens_generated == 1


def test_immediate_eos_yields_empty_stop() -> None:
    backend = FakeBackend(2, {"p": [EOS]})
    sched = ContinuousBatchScheduler(backend)
    run = sched.run([BatchRequest(prompt="p", max_tokens=5)])

    (res,) = run.results
    assert res.token_ids == ()
    assert res.completion_token_count == 0
    assert res.finish_reason == FINISH_STOP
    assert run.stats.tokens_generated == 0
    assert run.stats.steps == 0


def test_default_max_tokens_used_when_request_omits_it() -> None:
    backend = FakeBackend(1, {"p": [1, 2, 3, 4, 5, 6]})
    sched = ContinuousBatchScheduler(backend, BatchConfig(n_seq_max=1, default_max_tokens=4))
    run = sched.run([BatchRequest(prompt="p")])

    (res,) = run.results
    assert res.completion_token_count == 4
    assert res.finish_reason == FINISH_LENGTH


def test_batch_fitting_one_wave_has_no_admission_waits() -> None:
    plans = {"a": [1, EOS], "b": [2, 3, EOS], "c": [4, EOS]}
    backend = FakeBackend(4, plans)
    sched = ContinuousBatchScheduler(backend, BatchConfig(n_seq_max=4, default_max_tokens=10))
    reqs = [BatchRequest(prompt=p, request_id=p) for p in ("a", "b", "c")]
    run = sched.run(reqs)

    assert [r.request_id for r in run.results] == ["a", "b", "c"]  # input order
    assert run.stats.admission_waits == 0
    assert run.stats.max_concurrency == 3  # all three coexisted in the first step
    assert run.stats.requests_completed == 3


def test_continuous_admission_reuses_freed_slots() -> None:
    # Two slots, four requests: the short ones free slots that the later ones are
    # admitted into mid-flight -- continuous batching, not a fixed wave.
    plans = {
        "short1": [1, EOS],
        "short2": [2, EOS],
        "long1": [3, 4, 5, 6, EOS],
        "long2": [7, 8, 9, EOS],
    }
    backend = FakeBackend(2, plans)
    sched = ContinuousBatchScheduler(backend, BatchConfig(n_seq_max=2, default_max_tokens=20))
    order = ["short1", "short2", "long1", "long2"]
    run = sched.run([BatchRequest(prompt=p, request_id=p) for p in order])

    assert [r.request_id for r in run.results] == order  # still input order
    assert all(r.finish_reason == FINISH_STOP for r in run.results)
    assert run.stats.max_concurrency == 2  # never exceeded n_seq_max
    assert run.stats.admission_waits == 2  # the two later requests waited for a slot
    assert run.stats.requests_completed == 4
    # short1/short2 admitted first (slots 0,1); the two longs reused freed slots.
    assert set(backend.freed) == {0, 1}


def test_results_content_detokenized() -> None:
    backend = FakeBackend(1, {"p": [11, 12, EOS]})
    sched = ContinuousBatchScheduler(backend)
    run = sched.run([BatchRequest(prompt="p", max_tokens=10)])
    assert run.results[0].text == "11 12"


def test_empty_request_list() -> None:
    backend = FakeBackend(2, {})
    run = ContinuousBatchScheduler(backend).run([])
    assert run.results == []
    assert run.stats.requests_completed == 0
    assert run.stats.steps == 0


# --------------------------------------------------------------------------- #
# GPU-gated integration test: the real correctness bar.
# --------------------------------------------------------------------------- #

# The rig's own model, never a model this repo names. $MODEL_DEFAULT is the
# default role declared in ci/runner/rig-local.sh; an unset variable gives a path
# that cannot exist, so the test below skips rather than failing on someone else's machine.
_MODEL_PATH = asset_path("MODEL_DEFAULT")


def _gpu_ready() -> bool:
    # Backend-agnostic: CUDA_VISIBLE_DEVICES means nothing to the Vulkan build,
    # and requiring it here SKIPS this gate on a rig that can run it.
    if not gpu_env_ready():
        return False
    if not _MODEL_PATH.exists():
        return False
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _gpu_ready(), reason="requires GPU env (LLAMA_CPP_LIB_PATH, a GPU device, MODEL_DEFAULT)")
def test_gpu_scheduler_matches_isolated_and_router() -> None:
    from inferference.models import EngineConfig
    from inferference.router import MultiLoRARouter, RouterConfig, RouteRequest

    n_ctx, n_seq_max, max_tokens = 2048, 4, 24
    prompts = [
        "The capital of France is",
        "Two plus two equals",
        "Water is made of hydrogen and",
        "The opposite of hot is",
    ]

    # Scheduler over its own continuous-batch engine.
    engine = LlamaContinuousBatchEngine(
        str(_MODEL_PATH),
        n_ctx=n_ctx,
        n_seq_max=n_seq_max,
        n_batch=128,
        n_gpu_layers=-1,
        seed=17018,
    )
    sched = ContinuousBatchScheduler(engine, BatchConfig(n_seq_max=n_seq_max, default_max_tokens=max_tokens))
    run = sched.run([BatchRequest(prompt=p, max_tokens=max_tokens, request_id=str(i)) for i, p in enumerate(prompts)])
    engine.close()

    assert len(run.results) == len(prompts)
    for res in run.results:
        assert res.completion_token_count > 0
        assert res.finish_reason in (FINISH_STOP, FINISH_LENGTH)
    assert run.stats.max_concurrency == len(prompts)  # all fit in one wave
    assert run.stats.admission_waits == 0

    # A batch that fits in one wave should match the router's fixed-wave decode.
    router = MultiLoRARouter(
        RouterConfig(
            engine=EngineConfig(model_path=str(_MODEL_PATH), n_ctx=n_ctx, n_gpu_layers=-1, n_batch=128, seed=17018),
            adapters=(),
            n_seq_max=n_seq_max,
            include_base=True,
        )
    )
    router_results = router.run([RouteRequest(prompt=p, adapter=None, max_tokens=max_tokens) for p in prompts])
    router.close()

    for sched_res, router_res in zip(run.results, router_results, strict=True):
        assert sched_res.token_ids == router_res.token_ids
        assert sched_res.finish_reason == router_res.finish_reason
