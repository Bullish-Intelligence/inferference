"""CPU tests for ``TraceRouter``: the wave plans, the forcing, the order of record and mask, and the reset.

No GPU and no native library. A stub fills the places where the router would call llama.cpp.
"""

from __future__ import annotations

import ctypes
import dataclasses
from typing import Any, cast

import numpy as np
import pytest

from inferference.benchkit.trace_router import (
    TraceRouter,
    cached_legs,
    cached_state_pairs,
    cached_wave_plan,
    wave_plan,
)
from inferference.mixed_batch_gates import Trace, measure_pairs, row_step
from inferference.router import MultiLoRARouter, RouteRequest

V = 40


def _req(prompt: str, adapter: str | None) -> RouteRequest:
    return RouteRequest(prompt=prompt, adapter=adapter, max_tokens=4)


def test_wave_plan_groups_by_adapter_for_context_pool_and_chunks_for_seq_routed() -> None:
    reqs = [_req("a", "x"), _req("b", None), _req("c", "x"), _req("d", None), _req("e", "x")]
    assert wave_plan("context_pool", 8, reqs) == [[0, 2, 4], [1, 3]]
    assert wave_plan("context_pool", 2, reqs) == [[0, 2], [4], [1, 3]]
    assert wave_plan("seq_routed", 2, reqs) == [[0, 1], [2, 3], [4]]


def test_cached_wave_plan_lists_cached_cold_then_one_baseline_per_request() -> None:
    reqs = [_req("a", "x"), _req("b", None), _req("c", "x")]
    assert cached_wave_plan(8, reqs) == [
        ("cached", [0, 2]),
        ("cold", [0, 2]),
        ("baseline", [0]),
        ("baseline", [2]),
        ("cached", [1]),
        ("cold", [1]),
        ("baseline", [1]),
    ]


class _Matcher:
    def __init__(self, accept: bool = True) -> None:
        self.accept = accept
        self.accepted: list[int] = []

    def accept_token(self, token: int) -> bool:
        self.accepted.append(token)
        return self.accept


def _bare(backend: str = "context_pool") -> TraceRouter:
    """A TraceRouter with no model: only the state that the tracing code reads."""
    router = object.__new__(TraceRouter)
    router._top_k = 16
    router._waves = []
    router._cur = []
    router._pre = 0
    router._wave_idx = -1
    router._forced = None
    router.backend = backend
    router.n_seq_max = 8
    router.n_vocab = V
    return router


def _logits(i: int, t: int) -> np.ndarray:
    return np.random.default_rng(1000 * i + t).normal(0.0, 3.0, V).astype(np.float32)


def test_trace_row_records_the_raw_row_before_the_fallback_masks_it() -> None:
    router = _bare()
    router._open_wave(1)
    row = _logits(0, 0)
    original = row.copy()

    def fallback() -> int:
        row[:] = -np.inf  # what the grammar mask does to the llama.cpp buffer
        return 7

    token = router._trace_row(0, row, None, fallback)
    step = router._cur[0][0]
    assert token == 7 and step.token == 7
    assert step.argmax == int(np.argmax(original)) and step.logits[0] == float(original[step.ids[0]])


def test_trace_row_takes_the_forced_token_and_the_reference_probe_ids() -> None:
    router = _bare()
    ref_row = _logits(0, 0)
    ref = Trace((row_step(ref_row),))
    router._forced = [[ref]]
    router._open_wave(1)
    matcher = _Matcher()
    got = router._trace_row(0, _logits(1, 0), matcher, lambda: pytest.fail("a forced step must not sample"))
    assert got == ref.steps[0].token and matcher.accepted == [got]
    assert router._cur[0][0].ids == ref.steps[0].ids  # read at the reference ids, not its own top


def test_trace_row_fails_closed_when_the_grammar_rejects_a_forced_token() -> None:
    router = _bare()
    router._forced = [[Trace((row_step(_logits(0, 0)),))]]
    router._open_wave(1)
    with pytest.raises(RuntimeError, match="grammar rejected the forced token"):
        router._trace_row(0, _logits(0, 0), _Matcher(accept=False), lambda: 0)


def test_trace_row_samples_freely_once_the_reference_is_exhausted() -> None:
    router = _bare()
    router._forced = [[Trace((row_step(_logits(0, 0)),))]]
    router._open_wave(1)
    router._trace_row(0, _logits(0, 0), None, lambda: 0)
    assert router._trace_row(0, _logits(0, 1), None, lambda: 5) == 5


def test_a_sampler_is_refused() -> None:
    router = _bare()
    router._open_wave(1)
    with pytest.raises(RuntimeError, match="greedy"):
        router._sample_row(None, 0, None, object())


def test_slot_counts_prefill_samples_in_order_and_decode_samples_by_row() -> None:
    router = _bare()
    router._open_wave(3)
    assert [router._slot(-1) for _ in range(3)] == [0, 1, 2]
    assert router._slot(2) == 2 and router._slot(0) == 0


class _Stub(TraceRouter):
    """``run`` plays the waves of the plan with a known logits row per (request, step)."""

    steps = 3

    def run(self, requests: list[RouteRequest], *, constrained: bool = False) -> list[Any]:
        plan = wave_plan(self.backend, self.n_seq_max, requests)
        for wave in plan:
            self._open_wave(len(wave))
            for t in range(self.steps):
                for slot, j in enumerate(wave):
                    row = _logits(j, t)
                    self._trace_row(slot, row, None, lambda r=row: int(np.argmax(r)))
        return []


def _stub(backend: str) -> _Stub:
    router = object.__new__(_Stub)
    router.__dict__.update(_bare(backend).__dict__)
    return router


def test_trace_returns_one_trace_per_request_in_request_order() -> None:
    router = _stub("context_pool")
    reqs = [_req("a", "x"), _req("b", None), _req("c", "x")]
    traces = router.trace(reqs)
    assert len(traces) == 3
    for j, trace in enumerate(traces):  # the context_pool plan runs requests 0 and 2 before 1
        assert [s.argmax for s in trace.steps] == [int(np.argmax(_logits(j, t))) for t in range(3)]


def test_trace_forces_the_reference_and_checks_the_wave_count() -> None:
    router = _stub("seq_routed")
    reqs = [_req("a", "x"), _req("b", None)]
    free = router.trace(reqs)
    forced = router.trace(reqs, reference=free)
    assert [t.tokens for t in forced] == [t.tokens for t in free]
    with pytest.raises(ValueError, match="references for"):
        router.trace(reqs, reference=free[:1])

    class _Short(_Stub):
        def run(self, requests: list[RouteRequest], *, constrained: bool = False) -> list[Any]:
            return []

    short = object.__new__(_Short)
    short.__dict__.update(_bare("seq_routed").__dict__)
    with pytest.raises(RuntimeError, match="waves, the plan has"):
        short.trace(reqs)


def test_mixes_adapters_follows_the_backend() -> None:
    assert _bare("seq_routed").mixes_adapters and not _bare("context_pool").mixes_adapters


class _FakeC:
    def __init__(self) -> None:
        self.cleared: list[tuple[Any, bool]] = []

    def llama_get_memory(self, ctx: Any) -> Any:
        return ("mem", ctx)

    def llama_memory_clear(self, mem: Any, data: bool) -> None:
        self.cleared.append((mem, data))


def test_reset_state_clears_every_context_with_its_data() -> None:
    router = object.__new__(MultiLoRARouter)
    router.C = _FakeC()
    router._ctx = {None: "base", "a": "ctx-a"}
    router._seq_ctx = "seq"
    router.reset_state()
    assert router.C.cleared == [(("mem", "base"), True), (("mem", "ctx-a"), True), (("mem", "seq"), True)]
    router._seq_ctx = None
    router.C.cleared.clear()
    router.reset_state()
    assert len(router.C.cleared) == 2


# ---- the cached gates, driven by a router whose legs are scripted ----


class _CachedStub(TraceRouter):
    """``trace_cached`` is scripted: each leg is the isolated trace plus a fixed offset."""

    def __init__(self, leg_offset: float) -> None:
        self.leg_offset = leg_offset
        self.resets = 0

    def reset_state(self) -> None:
        self.resets += 1

    def trace_cached(
        self, requests: Any, prefix_token_ids: Any, *, constrained: bool = False, reference: Any = None
    ) -> Any:
        from inferference.benchkit.trace_router import CachedTrace

        n = len(requests)
        base = [Trace(tuple(row_step(_logits(i, t)) for t in range(3))) for i in range(n)]

        def leg(offset: float) -> list[Trace]:
            out = []
            for i in range(n):
                steps = []
                for t in range(3):
                    ref = None if reference is None else reference[i].steps[t]
                    steps.append(
                        row_step(
                            _logits(i, t) + np.float32(offset),
                            token=None if ref is None else ref.token,
                            probe_ids=None if ref is None else ref.ids,
                        )
                    )
                out.append(Trace(tuple(steps)))
            return out

        return CachedTrace(
            result=cast(Any, None),
            cached=leg(self.leg_offset),
            cold=leg(0.0),
            baseline=base if reference is None else leg(0.0),
        )


class _SkewedStub(_CachedStub):
    """The cached leg of the forced pass has its logits skewed on every second probe id."""

    def trace_cached(
        self, requests: Any, prefix_token_ids: Any, *, constrained: bool = False, reference: Any = None
    ) -> Any:
        out = super().trace_cached(requests, prefix_token_ids, constrained=constrained, reference=reference)
        if reference is not None:
            out.cached = [
                Trace(
                    tuple(
                        dataclasses.replace(s, logits=tuple(x + 3.0 * (k % 2) for k, x in enumerate(s.logits)))
                        for s in t.steps
                    )
                )
                for t in out.cached
            ]
        return out


class _FlakyStub(_CachedStub):
    """Skews the leg named in ``leg`` on one call only, so two passes disagree."""

    def __init__(self, leg: str, on_call: int) -> None:
        super().__init__(0.0)
        self.leg, self.on_call, self.calls = leg, on_call, 0

    def trace_cached(
        self, requests: Any, prefix_token_ids: Any, *, constrained: bool = False, reference: Any = None
    ) -> Any:
        self.calls += 1
        out = super().trace_cached(requests, prefix_token_ids, constrained=constrained, reference=reference)
        if self.calls == self.on_call:
            setattr(
                out,
                self.leg,
                [
                    Trace(tuple(dataclasses.replace(s, lse=s.lse + 1e-3) for s in t.steps))
                    for t in getattr(out, self.leg)
                ],
            )
        return out


def test_cached_legs_pass_on_exact_legs_and_measure_a_large_difference_without_failing() -> None:
    reqs = [_req("a", None), _req("b", "x")]
    router = _SkewedStub(0.0)  # the cached leg is far from the reference, but the same on every pass
    findings, pairs = cached_legs(router, reqs, [1, 2, 3])
    assert findings == []  # a large difference is not a finding
    assert router.resets == 3  # a clean state before each of the three passes
    assert len(pairs) == 4 and {p.composition for p in pairs} == {"run_cached/cached", "run_cached/cold"}
    cached = measure_pairs([p for p in pairs if p.composition == "run_cached/cached"])["base"]
    assert cached["max"] > 2.0  # measured


def test_cached_legs_fail_when_the_cached_leg_is_not_deterministic() -> None:
    findings, _ = cached_legs(_FlakyStub("cached", on_call=3), [_req("a", None)], [1, 2, 3])
    assert any("cached leg, pass 2 against pass 3" in f for f in findings)


def test_cached_legs_fail_when_the_reference_replay_changes() -> None:
    findings, _ = cached_legs(_FlakyStub("baseline", on_call=2), [_req("a", None)], [1, 2, 3])
    assert any("reference leg replay" in f for f in findings)


def test_constrained_cached_legs_take_the_cold_leg_as_reference() -> None:
    findings, pairs = cached_legs(_FlakyStub("cold", on_call=2), [_req("a", None)], [1], constrained=True)
    assert any("reference leg replay" in f for f in findings)
    assert {p.composition for p in pairs} == {"run_cached/cached"}  # no cold pair: it is the reference


class _RecordingRouter(TraceRouter):
    calls: list[str]

    def _seq_rm(self, ctx: Any, seq_id: int) -> None:
        self.calls.append(f"rm{seq_id}")

    def reset_state(self) -> None:
        self.calls.append("reset")


def test_cached_state_pairs_resets_first_and_reads_the_cold_probe_ids_in_the_cached_step() -> None:
    router = object.__new__(_RecordingRouter)
    router.__dict__.update(_bare().__dict__)
    router.C = _StateC()
    router._ctx = {None: "ctx"}
    router._seq_ctx = None
    router.calls = []
    bridge = _Bridge(router.C)
    pairs = cached_state_pairs(router, bridge, None, [1, 2, 3], [[4], [5]])
    assert router.calls[0] == "reset"
    assert len(pairs) == 2
    for cold, cached in pairs:
        assert cached.steps[0].ids == cold.steps[0].ids and cached.steps[0].token == cold.steps[0].token


class _StateC:
    """The logits row depends on how many tokens the last decode held: cold sees more than cached."""

    def __init__(self) -> None:
        self.last_n = 0
        self._row = np.zeros(V, dtype=np.float32)

    def llama_get_logits_ith(self, ctx: Any, row: int) -> Any:
        self._row = _logits(self.last_n, 0)
        return self._row.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class _Bridge:
    def __init__(self, c: _StateC) -> None:
        self.c = c

    def decode_tokens(self, ctx: Any, tokens: list[int], seq: int, pos: int) -> None:
        self.c.last_n = len(tokens)

    def capture_seq_state(self, ctx: Any, seq: int) -> bytes:
        return b"blob"

    def restore_seq_state(self, ctx: Any, blob: bytes, seq: int) -> int:
        return 1
