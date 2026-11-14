"""A router that records its logits and can replay tokens: the engine for the mixed-batch gates.

``TraceRouter`` is a :class:`inferference.router.MultiLoRARouter` that, at every sampled
step, reads the logits row and stores a :class:`inferference.mixed_batch_gates.Step`.
It can also take its tokens from a reference trace (teacher forcing). Both runs of a
comparison then read the same tokens, so their logits are comparable at every step.

The recording reads the row before the grammar mask touches it. It changes no decode
state. Only greedy requests are traced.

The pure functions (:func:`wave_plan`, :func:`cached_wave_plan`) say which request sits in
which wave and slot. They mirror the router, and ``trace`` checks the wave count, so a
router change that moves requests fails loudly.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from inferference.mixed_batch_gates import (
    TOP_K,
    Pair,
    Step,
    Trace,
    check_exact,
    row_step,
)
from inferference.router import CachedRunResult, MultiLoRARouter, RouterConfig, RouteRequest

__all__ = [
    "CachedTrace",
    "TraceRouter",
    "cached_state_pairs",
    "pairs_from_state",
    "cached_wave_plan",
    "cached_legs",
    "wave_plan",
]


def wave_plan(backend: str, n_seq_max: int, requests: Sequence[RouteRequest]) -> list[list[int]]:
    """Request indexes per wave, in the order the router runs the waves."""
    if backend == "seq_routed":
        idx = list(range(len(requests)))
        return [idx[w : w + n_seq_max] for w in range(0, len(idx), n_seq_max)]
    groups: dict[str | None, list[int]] = {}
    for i, r in enumerate(requests):
        groups.setdefault(r.adapter, []).append(i)
    plan: list[list[int]] = []
    for idxs in groups.values():
        for w in range(0, len(idxs), n_seq_max):
            plan.append(idxs[w : w + n_seq_max])
    return plan


def cached_wave_plan(n_seq_max: int, requests: Sequence[RouteRequest]) -> list[tuple[str, list[int]]]:
    """``(kind, request indexes)`` per wave of ``run_cached``, in order.

    For each adapter group and each wave: the cached wave, the cold wave, then one
    isolated baseline per request of the wave.
    """
    plan: list[tuple[str, list[int]]] = []
    for wave in wave_plan("context_pool", n_seq_max, requests):
        plan.append(("cached", wave))
        plan.append(("cold", wave))
        plan.extend(("baseline", [j]) for j in wave)
    return plan


@dataclass
class CachedTrace:
    """Result and traces of one traced ``run_cached``."""

    result: CachedRunResult
    cached: list[Trace]
    cold: list[Trace]
    baseline: list[Trace]


class TraceRouter(MultiLoRARouter):
    """Records a trace per sequence. Optionally forces the tokens of a reference trace."""

    def __init__(self, config: RouterConfig, *, top_k: int = TOP_K) -> None:
        self._top_k = top_k
        self._waves: list[list[list[Step]]] = []
        self._cur: list[list[Step]] = []
        self._pre = 0
        self._wave_idx = -1
        self._forced: list[list[Trace]] | None = None
        super().__init__(config)

    @property
    def mixes_adapters(self) -> bool:
        return self.backend == "seq_routed"

    # ---- wave bookkeeping ----

    def _open_wave(self, n_rows: int) -> None:
        self._cur = [[] for _ in range(n_rows)]
        self._waves.append(self._cur)
        self._wave_idx = len(self._waves) - 1
        self._pre = 0

    def _generate_wave(
        self, ctx: Any, prompts: list[list[int]], max_tokens: int, matchers: Any, samplers: Any = None
    ) -> Any:
        self._open_wave(len(prompts))
        return super()._generate_wave(ctx, prompts, max_tokens, matchers, samplers)

    def _generate_wave_restored(
        self,
        ctx: Any,
        blob: bytes,
        prefix_len: int,
        suffixes: list[list[int]],
        max_tokens: int,
        matchers: Any,
        bridge: Any,
    ) -> Any:
        self._open_wave(len(suffixes))
        return super()._generate_wave_restored(ctx, blob, prefix_len, suffixes, max_tokens, matchers, bridge)

    # ---- the traced sample ----

    def _slot(self, row: int) -> int:
        """The slot of a sample. The prefill samples (row -1) arrive in slot order."""
        if row == -1:
            slot = self._pre
            self._pre += 1
            return slot
        return row

    def _reference_step(self, slot: int) -> Step | None:
        if self._forced is None:
            return None
        ref = self._forced[self._wave_idx][slot]
        t = len(self._cur[slot])
        return ref.steps[t] if t < len(ref) else None

    def _trace_row(self, slot: int, logits: Any, matcher: Any | None, fallback: Callable[[], int]) -> int:
        """Record the raw row, then take the forced token or the router's own sample."""
        ref = self._reference_step(slot)
        step = row_step(logits, probe_ids=None if ref is None else ref.ids, k=self._top_k)
        if ref is None:
            token = fallback()  # applies the grammar mask in place, after the record
        else:
            token = ref.token
            if matcher is not None and matcher.accept_token(token) is False:
                raise RuntimeError(f"grammar rejected the forced token {token} (fail-closed)")
        self._cur[slot].append(dataclasses.replace(step, token=token))
        return token

    def _sample_row(self, ctx: Any, row: int, matcher: Any | None = None, sampler: Any | None = None) -> int:
        if sampler is not None:
            raise RuntimeError("TraceRouter traces greedy requests only")
        slot = self._slot(row)
        ptr = self.C.llama_get_logits_ith(ctx, row)
        logits = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))
        parent = super()._sample_row
        return self._trace_row(slot, logits, matcher, lambda: parent(ctx, row, matcher, sampler))

    # ---- the traced runs ----

    def trace(self, requests: Sequence[RouteRequest], *, reference: Sequence[Trace] | None = None) -> list[Trace]:
        """Run the requests as one workload and return one trace per request.

        With ``reference``, request ``i`` takes the tokens of ``reference[i]`` and reads
        the logits at its probe ids. Greedy requests only.
        """
        reqs = list(requests)
        if reference is not None and len(reference) != len(reqs):
            raise ValueError(f"{len(reference)} references for {len(reqs)} requests")
        plan = wave_plan(self.backend, self.n_seq_max, reqs)
        self._forced = None if reference is None else [[reference[j] for j in wave] for wave in plan]
        self._waves = []
        try:
            self.run(reqs)
        finally:
            self._forced = None
        if len(self._waves) != len(plan):
            raise RuntimeError(f"the router ran {len(self._waves)} waves, the plan has {len(plan)}")
        out: list[Trace | None] = [None] * len(reqs)
        for w, wave in enumerate(plan):
            if len(self._waves[w]) != len(wave):
                raise RuntimeError(f"wave {w} has {len(self._waves[w])} rows, the plan has {len(wave)}")
            for slot, j in enumerate(wave):
                out[j] = Trace(tuple(self._waves[w][slot]))
        return [t for t in out if t is not None]

    def trace_cached(
        self,
        requests: Sequence[RouteRequest],
        prefix_token_ids: Sequence[int],
        *,
        constrained: bool = False,
        reference: Sequence[Trace] | None = None,
    ) -> CachedTrace:
        """``run_cached`` with a trace per request and per leg (cached, cold, baseline).

        With ``reference``, every leg of request ``i`` takes the tokens of ``reference[i]``.
        """
        reqs = list(requests)
        plan = cached_wave_plan(self.n_seq_max, reqs)
        self._forced = None if reference is None else [[reference[j] for j in idxs] for _, idxs in plan]
        self._waves = []
        try:
            result = self.run_cached(reqs, prefix_token_ids, constrained=constrained)
        finally:
            self._forced = None
        if len(self._waves) != len(plan):
            raise RuntimeError(f"run_cached ran {len(self._waves)} waves, the plan has {len(plan)}")
        legs: dict[str, list[Trace | None]] = {k: [None] * len(reqs) for k in ("cached", "cold", "baseline")}
        for w, (kind, idxs) in enumerate(plan):
            for slot, j in enumerate(idxs):
                legs[kind][j] = Trace(tuple(self._waves[w][slot]))
        return CachedTrace(
            result=result,
            cached=[t for t in legs["cached"] if t is not None],
            cold=[t for t in legs["cold"] if t is not None],
            baseline=[t for t in legs["baseline"] if t is not None],
        )


def cached_legs(
    router: TraceRouter,
    requests: Sequence[RouteRequest],
    prefix_token_ids: Sequence[int],
    *,
    constrained: bool = False,
) -> tuple[list[str], list[Pair]]:
    """Cached restore against the full-prompt run: exact findings and measured pairs.

    Three passes, each from a reset state. Pass 1 free-runs and gives the reference: the isolated
    batch-1 full-prompt baseline in plain mode, the cold leg in constrained mode (the baseline
    leg has no grammar). Passes 2 and 3 take the reference tokens (teacher forcing).

    Exact findings (gates): the reference leg, replayed in pass 2, is bit-identical (the forcing
    changes nothing it should not); the cached leg of pass 2 and of pass 3 are bit-identical
    (the restore is deterministic).

    Pairs (measured, never gated): the reference against the cached leg, and in plain mode
    against the cold leg.
    """
    reqs = list(requests)
    router.reset_state()
    first = router.trace_cached(reqs, prefix_token_ids, constrained=constrained)
    ref = first.cold if constrained else first.baseline
    router.reset_state()
    second = router.trace_cached(reqs, prefix_token_ids, constrained=constrained, reference=ref)
    router.reset_state()
    third = router.trace_cached(reqs, prefix_token_ids, constrained=constrained, reference=ref)
    findings: list[str] = []
    pairs: list[Pair] = []
    replay = second.cold if constrained else second.baseline
    for i, r in enumerate(reqs):
        tag = f"cached request {i} ({r.adapter or 'base'})"
        findings += check_exact(f"{tag} reference leg replay", ref[i], replay[i])
        findings += check_exact(f"{tag} cached leg, pass 2 against pass 3", second.cached[i], third.cached[i])
        pairs.append(Pair(r.prompt, "run_cached/cached", r.adapter, ref[i], second.cached[i]))
        if not constrained:
            pairs.append(Pair(r.prompt, "run_cached/cold", r.adapter, ref[i], second.cold[i]))
    return findings, pairs


def cached_state_pairs(
    router: TraceRouter,
    bridge: Any,
    adapter: str | None,
    prefix: Sequence[int],
    suffix_lists: Sequence[Sequence[int]],
) -> list[tuple[Trace, Trace]]:
    """Cold full prefill against restore-then-suffix, one next-token step each.

    The flow is the one of the shared-prefix restore test: capture the prefix state once,
    then for each suffix run the cold path in slot ``i`` and the cached path in slot ``i``.
    Returns ``(cold, cached)`` one-step traces. The cached step reads the cold step's probe
    ids. The state is reset first, so history does not enter the comparison.
    """
    router.reset_state()
    ctx = router._ctx[adapter]
    prefix = list(prefix)
    bridge.decode_tokens(ctx, prefix, 0, 0)
    blob = bridge.capture_seq_state(ctx, 0)
    router._seq_rm(ctx, 0)

    def read(probe: Sequence[int] | None) -> Step:
        ptr = router.C.llama_get_logits_ith(ctx, -1)
        return row_step(np.ctypeslib.as_array(ptr, shape=(router.n_vocab,)), probe_ids=probe, k=TOP_K)

    cold: list[Step] = []
    for i, suffix in enumerate(suffix_lists):
        router._seq_rm(ctx, i)
        bridge.decode_tokens(ctx, prefix + list(suffix), i, 0)
        cold.append(read(None))
    pairs: list[tuple[Trace, Trace]] = []
    for i, suffix in enumerate(suffix_lists):
        router._seq_rm(ctx, i)
        code = bridge.restore_seq_state(ctx, blob, i)
        if code <= 0:
            raise RuntimeError(f"restore_seq_state returned {code}: the restore was rejected (fail-closed)")
        bridge.decode_tokens(ctx, list(suffix), i, len(prefix))
        got = dataclasses.replace(read(cold[i].ids), token=cold[i].token)
        pairs.append((Trace((cold[i],)), Trace((got,))))
    for i in range(len(suffix_lists)):
        router._seq_rm(ctx, i)
    return pairs


def pairs_from_state(adapter: str | None, pairs: Sequence[tuple[Trace, Trace]], composition: str) -> list[Pair]:
    """Wrap ``cached_state_pairs`` output (cold, cached) as measured pairs."""
    return [Pair(f"state-{i}", composition, adapter, a, b) for i, (a, b) in enumerate(pairs)]
