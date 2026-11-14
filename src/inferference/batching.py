"""Our own continuous / dynamic batching layer over llama.cpp (Pillar 4).

The built-in llama.cpp server batches for us; this module does *not* use it.
Instead it owns the admission and scheduling policy directly on top of a single
shared ``llama_context`` with ``n_seq_max`` sequence slots, so the mechanics of
continuous batching are visible and testable rather than hidden behind a server.

The teaching point (PLAN standing rule) is **control and understanding, not
beating C++ throughput**:

* One context, ``n_seq_max`` slots.  At most ``n_seq_max`` sequences are ever in
  flight; the rest wait in a FIFO admission queue.
* Every ``llama_decode`` step advances *all currently active slots by one token*.
  Sequences at different positions coexist in the same step -- that is what makes
  this *continuous* batching rather than the router's fixed waves: when a slot's
  sequence finishes (EOS / ``max_tokens``) it is freed (``llama_memory_seq_rm``)
  and the next waiting request is admitted into that slot mid-flight.
* Observability is first class: a :class:`BatchStats` records decode steps,
  prefills, tokens generated, the peak concurrency actually reached, and how many
  requests had to wait for a slot.

The scheduling policy lives entirely in :class:`ContinuousBatchScheduler` and is
expressed against a small :class:`BatchDecodeBackend` protocol, so the admission
logic is unit-tested with an in-memory fake and no GPU.  The real ctypes backend
:class:`LlamaContinuousBatchEngine` reuses the proven decode primitives from
:mod:`inferference.router` (own-batch multi-seq decode, seq_id per
row, per-seq greedy argmax); it is imported lazily like ``decode.py`` so a
core-only install never touches the native library.

Boundary objects (:class:`BatchConfig`, :class:`BatchRequest`) are Pydantic;
:class:`~inferference.models.GenerationResult` is the result type.
The hot path passes plain token ids and numpy views.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .decode import FINISH_LENGTH, FINISH_STOP
from .models import GenerationResult


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BatchConfig(_BoundaryModel):
    """Admission / scheduling policy for a :class:`ContinuousBatchScheduler`.

    ``n_seq_max`` is the number of sequence slots the shared context exposes and
    therefore the maximum concurrency.  ``default_max_tokens`` applies to any
    request that does not set its own budget.  Only FIFO admission is implemented;
    the field is typed as a ``Literal`` so a future policy is an explicit, typed
    extension rather than a silent string.
    """

    n_seq_max: int = Field(default=8, gt=0)
    default_max_tokens: int = Field(default=64, gt=0)
    admission_policy: Literal["fifo"] = "fifo"


class BatchRequest(_BoundaryModel):
    """One request entering the continuous batch.

    ``max_tokens=None`` defers to :attr:`BatchConfig.default_max_tokens`, so the
    common case stays terse while a request may still cap itself. ``adapter`` names
    the LoRA adapter the request should decode under (``None`` = raw base); it is
    honored only by a seq-routing-capable backend (the P2 fork), where a single
    decode step can mix adapters — a stock backend ignores it.
    """

    prompt: str
    max_tokens: int | None = Field(default=None, gt=0)
    request_id: str | None = None
    adapter: str | None = None


@dataclass(frozen=True, slots=True)
class BatchStats:
    """Observable counters for one :meth:`ContinuousBatchScheduler.run`.

    * ``steps`` -- batched ``llama_decode`` calls (each advances every active slot
      one token).
    * ``prefills`` -- prompt-decode calls (one per admitted request).
    * ``tokens_generated`` -- total content tokens emitted across all requests.
    * ``max_concurrency`` -- peak number of slots decoded together in one step;
      the concrete evidence that sequences at different positions coexisted.
    * ``admission_waits`` -- requests that could not be admitted immediately and
      waited for a slot to free (0 when the whole batch fits in ``n_seq_max``).
    * ``requests_completed`` -- requests carried to a terminal ``finish_reason``.
    """

    steps: int
    prefills: int
    tokens_generated: int
    max_concurrency: int
    admission_waits: int
    requests_completed: int


@dataclass(frozen=True, slots=True)
class BatchRun:
    """Results (in request order) plus the run's observability counters."""

    results: list[GenerationResult]
    stats: BatchStats


@runtime_checkable
class BatchDecodeBackend(Protocol):
    """The minimal decode surface the scheduler drives.

    Kept deliberately small so the scheduling policy can be exercised with an
    in-memory fake: the backend owns tokenization, the shared context, per-slot
    prefill, the one-token-per-active-slot batched step, and greedy sampling;
    the scheduler owns *only* admission and lifecycle.
    """

    @property
    def n_seq_max(self) -> int:
        """Number of sequence slots (maximum concurrency)."""

    @property
    def eos(self) -> int:
        """The end-of-sequence token id that marks a clean stop."""

    def tokenize(self, prompt: str) -> list[int]:
        """Tokenize a prompt (with BOS) into the ids the loop decodes."""

    def detokenize(self, tokens: Sequence[int]) -> str:
        """Render generated token ids back to text."""

    def free_slot(self, seq_id: int) -> None:
        """Drop all KV for ``seq_id`` so the slot can be reused."""

    def prefill(self, tokens: Sequence[int], seq_id: int) -> int:
        """Decode a prompt into ``seq_id`` and return the greedy first token."""

    def step(self, active: Sequence[tuple[int, int, int]]) -> list[int]:
        """Advance active slots one token in a single batched decode.

        ``active`` is ``(seq_id, current_token, position)`` per slot; returns the
        greedy next token for each, in the same order.
        """


@dataclass(slots=True)
class _Slot:
    """Mutable per-slot decode state while a request is in flight."""

    seq_id: int  # physical slot id (== the shared context sequence slot)
    index: int  # request's position in the submitted list (output order)
    request: BatchRequest
    max_tokens: int
    prompt_len: int
    tokens: list[int]
    cur: int  # token to decode next (already recorded in ``tokens`` unless EOS)
    pos: int  # KV position of ``cur``
    finish: str | None = None


class ContinuousBatchScheduler:
    """FIFO continuous batching over one shared ``n_seq_max``-slot context.

    :meth:`run` accepts a batch of requests, keeps at most ``n_seq_max`` in flight,
    and admits a waiting request into any slot the instant its predecessor
    finishes -- so a step's active set mixes freshly admitted short-prompt
    sequences with long-running ones.  Results come back in submission order with
    honest ``finish_reason`` (``stop`` for EOS, ``length`` for a ``max_tokens``
    cut), alongside a :class:`BatchStats`.
    """

    def __init__(self, backend: BatchDecodeBackend, config: BatchConfig | None = None) -> None:
        self.backend = backend
        self.config = config or BatchConfig(n_seq_max=backend.n_seq_max)
        self.n_seq_max = backend.n_seq_max

    def _max_tokens(self, request: BatchRequest) -> int:
        return request.max_tokens if request.max_tokens is not None else self.config.default_max_tokens

    def _record(self, slot: _Slot, token: int) -> bool:
        """Append a sampled content token or mark the slot finished.

        Returns ``True`` when the token was counted as content.  EOS is the
        boundary, not content, so it is never appended; reaching the per-request
        ``max_tokens`` is a ``length`` cut on the last content token.
        """
        if token == self.backend.eos:
            slot.finish = FINISH_STOP
            return False
        slot.tokens.append(token)
        if len(slot.tokens) >= slot.max_tokens:
            slot.finish = FINISH_LENGTH
        return True

    def _finalize(self, slot: _Slot) -> GenerationResult:
        text = self.backend.detokenize(slot.tokens)
        return GenerationResult(
            text=text,
            token_ids=tuple(slot.tokens),
            prompt_token_count=slot.prompt_len,
            completion_token_count=len(slot.tokens),
            finish_reason=slot.finish or FINISH_LENGTH,
            request_id=slot.request.request_id,
        )

    def run(self, requests: Sequence[BatchRequest]) -> BatchRun:
        """Continuously batch ``requests`` to completion; results in input order."""
        reqs = list(requests)
        results: list[GenerationResult | None] = [None] * len(reqs)
        waiting: deque[int] = deque(range(len(reqs)))
        slots: list[_Slot | None] = [None] * self.n_seq_max

        counters = {
            "steps": 0,
            "prefills": 0,
            "tokens_generated": 0,
            "max_concurrency": 0,
            "admission_waits": 0,
            "requests_completed": 0,
        }

        def complete(slot: _Slot) -> None:
            results[slot.index] = self._finalize(slot)
            slots[slot.seq_id] = None
            self.backend.free_slot(slot.seq_id)
            counters["requests_completed"] += 1

        def admit() -> None:
            """Fill every free slot with the next waiting request (FIFO)."""
            for sid in range(self.n_seq_max):
                if slots[sid] is not None or not waiting:
                    continue
                idx = waiting.popleft()
                req = reqs[idx]
                if counters["steps"] > 0:
                    counters["admission_waits"] += 1  # a slot had to free before this fit
                self.backend.free_slot(sid)  # clean slate for a reused slot
                # Mixed-adapter admission (P2 fork): assign this slot its request's
                # adapter before prefill so the whole in-flight decode carries the mix.
                # Stock backends have no such method; the wave is single-adapter.
                if getattr(self.backend, "supports_seq_routing", False):
                    self.backend.set_slot_adapter(sid, req.adapter)
                tokens = self.backend.tokenize(req.prompt)
                first = self.backend.prefill(tokens, sid)
                counters["prefills"] += 1
                slot = _Slot(
                    seq_id=sid,
                    index=idx,
                    request=req,
                    max_tokens=self._max_tokens(req),
                    prompt_len=len(tokens),
                    tokens=[],
                    cur=first,
                    pos=len(tokens),
                )
                slots[sid] = slot
                if self._record(slot, first):
                    counters["tokens_generated"] += 1
                if slot.finish is not None:
                    complete(slot)  # finished at prefill; slot freed for reuse

        admit()
        while any(s is not None for s in slots):
            active = [s for s in slots if s is not None]
            batch = [(s.seq_id, s.cur, s.pos) for s in active]
            sampled = self.backend.step(batch)
            counters["steps"] += 1
            counters["max_concurrency"] = max(counters["max_concurrency"], len(active))
            for slot, token in zip(active, sampled, strict=True):
                slot.pos += 1  # advance KV position only after the decode
                slot.cur = token
                if self._record(slot, token):
                    counters["tokens_generated"] += 1
                if slot.finish is not None:
                    complete(slot)
            admit()

        stats = BatchStats(**counters)
        return BatchRun(results=[r for r in results if r is not None], stats=stats)


class LlamaContinuousBatchEngine:
    """ctypes :class:`BatchDecodeBackend` over one owned ``n_seq_max``-slot context.

    Reuses the router's own-batch decode primitives: per-slot prefill in
    ``n_batch`` chunks, one ``llama_decode`` advancing every active slot a token,
    per-slot greedy argmax over the zero-copy logits view.  Imported lazily so a
    core-only install never loads the native library.
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int,
        n_seq_max: int = 8,
        n_batch: int = 512,
        n_gpu_layers: int = 0,
        seed: int | None = None,
        adapters: Sequence[tuple[str, str, float]] = (),
    ) -> None:
        import numpy as np
        from llama_cpp import Llama, llama_cpp

        self._np = np
        self.C = llama_cpp
        # Optional P2-fork mixed-adapter routing. Absent a fork lib (or no adapters) the
        # engine is the plain single-context batcher and ``supports_seq_routing`` is
        # False, so the scheduler never calls the routing hook.
        self._seq_binding: Any = None
        self._seq_index: dict[str, int] = {}
        binding = None
        if adapters:
            from .seq_adapter_coverage import refuse_dropped_adapters
            from .seq_routing import SeqRoutingBinding, SeqRoutingUnavailable
            from .tied_head_adapter import refuse_tied_head_adapters

            try:
                binding = SeqRoutingBinding(self.C)
            except SeqRoutingUnavailable:
                binding = None
            if binding is not None:
                # An adapter on token_embd.weight aborts the process when the base model ties
                # its output head. Refuse before the model loads.
                refuse_tied_head_adapters(model_path, ((name, gguf_path) for name, gguf_path, _scale in adapters))
                # The fork's seq path drops the token-embedding and MoE-expert parts of
                # an adapter without a message. Refuse before the model loads.
                refuse_dropped_adapters((name, gguf_path) for name, gguf_path, _scale in adapters)
        self._n_seq_max = n_seq_max
        self.n_batch = n_batch
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
        )
        self.model = self.llm._model.model
        self.n_vocab = self.llm._n_vocab
        self._eos = int(self.llm.token_eos())

        p = self.C.llama_context_default_params()
        p.n_ctx = n_ctx
        p.n_batch = n_batch
        p.n_ubatch = n_batch
        p.n_seq_max = n_seq_max
        self.ctx = self.C.llama_new_context_with_model(self.model, p)
        if not self.ctx:
            raise RuntimeError("llama_new_context_with_model returned NULL (out of VRAM?)")

        # Register the ordered pool once and let each slot pick its adapter.
        if binding is not None:
            ptrs = []
            for name, gguf_path, _scale in adapters:
                ptr = self.C.llama_adapter_lora_init(self.model, gguf_path.encode("utf-8"))
                if not ptr:
                    raise RuntimeError(f"failed to load adapter {name!r} from {gguf_path}")
                self._seq_index[name] = len(ptrs)
                ptrs.append(ptr)
            binding.set_seq_adapters(self.ctx, ptrs)
            self._seq_binding = binding

    @property
    def n_seq_max(self) -> int:
        return self._n_seq_max

    @property
    def eos(self) -> int:
        return self._eos

    @property
    def supports_seq_routing(self) -> bool:
        """True when a fork adapter pool is registered (mixed-adapter admission)."""
        return self._seq_binding is not None

    def set_slot_adapter(self, seq_id: int, adapter: str | None) -> None:
        """Route slot ``seq_id`` to ``adapter`` (``None`` = base) before its prefill."""
        if self._seq_binding is None:
            return
        from .seq_routing import NO_ADAPTER

        idx = NO_ADAPTER if adapter is None else self._seq_index[adapter]
        self._seq_binding.set_seq_adapter(self.ctx, seq_id, idx)

    def tokenize(self, prompt: str) -> list[int]:
        return list(self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))

    def detokenize(self, tokens: Sequence[int]) -> str:
        return self.llm.detokenize(list(tokens)).decode("utf-8", errors="replace")

    def free_slot(self, seq_id: int) -> None:
        self.C.llama_memory_seq_rm(self.C.llama_get_memory(self.ctx), seq_id, -1, -1)

    def _argmax_row(self, row: int) -> int:
        ptr = self.C.llama_get_logits_ith(self.ctx, row)
        logits = self._np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))
        return int(self._np.argmax(logits))

    def prefill(self, tokens: Sequence[int], seq_id: int) -> int:
        toks = list(tokens)
        n = len(toks)
        for off in range(0, n, self.n_batch):
            chunk = toks[off : off + self.n_batch]
            m = len(chunk)
            batch = self.C.llama_batch_init(m, 0, 1)
            batch.n_tokens = m
            for i, t in enumerate(chunk):
                batch.token[i] = t
                batch.pos[i] = off + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq_id
                batch.logits[i] = 1 if off + m == n and i == m - 1 else 0
            rc = self.C.llama_decode(self.ctx, batch)
            self.C.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"prefill llama_decode rc={rc}")
        return self._argmax_row(-1)

    def step(self, active: Sequence[tuple[int, int, int]]) -> list[int]:
        s = len(active)
        batch = self.C.llama_batch_init(s, 0, 1)
        batch.n_tokens = s
        for row, (seq_id, token, pos) in enumerate(active):
            batch.token[row] = token
            batch.pos[row] = pos
            batch.n_seq_id[row] = 1
            batch.seq_id[row][0] = seq_id
            batch.logits[row] = 1
        rc = self.C.llama_decode(self.ctx, batch)
        if rc != 0:
            self.C.llama_batch_free(batch)
            raise RuntimeError(f"batched llama_decode rc={rc}")
        out = [self._argmax_row(row) for row in range(s)]
        self.C.llama_batch_free(batch)
        return out

    def close(self) -> None:
        self.C.llama_free(self.ctx)
        self.llm.close()

    def __enter__(self) -> LlamaContinuousBatchEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
