"""Multi-LoRA agent-router surface — the project-17 flagship as a library API.

One base model loaded once; one pinned-adapter ``llama_context`` per adapter (plus
an optional base context). A request carries an adapter name; the router routes it
to that adapter's context, batches concurrent requests within a context (own-batch
multi-sequence decode), and multiplexes across contexts. Optional grammar-constrained
decoding reuses :mod:`inferference.grammar`.

Design boundaries (repo standing rule): Pydantic validates at the edges
(:class:`RouterConfig`, :class:`RouteRequest`, :class:`RouteResult`); the decode hot
path passes plain token ids and numpy logit views. This mirrors
:mod:`inferference.decode`, extended from one sequence to a routed,
batched wave.

The research provenance (GPU-validated, exact-match vs isolated baselines) lives in
an earlier internal harness tree; this module is the teaching-quality library form.
"""

from __future__ import annotations

import copy
import ctypes
import json
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .decode import FINISH_LENGTH, FINISH_STOP
from .grammar import JsonSchemaGrammar, apply_packed_bitmask_inplace
from .models import EngineConfig, GenerationResult
from .prefix_cache_live import LlamaSeqStateBridge
from .seq_adapter_coverage import refuse_dropped_adapters
from .seq_routing import NO_ADAPTER, SeqRoutingBinding, SeqRoutingUnavailable
from .tied_head_adapter import refuse_tied_head_adapters

BASE: None = None  # sentinel adapter name meaning "no adapter / raw base model"


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdapterSpec(_BoundaryModel):
    """One LoRA adapter in the router's pool."""

    name: str
    gguf_path: str
    scale: float = Field(default=1.0, gt=0)


Backend = Literal["context_pool", "seq_routed", "auto"]


class RouterConfig(_BoundaryModel):
    """Everything needed to stand up a router over one base model.

    ``backend`` selects how a mixed-adapter workload is served:

    * ``context_pool`` — one pinned-adapter ``llama_context`` per adapter; requests
      are grouped by adapter and each group multiplexed on its context (the
      always-available shipping path).
    * ``seq_routed`` — the P2 fork's true mixed-batch path: one context, the ordered
      adapter pool registered once, each sequence assigned its adapter, a single
      ``llama_decode`` carrying the mix. Requires a fork lib; raises at construction
      if the capability is absent (explicit opt-in). It also raises
      :class:`~inferference.seq_adapter_coverage.SeqRoutedAdapterRefused` for an
      adapter that targets the token embedding or the MoE expert matrices: this
      path would drop those parts without a message.
    * ``auto`` (default) — ``seq_routed`` when the loaded lib reports the routing
      capability, else ``context_pool``. Fail-closed: a stock lib silently uses the
      context-pool path; missing capability is never an inference failure. When
      ``auto`` resolves to ``seq_routed``, the adapter refusal above applies; set
      ``context_pool`` to serve such an adapter.

    Every backend raises :class:`~inferference.tied_head_adapter.TiedHeadAdapterRefused`
    for an adapter on ``token_embd.weight`` when the base model ties its output head:
    the process would abort at the first decode. This refusal has no override and no
    backend that avoids it.
    """

    engine: EngineConfig
    adapters: tuple[AdapterSpec, ...]
    n_seq_max: int = Field(default=8, gt=0)
    include_base: bool = True
    backend: Backend = "auto"


class RouteRequest(_BoundaryModel):
    """A generation request tagged with the adapter it should route to.

    Sampling fields (Stage 2, llgym Phase 1): all ``None`` means greedy
    argmax — the historical default, so existing callers are untouched.
    Setting any of ``temperature``/``top_p``/``top_k``/``seed`` builds a
    llama.cpp sampler chain for the request (see :func:`build_sampler_chain`);
    ``temperature == 0`` stays greedy (deterministic, matching
    llama-cpp-python's ``temp == 0 -> add_greedy``).
    """

    prompt: str
    adapter: str | None = None  # None -> base context
    max_tokens: int = Field(default=64, gt=0)
    request_id: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, gt=0)
    seed: int | None = None
    # Grammar-constrained runs only. False (default) forces compact JSON. True lets
    # the model emit whitespace between JSON elements, bounded per place by
    # ``grammar.MAX_WHITESPACE_CNT``. An unconstrained run ignores it.
    allow_whitespace: bool = False


def request_uses_sampling(
    *, temperature: float | None, top_p: float | None, top_k: int | None, seed: int | None
) -> bool:
    """True when a request's sampling fields imply a non-greedy decode.

    Greedy (argmax) when no sampling parameter is set, or when the only
    sampling signal is ``temperature == 0`` (deterministic, matching
    llama-cpp-python's ``temp == 0 -> add_greedy``).  Any other combination
    (temperature != 0, or top_p/top_k/seed set) requires a sampler chain.
    """
    return not (temperature == 0 or (temperature is None and top_p is None and top_k is None and seed is None))


# llama.cpp's random-seed sentinel: ``get_rng_seed`` (src/llama-sampler.cpp:340)
# randomizes ONLY when the seed equals LLAMA_DEFAULT_SEED (include/llama.h:37);
# any other seed (including 0) is fixed. llama-cpp-python's ``seed or
# LLAMA_DEFAULT_SEED`` (llama.py:309) therefore treats None/0 as "random per
# request" — the same contract an unseeded sampler chain must honor, or every
# temp!=0 request is silently deterministic (rig finding 2026-08-05).
_LLAMA_DEFAULT_SEED = 0xFFFFFFFF


def build_sampler_chain(
    C: Any,
    engine_seed: int | None,
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
) -> Any | None:
    """Build a per-request llama.cpp sampler chain, or None for greedy argmax.

    Mirrors the ``OwnedLlamaDecoder`` sampler pattern (``decode.py:104-105``:
    ``llama_sampler_chain_init`` + ``chain_add`` + apply/accept at sampling
    time). 0.3.34 has **no** ``llama_sampler_init_seed`` — the seed lives in
    ``llama_sampler_init_dist(seed)`` (same mechanism llama-cpp-python's
    ``sample()`` uses: ``temp == 0 -> greedy``, else ``top_k/top_p/temp/dist``).

    Greedy (returns ``None``) when no sampling parameter is set, or when the
    only sampling signal is ``temperature == 0``.  Otherwise the chain is, in
    application order: top_k (if set) -> top_p (if set) -> temperature (if set
    and != 0) -> dist(seed).  A missing temperature means T=1.0 (no temp
    sampler); a missing seed falls back to the engine seed, then the random
    sentinel (``LLAMA_DEFAULT_SEED``) — unseeded sampling is random per chain
    build, matching llama-cpp-python's ``seed or LLAMA_DEFAULT_SEED``.
    """
    if not request_uses_sampling(temperature=temperature, top_p=top_p, top_k=top_k, seed=seed):
        return None
    chain = C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
    if top_k is not None:
        C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_k(top_k))
    if top_p is not None:
        C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_p(top_p, 1))
    if temperature is not None and temperature != 0:
        C.llama_sampler_chain_add(chain, C.llama_sampler_init_temp(temperature))
    dist_seed = seed if seed is not None else (engine_seed if engine_seed is not None else _LLAMA_DEFAULT_SEED)
    C.llama_sampler_chain_add(chain, C.llama_sampler_init_dist(dist_seed))
    return chain


class RouteResult(GenerationResult):
    """A completed route: the generation plus which adapter served it."""

    adapter: str | None = None
    decision: dict[str, Any] | None = None  # parsed JSON when grammar-constrained


@dataclass
class CachedRunResult:
    """One ``run_cached`` execution: cached + cold waves and isolated baselines.

    ``cached`` / ``cold`` / ``baselines`` are parallel to the input request
    order; the wall times are CUDA-synchronized totals across all adapters and
    waves (cached = restore + suffix-only, cold = full re-prefill).
    """

    cached: list[RouteResult]
    cold: list[RouteResult]
    baselines: list[RouteResult]
    cached_wall_ns: int
    cold_wall_ns: int


class MultiLoRARouter:
    """Shared base model, one pinned-adapter context per adapter, batched routing."""

    def __init__(self, config: RouterConfig) -> None:
        from llama_cpp import Llama, llama_cpp

        self.config = config
        self.C = llama_cpp
        # ---- backend selection (fail-closed) ----
        # ``auto`` uses the fork path only when the loaded lib exports it; ``seq_routed``
        # is an explicit opt-in that raises here if the fork is absent; ``context_pool``
        # never touches the fork surface. Resolved before the model loads, so a refusal
        # below costs no VRAM.
        self.backend = self._resolve_backend(config.backend)
        # Every backend: an adapter on token_embd.weight aborts the process when the base
        # model ties its output head. Refuse it now; there is no override.
        refuse_tied_head_adapters(config.engine.model_path, ((s.name, s.gguf_path) for s in config.adapters))
        if self.backend == "seq_routed":
            # seq_routed drops the token-embedding and MoE-expert parts of an adapter
            # without a message. Refuse such an adapter now; there is no override.
            refuse_dropped_adapters((s.name, s.gguf_path) for s in config.adapters)
        eng = config.engine
        self.n_seq_max = config.n_seq_max
        self.n_batch = eng.n_batch
        self.llm = Llama(
            model_path=eng.model_path,
            n_ctx=eng.n_ctx,
            n_batch=eng.n_batch,
            n_gpu_layers=eng.n_gpu_layers,
            seed=eng.seed,
            verbose=False,
        )
        self.model = self.llm._model.model
        self.n_vocab = self.llm._n_vocab
        self._eos = int(self.llm.token_eos())

        self._adapter_ptr: dict[str, Any] = {}
        for spec in config.adapters:
            ptr = self.C.llama_adapter_lora_init(self.model, spec.gguf_path.encode("utf-8"))
            if not ptr:
                raise RuntimeError(f"failed to load adapter {spec.name!r} from {spec.gguf_path}")
            self._adapter_ptr[spec.name] = ptr
        self._scale = {s.name: s.scale for s in config.adapters}

        self._seq_binding: SeqRoutingBinding | None = None
        self._seq_ctx: Any = None
        # Ordered adapter pool: pool index == routing id. Fixed at construction so a
        # sequence's index is stable across waves. BASE maps to the -1 sentinel.
        self._seq_pool: tuple[str, ...] = tuple(s.name for s in config.adapters)
        self._seq_index: dict[str, int] = {name: i for i, name in enumerate(self._seq_pool)}

        self._ctx: dict[str | None, Any] = {}
        try:
            if self.backend == "seq_routed":
                self._setup_seq_routed()
            else:
                self._setup_context_pool()
        except Exception:
            # Fail-closed: a failed backend setup must not leak the loaded model
            # (a multi-GB GPU allocation on a 12 GB rig).
            self.llm.close()
            raise
        self._grammar: JsonSchemaGrammar | None = None  # compact grammar (the default)
        self._grammar_schema: dict[str, Any] | None = None  # the router's own copy
        self._grammar_whitespace: JsonSchemaGrammar | None = None  # compiled on first use
        # Lazy per-request sampler scratch buffers (allocated only when a request
        # asks for sampling — greedy routers never touch this).
        self._candidate_type: Any | None = None
        self._candidates: Any | None = None
        self._candidate_view: Any | None = None
        self._candidate_ids: Any | None = None

    # ---- backend setup ----

    def _resolve_backend(self, requested: Backend) -> str:
        """Pick the concrete backend, honoring the fail-closed rule for ``auto``."""
        lib = getattr(self.C, "_lib", None)
        from .seq_routing import library_supports_seq_routing

        capable = lib is not None and library_supports_seq_routing(lib)
        if requested == "seq_routed":
            if not capable:
                raise SeqRoutingUnavailable(
                    "backend='seq_routed' requires the P2 fork lib "
                    "(llama_set_seq_adapters/llama_set_seq_adapter absent)"
                )
            return "seq_routed"
        if requested == "auto":
            return "seq_routed" if capable else "context_pool"
        return "context_pool"

    def _setup_context_pool(self) -> None:
        names: list[str | None] = ([BASE] if self.config.include_base else []) + list(self._seq_pool)
        created: list[tuple[str | None, Any]] = []
        try:
            for name in names:
                ctx = self._make_ctx()
                created.append((name, ctx))
                if name is not BASE:
                    self._pin_adapter(ctx, self._adapter_ptr[name], self._scale[name])
                self._ctx[name] = ctx
        except Exception:
            # Fail-closed: a partial context-pool build must not leak the created
            # contexts (each is a multi-GB KV reservation on a 12 GB rig).
            for _, ctx in created:
                self.C.llama_free(ctx)
            self._ctx.clear()
            raise

    def _setup_seq_routed(self) -> None:
        """One context; register the ordered adapter pool once via the fork binding.

        The fork's ``llama_set_seq_adapters`` takes no per-adapter scale — the masked
        path applies each adapter's built-in alpha scaling. So an ``AdapterSpec.scale``
        override is honored only on the ``context_pool`` backend; on ``seq_routed`` a
        non-default scale is ignored. Warn rather than silently diverge.
        """
        off_default = [s.name for s in self.config.adapters if s.scale != 1.0]
        if off_default:
            import warnings

            warnings.warn(
                f"seq_routed backend ignores AdapterSpec.scale overrides for {off_default} "
                "(the fork applies built-in adapter alpha); use backend='context_pool' to "
                "apply custom scales.",
                stacklevel=2,
            )
        self._seq_binding = SeqRoutingBinding(self.C)
        try:
            self._seq_ctx = self._make_ctx()
            self._seq_binding.set_seq_adapters(self._seq_ctx, [self._adapter_ptr[name] for name in self._seq_pool])
        except Exception:
            if self._seq_ctx is not None:
                self.C.llama_free(self._seq_ctx)
                self._seq_ctx = None
            raise

    # ---- setup ----

    def _make_ctx(self) -> Any:
        p = self.C.llama_context_default_params()
        p.n_ctx = self.config.engine.n_ctx
        p.n_batch = self.n_batch
        p.n_ubatch = self.n_batch
        p.n_seq_max = self.n_seq_max
        # The llama.cpp default (kv_unified=False, PR #17046) partitions the
        # context across n_seq_max: n_ctx_seq = n_ctx / n_seq_max — with the
        # router's n_seq_max=8 a requested n_ctx=4096 silently becomes 512
        # tokens per sequence and every generation beyond ~500 total tokens
        # fails "failed to find a memory slot" (llgym Stage-6 rig finding).
        # Unified KV gives each sequence the full n_ctx; concurrent sequences
        # in one context still arbitrate cells via the memory manager.
        p.kv_unified = True
        ctx = self.C.llama_new_context_with_model(self.model, p)
        if not ctx:
            raise RuntimeError("llama_new_context_with_model returned NULL (out of VRAM?)")
        return ctx

    def _pin_adapter(self, ctx: Any, ptr: Any, scale: float) -> None:
        arr = (self.C.llama_adapter_lora_p_ctypes * 1)(ptr)
        scales = (ctypes.c_float * 1)(scale)
        if self.C.llama_set_adapters_lora(ctx, arr, 1, scales) != 0:
            raise RuntimeError("llama_set_adapters_lora failed")

    def enable_grammar(self, tokenizer: Any, schema: dict[str, Any]) -> None:
        """Compile a JSON-schema grammar; constrained runs mask to it per sequence.

        The compact grammar compiles now. The whitespace grammar compiles on the
        first request with ``allow_whitespace=True``, with the compiler of the compact
        grammar. The router keeps its own copy of ``schema``: a later change to the
        caller's dict does not change the whitespace grammar.
        """
        schema = copy.deepcopy(schema)
        self._grammar = JsonSchemaGrammar.from_huggingface(tokenizer, schema, vocab_size=self.n_vocab)
        self._grammar_schema = schema
        self._grammar_whitespace = None

    def _grammar_for(self, request: RouteRequest) -> JsonSchemaGrammar:
        """Return the compiled grammar that matches the request's whitespace option."""
        if self._grammar is None or self._grammar_schema is None:
            raise RuntimeError("a constrained request requires enable_grammar(...) first")
        if not request.allow_whitespace:
            return self._grammar
        if self._grammar_whitespace is None:
            self._grammar_whitespace = self._grammar.with_whitespace(
                copy.deepcopy(self._grammar_schema), allow_whitespace=True
            )
        return self._grammar_whitespace

    # ---- decode primitives (plain tokens / numpy on the hot path) ----

    def _seq_rm(self, ctx: Any, seq_id: int) -> None:
        self.C.llama_memory_seq_rm(self.C.llama_get_memory(ctx), seq_id, -1, -1)

    def reset_state(self) -> None:
        """Clear the KV memory of every context, data included.

        The caller must be idle: call this only when no ``run``, ``run_cached`` or
        ``run_stream`` is in progress on this router. The call is not safe against a
        running decode.

        A wave removes only its own sequences. Sequences of an earlier wave stay in the
        unified KV cache, and they set where the next sequence lands and how many cells
        attention reads. On a Vulkan build that changes the logits. A gate that compares
        two runs calls this before each run, so both start from the same state.
        """
        contexts = list(self._ctx.values())
        if self._seq_ctx is not None:
            contexts.append(self._seq_ctx)
        for ctx in contexts:
            self.C.llama_memory_clear(self.C.llama_get_memory(ctx), True)

    def _prefill(self, ctx: Any, tokens: list[int], seq_id: int) -> None:
        n = len(tokens)
        for off in range(0, n, self.n_batch):
            chunk = tokens[off : off + self.n_batch]
            m = len(chunk)
            batch = self.C.llama_batch_init(m, 0, 1)
            batch.n_tokens = m
            for i, t in enumerate(chunk):
                batch.token[i] = t
                batch.pos[i] = off + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq_id
                batch.logits[i] = 1 if off + m == n and i == m - 1 else 0
            rc = self.C.llama_decode(ctx, batch)
            self.C.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"prefill llama_decode rc={rc}")

    def _candidate_array(self, logits: Any) -> Any:
        """Wrap zero-copy logits in llama.cpp's candidate representation.

        Same scratch-buffer pattern as ``decode.py:135`` — one C-compatible
        array reused across rows/requests, refilled vectorially.
        """
        import numpy as np

        candidate_type = self._candidate_type
        if candidate_type is None:
            candidate_type = self.C.llama_token_data * self.n_vocab
            self._candidate_type = candidate_type
            candidates = candidate_type()
            self._candidates = candidates
            view = np.ctypeslib.as_array(candidates)
            self._candidate_view = view
            ids = np.arange(self.n_vocab, dtype=view["id"].dtype)
            self._candidate_ids = ids
        else:
            candidates = self._candidates
            view = self._candidate_view
            ids = self._candidate_ids
            assert candidates is not None and view is not None and ids is not None
        view["id"] = ids
        view["logit"] = logits
        view["p"].fill(0.0)
        return self.C.llama_token_data_array(candidates, len(candidates), -1, False)

    def _sample_row(self, ctx: Any, row: int, matcher: Any | None = None, sampler: Any | None = None) -> int:
        ptr = self.C.llama_get_logits_ith(ctx, row)
        logits = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))
        if matcher is not None:
            import xgrammar as xgr

            bitmask = np.zeros(xgr.get_bitmask_shape(1, self.n_vocab), dtype=np.int32)
            if matcher.fill_next_token_bitmask(bitmask):
                apply_packed_bitmask_inplace(logits, bitmask[0], self.n_vocab)
        if sampler is None:
            tok = int(np.argmax(logits))
        else:
            array = self._candidate_array(logits)
            self.C.llama_sampler_apply(sampler, ctypes.byref(array))
            tok = int(array.data[array.selected].id)
            self.C.llama_sampler_accept(sampler, tok)
        if matcher is not None and matcher.accept_token(tok) is False:
            raise RuntimeError(f"grammar rejected token {tok} (fail-closed)")
        return tok

    def _advance(self, out: list[int], tok: int, matcher: Any | None) -> str | None:
        """Record a sampled token; return a finish_reason if the sequence is done.

        Grammar completion or a sampled EOS is a clean ``stop``; otherwise the
        sequence continues (``None``) until ``max_tokens`` yields a ``length`` cut.
        """
        if matcher is not None:
            out.append(tok)
            return FINISH_STOP if matcher.is_terminated() else None
        if tok == self._eos:
            return FINISH_STOP  # EOS is the boundary, not content — do not emit it
        out.append(tok)
        return None

    def _iter_wave(
        self,
        ctx: Any,
        prompts: list[list[int]],
        max_tokens: int,
        matchers: list[Any] | None,
        samplers: list[Any | None] | None = None,
    ) -> Any:
        """Streaming wave generator: yields one step per decode.

        Each step is a list (parallel to the sequences) of
        ``(token, finish_reason_or_None)``: ``token`` is the token to *emit*
        (None when the slot already finished, or when the stop was an EOS that
        is not content), and ``finish_reason`` is "stop"/"length" the step the
        sequence ends, else None.  A final flush step reports
        ``finish or FINISH_LENGTH`` for every slot so stream consumers observe
        the same contract as :meth:`_generate_wave` (a ``max_tokens`` cutoff is
        ``length``, a clean end is ``stop``).

        seq_id == row invariant, batched decode, and per-row sampling mirror
        the materialized :meth:`_generate_wave` exactly — this generator is the
        single wave-loop implementation and the non-stream path materializes
        it, so both paths cannot drift.  Sampler chains passed in are owned by
        the wave: they are freed here (finally), including on early close
        (client disconnect aborts the generation cleanly).
        """
        s = len(prompts)
        try:
            for i in range(s):
                self._seq_rm(ctx, i)
            out: list[list[int]] = [[] for _ in range(s)]
            cur: list[int] = [0] * s
            pos: list[int] = [0] * s
            finish: list[str | None] = [None] * s
            for i in range(s):
                self._prefill(ctx, prompts[i], i)
                cur[i] = self._sample_row(ctx, -1, matchers[i] if matchers else None, samplers[i] if samplers else None)
                pos[i] = len(prompts[i])
                before = len(out[i])
                finish[i] = self._advance(out[i], cur[i], matchers[i] if matchers else None)
                step: list[tuple[int | None, str | None]] = [(None, None)] * s
                step[i] = self._step(out, before, finish, i)
                yield step
            for _ in range(max_tokens - 1):
                if all(f is not None for f in finish):
                    break
                batch = self.C.llama_batch_init(s, 0, 1)
                batch.n_tokens = s
                for i in range(s):
                    batch.token[i] = cur[i]
                    batch.pos[i] = pos[i]
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = i
                    batch.logits[i] = 1
                rc = self.C.llama_decode(ctx, batch)
                self.C.llama_batch_free(batch)
                if rc != 0:
                    raise RuntimeError(f"batched llama_decode rc={rc}")
                step: list[tuple[int | None, str | None]] = []
                for i in range(s):
                    pos[i] += 1
                    if finish[i] is not None:
                        step.append((None, None))
                        continue
                    cur[i] = self._sample_row(
                        ctx, i, matchers[i] if matchers else None, samplers[i] if samplers else None
                    )
                    before = len(out[i])
                    finish[i] = self._advance(out[i], cur[i], matchers[i] if matchers else None)
                    step.append(self._step(out, before, finish, i))
                yield step
            # Flush: a wave that ran out of tokens ends unfinished slots as
            # ``length`` (the non-stream contract) — visible to stream
            # consumers instead of being inferred from generator exhaustion.
            yield [(None, finish[i] or FINISH_LENGTH) for i in range(s)]
        finally:
            if samplers:
                for smp in samplers:
                    if smp is not None:
                        self.C.llama_sampler_free(smp)

    @staticmethod
    def _step(out: list[list[int]], before: int, finish: list[str | None], i: int) -> tuple[int | None, str | None]:
        """The emit view of one sampled token: (token or None, finish or None)."""
        emitted = out[i][before:]  # _advance appended unless it was an EOS stop
        return (emitted[0] if emitted else None, finish[i])

    def _generate_wave(
        self,
        ctx: Any,
        prompts: list[list[int]],
        max_tokens: int,
        matchers: list[Any] | None,
        samplers: list[Any | None] | None = None,
    ) -> list[tuple[list[int], str]]:
        """Greedy/sampled batched decode of up to n_seq_max sequences; seq_id == row.

        Returns ``(tokens, finish_reason)`` per sequence so truncation is
        observable.  Finished slots ride along (frozen) to keep the seq_id==row
        invariant until all finish or ``max_tokens`` is hit.  This is the
        materialized wrapper over the streaming :meth:`_iter_wave` — the public
        surface and semantics are unchanged for existing callers.
        """
        s = len(prompts)
        out: list[list[int]] = [[] for _ in range(s)]
        finish: list[str | None] = [None] * s
        for step in self._iter_wave(ctx, prompts, max_tokens, matchers, samplers):
            for i, (tok, fin) in enumerate(step):
                if tok is not None:
                    out[i].append(tok)
                if fin is not None:
                    finish[i] = fin
        return [(out[i], finish[i] or FINISH_LENGTH) for i in range(s)]

    # ---- public API ----

    def tokenize(self, text: str) -> list[int]:
        """Tokenize with a leading BOS (the shared-prefix convention)."""
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True))

    def tokenize_suffix(self, text: str) -> list[int]:
        """Suffix tokens WITHOUT a leading BOS (the shared prefix carried it)."""
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def run(self, requests: list[RouteRequest], *, constrained: bool = False) -> list[RouteResult]:
        """Route + batch a mixed workload; optionally constrain to the grammar.

        Same public surface on both backends. ``context_pool`` groups by adapter and
        multiplexes across contexts; ``seq_routed`` carries a mix of adapters in each
        single-``llama_decode`` wave.
        """
        if constrained and self._grammar is None:
            raise RuntimeError("constrained=True requires enable_grammar(...) first")
        for r in requests:
            if r.adapter is not BASE and r.adapter not in self._adapter_ptr:
                raise KeyError(f"unknown adapter {r.adapter!r}")
        if self.backend == "seq_routed":
            return self._run_seq_routed(requests, constrained=constrained)
        return self._run_context_pool(requests, constrained=constrained)

    def _finalize(
        self, r: RouteRequest, prompt: list[int], gen: tuple[list[int], str], constrained: bool
    ) -> RouteResult:
        toks, finish_reason = gen
        toks = toks[: r.max_tokens]
        text = self.llm.detokenize(toks).decode("utf-8", errors="replace")
        # Only parse when the grammar actually completed; a length cut leaves the
        # JSON truncated, and ``validated`` must reflect that.
        decision = None
        if constrained and finish_reason == FINISH_STOP:
            try:
                decision = json.loads(text)
            except ValueError:
                decision = None
        return RouteResult(
            text=text,
            token_ids=tuple(toks),
            prompt_token_count=len(prompt),
            completion_token_count=len(toks),
            finish_reason=finish_reason,
            request_id=r.request_id,
            adapter=r.adapter,
            decision=decision,
            validated=(decision is not None) if constrained else None,
        )

    def _run_context_pool(self, requests: list[RouteRequest], *, constrained: bool) -> list[RouteResult]:
        by_adapter: dict[str | None, list[int]] = {}
        for idx, r in enumerate(requests):
            by_adapter.setdefault(r.adapter, []).append(idx)
        results: dict[int, RouteResult] = {}
        for adapter, idxs in by_adapter.items():
            ctx = self._ctx[adapter]
            for w in range(0, len(idxs), self.n_seq_max):
                wave = idxs[w : w + self.n_seq_max]
                prompts = [self.tokenize(requests[j].prompt) for j in wave]
                matchers = [self._grammar_for(requests[j]).new_matcher() for j in wave] if constrained else None
                samplers = [self._build_sampler(requests[j]) for j in wave]
                gen = self._generate_wave(ctx, prompts, max(requests[j].max_tokens for j in wave), matchers, samplers)
                for slot, j in enumerate(wave):
                    results[j] = self._finalize(requests[j], prompts[slot], gen[slot], constrained)
        return [results[i] for i in range(len(requests))]

    def _run_seq_routed(self, requests: list[RouteRequest], *, constrained: bool) -> list[RouteResult]:
        """True mixed-batch: each wave is one ``llama_decode`` over a mix of adapters.

        Requests keep submission order; each wave of up to ``n_seq_max`` sequences
        assigns row ``i`` its own adapter (``-1`` for BASE) before decoding, so a
        single decode carries the mix — no per-adapter context, no ``n_seq_max`` split.
        """
        assert self._seq_binding is not None and self._seq_ctx is not None
        ctx = self._seq_ctx
        results: list[RouteResult] = []
        for w in range(0, len(requests), self.n_seq_max):
            wave = requests[w : w + self.n_seq_max]
            for i, r in enumerate(wave):
                idx = NO_ADAPTER if r.adapter is BASE else self._seq_index[r.adapter]
                self._seq_binding.set_seq_adapter(ctx, i, idx)
            prompts = [self.tokenize(r.prompt) for r in wave]
            matchers = [self._grammar_for(r).new_matcher() for r in wave] if constrained else None
            samplers = [self._build_sampler(r) for r in wave]
            gen = self._generate_wave(ctx, prompts, max(r.max_tokens for r in wave), matchers, samplers)
            for i, r in enumerate(wave):
                results.append(self._finalize(r, prompts[i], gen[i], constrained))
        return results

    def _build_sampler(self, request: RouteRequest) -> Any | None:
        """Per-request sampler chain (None -> greedy argmax, back-compat)."""
        return build_sampler_chain(
            self.C,
            self.config.engine.seed,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=request.seed,
        )

    def run_stream(self, request: RouteRequest, *, constrained: bool = False) -> Any:
        """Stream ONE request: yields ``(token, finish_reason_or_None)`` per step.

        ``token`` is the token to emit (None on a stop step, e.g. an EOS that
        is not content); ``finish_reason`` turns non-None exactly the step the
        sequence ends ("stop" clean / "length" max_tokens — the same contract
        as :meth:`run`).  Closing the generator early (client disconnect)
        aborts the wave and frees the request's sampler chain.

        Streaming requires ``backend="context_pool"`` unless the fork lib is
        loaded (``seq_routed`` sets the sequence's adapter the same way
        :meth:`_run_seq_routed` does).  Grammar-constrained streaming works via
        the same per-sequence matcher as the batched path.
        """
        if constrained and self._grammar is None:
            raise RuntimeError("constrained=True requires enable_grammar(...) first")
        if request.adapter is not BASE and request.adapter not in self._adapter_ptr:
            raise KeyError(f"unknown adapter {request.adapter!r}")
        if self.backend == "seq_routed":
            assert self._seq_binding is not None and self._seq_ctx is not None
            idx = NO_ADAPTER if request.adapter is BASE else self._seq_index[request.adapter]
            self._seq_binding.set_seq_adapter(self._seq_ctx, 0, idx)
            ctx = self._seq_ctx
        else:
            ctx = self._ctx[request.adapter]
        prompts = [self.tokenize(request.prompt)]
        matchers = [self._grammar_for(request).new_matcher()] if constrained else None
        samplers = [self._build_sampler(request)]
        inner = self._iter_wave(ctx, prompts, request.max_tokens, matchers, samplers)
        try:
            for step in inner:
                yield step[0][0], step[0][1]
        finally:
            inner.close()

    # ---- cached shared-prefix routing (reference context_pool_router_cached) ----

    def _generate_wave_restored(
        self,
        ctx: Any,
        blob: bytes,
        prefix_len: int,
        suffixes: list[list[int]],
        max_tokens: int,
        matchers: list[Any] | None,
        bridge: LlamaSeqStateBridge,
    ) -> list[tuple[list[int], str]]:
        """Greedy batched decode after restoring ONE shared-prefix KV blob.

        Every seq slot gets the same restored prefix (fail-closed: a
        ``set_data == 0`` reject aborts before any decode), then decodes only
        its suffix at explicit positions right after the restored KV and
        generates. Keeps ``_generate_wave``'s seq_id == row invariant and the
        finish-reason contract.
        """
        s = len(suffixes)
        if s > self.n_seq_max:
            raise ValueError(f"{s} sequences exceed n_seq_max {self.n_seq_max}")
        for i in range(s):
            self._seq_rm(ctx, i)
            bridge.restore_blob_into_seq(ctx, blob, i)
        out: list[list[int]] = [[] for _ in range(s)]
        cur: list[int] = [0] * s
        pos: list[int] = [0] * s
        finish: list[str | None] = [None] * s
        for i in range(s):
            bridge.decode_tokens(ctx, suffixes[i], i, prefix_len)
            cur[i] = self._sample_row(ctx, -1, matchers[i] if matchers else None)
            pos[i] = prefix_len + len(suffixes[i])
            finish[i] = self._advance(out[i], cur[i], matchers[i] if matchers else None)
        for _ in range(max_tokens - 1):
            if all(f is not None for f in finish):
                break
            batch = self.C.llama_batch_init(s, 0, 1)
            batch.n_tokens = s
            for i in range(s):
                batch.token[i] = cur[i]
                batch.pos[i] = pos[i]
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = i
                batch.logits[i] = 1
            rc = self.C.llama_decode(ctx, batch)
            self.C.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"batched llama_decode rc={rc}")
            for i in range(s):
                pos[i] += 1
                if finish[i] is not None:
                    continue
                cur[i] = self._sample_row(ctx, i, matchers[i] if matchers else None)
                finish[i] = self._advance(out[i], cur[i], matchers[i] if matchers else None)
        return [(out[i], finish[i] or FINISH_LENGTH) for i in range(s)]

    def _baseline_generate(self, adapter: str | None, tokens: list[int], max_tokens: int) -> tuple[list[int], str]:
        """Isolated batch-1 ground truth: fresh context, full prompt, greedy."""
        ctx = self._make_ctx()
        try:
            if adapter is not BASE:
                self._pin_adapter(ctx, self._adapter_ptr[adapter], self._scale[adapter])
            return self._generate_wave(ctx, [tokens], max_tokens, None)[0]
        finally:
            self.C.llama_free(ctx)

    def run_cached(
        self,
        requests: list[RouteRequest],
        prefix_token_ids: Sequence[int],
        *,
        constrained: bool = False,
    ) -> CachedRunResult:
        """Cached shared-prefix routing: prefill the prefix ONCE per adapter, then
        restore it into every request slot and decode only each request's suffix.

        ``request.prompt`` is the SUFFIX text (the prefix is provided separately
        as token ids; the full prompt is prefix + suffix). Requires
        ``backend="context_pool"`` (the per-adapter context cache);
        ``constrained=True`` requires ``enable_grammar(...)`` first. Returns the
        cached wave, the cold full re-prefill wave, isolated batch-1 baselines,
        and the synchronized cached/cold wall times.
        """
        if self.backend != "context_pool":
            raise RuntimeError("run_cached requires backend='context_pool' (per-adapter prefix cache)")
        if constrained and self._grammar is None:
            raise RuntimeError("constrained=True requires enable_grammar(...) first")
        sampling = [
            r
            for r in requests
            if request_uses_sampling(temperature=r.temperature, top_p=r.top_p, top_k=r.top_k, seed=r.seed)
        ]
        if sampling:
            # Fail-closed: the cached legs decode with the greedy wave; silently
            # ignoring a request's sampling params would be a fidelity lie.
            raise ValueError(
                "run_cached does not support sampling parameters "
                f"(requests {[r.request_id for r in sampling]}); pass all-None sampling fields"
            )
        prefix = list(prefix_token_ids)
        if not prefix:
            raise ValueError("prefix_token_ids must contain at least one token")
        for r in requests:
            if r.adapter is not BASE and r.adapter not in self._adapter_ptr:
                raise KeyError(f"unknown adapter {r.adapter!r}")
        bridge = LlamaSeqStateBridge(n_batch=self.n_batch, n_vocab=self.n_vocab, native=self.C)

        by_adapter: dict[str | None, list[int]] = {}
        for idx, r in enumerate(requests):
            by_adapter.setdefault(r.adapter, []).append(idx)

        cached: dict[int, RouteResult] = {}
        cold: dict[int, RouteResult] = {}
        baselines: dict[int, RouteResult] = {}
        cached_wall_ns = 0
        cold_wall_ns = 0

        for adapter, idxs in by_adapter.items():
            ctx = self._ctx[adapter]
            # Cache the shared prefix ONCE in this adapter's context.
            self._seq_rm(ctx, 0)
            self._prefill(ctx, prefix, 0)
            blob = bridge.capture_seq_state(ctx, 0)
            self._seq_rm(ctx, 0)

            for w in range(0, len(idxs), self.n_seq_max):
                wave = idxs[w : w + self.n_seq_max]
                suffixes = [self.tokenize_suffix(requests[j].prompt) for j in wave]
                max_tokens = max(requests[j].max_tokens for j in wave)
                matchers_c = [self._grammar_for(requests[j]).new_matcher() for j in wave] if constrained else None
                matchers_cold = [self._grammar_for(requests[j]).new_matcher() for j in wave] if constrained else None
                fulls = [prefix + s for s in suffixes]

                self.C.llama_synchronize(ctx)
                t0 = perf_counter_ns()
                gen_c = self._generate_wave_restored(ctx, blob, len(prefix), suffixes, max_tokens, matchers_c, bridge)
                self.C.llama_synchronize(ctx)
                cached_wall_ns += perf_counter_ns() - t0

                self.C.llama_synchronize(ctx)
                t0 = perf_counter_ns()
                gen_cold = self._generate_wave(ctx, fulls, max_tokens, matchers_cold)
                self.C.llama_synchronize(ctx)
                cold_wall_ns += perf_counter_ns() - t0

                for slot, j in enumerate(wave):
                    base_gen = self._baseline_generate(adapter, fulls[slot], max_tokens)
                    cached[j] = self._finalize(requests[j], fulls[slot], gen_c[slot], constrained)
                    cold[j] = self._finalize(requests[j], fulls[slot], gen_cold[slot], constrained)
                    baselines[j] = self._finalize(requests[j], fulls[slot], base_gen, constrained=False)

        return CachedRunResult(
            cached=[cached[i] for i in range(len(requests))],
            cold=[cold[i] for i in range(len(requests))],
            baselines=[baselines[i] for i in range(len(requests))],
            cached_wall_ns=cached_wall_ns,
            cold_wall_ns=cold_wall_ns,
        )

    def close(self) -> None:
        for ctx in self._ctx.values():
            self.C.llama_free(ctx)
        self._ctx.clear()
        if self._seq_ctx is not None:
            self.C.llama_free(self._seq_ctx)
            self._seq_ctx = None
        self.llm.close()

    def __enter__(self) -> MultiLoRARouter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
