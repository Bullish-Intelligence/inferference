"""A small, explicit llama.cpp decode loop.

This module deliberately uses ``llama_decode`` and the sampler C API instead
of the high-level completion helper.  It is the teaching reference for later
grammar integration: a mask hook receives the zero-copy logits view before
sampler transforms run.

The two original callbacks (``logits_hook``/``token_hook``) still work, but the
loop now also drives a composable :class:`~.middleware.MiddlewarePipeline`, so
new decode concerns plug in as stage-aware middleware instead of new callback
parameters (Pillar 3).
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .middleware import MiddlewarePipeline


class LogitsHook(Protocol):
    """Mutate a logits vector before sampler application."""

    def __call__(self, logits: Any) -> None: ...


class TokenHook(Protocol):
    """Observe exactly the token selected by the sampler."""

    def __call__(self, token: int) -> None: ...


class SynchronizeHook(Protocol):
    """Synchronize the active CUDA stream before timing a GPU wall interval."""

    def __call__(self) -> None: ...


# Why generation stopped.  ``stop`` means a stop token was sampled (a clean
# end-of-sequence); ``length`` means ``max_tokens`` was reached first.  Making
# this explicit is what turns a truncated generation from a silent, malformed
# output into an observable outcome the caller can reject.
FINISH_STOP = "stop"
FINISH_LENGTH = "length"


@dataclass(frozen=True, slots=True)
class DecodeOutcome:
    """The tokens produced by one owned decode plus why the loop stopped."""

    tokens: list[int]
    finish_reason: str
    stop_token: int | None = None


@dataclass(frozen=True, slots=True)
class DecodeText:
    """Detokenized text plus the structured outcome that produced it."""

    text: str
    finish_reason: str
    stop_token: int | None = None
    completion_token_count: int = 0


def _llama_cpp() -> Any:
    """Import lazily so core-only installs do not import the native library."""
    import llama_cpp

    return llama_cpp


class OwnedLlamaDecoder:
    """Single-sequence owned decode loop backed by a ``llama_cpp.Llama``.

    ``Llama`` remains responsible only for model lifecycle and tokenization.
    Evaluation, zero-copy logits access, sampling, and sampler acceptance are
    owned here.  The sampler is accepted exactly once per output token.
    """

    def __init__(self, llm: Any, sampler: Any | None = None) -> None:
        self.llm = llm
        self._native = _llama_cpp()
        self._owns_sampler = sampler is None
        self.sampler = sampler if sampler is not None else self._new_greedy_sampler()
        self._batch = self._native.llama_batch_init(1, 0, 1)
        # Candidate IDs never depend on the request.  Keep one C-compatible
        # array and refill it vectorially; rebuilding 248k Python structs per
        # token would dominate a GPU benchmark and obscure mask cost.
        import numpy as np

        self._candidate_type = self._native.llama_token_data * self.llm.n_vocab()
        self._candidates = self._candidate_type()
        self._candidate_view = np.ctypeslib.as_array(self._candidates)
        self._candidate_ids = np.arange(self.llm.n_vocab(), dtype=self._candidate_view["id"].dtype)
        self._closed = False

    def _new_greedy_sampler(self) -> Any:
        chain = self._native.llama_sampler_chain_init(self._native.llama_sampler_chain_default_params())
        self._native.llama_sampler_chain_add(chain, self._native.llama_sampler_init_greedy())
        return chain

    def close(self) -> None:
        """Free the owned batch and, when applicable, sampler chain."""
        if self._closed:
            return
        self._native.llama_batch_free(self._batch)
        if self._owns_sampler:
            self._native.llama_sampler_free(self.sampler)
        self._closed = True

    def __enter__(self) -> OwnedLlamaDecoder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _decode_one(self, token: int, position: int) -> None:
        batch = self._batch
        batch.n_tokens = 1
        batch.token[0] = token
        batch.pos[0] = position
        batch.n_seq_id[0] = 1
        batch.seq_id[0][0] = 0
        batch.logits[0] = 1
        result = self._native.llama_decode(self.llm._ctx.ctx, batch)
        if result != 0:
            raise RuntimeError(f"llama_decode failed with code {result}")

    def _candidate_array(self) -> tuple[Any, Any]:
        """Copy logits into llama.cpp's candidate representation for sampling."""
        import numpy as np

        logits_pointer = self._native.llama_get_logits_ith(self.llm._ctx.ctx, 0)
        logits = np.ctypeslib.as_array(logits_pointer, shape=(self.llm.n_vocab(),))
        self._candidate_view["id"] = self._candidate_ids
        self._candidate_view["logit"] = logits
        self._candidate_view["p"].fill(0.0)
        array = self._native.llama_token_data_array(self._candidates, len(self._candidates), -1, False)
        return logits, array

    def generate_tokens(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int,
        logits_hook: LogitsHook | None = None,
        token_hook: TokenHook | None = None,
        stop_tokens: frozenset[int] = frozenset(),
        middleware: MiddlewarePipeline | Iterable[Any] | None = None,
        benchmark: Any | None = None,
        synchronize: SynchronizeHook | None = None,
    ) -> DecodeOutcome:
        """Generate greedily or with the supplied sampler, without completion APIs.

        Returns a :class:`DecodeOutcome` whose ``finish_reason`` distinguishes a
        clean stop-token end from a ``max_tokens`` cutoff, so callers never have
        to infer truncation from a downstream parse failure.

        ``middleware`` composes stage-aware :class:`~.middleware.DecodeMiddleware`
        into the loop.  It is dispatched *after* the legacy ``logits_hook`` and
        ``token_hook`` at each stage, so both APIs coexist; a middleware may
        mutate the logits view or request a stop from ``on_token``.
        """
        if self._closed:
            raise RuntimeError("OwnedLlamaDecoder is closed")
        if not prompt_tokens:
            raise ValueError("prompt_tokens must contain at least one token")
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")

        from .middleware import as_pipeline

        pipeline = as_pipeline(middleware)
        pipeline.on_prompt(prompt_tokens)

        # A new decoder is normally paired with a fresh context.  Reset is
        # necessary for hybrid models, and keeps lifecycle ownership explicit.
        self.llm.reset()
        if benchmark is None:
            for position, token in enumerate(prompt_tokens):
                self._decode_one(token, position)
        else:
            from time import perf_counter_ns

            prefill_started_ns = perf_counter_ns()
            with benchmark.measure("prefill_enqueue"):
                for position, token in enumerate(prompt_tokens):
                    self._decode_one(token, position)
            if synchronize is not None:
                synchronize()
            benchmark.add_ns("prefill_wall", perf_counter_ns() - prefill_started_ns)

        generated: list[int] = []
        next_position = len(prompt_tokens)
        finish_reason = FINISH_LENGTH
        stop_token: int | None = None
        generation_measurement = benchmark.measure("generation_wall") if benchmark is not None else nullcontext()
        with generation_measurement:
            for _ in range(max_tokens):
                token_started_ns = None
                if benchmark is not None:
                    from time import perf_counter_ns

                    token_started_ns = perf_counter_ns()
                candidate_measurement = benchmark.measure("candidate_array") if benchmark is not None else nullcontext()
                with candidate_measurement:
                    logits, candidates = self._candidate_array()
                if logits_hook is not None:
                    logits_hook(logits)
                pipeline.on_logits(logits)
                # Candidate logits are copied after masking to make the order
                # visible and unambiguous to readers of the loop.
                if hasattr(self, "_candidate_view"):
                    self._candidate_view["logit"] = logits
                else:  # Test doubles may bypass __init__ and provide candidates directly.
                    for index, logit in enumerate(logits):
                        candidates.data[index].logit = float(logit)
                sample_measurement = benchmark.measure("sampler_apply") if benchmark is not None else nullcontext()
                with sample_measurement:
                    self._native.llama_sampler_apply(self.sampler, ctypes.byref(candidates))
                token = int(candidates.data[candidates.selected].id)
                # This is the sole llama sampler acceptance; the convenience
                # sampling function would accept internally.
                accept_measurement = benchmark.measure("sampler_accept") if benchmark is not None else nullcontext()
                with accept_measurement:
                    self._native.llama_sampler_accept(self.sampler, token)
                if token_hook is not None:
                    matcher_measurement = (
                        benchmark.measure("matcher_accept") if benchmark is not None else nullcontext()
                    )
                    with matcher_measurement:
                        token_hook(token)
                middleware_stop = pipeline.on_token(token)
                if token in stop_tokens or middleware_stop:
                    finish_reason = FINISH_STOP
                    stop_token = token
                    break
                generated.append(token)
                self._decode_one(token, next_position)
                if synchronize is not None:
                    synchronize()
                if benchmark is not None:
                    from time import perf_counter_ns

                    benchmark.record_token_latency_ns(perf_counter_ns() - token_started_ns)
                next_position += 1
        outcome = DecodeOutcome(tokens=generated, finish_reason=finish_reason, stop_token=stop_token)
        pipeline.on_finish(outcome)
        return outcome

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int,
        logits_hook: LogitsHook | None = None,
        token_hook: TokenHook | None = None,
        middleware: MiddlewarePipeline | Iterable[Any] | None = None,
        benchmark: Any | None = None,
        synchronize: SynchronizeHook | None = None,
    ) -> DecodeText:
        """Tokenize, run the owned loop, and detokenize its generated tokens.

        The returned :class:`DecodeText` carries ``finish_reason`` so a caller
        (or the soak harness) can flag a ``length`` cutoff before attempting to
        parse a likely-truncated result.
        """
        prompt_tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        outcome = self.generate_tokens(
            prompt_tokens,
            max_tokens=max_tokens,
            logits_hook=logits_hook,
            token_hook=token_hook,
            stop_tokens=frozenset({self.llm.token_eos()}),
            middleware=middleware,
            benchmark=benchmark,
            synchronize=synchronize,
        )
        if benchmark is None:
            text = self.llm.detokenize(outcome.tokens, special=True).decode("utf-8", errors="replace")
        else:
            with benchmark.measure("detokenize"):
                text = self.llm.detokenize(outcome.tokens, special=True).decode("utf-8", errors="replace")
        return DecodeText(
            text=text,
            finish_reason=outcome.finish_reason,
            stop_token=outcome.stop_token,
            completion_token_count=len(outcome.tokens),
        )
