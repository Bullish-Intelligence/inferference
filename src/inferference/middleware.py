"""Composable decode middleware for the owned llama.cpp loop.

Pillar 3 treats inference as a *workflow you can add to, intercept, and
modify*.  Rather than threading one ad-hoc callback per concern through
:meth:`OwnedLlamaDecoder.generate_tokens`, a decode is expressed as an ordered
list of :class:`DecodeMiddleware`.  Each middleware sees the stages of a single
owned decode and may observe or mutate them:

* ``on_prompt(tokens)`` — once, before prefill.  Observe the full prompt.
* ``on_logits(logits)`` — every step, on the zero-copy logits *view* before the
  sampler runs.  Mutate in place to bias/mask sampling (grammar masking lives
  here).
* ``on_token(token)`` — every step, on the exact token the sampler selected.
  Return a truthy value to request that generation stop.
* ``on_finish(outcome)`` — once, after the loop ends, with the final
  :class:`DecodeOutcome`.

The hot-path stages (``on_logits``/``on_token``) stay plain: a numpy view and a
Python ``int``.  Boundary/config objects are Pydantic elsewhere; nothing here
allocates per token beyond what a middleware itself chooses to do.  This module
imports no native library and no numpy at import time, matching ``decode.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DecodeMiddleware(Protocol):
    """One pluggable stage-aware participant in an owned decode.

    Every method is optional: a middleware implements only the stages it cares
    about.  The :class:`MiddlewarePipeline` dispatches each stage to whichever
    middleware define it, so a logits-only masker and a token-only observer can
    coexist without either implementing the other's hooks.
    """

    def on_prompt(self, tokens: Sequence[int]) -> None:
        """Observe the prompt tokens once, before prefill."""

    def on_logits(self, logits: Any) -> None:
        """Mutate the zero-copy logits view in place before sampling."""

    def on_token(self, token: int) -> bool | None:
        """Observe the selected token; return truthy to request a stop."""

    def on_finish(self, outcome: Any) -> None:
        """Observe the final :class:`DecodeOutcome` after the loop ends."""


class BaseDecodeMiddleware:
    """No-op base so subclasses override only the stages they need.

    Subclassing is optional -- any object with a matching subset of methods is
    a valid middleware -- but inheriting keeps a middleware future-proof if new
    stages are added to the pipeline.
    """

    def on_prompt(self, tokens: Sequence[int]) -> None:  # noqa: D102
        return None

    def on_logits(self, logits: Any) -> None:  # noqa: D102
        return None

    def on_token(self, token: int) -> bool | None:  # noqa: D102
        return None

    def on_finish(self, outcome: Any) -> None:  # noqa: D102
        return None


class MiddlewarePipeline:
    """Compose an ordered list of middleware and dispatch each decode stage.

    Ordering is significant for mutating stages: ``on_logits`` runs the
    middleware left to right, so a later masker sees the biases applied by an
    earlier one.  Observing stages (``on_prompt``/``on_token``/``on_finish``)
    are always dispatched to *every* middleware even when one requests a stop,
    so observers never miss the final token.
    """

    __slots__ = ("_middleware",)

    def __init__(self, middleware: Iterable[Any] = ()) -> None:
        self._middleware = list(middleware)

    def __len__(self) -> int:
        return len(self._middleware)

    def __iter__(self) -> Any:
        return iter(self._middleware)

    @property
    def middleware(self) -> list[Any]:
        """The ordered middleware, for introspection in tests and tooling."""
        return list(self._middleware)

    def on_prompt(self, tokens: Sequence[int]) -> None:
        for mw in self._middleware:
            hook = getattr(mw, "on_prompt", None)
            if hook is not None:
                hook(tokens)

    def on_logits(self, logits: Any) -> None:
        for mw in self._middleware:
            hook = getattr(mw, "on_logits", None)
            if hook is not None:
                hook(logits)

    def on_token(self, token: int) -> bool:
        """Dispatch the selected token; return True if any middleware stops.

        Every middleware is called even after a stop is requested, so a pure
        observer downstream of a stopping middleware still sees the token.
        """
        stop = False
        for mw in self._middleware:
            hook = getattr(mw, "on_token", None)
            if hook is not None and hook(token):
                stop = True
        return stop

    def on_finish(self, outcome: Any) -> None:
        for mw in self._middleware:
            hook = getattr(mw, "on_finish", None)
            if hook is not None:
                hook(outcome)


def as_pipeline(middleware: MiddlewarePipeline | Iterable[Any] | None) -> MiddlewarePipeline:
    """Normalize the decoder's ``middleware`` argument to a pipeline.

    Accepts ``None`` (empty pipeline), an existing :class:`MiddlewarePipeline`
    (returned as-is), or any iterable of middleware.
    """
    if middleware is None:
        return MiddlewarePipeline()
    if isinstance(middleware, MiddlewarePipeline):
        return middleware
    return MiddlewarePipeline(middleware)


class CallbackMiddleware(BaseDecodeMiddleware):
    """Adapter that lifts the legacy ``logits_hook``/``token_hook`` callbacks.

    The owned decoder still exposes the two original callback parameters; this
    adapter lets them participate in a pipeline unchanged, which is how the two
    ad-hoc hooks are unified without breaking their call sites.  A ``token_hook``
    is a pure observer, so this never requests a stop.
    """

    __slots__ = ("_logits_hook", "_token_hook")

    def __init__(self, logits_hook: Any | None = None, token_hook: Any | None = None) -> None:
        self._logits_hook = logits_hook
        self._token_hook = token_hook

    def on_logits(self, logits: Any) -> None:
        if self._logits_hook is not None:
            self._logits_hook(logits)

    def on_token(self, token: int) -> bool | None:
        if self._token_hook is not None:
            self._token_hook(token)
        return None


class StopTokenMiddleware(BaseDecodeMiddleware):
    """Request a stop when one of a fixed set of tokens is selected.

    A small, self-contained example of a token-stage middleware that changes
    control flow, mirroring the decoder's built-in ``stop_tokens`` set but as a
    pluggable unit.
    """

    __slots__ = ("_stop_tokens",)

    def __init__(self, stop_tokens: Iterable[int]) -> None:
        self._stop_tokens = frozenset(stop_tokens)

    def on_token(self, token: int) -> bool:
        return token in self._stop_tokens


class GrammarMiddleware(BaseDecodeMiddleware):
    """Unify a :class:`JsonSchemaGrammar` matcher under the pipeline.

    ``on_logits`` fills and applies the next-token mask; ``on_token`` advances
    the matcher, failing closed if a masked token is somehow selected.  It wraps
    the grammar's existing ``logits_hook``/``token_hook`` factories, so the
    pipeline and the direct-hook API produce identical masking -- the middleware
    is a thin composition, not a reimplementation.
    """

    __slots__ = ("_apply_logits", "_accept_token")

    def __init__(self, grammar: Any, matcher: Any, *, benchmark: Any | None = None) -> None:
        self._apply_logits = grammar.logits_hook(matcher, benchmark=benchmark)
        self._accept_token = grammar.token_hook(matcher)

    def on_logits(self, logits: Any) -> None:
        self._apply_logits(logits)

    def on_token(self, token: int) -> bool | None:
        self._accept_token(token)
        return None
