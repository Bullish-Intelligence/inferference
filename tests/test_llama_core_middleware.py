from __future__ import annotations

import ctypes
import types

import numpy as np
import pytest

from inferference.decode import FINISH_LENGTH, FINISH_STOP, OwnedLlamaDecoder
from inferference.middleware import (
    BaseDecodeMiddleware,
    CallbackMiddleware,
    DecodeMiddleware,
    GrammarMiddleware,
    MiddlewarePipeline,
    StopTokenMiddleware,
    as_pipeline,
)


class _ArgmaxCandidates:
    """Candidate surface whose selection is the argmax of the (masked) logits."""

    def __init__(self, logits: np.ndarray) -> None:
        self.data = [types.SimpleNamespace(id=i, logit=float(v)) for i, v in enumerate(logits)]
        self.selected = int(np.argmax(logits))


def _argmax_decoder(step_logits: list[np.ndarray]) -> OwnedLlamaDecoder:
    """Decoder whose per-step logits are scripted; selection is the argmax.

    The candidate array reflects logits *after* hooks mutate the view, so a
    masking middleware can steer the selection -- exactly what the real loop
    does when it copies the masked view into the candidate logits.
    """
    decoder = OwnedLlamaDecoder.__new__(OwnedLlamaDecoder)
    decoder._closed = False
    decoder._owns_sampler = False
    decoder.sampler = object()
    steps = iter(step_logits)
    decoder.llm = types.SimpleNamespace(reset=lambda: None)

    state: dict[str, _ArgmaxCandidates] = {}

    def _candidate_array(self: OwnedLlamaDecoder) -> tuple[np.ndarray, _ArgmaxCandidates]:
        logits = np.array(next(steps), dtype=np.float32)
        # Selection is deferred until after hooks mutate the returned view.
        candidates = _ArgmaxCandidates(logits)
        state["last"] = candidates
        return logits, candidates

    def _reselect(candidates: _ArgmaxCandidates) -> None:
        candidates.selected = int(np.argmax([d.logit for d in candidates.data]))

    decoder._candidate_array = types.MethodType(_candidate_array, decoder)  # type: ignore[assignment]
    decoder._decode_one = types.MethodType(lambda self, token, position: None, decoder)  # type: ignore[assignment]
    decoder._native = types.SimpleNamespace(
        # The real loop copies the masked view into candidate logits before
        # sampling; mirror that here so masking affects the argmax selection.
        llama_sampler_apply=lambda *_: _reselect(state["last"]),
        llama_sampler_accept=lambda *_: None,
    )
    return decoder


@pytest.fixture(autouse=True)
def _identity_byref(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ctypes, "byref", lambda arg: arg)


# --- pipeline unit tests -------------------------------------------------


def test_base_middleware_satisfies_protocol() -> None:
    assert isinstance(BaseDecodeMiddleware(), DecodeMiddleware)


def test_as_pipeline_normalizes_inputs() -> None:
    empty = as_pipeline(None)
    assert isinstance(empty, MiddlewarePipeline) and len(empty) == 0

    existing = MiddlewarePipeline([BaseDecodeMiddleware()])
    assert as_pipeline(existing) is existing

    from_list = as_pipeline([BaseDecodeMiddleware(), BaseDecodeMiddleware()])
    assert len(from_list) == 2


def test_pipeline_preserves_order_on_logits() -> None:
    calls: list[str] = []

    class Named(BaseDecodeMiddleware):
        def __init__(self, name: str) -> None:
            self.name = name

        def on_logits(self, logits: object) -> None:
            calls.append(self.name)

    MiddlewarePipeline([Named("a"), Named("b"), Named("c")]).on_logits(object())
    assert calls == ["a", "b", "c"]


def test_pipeline_on_logits_exposes_mutable_view() -> None:
    class Zeroer(BaseDecodeMiddleware):
        def on_logits(self, logits: object) -> None:
            logits[:] = 0.0  # type: ignore[index]

    view = np.arange(5, dtype=np.float32)
    MiddlewarePipeline([Zeroer()]).on_logits(view)
    assert np.all(view == 0.0)


def test_pipeline_on_token_stops_if_any_middleware_requests() -> None:
    seen: list[int] = []

    class Observer(BaseDecodeMiddleware):
        def on_token(self, token: int) -> bool:
            seen.append(token)
            return False

    pipeline = MiddlewarePipeline([StopTokenMiddleware({42}), Observer()])
    assert pipeline.on_token(7) is False
    # Even though the first middleware stops, the observer still sees the token.
    assert pipeline.on_token(42) is True
    assert seen == [7, 42]


def test_pipeline_on_finish_dispatches_outcome() -> None:
    received: list[object] = []

    class Recorder(BaseDecodeMiddleware):
        def on_finish(self, outcome: object) -> None:
            received.append(outcome)

    sentinel = object()
    MiddlewarePipeline([Recorder()]).on_finish(sentinel)
    assert received == [sentinel]


# --- integration with the owned decode loop ------------------------------


def test_decoder_drives_prompt_and_finish_stages() -> None:
    events: list[str] = []

    class Tracer(BaseDecodeMiddleware):
        def on_prompt(self, tokens: object) -> None:
            events.append(f"prompt:{list(tokens)}")

        def on_finish(self, outcome: object) -> None:
            events.append(f"finish:{outcome.finish_reason}")  # type: ignore[attr-defined]

    decoder = _argmax_decoder([np.array([0.0, 1.0]), np.array([0.0, 1.0])])
    decoder.generate_tokens([5], max_tokens=2, middleware=[Tracer()])
    assert events == ["prompt:[5]", "finish:length"]


def test_middleware_on_token_can_request_stop() -> None:
    decoder = _argmax_decoder([np.array([0.0, 1.0]), np.array([2.0, 0.0]), np.array([0.0, 3.0])])
    # Argmax picks token 1, then 0, then 1; stop when token 0 is selected.
    outcome = decoder.generate_tokens([5], max_tokens=8, middleware=[StopTokenMiddleware({0})])
    assert outcome.tokens == [1]
    assert outcome.finish_reason == FINISH_STOP
    assert outcome.stop_token == 0


def test_middleware_masking_steers_selection() -> None:
    class MaskToken0(BaseDecodeMiddleware):
        def on_logits(self, logits: object) -> None:
            logits[1] = -np.inf  # type: ignore[index]

    # Without the mask argmax would pick token 1; the middleware forces token 0.
    decoder = _argmax_decoder([np.array([0.0, 5.0])])
    outcome = decoder.generate_tokens([5], max_tokens=1, middleware=[MaskToken0()])
    assert outcome.tokens == [0]
    assert outcome.finish_reason == FINISH_LENGTH


def test_callback_middleware_bridges_legacy_hooks() -> None:
    logits_seen: list[int] = []
    tokens_seen: list[int] = []

    mw = CallbackMiddleware(
        logits_hook=lambda logits: logits_seen.append(len(logits)),
        token_hook=lambda token: tokens_seen.append(token),
    )
    decoder = _argmax_decoder([np.array([0.0, 1.0])])
    decoder.generate_tokens([5], max_tokens=1, middleware=[mw])
    assert logits_seen == [2]
    assert tokens_seen == [1]


# --- grammar middleware --------------------------------------------------


class _FakeGrammar:
    """Grammar whose logits_hook masks a fixed forbidden token; token_hook records."""

    def __init__(self, forbidden: int) -> None:
        self.forbidden = forbidden
        self.accepted: list[int] = []

    def logits_hook(self, matcher: object, *, benchmark: object | None = None) -> object:
        def apply(logits: object) -> None:
            logits[self.forbidden] = -np.inf  # type: ignore[index]

        return apply

    def token_hook(self, matcher: object) -> object:
        def accept(token: int) -> None:
            self.accepted.append(token)

        return accept


def test_grammar_middleware_masks_like_direct_hooks() -> None:
    grammar = _FakeGrammar(forbidden=1)
    mw = GrammarMiddleware(grammar, matcher=object())

    # Direct-hook equivalence: applying the middleware's on_logits matches the
    # grammar's own logits_hook output on the same view.
    direct_view = np.array([0.0, 5.0, 1.0], dtype=np.float32)
    mw_view = direct_view.copy()
    grammar.logits_hook(object())(direct_view)
    mw.on_logits(mw_view)
    assert np.array_equal(np.isneginf(direct_view), np.isneginf(mw_view))

    # And in the loop the mask steers selection away from the forbidden token.
    decoder = _argmax_decoder([np.array([0.0, 5.0, 1.0])])
    outcome = decoder.generate_tokens([5], max_tokens=1, middleware=[mw])
    assert outcome.tokens == [2]
    assert grammar.accepted == [2]
