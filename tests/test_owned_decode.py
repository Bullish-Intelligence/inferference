from __future__ import annotations

import ctypes
import types

import pytest

from inferference.decode import (
    FINISH_LENGTH,
    FINISH_STOP,
    DecodeOutcome,
    OwnedLlamaDecoder,
)


class _FakeCandidates:
    """Minimal stand-in for llama.cpp's token_data_array selection surface."""

    def __init__(self, chosen: int) -> None:
        self.selected = 0
        self.data = [types.SimpleNamespace(id=chosen, logit=0.0)]


def _fake_decoder(script: list[int]) -> OwnedLlamaDecoder:
    """Build a decoder whose native/model layers are replaced by fakes.

    ``script`` is the sequence of token ids the sampler will 'select', so the
    finish-reason logic can be exercised with no llama.cpp present.
    """
    decoder = OwnedLlamaDecoder.__new__(OwnedLlamaDecoder)
    decoder._closed = False
    decoder._owns_sampler = False
    decoder.sampler = object()
    steps = iter(script)

    decoder.llm = types.SimpleNamespace(reset=lambda: None)
    decoder._native = types.SimpleNamespace(
        llama_sampler_apply=lambda *_: None,
        llama_sampler_accept=lambda *_: None,
    )
    decoder._decode_one = types.MethodType(lambda self, token, position: None, decoder)  # type: ignore[assignment]
    decoder._candidate_array = types.MethodType(  # type: ignore[assignment]
        lambda self: ([], _FakeCandidates(next(steps))), decoder
    )
    return decoder


@pytest.fixture(autouse=True)
def _identity_byref(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fake candidate array is not a ctypes instance; the sampler call it is
    # passed to is a no-op fake, so bypass byref's type check for these tests.
    monkeypatch.setattr(ctypes, "byref", lambda arg: arg)


def test_generate_tokens_reports_stop_when_stop_token_sampled() -> None:
    decoder = _fake_decoder([11, 12, 99])
    outcome = decoder.generate_tokens([1], max_tokens=8, stop_tokens=frozenset({99}))

    assert isinstance(outcome, DecodeOutcome)
    assert outcome.tokens == [11, 12]
    assert outcome.finish_reason == FINISH_STOP
    assert outcome.stop_token == 99


def test_generate_tokens_reports_length_on_max_tokens_cutoff() -> None:
    decoder = _fake_decoder([11, 12, 13])
    outcome = decoder.generate_tokens([1], max_tokens=3, stop_tokens=frozenset({99}))

    assert outcome.tokens == [11, 12, 13]
    assert outcome.finish_reason == FINISH_LENGTH
    assert outcome.stop_token is None


def test_owned_loop_contains_one_explicit_sampler_accept() -> None:
    """Gate 1 regression: the teaching loop must not delegate to sample()."""
    source = OwnedLlamaDecoder.generate_tokens.__doc__
    assert source is not None

    # The implementation's public contract is independently pinned here.  A
    # native smoke (when the model is available) tests the C API separately.
    import inspect

    implementation = inspect.getsource(OwnedLlamaDecoder.generate_tokens)
    assert implementation.count(".llama_sampler_accept(") == 1
    assert "llama_sampler_sample" not in implementation
