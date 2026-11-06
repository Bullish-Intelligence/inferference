"""CPU-only contracts for the scaled per-sequence routing call.

``llama_set_seq_adapter_scaled`` is D2 option B: the per-sequence LoRA scale
rides the per-decode routing mask instead of the adapter pool, so two sequences
can use one pool slot at two scales inside one ``llama_decode``, and changing a
scale forces no graph re-reserve.

It is a capability separate from the P2 pair. An ``integration`` lib exports
``llama_set_seq_adapters`` and ``llama_set_seq_adapter`` and not this one, so the
binding must keep working there and refuse only the scaled call. No native lib
and no GPU: these use the same fake-lib pattern as the P2 suite tests.
"""

from __future__ import annotations

import ctypes

import pytest

from inferference.seq_routing import (
    NO_ADAPTER,
    SEQ_ADAPTER_SCALED_SYMBOL,
    SeqRoutingBinding,
    SeqRoutingUnavailable,
    library_supports_seq_adapter_scale,
)


class _FakeSymbol:
    """A ctypes-function stand-in: callable, with settable argtypes/restype."""

    def __init__(self, name: str, calls: list[tuple]) -> None:
        self._name = name
        self._calls = calls
        self.argtypes: list[type] = []
        self.restype: type | None = None

    def __call__(self, *args):  # noqa: ANN002
        self._calls.append((self._name, *args))
        return 0


class _FakeLibPair:
    """An integration libllama: the P2 pair, and no scaled call."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.llama_set_seq_adapters = _FakeSymbol("set_seq_adapters", self.calls)
        self.llama_set_seq_adapter = _FakeSymbol("set_seq_adapter", self.calls)


class _FakeLibScaled(_FakeLibPair):
    """A stage-4 libllama: the pair plus the scaled call."""

    def __init__(self) -> None:
        super().__init__()
        self.llama_set_seq_adapter_scaled = _FakeSymbol("set_seq_adapter_scaled", self.calls)


class _FakeC:
    """Minimal stand-in for the ``llama_cpp.llama_cpp`` ctypes namespace."""

    llama_context_p_ctypes = ctypes.c_void_p
    llama_adapter_lora_p_ctypes = ctypes.c_void_p

    def __init__(self, lib: object) -> None:
        self._lib = lib


def test_scale_capability_probe() -> None:
    assert library_supports_seq_adapter_scale(_FakeLibScaled()) is True
    assert library_supports_seq_adapter_scale(_FakeLibPair()) is False


def test_binding_still_works_against_a_lib_without_the_scaled_call() -> None:
    """The pair is the bind-time requirement; the scaled call is not."""
    binding = SeqRoutingBinding(_FakeC(_FakeLibPair()))
    assert binding.supports_scale is False
    binding.set_seq_adapter(ctypes.c_void_p(0x1234), 0, 1)


def test_scaled_call_refuses_when_the_symbol_is_absent() -> None:
    binding = SeqRoutingBinding(_FakeC(_FakeLibPair()))
    with pytest.raises(SeqRoutingUnavailable, match=SEQ_ADAPTER_SCALED_SYMBOL):
        binding.set_seq_adapter_scaled(ctypes.c_void_p(0x1234), 0, 1, 0.5)


def test_scaled_call_passes_the_scale_through() -> None:
    lib = _FakeLibScaled()
    binding = SeqRoutingBinding(_FakeC(lib))
    assert binding.supports_scale is True
    ctx = ctypes.c_void_p(0x1234)
    binding.set_seq_adapter_scaled(ctx, 0, 1, 0.25)
    binding.set_seq_adapter_scaled(ctx, 1, 1, 0.75)
    binding.set_seq_adapter_scaled(ctx, 2, NO_ADAPTER, 1.0)

    assert [c[0] for c in lib.calls] == ["set_seq_adapter_scaled"] * 3
    # Two sequences, one pool slot, two scales: the case option A cannot serve.
    assert (lib.calls[0][2], lib.calls[0][3], lib.calls[0][4]) == (0, 1, 0.25)
    assert (lib.calls[1][2], lib.calls[1][3], lib.calls[1][4]) == (1, 1, 0.75)
    assert lib.calls[2][3] == NO_ADAPTER


def test_scaled_call_binds_a_float_argtype() -> None:
    """The scale must cross as c_float, or the C side reads a converted int."""
    lib = _FakeLibScaled()
    SeqRoutingBinding(_FakeC(lib))
    assert lib.llama_set_seq_adapter_scaled.argtypes[-1] is ctypes.c_float
    assert lib.llama_set_seq_adapter_scaled.restype is ctypes.c_int32
