"""Hats loop-routing binding contract tests (fake-lib, no GPU).

Mirrors ``suites/p2-mixed-batch-lora/tests/test_seq_routing.py``'s fake-lib
pattern: the hats surface (``LOOP_ROUTING_SYMBOLS`` / ``LoopRoutingBinding`` /
``LoopRoutingUnavailable``) must bind on a hats-capable lib and raise the
documented error on a lib without the symbols — absent-safe, never a crash,
never an import-time failure.
"""

from __future__ import annotations

import ctypes

import pytest

from inferference.seq_routing import (
    LOOP_ROUTING_SYMBOLS,
    LoopRoutingBinding,
    LoopRoutingUnavailable,
    library_supports_loop_routing,
)


class _FakeSymbol:
    def __init__(self, name: str, calls: list[tuple]) -> None:
        self._name = name
        self._calls = calls
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):  # noqa: ANN002
        self._calls.append((self._name, *args))
        return 0


class _FakeLibHats:
    """A hats libllama surface: both loop symbols present, capturing calls."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.llama_set_loop_adapters = _FakeSymbol("set_loop_adapters", self.calls)
        self.llama_set_loop_adapter = _FakeSymbol("set_loop_adapter", self.calls)


class _FakeLibStock:
    """A stock libllama surface: no loop symbols, no P2 symbols."""


class _FakeC:
    """Minimal stand-in for the ``llama_cpp.llama_cpp`` ctypes namespace."""

    llama_context_p_ctypes = ctypes.c_void_p
    llama_adapter_lora_p_ctypes = ctypes.c_void_p

    def __init__(self, lib: object) -> None:
        self._lib = lib


def test_loop_routing_symbols_are_the_hats_pair() -> None:
    assert LOOP_ROUTING_SYMBOLS == ("llama_set_loop_adapters", "llama_set_loop_adapter")


def test_library_capability_probe() -> None:
    assert library_supports_loop_routing(_FakeLibHats()) is True
    assert library_supports_loop_routing(_FakeLibStock()) is False


def test_binding_raises_documented_error_on_stock_lib() -> None:
    """Capability-absent path raises the documented error (no GPU)."""
    with pytest.raises(LoopRoutingUnavailable, match="hats"):
        LoopRoutingBinding(_FakeC(_FakeLibStock()))


def test_binding_binds_and_routes_on_hats_lib() -> None:
    lib = _FakeLibHats()
    binding = LoopRoutingBinding(_FakeC(lib))
    ctx = ctypes.c_void_p(0x1234)
    binding.set_loop_adapters(ctx, [ctypes.c_void_p(1), ctypes.c_void_p(2)])
    binding.set_loop_adapter(ctx, 0, 0)
    binding.set_loop_adapter(ctx, 1, -1)  # base sentinel for a loop step
    kinds = [c[0] for c in lib.calls]
    assert kinds == ["set_loop_adapters", "set_loop_adapter", "set_loop_adapter"]
    assert lib.calls[1][3] == 0  # (name, ctx, loop_step, adapter_idx)
    assert lib.calls[2][3] == -1


def test_hats_and_p2_surfaces_are_independent() -> None:
    """A hats lib need not expose the P2 seq surface (and vice versa)."""
    from inferference.seq_routing import library_supports_seq_routing

    assert library_supports_seq_routing(_FakeLibHats()) is False
    assert library_supports_loop_routing(_FakeLibStock()) is False
