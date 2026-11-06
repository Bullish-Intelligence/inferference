"""Mixed-batch per-sequence LoRA routing — the P2 fork's ctypes surface (Pillar).

The private P2 fork of the pinned llama.cpp adds two C entry points that let a
single ``llama_decode`` carry a *mix* of adapters — different sequences in one
ubatch use different LoRA adapters (the vLLM/Punica capability):

* ``llama_set_seq_adapters(ctx, adapters, n)`` — register the ordered adapter pool
  once; the pool index becomes the routing id.
* ``llama_set_seq_adapter(ctx, seq_id, idx)`` — assign a sequence to a pool slot
  (``-1`` = no adapter / raw base model).

The hand-bound ctypes below came from an earlier internal router prototype
(``enable_seq_routing`` /
``run_seq_routed``) into the library behind a narrow, capability-guarded surface.

Fail-closed contract (repo standing rule): the routing symbols only exist on a
fork lib. On a stock lib the guard raises :class:`SeqRoutingUnavailable` at *bind*
time; callers (the router's ``auto`` backend) treat that as "capability absent" and
fall back to the context-pool path — a missing capability is never an inference
failure. Pydantic stays at the config boundary; this binding is plain ctypes.
"""

from __future__ import annotations

import ctypes
from typing import Any

# The two fork-only symbols. A lib is fork-capable iff it exports both.
SEQ_ROUTING_SYMBOLS: tuple[str, ...] = ("llama_set_seq_adapters", "llama_set_seq_adapter")

# The scaled form: the per-sequence
# scale rides the per-decode routing mask, so two sequences can use one pool slot
# at two scales inside one ``llama_decode``. It is a SEPARATE capability from the
# pair above: an ``integration`` lib carries the pair and not this symbol.
#
# The bindgen cdef is deliberately NOT extended with it — the hats precedent
# (``LOOP_ROUTING_SYMBOLS`` below, step-3 decision): the fork-paired cdef must
# keep compiling against builds that lack the symbol, so it is proven by direct
# resolution instead.
SEQ_ADAPTER_SCALED_SYMBOL: str = "llama_set_seq_adapter_scaled"

# Sentinel adapter index meaning "no adapter / raw base model" (matches the fork's
# seq_adapter_map default and llama_set_seq_adapter's -1 contract).
NO_ADAPTER: int = -1


class SeqRoutingUnavailable(RuntimeError):
    """Raised when the loaded libllama lacks the P2-fork routing symbols.

    Signals *missing capability*, not a fault: the ``auto`` router backend catches
    it and falls back to the context-pool path.
    """


def library_supports_seq_routing(lib: Any) -> bool:
    """True iff ``lib`` (a loaded ctypes CDLL) exports both fork routing symbols."""
    return all(hasattr(lib, name) for name in SEQ_ROUTING_SYMBOLS)


def library_supports_seq_adapter_scale(lib: Any) -> bool:
    """True iff ``lib`` exports the scaled per-sequence routing call."""
    return hasattr(lib, SEQ_ADAPTER_SCALED_SYMBOL)


class SeqRoutingBinding:
    """Typed ctypes binding for the P2 fork's per-sequence LoRA routing calls.

    Construct from the ``llama_cpp.llama_cpp`` module (the high-level package's
    ctypes namespace, exposing ``_lib`` and the shared ``argtypes`` aliases). The
    constructor probes for the fork symbols and installs ``argtypes`` / ``restype``
    exactly as the benchmark reference bound them; if the symbols are absent it
    raises :class:`SeqRoutingUnavailable` and touches nothing.
    """

    def __init__(self, llama_cpp_module: Any) -> None:
        self._C = llama_cpp_module
        lib = getattr(llama_cpp_module, "_lib", None)
        if lib is None:
            raise SeqRoutingUnavailable("llama_cpp module exposes no loaded library (_lib is None)")
        if not library_supports_seq_routing(lib):
            raise SeqRoutingUnavailable(
                "loaded libllama has no "
                f"{'/'.join(SEQ_ROUTING_SYMBOLS)} — need the P2 fork build "
                "(set LLAMA_CPP_LIB_PATH to a p2fork lib dir)"
            )
        self._lib = lib

        # argtypes/restypes copied verbatim from context_pool_router.py:523-535.
        lib.llama_set_seq_adapters.argtypes = [
            self._C.llama_context_p_ctypes,
            ctypes.POINTER(self._C.llama_adapter_lora_p_ctypes),
            ctypes.c_size_t,
        ]
        lib.llama_set_seq_adapters.restype = ctypes.c_int32
        lib.llama_set_seq_adapter.argtypes = [
            self._C.llama_context_p_ctypes,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.llama_set_seq_adapter.restype = ctypes.c_int32

        # The scaled call is optional: bind it only when the lib carries it, so
        # this class keeps working against an integration lib.
        self.supports_scale = library_supports_seq_adapter_scale(lib)
        if self.supports_scale:
            lib.llama_set_seq_adapter_scaled.argtypes = [
                self._C.llama_context_p_ctypes,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_float,
            ]
            lib.llama_set_seq_adapter_scaled.restype = ctypes.c_int32

    def set_seq_adapters(self, ctx: Any, adapter_ptrs: list[Any]) -> None:
        """Register the ordered adapter pool on ``ctx`` (index = routing id)."""
        n = len(adapter_ptrs)
        arr = (self._C.llama_adapter_lora_p_ctypes * n)(*adapter_ptrs)
        if self._lib.llama_set_seq_adapters(ctx, arr, n) != 0:
            raise RuntimeError("llama_set_seq_adapters failed")

    def set_seq_adapter(self, ctx: Any, seq_id: int, adapter_idx: int) -> None:
        """Route sequence ``seq_id`` to pool slot ``adapter_idx`` (``-1`` = base).

        A lib carrying the stage 0.5 node-budget fix returns ``-1`` for a
        ``seq_id`` at or past ``n_seq_max`` and this raises. A lib without the fix
        returns ``0`` for the same rejected call, and the out-of-range sequence
        then reads ``seq_adapter_map`` past its end (S11-P1). The raise is the
        intended new behaviour, not a regression.

        The call also resets the sequence's scale to ``1.0``. Use
        :meth:`set_seq_adapter_scaled` to keep a scale.
        """
        if self._lib.llama_set_seq_adapter(ctx, seq_id, adapter_idx) != 0:
            raise RuntimeError(f"llama_set_seq_adapter(seq={seq_id}, idx={adapter_idx}) failed")

    def set_seq_adapter_scaled(self, ctx: Any, seq_id: int, adapter_idx: int, scale: float) -> None:
        """Route ``seq_id`` to pool slot ``adapter_idx`` at ``scale``.

        The scale rides the per-decode mask, so two sequences may share one pool
        slot at two scales inside one ``llama_decode``, and changing a scale
        forces no graph re-reserve (S9: 1.011x against 2.175x for the pool-side
        form). ``scale`` 1.0 is bit-identical to :meth:`set_seq_adapter`.

        Raises :class:`SeqRoutingUnavailable` when the lib lacks the symbol.
        """
        if not self.supports_scale:
            raise SeqRoutingUnavailable(
                f"loaded libllama has no {SEQ_ADAPTER_SCALED_SYMBOL} - "
                "need a build carrying the scaled per-sequence routing surface"
            )
        if self._lib.llama_set_seq_adapter_scaled(ctx, seq_id, adapter_idx, scale) != 0:
            raise RuntimeError(f"llama_set_seq_adapter_scaled(seq={seq_id}, idx={adapter_idx}, scale={scale}) failed")


# --------------------------------------------------------------------------- #
# Per-loop-step LoRA routing ("hats")
# --------------------------------------------------------------------------- #
# The spike branch ``feat/p2-nanbeige-hats`` (a branch name, not a model
# requirement — this works with any model) extends the P2 mechanism with two
# additive ``extern "C"`` decls (same class of change as the P2 pair; ABI
# additive-only):
#
# * ``llama_set_loop_adapters(ctx, adapters, n)`` — register the ordered adapter
#   pool once; the pool index becomes the per-loop-step routing id (the fork's
#   ``loop_hat_map`` is sized ``loop_count()``, ``-1`` = base).
# * ``llama_set_loop_adapter(ctx, loop_step, idx)`` — assign a loop step to a
#   pool slot (``-1`` = no adapter / raw base model), mirroring the P2 pair.
#
# The bindgen cdef is deliberately NOT extended with these symbols: the
# fork-paired cdef must keep passing against ``integration``
# builds that lack hats, so the hats surface is proven by direct symbol
# resolution, exactly like the spike's proof. The surface below mirrors the P2
# pair's capability-guarded contract: absent on the integration lib at *bind*
# time (never at import), fail-closed.

# The two hats-only symbols. A lib is hats-capable iff it exports both.
LOOP_ROUTING_SYMBOLS: tuple[str, ...] = ("llama_set_loop_adapters", "llama_set_loop_adapter")

# Sentinel meaning "no adapter / raw base model" for a loop step — same contract
# as the fork's ``loop_hat_map`` default and ``llama_set_loop_adapter``'s -1.
NO_LOOP_ADAPTER: int = NO_ADAPTER


class LoopRoutingUnavailable(RuntimeError):
    """Raised when the loaded libllama lacks the hats loop-routing symbols.

    Signals *missing capability*, not a fault: the hats surface is opt-in, and a
    caller on an integration/stock lib treats this as "capability absent" (the
    H1 hats graph path simply cannot be driven) — a missing capability is never
    an inference failure.
    """


def library_supports_loop_routing(lib: Any) -> bool:
    """True iff ``lib`` (a loaded ctypes CDLL) exports both hats routing symbols."""
    return all(hasattr(lib, name) for name in LOOP_ROUTING_SYMBOLS)


class LoopRoutingBinding:
    """Typed ctypes binding for the hats fork's per-loop-step LoRA routing calls.

    Construct from the ``llama_cpp.llama_cpp`` module, exactly like
    :class:`SeqRoutingBinding`. The constructor probes for the hats symbols and
    installs ``argtypes`` / ``restype`` mirroring the P2 pair (the H1 prototype
    reuses the ``seq_loras`` pool, so the C signatures are identical); if the
    symbols are absent it raises :class:`LoopRoutingUnavailable` and touches
    nothing.
    """

    def __init__(self, llama_cpp_module: Any) -> None:
        self._C = llama_cpp_module
        lib = getattr(llama_cpp_module, "_lib", None)
        if lib is None:
            raise LoopRoutingUnavailable("llama_cpp module exposes no loaded library (_lib is None)")
        if not library_supports_loop_routing(lib):
            raise LoopRoutingUnavailable(
                "loaded libllama has no "
                f"{'/'.join(LOOP_ROUTING_SYMBOLS)} — need the hats fork build "
                "(feat/p2-nanbeige-hats; set LLAMA_CPP_LIB_PATH to a hats lib dir)"
            )
        self._lib = lib

        # argtypes/restypes mirror the P2 pair (identical C signatures).
        lib.llama_set_loop_adapters.argtypes = [
            self._C.llama_context_p_ctypes,
            ctypes.POINTER(self._C.llama_adapter_lora_p_ctypes),
            ctypes.c_size_t,
        ]
        lib.llama_set_loop_adapters.restype = ctypes.c_int32
        lib.llama_set_loop_adapter.argtypes = [
            self._C.llama_context_p_ctypes,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.llama_set_loop_adapter.restype = ctypes.c_int32

    def set_loop_adapters(self, ctx: Any, adapter_ptrs: list[Any]) -> None:
        """Register the ordered adapter pool on ``ctx`` (index = loop-step id)."""
        n = len(adapter_ptrs)
        arr = (self._C.llama_adapter_lora_p_ctypes * n)(*adapter_ptrs)
        if self._lib.llama_set_loop_adapters(ctx, arr, n) != 0:
            raise RuntimeError("llama_set_loop_adapters failed")

    def set_loop_adapter(self, ctx: Any, loop_step: int, adapter_idx: int) -> None:
        """Route loop step ``loop_step`` to pool slot ``adapter_idx`` (``-1`` = base)."""
        if self._lib.llama_set_loop_adapter(ctx, loop_step, adapter_idx) != 0:
            raise RuntimeError(f"llama_set_loop_adapter(loop_step={loop_step}, idx={adapter_idx}) failed")
