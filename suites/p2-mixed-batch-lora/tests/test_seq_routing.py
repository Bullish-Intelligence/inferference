"""CPU-only contracts for the P2-fork mixed-batch multi-LoRA runtime.

No native library or GPU: these cover the capability-aware fingerprint key, the
``SeqRoutingBinding`` capability guard, the fail-closed ``auto`` backend selection,
and the boundary surface of the seq-routing config. The token-exact GPU gate lives
in ``test_seq_routing_gpu.py`` behind the GPU env.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import pytest

from inferference.diagnostics import collect_runtime_diagnostics
from inferference.fingerprint import ArtifactIdentity, LlamaEngineFingerprint
from inferference.gpu_env import GPU_BACKEND_LABEL
from inferference.seq_routing import (
    NO_ADAPTER,
    SeqRoutingBinding,
    SeqRoutingUnavailable,
    library_supports_seq_routing,
)


def _fingerprint(*, seq_adapter_routing: bool = False) -> LlamaEngineFingerprint:
    artifact = ArtifactIdentity(path="/models/model.gguf", sha256="a" * 64, size_bytes=1, mtime_ns=1, inode=1)
    return LlamaEngineFingerprint(
        model=artifact,
        tokenizer=artifact,
        llama_cpp_python_version="0.3.34",
        llama_cpp_commit="c588c4f47",
        backend=GPU_BACKEND_LABEL,
        n_ctx=2048,
        seq_adapter_routing=seq_adapter_routing,
    )


# ---- Workstream B: capability-aware fingerprint ----


def test_fork_and_stock_fingerprints_have_distinct_cache_keys() -> None:
    """B4 — a fork-built engine and a stock engine must never share cache state."""
    stock = _fingerprint(seq_adapter_routing=False)
    fork = _fingerprint(seq_adapter_routing=True)
    assert stock.cache_key() != fork.cache_key()
    # Otherwise identical inputs -> the routing capability is the only difference.
    assert stock.model_copy(update={"seq_adapter_routing": True}).cache_key() == fork.cache_key()


def test_fingerprint_defaults_to_stock_capability() -> None:
    assert _fingerprint().seq_adapter_routing is False


def test_diagnostics_surface_routing_capability_from_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B3 — the routing capability is visible in runtime diagnostics."""
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    (tmp_path / "build-manifest.json").write_text(
        json.dumps({"llama_cpp_commit": "c588c4f47", "profile": "p2fork", "seq_adapter_routing": True})
    )
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", str(library_dir))
    diagnostics = collect_runtime_diagnostics()
    assert diagnostics.seq_adapter_routing is True


def test_diagnostics_routing_capability_absent_when_manifest_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    (tmp_path / "build-manifest.json").write_text(json.dumps({"profile": "gpu-profile"}))
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", str(library_dir))
    assert collect_runtime_diagnostics().seq_adapter_routing is None


# ---- Workstream C: SeqRoutingBinding capability guard ----


class _FakeLibStock:
    """A stock libllama: no fork routing symbols."""


class _FakeSymbol:
    """A ctypes-function stand-in: callable with settable argtypes/restype."""

    def __init__(self, name: str, calls: list[tuple]) -> None:
        self._name = name
        self._calls = calls
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):  # noqa: ANN002
        self._calls.append((self._name, *args))
        return 0


class _FakeLibFork:
    """A fork libllama surface: both symbols present, capturing their calls."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.llama_set_seq_adapters = _FakeSymbol("set_seq_adapters", self.calls)
        self.llama_set_seq_adapter = _FakeSymbol("set_seq_adapter", self.calls)


class _FakeC:
    """Minimal stand-in for the ``llama_cpp.llama_cpp`` ctypes namespace."""

    llama_context_p_ctypes = ctypes.c_void_p
    llama_adapter_lora_p_ctypes = ctypes.c_void_p

    def __init__(self, lib: object) -> None:
        self._lib = lib


def test_library_capability_probe() -> None:
    assert library_supports_seq_routing(_FakeLibFork()) is True
    assert library_supports_seq_routing(_FakeLibStock()) is False


def test_binding_raises_documented_error_on_stock_lib() -> None:
    """C3 — capability-absent path raises the documented error (no GPU)."""
    with pytest.raises(SeqRoutingUnavailable, match="P2 fork"):
        SeqRoutingBinding(_FakeC(_FakeLibStock()))


def test_binding_binds_and_routes_on_fork_lib() -> None:
    lib = _FakeLibFork()
    binding = SeqRoutingBinding(_FakeC(lib))
    ctx = ctypes.c_void_p(0x1234)
    binding.set_seq_adapters(ctx, [ctypes.c_void_p(1), ctypes.c_void_p(2)])
    binding.set_seq_adapter(ctx, 0, 1)
    binding.set_seq_adapter(ctx, 1, NO_ADAPTER)
    kinds = [c[0] for c in lib.calls]
    assert kinds == ["set_seq_adapters", "set_seq_adapter", "set_seq_adapter"]
    assert lib.calls[1][3] == 1
    assert lib.calls[2][3] == NO_ADAPTER  # base sentinel routed as -1


# ---- Workstream D: fail-closed backend selection ----


def _resolver(capable: bool):
    """Build a bare MultiLoRARouter shell to exercise _resolve_backend in isolation."""
    from inferference.router import MultiLoRARouter

    shell = MultiLoRARouter.__new__(MultiLoRARouter)
    shell.C = _FakeC(_FakeLibFork() if capable else _FakeLibStock())
    return shell


def test_auto_selects_seq_routed_only_when_capable() -> None:
    """D2 — auto picks seq_routed on a fork lib, context_pool on a stock lib."""
    assert _resolver(capable=True)._resolve_backend("auto") == "seq_routed"
    assert _resolver(capable=False)._resolve_backend("auto") == "context_pool"


def test_auto_fallback_surfaces_no_error() -> None:
    """E3/D — capability-absent fallback selects context_pool, never raises."""
    assert _resolver(capable=False)._resolve_backend("auto") == "context_pool"


def test_explicit_seq_routed_fails_closed_without_fork() -> None:
    with pytest.raises(SeqRoutingUnavailable):
        _resolver(capable=False)._resolve_backend("seq_routed")


def test_explicit_context_pool_ignores_capability() -> None:
    assert _resolver(capable=True)._resolve_backend("context_pool") == "context_pool"
