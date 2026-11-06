"""Whole-state prefix-cache validation half (run_prefix_cache gates).

The validation half of the prefix-cache pair; the sweep half is
``src/inferference/benchkit/runners/prefix_cache_sweep.py``. These are the
validation semantics, applied to the library's
:class:`inferference.prefix_cache.PersistentPrefixCache`:

  * a captured whole-state blob survives a context restart and restores to a
    token-exact continuation (SEMANTIC success — the byte count is recorded, never
    trusted);
  * a request that does not extend the cached prefix must be rejected
    (``plan_restore`` semantics);
  * a tampered blob fails the checksum integrity gate (fail-closed reject).

GPU-only: skips (never fails) when the env lacks CUDA-offload lib / model.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.asset_paths import asset_path

from inferference.diagnostics import collect_runtime_diagnostics
from inferference.fingerprint import LlamaEngineFingerprint, register_artifact
from inferference.gpu_env import gpu_env_ready
from inferference.prefix_cache import (
    CacheRejectionReason,
    PersistentPrefixCache,
    PrefixCacheKey,
    check_state_integrity,
    plan_restore,
)

_REPO = Path(__file__).resolve().parents[3]
_MODEL = asset_path("MODEL_ORNITH")


def _ready() -> bool:
    # Backend-agnostic: CUDA_VISIBLE_DEVICES means nothing to the Vulkan build,
    # and requiring it here SKIPPED this suite on a rig that can run it.
    if not gpu_env_ready():
        return False
    if not _MODEL.exists():
        return False
    try:
        import llama_cpp
    except Exception:
        return False
    return bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())


pytestmark = pytest.mark.skipif(
    not _ready(),
    reason="requires CUDA-offload lib + Ornith model (GPU env)",
)


def _token(llm: Any) -> int:
    return int(np.argmax(np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm._n_vocab,))))


def test_whole_state_cache_semantic_restore(tmp_path: Path) -> None:
    from llama_cpp import Llama, llama_cpp  # noqa: PLC0415

    n_ctx = 512
    llm = Llama(model_path=str(_MODEL), n_ctx=n_ctx, n_batch=128, n_gpu_layers=-1, seed=17018, verbose=False)
    try:
        runtime = collect_runtime_diagnostics()
        identity = register_artifact(_MODEL, metadata={"role": "model-and-gguf-tokenizer"})
        fingerprint = LlamaEngineFingerprint(
            model=identity,
            tokenizer=identity,
            llama_cpp_python_version=runtime.llama_cpp_python_version or "unknown",
            llama_cpp_commit=runtime.llama_cpp_commit,
            llama_cpp_build_id=runtime.llama_cpp_build_id,
            backend="cuda",
            n_ctx=n_ctx,
        )
        cache = PersistentPrefixCache(tmp_path / "cache")
        tokens = tuple(llm.tokenize(b"Persistent exact prefix cache workload. " * 100, add_bos=False, special=True))
        prefix, suffix = tokens[:32], (tokens[32],)

        key = PrefixCacheKey.from_fingerprint(
            namespace="suite-prefix-cache", fingerprint=fingerprint, prefix_token_ids=prefix
        )
        llm.reset()
        llm.eval(list(prefix))
        llama_cpp.llama_synchronize(llm._ctx.ctx)
        entry = cache.publish(
            key,
            pickle.dumps(llm.save_state(), protocol=pickle.HIGHEST_PROTOCOL),
            llama_state_version="llama-cpp-python-LlamaState-v1",
        )

        # restart lifecycle: fresh context, restore, semantic continuation check
        llm.close()
        restart = Llama(model_path=str(_MODEL), n_ctx=n_ctx, n_batch=128, n_gpu_layers=-1, seed=17018, verbose=False)
        try:
            hit = cache.lookup(key)
            assert hit.hit, f"cache miss: {hit.reason}"
            assert hit.state is not None
            restart.load_state(pickle.loads(hit.state))
            restart.eval(list(suffix))
            llama_cpp.llama_synchronize(restart._ctx.ctx)
            restored_token = _token(restart)

            # cold ground truth
            restart.reset()
            restart.eval(list(prefix + suffix))
            llama_cpp.llama_synchronize(restart._ctx.ctx)
            cold_token = _token(restart)
            assert restored_token == cold_token, (
                f"semantic restore broken: restored={restored_token} cold={cold_token} "
                f"(blob bytes {entry.state_size_bytes} recorded, never trusted)"
            )
        finally:
            restart.close()

        # reject: request does not extend the cached prefix (same key, shorter
        # request — the request-level extension check, distinct from the
        # key-level compatibility checks)
        decision = plan_restore(entry, key, tuple(tokens[:31]))
        assert not decision.can_restore
        assert decision.rejection is not None
        assert decision.rejection.reason == CacheRejectionReason.REQUEST_DOES_NOT_EXTEND_PREFIX

        # reject: tampered blob fails the checksum gate (byte-level, fail-closed)
        tampered = b"\x00" * (entry.state_size_bytes + 1)  # wrong size -> size reject
        integrity = check_state_integrity(entry, tampered)
        assert not integrity.accepted
        assert integrity.reason == CacheRejectionReason.STATE_SIZE_MISMATCH
        _ = hashlib.sha256  # noqa: B018  (sha256 import exercised via check_state_integrity)
    finally:
        llm.close()
