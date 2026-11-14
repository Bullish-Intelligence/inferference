"""Seq-state restore validation semantics (run_seq_reuse gates).

The validation half of the seq-reuse pair; the perf half is
``src/inferference/benchkit/runners/seq_reuse.py``.
The state caveat is load-bearing: **byte-count success is not
semantic success** — ``llama_state_seq_set_data`` returns the number of bytes
read, which says nothing about whether the restored KV is valid for THIS
context. This gate asserts:

  * restore into a *nonzero* sequence slot of a *live multi-sequence* context
    without disturbing its neighbours (seq 0 live, blob restored into seq 1,
    both continuations match their isolated baselines);
  * the ``n_seq_max`` portability reject — a blob captured under one
    ``n_seq_max`` loads into a context with a different ``n_seq_max`` only via
    ``set_data`` returning 0 (the detectable, fail-closed reject; the byte
    count of the SUCCESS case is recorded, never trusted as semantic proof).

GPU-only: skips (never fails) when the env lacks GPU-offload lib / model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.asset_paths import asset_path

from inferference.gpu_env import gpu_env_ready
from inferference.prefix_cache_live import LlamaSeqStateBridge

_REPO = Path(__file__).resolve().parents[3]
_MODEL = asset_path("MODEL_DEFAULT")


def _ready() -> bool:
    # Backend-agnostic: a vendor device variable means nothing to a build for another backend,
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
    reason="requires GPU-offload lib + MODEL_DEFAULT (GPU env)",
)


def test_restore_into_live_slot_without_disturbing_neighbours() -> None:
    from llama_cpp import Llama, llama_cpp  # noqa: PLC0415

    llm = Llama(model_path=str(_MODEL), n_ctx=2048, n_batch=128, n_gpu_layers=-1, seed=17018, verbose=False)
    bridge = LlamaSeqStateBridge(n_batch=128, n_vocab=llm._n_vocab)
    model = llm._model.model
    try:
        toks = list(llm.tokenize(b"Multi sequence router reuse workload. " * 60, add_bos=False, special=True))
        pa, sa = toks[:24], [toks[24]]  # sequence A: the live neighbour
        pb, sb = toks[40:72], [toks[72]]  # sequence B: restored

        def isolated(prefix: list[int], suffix: list[int]) -> int:
            c = llama_cpp.llama_context_default_params()
            c.n_ctx = 2048
            c.n_batch = 128
            c.n_ubatch = 128
            c.n_seq_max = 1
            ctx = llama_cpp.llama_new_context_with_model(model, c)
            bridge.decode_tokens(ctx, prefix, 0, 0)
            bridge.decode_tokens(ctx, suffix, 0, len(prefix))
            llama_cpp.llama_synchronize(ctx)
            tok = bridge.last_token(ctx)
            llama_cpp.llama_free(ctx)
            return tok

        base_a = isolated(pa, sa)
        base_b = isolated(pb, sb)

        # capture B under matched n_seq_max=2 from seq 1
        p = llama_cpp.llama_context_default_params()
        p.n_ctx = 2048
        p.n_batch = 128
        p.n_ubatch = 128
        p.n_seq_max = 2
        src = llama_cpp.llama_new_context_with_model(model, p)
        bridge.decode_tokens(src, pb, 1, 0)
        llama_cpp.llama_synchronize(src)
        blob = bridge.capture_seq_state(src, 1)
        llama_cpp.llama_free(src)

        # live multi-seq context: A already decoded on seq 0; restore B into seq 1
        tgt = llama_cpp.llama_new_context_with_model(model, p)
        try:
            bridge.decode_tokens(tgt, pa, 0, 0)
            llama_cpp.llama_synchronize(tgt)
            code = bridge.restore_seq_state(tgt, blob, 1)
            assert code > 0, f"set_data rejected matched blob: code={code}"
            blob_bytes_recorded = code  # byte count — never trusted as semantic proof
            bridge.decode_tokens(tgt, sb, 1, len(pb))
            llama_cpp.llama_synchronize(tgt)
            restored_b = bridge.last_token(tgt)
            assert restored_b == base_b, f"restored B diverged: {restored_b} != {base_b}"
            # neighbour A still intact
            bridge.decode_tokens(tgt, sa, 0, len(pa))
            llama_cpp.llama_synchronize(tgt)
            neighbour_a = bridge.last_token(tgt)
            assert neighbour_a == base_a, f"neighbour A disturbed: {neighbour_a} != {base_a}"
            assert blob_bytes_recorded == len(blob)  # recorded, informational
        finally:
            llama_cpp.llama_free(tgt)

        # n_seq_max portability reject: capture under nsm=4, restore into nsm=2
        q = llama_cpp.llama_context_default_params()
        q.n_ctx = 2048
        q.n_batch = 128
        q.n_ubatch = 128
        q.n_seq_max = 4
        src4 = llama_cpp.llama_new_context_with_model(model, q)
        bridge.decode_tokens(src4, pb, 1, 0)
        llama_cpp.llama_synchronize(src4)
        blob4 = bridge.capture_seq_state(src4, 1)
        llama_cpp.llama_free(src4)
        tgt2 = llama_cpp.llama_new_context_with_model(model, p)
        try:
            reject_code = bridge.restore_seq_state(tgt2, blob4, 1)
            assert reject_code == 0, "n_seq_max portability broken: cross-nsm load must be a detectable reject"
        finally:
            llama_cpp.llama_free(tgt2)
    finally:
        llm.close()
