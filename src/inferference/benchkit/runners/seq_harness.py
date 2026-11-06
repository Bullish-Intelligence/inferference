"""Shared low-level seq-state harness for the state-restore bench runners.

The reference runners each carried a private ``Harness`` (same shape in
``run_seq_batch_breakeven.py`` and ``run_seq_reuse.py``). The port factors it
once: model load, custom-``n_seq_max`` context creation, own-batch decode with
explicit positions, per-seq capture/restore via
:class:`inferference.prefix_cache_live.LlamaSeqStateBridge`, a batched
multi-seq generation step, and ``llama_synchronize`` sync. GPU-only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from inferference.prefix_cache_live import LlamaSeqStateBridge


class SeqHarness:
    """Model + raw-ctx factory; seq-state primitives ride on LlamaSeqStateBridge."""

    def __init__(self, model_path: str, n_ctx: int, n_batch: int, seed: int) -> None:
        from llama_cpp import Llama, llama_cpp  # noqa: PLC0415

        self.llama_cpp = llama_cpp
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_batch=n_batch, n_gpu_layers=-1, seed=seed, verbose=False)
        self.model = self.llm._model.model
        self.n_vocab = self.llm._n_vocab
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.bridge = LlamaSeqStateBridge(n_batch=n_batch, n_vocab=self.n_vocab, native=self.llama_cpp)

    def tokenize(self, text: bytes) -> list[int]:
        return list(self.llm.tokenize(text, add_bos=False, special=True))

    def make_ctx(self, n_seq_max: int) -> Any:
        p = self.llama_cpp.llama_context_default_params()
        p.n_ctx = self.n_ctx
        p.n_batch = self.n_batch
        p.n_ubatch = self.n_batch
        p.n_seq_max = n_seq_max
        return self.llama_cpp.llama_new_context_with_model(self.model, p)

    def free(self, ctx: Any) -> None:
        self.llama_cpp.llama_free(ctx)

    def sync(self, ctx: Any) -> None:
        self.llama_cpp.llama_synchronize(ctx)

    def seq_rm(self, ctx: Any, seq_id: int) -> None:
        mem = self.llama_cpp.llama_get_memory(ctx)
        self.llama_cpp.llama_memory_seq_rm(mem, seq_id, -1, -1)

    def decode(self, ctx: Any, tokens: list[int], seq_id: int, start_pos: int) -> None:
        self.bridge.decode_tokens(ctx, tokens, seq_id, start_pos)

    def last_token(self, ctx: Any) -> int:
        return self.bridge.last_token(ctx)

    def get_seq(self, ctx: Any, seq_id: int) -> bytes:
        return self.bridge.capture_seq_state(ctx, seq_id)

    def set_seq(self, ctx: Any, blob: bytes, seq_id: int) -> int:
        return self.bridge.restore_seq_state(ctx, blob, seq_id)

    def baseline(self, n_seq_max: int, prefix: list[int], suffix: list[int]) -> int:
        ctx = self.make_ctx(n_seq_max)
        try:
            self.decode(ctx, prefix, 0, 0)
            self.decode(ctx, suffix, 0, len(prefix))
            self.sync(ctx)
            return self.last_token(ctx)
        finally:
            self.free(ctx)

    def batched_step(self, ctx: Any, cur: list[int], pos: list[int]) -> list[int]:
        """Decode one token for each of len(cur) sequences in a single batch."""
        lcpp = self.llama_cpp
        s = len(cur)
        batch = lcpp.llama_batch_init(s, 0, 1)
        batch.n_tokens = s
        for i in range(s):
            batch.token[i] = cur[i]
            batch.pos[i] = pos[i]
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = i
            batch.logits[i] = 1
        rc = lcpp.llama_decode(ctx, batch)
        if rc != 0:
            lcpp.llama_batch_free(batch)
            raise RuntimeError(f"batched llama_decode rc={rc}")
        nxt = []
        for i in range(s):
            ptr = lcpp.llama_get_logits_ith(ctx, i)
            nxt.append(int(np.argmax(np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)))))
        lcpp.llama_batch_free(batch)
        return nxt
