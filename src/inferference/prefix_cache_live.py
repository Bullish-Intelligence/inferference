"""Live bridge: wire the prefix-cache policy layer to llama.cpp per-seq KV state.

``prefix_cache.py`` is the pure policy/persistence layer (identity keys,
compatibility, the restore *plan*).  This module connects that policy to the
proven live mechanism from the project-17 research runners
(an earlier seq-reuse harness / ``context_pool_router.py``):

  * capture a shared prefix's per-sequence KV blob with
    ``llama_state_seq_get_data`` after decoding it once, and publish a
    :class:`~prefix_cache.PrefixCacheEntry`;
  * restore that blob with ``llama_state_seq_set_data`` (fail-closed on its
    ``0`` return), then execute the :class:`~prefix_cache.RestorePlan` by
    decoding the fresh suffix tokens in an own-batch with explicit positions so
    the continuation logits are valid.

Two correctness-critical rules from the GPU spikes are enforced here:

  * a captured blob only loads into a context with the **same** ``n_seq_max``
    (``set_data`` returns 0 otherwise) — checked explicitly *and* fail-closed;
  * saved state excludes the logits buffer, so a restore is only valid when
    followed by at least one suffix decode (the :class:`RestorePlan` invariant),
    and that suffix must be decoded in an own-batch at explicit positions
    starting right after the restored prefix, or a high-level ``Llama.eval``
    would ``kv_cache_seq_rm`` and wipe the restored cells.

``llama_cpp`` is imported lazily (like ``decode.py``) so importing this module
never pulls in the native library; the pure policy paths and fakes need no GPU.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from time import time_ns
from typing import Any, Protocol

from .fingerprint import LlamaEngineFingerprint
from .prefix_cache import (
    CacheLookup,
    CacheRejectionReason,
    PrefixCacheEntry,
    PrefixCacheKey,
    check_compatibility,
    check_state_integrity,
    plan_restore,
    restore_then_decode_suffix,
)

# Runtime fact recording the context ``n_seq_max`` a blob was captured under.
# A blob is only portable into a context with an identical value (proven rule).
N_SEQ_MAX_FACT = "n_seq_max"

# Recorded on entries so a reader can tell which mechanism produced the blob.
LLAMA_STATE_VERSION_SEQ = "llama_state_seq_v1"


def _llama_cpp() -> Any:
    """Import lazily so core-only installs never import the native library."""
    import llama_cpp

    return llama_cpp


class SeqRestoreRejected(RuntimeError):
    """Raised when ``llama_state_seq_set_data`` returns 0 (fail-closed reject)."""

    def __init__(self, seq_id: int) -> None:
        super().__init__(
            f"llama_state_seq_set_data returned 0 for seq {seq_id} (n_seq_max mismatch or incompatible blob)"
        )
        self.seq_id = seq_id


class PrefixStateCache(Protocol):
    """The publish/lookup surface both concrete caches expose to the bridge."""

    def publish(
        self,
        key: PrefixCacheKey,
        state: bytes,
        *,
        llama_state_version: str | None = ...,
        runtime_facts: dict[str, str] | None = ...,
    ) -> PrefixCacheEntry: ...

    def lookup(self, key: PrefixCacheKey) -> CacheLookup: ...


class InMemoryPrefixCache:
    """A dependency-free cache with the same contract as ``PersistentPrefixCache``.

    Reuses the exact compatibility and integrity checks from ``prefix_cache`` so
    a unit test exercises the same rejection logic as the filesystem store,
    without touching disk or SQLite.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[PrefixCacheEntry, bytes]] = {}

    def publish(
        self,
        key: PrefixCacheKey,
        state: bytes,
        *,
        llama_state_version: str | None = None,
        runtime_facts: dict[str, str] | None = None,
    ) -> PrefixCacheEntry:
        entry = PrefixCacheEntry(
            key,
            len(state),
            sha256(state).hexdigest(),
            f"mem://{key.storage_key}",
            time_ns(),
            time_ns(),
            llama_state_version,
            tuple(sorted((runtime_facts or {}).items())),
        )
        self._entries[key.storage_key] = (entry, bytes(state))
        return entry

    def lookup(self, key: PrefixCacheKey) -> CacheLookup:
        found = self._entries.get(key.storage_key)
        if found is None:
            return CacheLookup(None, None, False, "miss_not_found")
        entry, state = found
        compatible = check_compatibility(entry, key)
        if not compatible.accepted:
            return CacheLookup(None, None, False, str(compatible.reason))
        integrity = check_state_integrity(entry, state)
        if not integrity.accepted:
            return CacheLookup(entry, None, False, str(integrity.reason))
        return CacheLookup(entry, state, True, "hit")


@dataclass(frozen=True, slots=True)
class LiveRestoreResult:
    """Outcome of a live restore-then-decode attempt; failure is always data."""

    restored: bool
    reason: str
    next_token: int | None = None
    suffix_token_count: int = 0
    set_data_return: int | None = None

    @property
    def rejected(self) -> bool:
        return not self.restored


class LlamaSeqStateBridge:
    """Low-level ctypes wrapper for per-seq capture/restore and own-batch decode.

    Mirrors the proven primitives in the research runners (``_get_seq`` /
    ``_set_seq`` / own-batch ``decode``).  It operates on a raw ``llama_context``
    pointer so it composes with the router's context pool; it does not own the
    context lifecycle.
    """

    def __init__(self, *, n_batch: int, n_vocab: int, native: Any | None = None) -> None:
        if n_batch <= 0:
            raise ValueError("n_batch must be positive")
        if n_vocab <= 0:
            raise ValueError("n_vocab must be positive")
        self._native = native if native is not None else _llama_cpp()
        self.n_batch = n_batch
        self.n_vocab = n_vocab

    def capture_seq_state(self, ctx: Any, seq_id: int) -> bytes:
        """Copy sequence ``seq_id``'s KV state out with ``llama_state_seq_get_data``."""
        size = int(self._native.llama_state_seq_get_size(ctx, seq_id))
        buffer = (ctypes.c_uint8 * size)()
        copied = int(self._native.llama_state_seq_get_data(ctx, buffer, size, seq_id))
        return bytes(buffer[:copied])

    def restore_seq_state(self, ctx: Any, blob: bytes, seq_id: int) -> int:
        """Load a blob into ``seq_id``; return value 0 means the load failed."""
        array = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        return int(self._native.llama_state_seq_set_data(ctx, array, len(blob), seq_id))

    def restore_blob_into_seq(self, ctx: Any, blob: bytes, seq_id: int) -> int:
        """Restore a captured blob into ``seq_id``, fail-closed on rejection.

        Centralizes the ``set_data == 0`` reject rule for callers that restore
        the SAME prefix blob into MULTIPLE seq slots of one context (the
        context-pool router's batched multi-seq wave).  A ``0`` return means the
        blob is incompatible with the target context — most often a mismatched
        ``n_seq_max`` — so this raises :class:`SeqRestoreRejected` rather than
        letting an unrestored slot decode against stale KV.  Returns the number
        of bytes read on success.
        """
        code = self.restore_seq_state(ctx, blob, seq_id)
        if code == 0:
            raise SeqRestoreRejected(seq_id)
        return code

    def decode_tokens(self, ctx: Any, tokens: Sequence[int], seq_id: int, start_pos: int) -> None:
        """Own-batch decode with explicit positions; logits only on the last token.

        Explicit positions (``start_pos + i``) are what let a restored prefix's
        suffix decode land on the cells right after the restored KV without a
        high-level ``eval`` wiping them.
        """
        native = self._native
        count = len(tokens)
        if count == 0:
            raise ValueError("decode_tokens requires at least one token")
        for offset in range(0, count, self.n_batch):
            chunk = tokens[offset : offset + self.n_batch]
            width = len(chunk)
            batch = native.llama_batch_init(width, 0, 1)
            batch.n_tokens = width
            for index, token in enumerate(chunk):
                batch.token[index] = token
                batch.pos[index] = start_pos + offset + index
                batch.n_seq_id[index] = 1
                batch.seq_id[index][0] = seq_id
                batch.logits[index] = 0
            if offset + width == count:
                batch.logits[width - 1] = 1
            result = native.llama_decode(ctx, batch)
            native.llama_batch_free(batch)
            if result != 0:
                raise RuntimeError(f"llama_decode failed with code {result}")

    def last_token(self, ctx: Any) -> int:
        """Greedy argmax over the most recently produced logits row."""
        import numpy as np

        pointer = self._native.llama_get_logits_ith(ctx, -1)
        logits = np.ctypeslib.as_array(pointer, shape=(self.n_vocab,))
        return int(np.argmax(logits))


def capture_prefix(
    bridge: LlamaSeqStateBridge,
    ctx: Any,
    *,
    cache: PrefixStateCache,
    namespace: str,
    fingerprint: LlamaEngineFingerprint,
    prefix_token_ids: Sequence[int],
    n_seq_max: int,
    seq_id: int = 0,
) -> PrefixCacheEntry:
    """Decode a shared prefix once, capture its seq KV blob, and publish an entry.

    The capture ``n_seq_max`` is recorded as a runtime fact so a later restore
    can reject a portability mismatch *before* asking llama.cpp (which would also
    fail-closed).  Keyed by the exact prefix token IDs via the frozen engine
    fingerprint, so a different engine or token sequence can never collide.
    """
    prefix = tuple(prefix_token_ids)
    if not prefix:
        raise ValueError("prefix_token_ids must contain at least one token")
    bridge.decode_tokens(ctx, prefix, seq_id, 0)
    blob = bridge.capture_seq_state(ctx, seq_id)
    key = PrefixCacheKey.from_fingerprint(namespace=namespace, fingerprint=fingerprint, prefix_token_ids=prefix)
    return cache.publish(
        key,
        blob,
        llama_state_version=LLAMA_STATE_VERSION_SEQ,
        runtime_facts={N_SEQ_MAX_FACT: str(n_seq_max)},
    )


def restore_and_continue(
    bridge: LlamaSeqStateBridge,
    ctx: Any,
    *,
    cache: PrefixStateCache,
    key: PrefixCacheKey,
    request_token_ids: Sequence[int],
    n_seq_max: int,
    seq_id: int = 0,
) -> LiveRestoreResult:
    """Look up ``key``, restore its blob into ``seq_id``, then decode the suffix.

    All failure modes are returned as data (never raised) so a caller can fall
    back to a cold prefill.  The restore lifecycle is delegated to
    ``restore_then_decode_suffix`` so this bridge cannot skip the mandatory
    suffix decode.
    """
    lookup = cache.lookup(key)
    if not lookup.hit or lookup.entry is None or lookup.state is None:
        return LiveRestoreResult(False, lookup.reason)
    entry = lookup.entry

    facts = dict(entry.runtime_facts)
    captured_n_seq_max = facts.get(N_SEQ_MAX_FACT)
    if captured_n_seq_max != str(n_seq_max):
        return LiveRestoreResult(
            False,
            str(CacheRejectionReason.N_SEQ_MAX_MISMATCH),
        )

    decision = plan_restore(entry, key, tuple(request_token_ids))
    if decision.plan is None:
        assert decision.rejection is not None
        return LiveRestoreResult(False, str(decision.rejection.reason))
    plan = decision.plan

    captured: dict[str, int] = {}

    def load_state(state: bytes) -> None:
        code = bridge.restore_seq_state(ctx, state, seq_id)
        captured["set_data_return"] = code
        if code == 0:
            raise SeqRestoreRejected(seq_id)

    def decode_suffix(suffix: tuple[int, ...]) -> None:
        bridge.decode_tokens(ctx, suffix, seq_id, len(plan.cached_prefix_token_ids))

    try:
        restore_then_decode_suffix(plan, lookup.state, load_state=load_state, decode_suffix=decode_suffix)
    except SeqRestoreRejected:
        return LiveRestoreResult(
            False,
            str(CacheRejectionReason.STATE_SET_DATA_REJECTED),
            set_data_return=captured.get("set_data_return", 0),
        )

    return LiveRestoreResult(
        True,
        "hit",
        next_token=bridge.last_token(ctx),
        suffix_token_count=len(plan.uncached_suffix_token_ids),
        set_data_return=captured.get("set_data_return"),
    )
