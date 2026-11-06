from __future__ import annotations

import hashlib

from inferference.fingerprint import ArtifactIdentity, LlamaEngineFingerprint
from inferference.prefix_cache import (
    CacheRejectionReason,
    PrefixCacheEntry,
    PrefixCacheKey,
    check_compatibility,
    check_state_integrity,
    plan_restore,
)


def _fingerprint(model_digest: str = "a" * 64) -> LlamaEngineFingerprint:
    artifact = ArtifactIdentity(path="/models/model.gguf", sha256=model_digest, size_bytes=1, mtime_ns=1, inode=1)
    return LlamaEngineFingerprint(
        model=artifact,
        tokenizer=artifact,
        llama_cpp_python_version="0.3.34",
        llama_cpp_commit="b10103",
        backend="cpu",
        n_ctx=1024,
    )


def _entry(key: PrefixCacheKey, state: bytes = b"saved llama state") -> PrefixCacheEntry:
    return PrefixCacheEntry(key, len(state), hashlib.sha256(state).hexdigest())


def test_storage_keys_are_deterministic_and_exact_token_identity() -> None:
    fingerprint = _fingerprint()
    first = PrefixCacheKey.from_fingerprint(namespace="demo", fingerprint=fingerprint, prefix_token_ids=(1, 23))
    same = PrefixCacheKey.from_fingerprint(namespace="demo", fingerprint=fingerprint, prefix_token_ids=(1, 23))
    changed_tokens = PrefixCacheKey.from_fingerprint(
        namespace="demo", fingerprint=fingerprint, prefix_token_ids=(12, 3)
    )
    changed_engine = PrefixCacheKey.from_fingerprint(
        namespace="demo", fingerprint=_fingerprint("b" * 64), prefix_token_ids=(1, 23)
    )

    assert first.storage_key == same.storage_key
    assert first.storage_key != changed_tokens.storage_key
    assert first.storage_key != changed_engine.storage_key


def test_prompt_text_is_not_a_cache_identity_substitute() -> None:
    key = PrefixCacheKey.from_fingerprint(namespace="demo", fingerprint=_fingerprint(), prefix_token_ids=(101, 202))
    entry = _entry(key)
    same_prompt_different_tokens = PrefixCacheKey.from_fingerprint(
        namespace="demo", fingerprint=_fingerprint(), prefix_token_ids=(101, 203)
    )

    result = check_compatibility(entry, same_prompt_different_tokens)

    assert not result.accepted
    assert result.reason is CacheRejectionReason.PREFIX_TOKEN_IDS_MISMATCH


def test_fingerprint_mismatch_is_rejected() -> None:
    key = PrefixCacheKey.from_fingerprint(namespace="demo", fingerprint=_fingerprint(), prefix_token_ids=(1, 2))
    entry = _entry(key)
    incompatible = PrefixCacheKey.from_fingerprint(
        namespace="demo", fingerprint=_fingerprint("b" * 64), prefix_token_ids=(1, 2)
    )

    result = check_compatibility(entry, incompatible)

    assert not result.accepted
    assert result.reason is CacheRejectionReason.ENGINE_FINGERPRINT_MISMATCH


def test_corrupted_state_checksum_is_rejected() -> None:
    entry = _entry(PrefixCacheKey.from_fingerprint(namespace="demo", fingerprint=_fingerprint(), prefix_token_ids=(1,)))

    result = check_state_integrity(entry, b"x" * entry.state_size_bytes)

    assert not result.accepted
    assert result.reason is CacheRejectionReason.STATE_CHECKSUM_MISMATCH


def test_restore_plan_separates_checkpoint_prefix_from_uncached_suffix() -> None:
    key = PrefixCacheKey.from_fingerprint(namespace="demo", fingerprint=_fingerprint(), prefix_token_ids=(10, 20))

    decision = plan_restore(_entry(key), key, (10, 20, 30, 40))

    assert decision.can_restore
    assert decision.plan is not None
    assert decision.plan.cached_prefix_token_ids == (10, 20)
    assert decision.plan.uncached_suffix_token_ids == (30, 40)


def test_restore_plan_rejects_non_matching_request_prefix() -> None:
    key = PrefixCacheKey.from_fingerprint(namespace="demo", fingerprint=_fingerprint(), prefix_token_ids=(10, 20))

    decision = plan_restore(_entry(key), key, (10, 21, 30))

    assert not decision.can_restore
    assert decision.rejection is not None
    assert decision.rejection.reason is CacheRejectionReason.REQUEST_DOES_NOT_EXTEND_PREFIX


def test_restore_plan_requires_suffix_decode_to_prevent_stale_logits() -> None:
    key = PrefixCacheKey.from_fingerprint(namespace="demo", fingerprint=_fingerprint(), prefix_token_ids=(10, 20))

    decision = plan_restore(_entry(key), key, (10, 20))

    assert not decision.can_restore
    assert decision.rejection is not None
    assert decision.rejection.reason is CacheRejectionReason.SUFFIX_DECODE_REQUIRED
