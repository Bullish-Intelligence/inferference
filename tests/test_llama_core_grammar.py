from __future__ import annotations

import numpy as np
import pytest

from inferference.fingerprint import ArtifactIdentity, LlamaEngineFingerprint
from inferference.gpu_env import GPU_BACKEND_LABEL
from inferference.grammar import GrammarCompilerCache, JsonSchemaGrammar, apply_packed_bitmask_inplace


def _identity(name: str) -> ArtifactIdentity:
    return ArtifactIdentity(path=name, sha256="a" * 64, size_bytes=1, mtime_ns=1)


def _fingerprint(*, backend: str = "cpu") -> LlamaEngineFingerprint:
    return LlamaEngineFingerprint(
        model=_identity("model.gguf"),
        tokenizer=_identity("tokenizer.json"),
        llama_cpp_python_version="0.3.34",
        backend=backend,
        n_ctx=128,
    )


def test_packed_mask_masks_padded_model_vocabulary_ids() -> None:
    logits = np.arange(65, dtype=np.float32)
    # Permit tokens 0 and 64 only; model ids 10..63 emulate padded logits.
    mask = np.array([1, 0, 1], dtype=np.int32)

    apply_packed_bitmask_inplace(logits, mask, 65)

    assert logits[0] == 0
    assert logits[64] == 64
    assert np.isneginf(logits[1])
    assert np.isneginf(logits[63])


def test_packed_mask_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="smaller"):
        apply_packed_bitmask_inplace(np.zeros(4), np.zeros(1, dtype=np.int32), 33)


def test_matcher_token_hook_accepts_once_and_fails_closed() -> None:
    class Matcher:
        def __init__(self, accepted: bool) -> None:
            self.accepted = accepted
            self.tokens: list[int] = []

        def accept_token(self, token: int) -> bool:
            self.tokens.append(token)
            return self.accepted

    matcher = Matcher(True)
    JsonSchemaGrammar.token_hook(matcher)(17)
    assert matcher.tokens == [17]

    with pytest.raises(RuntimeError, match="rejected"):
        JsonSchemaGrammar.token_hook(Matcher(False))(18)


def test_grammar_cache_key_is_canonical_and_engine_scoped() -> None:
    schema_a = {"type": "object", "properties": {"city": {"type": "string"}}}
    schema_b = {"properties": {"city": {"type": "string"}}, "type": "object"}

    first = GrammarCompilerCache.key_for(_fingerprint(), schema_a, xgrammar_version="0.2.1")
    same = GrammarCompilerCache.key_for(_fingerprint(), schema_b, xgrammar_version="0.2.1")
    other_engine = GrammarCompilerCache.key_for(
        _fingerprint(backend=GPU_BACKEND_LABEL), schema_a, xgrammar_version="0.2.1"
    )

    assert first.digest == same.digest
    assert first.digest != other_engine.digest


@pytest.mark.runtime
def test_compiler_cache_reuses_only_matching_schema_and_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    import xgrammar as xgr

    calls = {"tokenizer": 0, "compiler": 0, "compile": 0}

    class TokenizerInfo:
        @staticmethod
        def from_huggingface(tokenizer: object, *, vocab_size: int) -> tuple[object, int]:
            calls["tokenizer"] += 1
            return tokenizer, vocab_size

    class Compiler:
        def __init__(self, tokenizer_info: object, *, cache_enabled: bool) -> None:
            calls["compiler"] += 1

        def compile_json_schema(self, schema: dict[str, object], *, strict_mode: bool, **options: object) -> object:
            calls["compile"] += 1
            return object()

    monkeypatch.setattr(xgr, "TokenizerInfo", TokenizerInfo)
    monkeypatch.setattr(xgr, "GrammarCompiler", Compiler)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.2.1")
    cache = GrammarCompilerCache()
    schema = {"type": "object"}

    first = cache.get_or_compile(_fingerprint(), object(), schema, vocab_size=65)
    same = cache.get_or_compile(_fingerprint(), object(), schema, vocab_size=65)
    other_schema = cache.get_or_compile(_fingerprint(), object(), {"type": "string"}, vocab_size=65)
    other_engine = cache.get_or_compile(_fingerprint(backend=GPU_BACKEND_LABEL), object(), schema, vocab_size=65)

    assert first is same
    assert other_schema is not first and other_engine is not first
    assert calls == {"tokenizer": 2, "compiler": 2, "compile": 3}
