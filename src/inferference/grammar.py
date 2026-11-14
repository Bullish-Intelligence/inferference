"""XGrammar integration helpers that keep torch out of the decode hot path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .fingerprint import LlamaEngineFingerprint


def apply_packed_bitmask_inplace(logits: Any, packed_mask: Any, vocab_size: int) -> None:
    """Set masked logits to ``-inf`` from XGrammar's packed int32 bitmask.

    Bit ``i`` is one when token ``i`` is allowed.  ``vocab_size`` must be the
    model logit width, not the Hugging Face tokenizer length: padded model IDs
    must be explicitly masked too.  NumPy is imported locally to preserve a
    lightweight shared-core import path.
    """
    import numpy as np

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if len(logits) < vocab_size:
        raise ValueError("logits is smaller than vocab_size")
    required_words = (vocab_size + 31) // 32
    if len(packed_mask) < required_words:
        raise ValueError("packed_mask is smaller than vocab_size")
    ids = np.arange(vocab_size, dtype=np.int64)
    words = np.asarray(packed_mask[:required_words], dtype=np.uint32)
    allowed = ((words[ids >> 5] >> (ids & 31)) & 1).astype(bool)
    logits[:vocab_size][~allowed] = -np.inf


# Compact JSON: no whitespace anywhere between tokens. xgrammar's default separators
# for ``any_whitespace=False`` are ``", "`` and ``": "``, which still force a space.
# The explicit pair removes it. One gap remains in xgrammar 0.2.1: the grammar of a
# free-form ``{}`` schema (root or property) still requires a space after a comma.
COMPACT_SEPARATORS: tuple[str, str] = (",", ":")

# Limit on the whitespace run at each gap between two JSON tokens, when a request opts in
# to whitespace. Each gap takes 0 to 8 whitespace characters, in any mix of space and
# newline. The ninth character has no valid next token, so the matcher rejects it.
# The limit is per gap: a document with N gaps may hold up to 8 * N whitespace
# characters. It does not bound the whole output. It does not apply to spaces inside a
# JSON string. xgrammar 0.2.1 has no limit by default (``max_whitespace_cnt=None``): a
# model may then loop on whitespace tokens until ``max_tokens``. 8 covers one newline
# plus 7 spaces, which is pretty-printed JSON at indent 2, three levels deep.
MAX_WHITESPACE_CNT = 8


def json_schema_compile_options(*, allow_whitespace: bool = False) -> dict[str, Any]:
    """Return the xgrammar ``compile_json_schema`` keywords for one whitespace mode.

    ``allow_whitespace=False`` (the default) gives compact JSON:
    ``any_whitespace=False`` with ``separators=(",", ":")``.
    ``allow_whitespace=True`` gives ``any_whitespace=True`` bounded by
    :data:`MAX_WHITESPACE_CNT`.
    """
    if allow_whitespace:
        return {"any_whitespace": True, "max_whitespace_cnt": MAX_WHITESPACE_CNT}
    return {"any_whitespace": False, "separators": COMPACT_SEPARATORS}


class JsonSchemaGrammar:
    """One compiled grammar and one fresh XGrammar matcher per sequence."""

    def __init__(self, compiler: Any, compiled: Any, *, vocab_size: int) -> None:
        self.compiler = compiler
        self.compiled = compiled
        self.vocab_size = vocab_size

    @classmethod
    def from_huggingface(
        cls,
        tokenizer: Any,
        schema: dict[str, Any],
        *,
        vocab_size: int,
        allow_whitespace: bool = False,
    ) -> JsonSchemaGrammar:
        """Compile strict JSON Schema with the true llama.cpp vocabulary width.

        The output is compact JSON unless ``allow_whitespace`` is true. Then
        whitespace is allowed, bounded by :data:`MAX_WHITESPACE_CNT`.
        """
        import xgrammar as xgr

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=True)
        compiled = compiler.compile_json_schema(
            schema, strict_mode=True, **json_schema_compile_options(allow_whitespace=allow_whitespace)
        )
        return cls(compiler, compiled, vocab_size=vocab_size)

    def with_whitespace(self, schema: dict[str, Any], *, allow_whitespace: bool) -> JsonSchemaGrammar:
        """Compile ``schema`` in another whitespace mode with this grammar's compiler.

        The compiler holds the tokenizer information. This call does not build it again.
        The matchers of the new grammar are independent of the matchers of this one.
        """
        compiled = self.compiler.compile_json_schema(
            schema, strict_mode=True, **json_schema_compile_options(allow_whitespace=allow_whitespace)
        )
        return type(self)(self.compiler, compiled, vocab_size=self.vocab_size)

    def new_matcher(self) -> Any:
        import xgrammar as xgr

        return xgr.GrammarMatcher(self.compiled)

    @staticmethod
    def token_hook(matcher: Any) -> Any:
        """Advance a matcher once, failing closed if a masked token is rejected."""

        def accept(token: int) -> None:
            accepted = matcher.accept_token(token)
            if accepted is False:
                raise RuntimeError(f"XGrammar rejected masked token {token}")

        return accept

    def logits_hook(self, matcher: Any, *, benchmark: Any | None = None) -> Any:
        """Return an owned-loop hook which fills and applies the next-token mask."""
        import numpy as np
        import xgrammar as xgr

        bitmask = np.zeros(xgr.get_bitmask_shape(1, self.vocab_size), dtype=np.int32)

        def apply(logits: Any) -> None:
            bitmask.fill(0)
            if benchmark is None:
                needs_mask = matcher.fill_next_token_bitmask(bitmask)
            else:
                with benchmark.measure("mask_creation"):
                    needs_mask = matcher.fill_next_token_bitmask(bitmask)
            if needs_mask:
                if benchmark is None:
                    apply_packed_bitmask_inplace(logits, bitmask[0], self.vocab_size)
                else:
                    with benchmark.measure("mask_application"):
                        apply_packed_bitmask_inplace(logits, bitmask[0], self.vocab_size)

        return apply


@dataclass(frozen=True, slots=True)
class GrammarCacheKey:
    """Stable key for a strict JSON grammar compiled for one engine identity."""

    engine_fingerprint: str
    canonical_schema: str
    strict_mode: bool
    xgrammar_version: str
    allow_whitespace: bool = False

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "engine_fingerprint": self.engine_fingerprint,
                "schema": self.canonical_schema,
                "strict_mode": self.strict_mode,
                "xgrammar_version": self.xgrammar_version,
                "allow_whitespace": self.allow_whitespace,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GrammarCompilerCache:
    """In-process compiler and compiled-grammar cache, never a matcher cache.

    A compiler is reused only for an identical engine fingerprint. Compiled
    grammars additionally require an identical canonical schema, strictness,
    xgrammar release, and whitespace mode. Matchers remain per-request state.
    """

    def __init__(self) -> None:
        self._compilers: dict[str, Any] = {}
        self._grammars: dict[str, JsonSchemaGrammar] = {}

    @staticmethod
    def key_for(
        fingerprint: LlamaEngineFingerprint,
        schema: dict[str, Any],
        *,
        strict_mode: bool = True,
        xgrammar_version: str,
        allow_whitespace: bool = False,
    ) -> GrammarCacheKey:
        return GrammarCacheKey(
            engine_fingerprint=fingerprint.cache_key(),
            canonical_schema=json.dumps(schema, separators=(",", ":"), sort_keys=True),
            strict_mode=strict_mode,
            xgrammar_version=xgrammar_version,
            allow_whitespace=allow_whitespace,
        )

    def get_or_compile(
        self,
        fingerprint: LlamaEngineFingerprint,
        tokenizer: Any,
        schema: dict[str, Any],
        *,
        vocab_size: int,
        strict_mode: bool = True,
        allow_whitespace: bool = False,
    ) -> JsonSchemaGrammar:
        """Return a compatible compiled grammar; do not share its matcher."""
        from importlib.metadata import version

        import xgrammar as xgr

        key = self.key_for(
            fingerprint,
            schema,
            strict_mode=strict_mode,
            xgrammar_version=version("xgrammar"),
            allow_whitespace=allow_whitespace,
        )
        cached = self._grammars.get(key.digest)
        if cached is not None:
            return cached
        compiler = self._compilers.get(key.engine_fingerprint)
        if compiler is None:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
            compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=True)
            self._compilers[key.engine_fingerprint] = compiler
        grammar = JsonSchemaGrammar(
            compiler,
            compiler.compile_json_schema(
                schema, strict_mode=strict_mode, **json_schema_compile_options(allow_whitespace=allow_whitespace)
            ),
            vocab_size=vocab_size,
        )
        self._grammars[key.digest] = grammar
        return grammar
