"""Compact JSON is the grammar default; whitespace is a bounded per-request opt-in.

xgrammar 0.2.1 compiles a JSON schema with ``any_whitespace=True`` and no bound by
default. A model can then fill ``max_tokens`` with whitespace tokens. These tests
cover the options, the cache key, the request type and the router selection with
no GPU. The ``runtime`` tests also drive a real xgrammar matcher over a small
synthetic vocabulary.
"""

from __future__ import annotations

from typing import Any

import pytest

from inferference.fingerprint import ArtifactIdentity, LlamaEngineFingerprint
from inferference.grammar import (
    COMPACT_SEPARATORS,
    MAX_WHITESPACE_CNT,
    GrammarCompilerCache,
    JsonSchemaGrammar,
    json_schema_compile_options,
)
from inferference.router import MultiLoRARouter, RouteRequest

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"tool": {"type": "string"}},
    "required": ["tool"],
}


def _fingerprint() -> LlamaEngineFingerprint:
    identity = ArtifactIdentity(path="m", sha256="a" * 64, size_bytes=1, mtime_ns=1)
    return LlamaEngineFingerprint(
        model=identity,
        tokenizer=identity,
        llama_cpp_python_version="0.3.34",
        backend="cpu",
        n_ctx=128,
    )


# ---- options, cache key, request type (no native stack) ----


def test_compact_is_the_default_option_set() -> None:
    assert json_schema_compile_options() == {"any_whitespace": False, "separators": COMPACT_SEPARATORS}
    assert COMPACT_SEPARATORS == (",", ":")


def test_whitespace_option_set_is_bounded() -> None:
    options = json_schema_compile_options(allow_whitespace=True)
    assert options == {"any_whitespace": True, "max_whitespace_cnt": MAX_WHITESPACE_CNT}
    assert 0 < MAX_WHITESPACE_CNT <= 16  # xgrammar requires a positive bound; keep it small


def test_cache_key_separates_compact_and_whitespace() -> None:
    compact = GrammarCompilerCache.key_for(_fingerprint(), SCHEMA, xgrammar_version="0.2.1")
    explicit = GrammarCompilerCache.key_for(_fingerprint(), SCHEMA, xgrammar_version="0.2.1", allow_whitespace=False)
    loose = GrammarCompilerCache.key_for(_fingerprint(), SCHEMA, xgrammar_version="0.2.1", allow_whitespace=True)
    assert compact.allow_whitespace is False
    assert compact.digest == explicit.digest
    assert compact.digest != loose.digest


def test_cache_compiles_each_whitespace_mode_once(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("xgrammar")
    import importlib.metadata

    import xgrammar as xgr

    compiled: list[dict[str, object]] = []

    class TokenizerInfo:
        @staticmethod
        def from_huggingface(tokenizer: object, *, vocab_size: int) -> object:
            return object()

    class Compiler:
        def __init__(self, tokenizer_info: object, *, cache_enabled: bool) -> None:
            pass

        def compile_json_schema(self, schema: dict[str, object], *, strict_mode: bool, **options: object) -> object:
            compiled.append(options)
            return object()

    monkeypatch.setattr(xgr, "TokenizerInfo", TokenizerInfo)
    monkeypatch.setattr(xgr, "GrammarCompiler", Compiler)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.2.1")
    cache = GrammarCompilerCache()

    compact = cache.get_or_compile(_fingerprint(), object(), SCHEMA, vocab_size=65)
    loose = cache.get_or_compile(_fingerprint(), object(), SCHEMA, vocab_size=65, allow_whitespace=True)
    assert cache.get_or_compile(_fingerprint(), object(), SCHEMA, vocab_size=65) is compact
    assert cache.get_or_compile(_fingerprint(), object(), SCHEMA, vocab_size=65, allow_whitespace=True) is loose
    assert compact is not loose
    assert compiled == [json_schema_compile_options(), json_schema_compile_options(allow_whitespace=True)]


def test_route_request_option_defaults_to_compact_and_round_trips() -> None:
    assert RouteRequest(prompt="p").allow_whitespace is False
    request = RouteRequest(prompt="p", allow_whitespace=True, request_id="r")
    assert request.allow_whitespace is True
    again = RouteRequest.model_validate(request.model_dump())
    assert again == request
    assert RouteRequest.model_validate_json(request.model_dump_json()).allow_whitespace is True
    # A payload that predates the option still validates and means compact.
    assert RouteRequest.model_validate({"prompt": "p"}).allow_whitespace is False


class _FakeCompiler:
    """Record each ``compile_json_schema`` call. No xgrammar."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def compile_json_schema(self, schema: dict[str, Any], **options: Any) -> tuple[str, bool]:
        self.calls.append((schema, options))
        return ("compiled", bool(options.get("any_whitespace")))


class _TaggedGrammar(JsonSchemaGrammar):
    """A grammar whose matcher says which grammar made it."""

    def new_matcher(self) -> Any:
        return "ws" if self.compiled[1] else "compact"


def _router_with_fake_grammar(
    monkeypatch: pytest.MonkeyPatch, schema: dict[str, Any]
) -> tuple[Any, _FakeCompiler, list[int]]:
    compiler = _FakeCompiler()
    built: list[int] = []

    def fake_from_huggingface(
        cls: type, tokenizer: object, schema: dict[str, Any], *, vocab_size: int, allow_whitespace: bool = False
    ) -> JsonSchemaGrammar:
        built.append(vocab_size)  # one call is one TokenizerInfo and one compiler
        compiled = compiler.compile_json_schema(
            schema, **json_schema_compile_options(allow_whitespace=allow_whitespace)
        )
        return _TaggedGrammar(compiler, compiled, vocab_size=vocab_size)

    monkeypatch.setattr(JsonSchemaGrammar, "from_huggingface", classmethod(fake_from_huggingface))
    router = object.__new__(MultiLoRARouter)
    router.n_vocab = 65
    router.enable_grammar(object(), schema)
    return router, compiler, built


def test_router_picks_the_grammar_by_request_option(monkeypatch: pytest.MonkeyPatch) -> None:
    router, compiler, built = _router_with_fake_grammar(monkeypatch, SCHEMA)
    assert built == [65] and len(compiler.calls) == 1  # compact compiles at enable time

    compact = router._grammar_for(RouteRequest(prompt="p"))
    assert compact is router._grammar
    assert len(compiler.calls) == 1  # whitespace waits for a request
    loose = router._grammar_for(RouteRequest(prompt="p", allow_whitespace=True))
    assert loose is not compact
    assert router._grammar_for(RouteRequest(prompt="q", allow_whitespace=True)) is loose
    assert [{k: v for k, v in opts.items() if k != "strict_mode"} for _, opts in compiler.calls] == [
        json_schema_compile_options(),
        json_schema_compile_options(allow_whitespace=True),
    ]
    assert all(opts["strict_mode"] is True for _, opts in compiler.calls[1:])

    router.enable_grammar(object(), SCHEMA)  # a new grammar drops the stale whitespace variant
    assert router._grammar_whitespace is None


def test_the_whitespace_grammar_reuses_the_compact_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    router, compiler, built = _router_with_fake_grammar(monkeypatch, SCHEMA)
    loose = router._grammar_for(RouteRequest(prompt="p", allow_whitespace=True))
    assert built == [65]  # no second TokenizerInfo and no second compiler
    assert loose.compiler is router._grammar.compiler is compiler
    assert loose.vocab_size == 65


def test_the_router_keeps_its_own_copy_of_the_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = {"type": "object", "properties": {"tool": {"type": "string"}}, "required": ["tool"]}
    router, compiler, _ = _router_with_fake_grammar(monkeypatch, schema)
    schema["properties"]["tool"] = {"type": "integer"}  # the caller changes its dict
    schema["required"].append("other")
    router._grammar_for(RouteRequest(prompt="p", allow_whitespace=True))
    compiled_schema = compiler.calls[-1][0]
    assert compiled_schema["properties"] == {"tool": {"type": "string"}} and compiled_schema["required"] == ["tool"]
    assert compiled_schema is not schema and compiler.calls[0][0] is not schema
    # A change to the schema that the first whitespace compile used does not leak into the next one.
    compiled_schema["required"].append("x")
    assert router._grammar_schema is not None and router._grammar_schema["required"] == ["tool"]


def test_grammar_for_without_enable_grammar_raises_runtime_error() -> None:
    router = object.__new__(MultiLoRARouter)
    router._grammar = None
    router._grammar_schema = None
    with pytest.raises(RuntimeError, match="enable_grammar"):
        router._grammar_for(RouteRequest(prompt="p"))
    with pytest.raises(RuntimeError, match="enable_grammar"):
        router._grammar_for(RouteRequest(prompt="p", allow_whitespace=True))


# ---- run, run_cached and run_stream pick the grammar per request ----


def _mixed_requests() -> list[RouteRequest]:
    return [
        RouteRequest(prompt="a", request_id="r0"),
        RouteRequest(prompt="b", request_id="r1", allow_whitespace=True),
        RouteRequest(prompt="c", request_id="r2", allow_whitespace=False),
        RouteRequest(prompt="d", request_id="r3", allow_whitespace=True),
    ]


_WANT = ["compact", "ws", "compact", "ws"]


def _stub_decode(router: Any, seen: list[list[Any] | None]) -> None:
    """Stub the native steps. ``seen`` collects the matcher list that each wave receives."""

    def generate(
        ctx: Any, prompts: list[Any], max_tokens: int, matchers: list[Any] | None, samplers: Any = None
    ) -> Any:
        seen.append(matchers)
        return [([1], "stop")] * len(prompts)

    router.n_seq_max = 8
    router.n_batch = 8
    router._ctx = {None: object()}
    router._adapter_ptr = {}
    router.tokenize = lambda text: [1]
    router.tokenize_suffix = lambda text: [1]
    router._build_sampler = lambda request: None
    router._generate_wave = generate
    router._finalize = lambda r, prompt, gen, constrained: r.request_id


def test_run_gives_each_request_its_own_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    router, _, _ = _router_with_fake_grammar(monkeypatch, SCHEMA)
    seen: list[list[Any] | None] = []
    _stub_decode(router, seen)
    router.backend = "context_pool"
    assert router.run(_mixed_requests(), constrained=True) == ["r0", "r1", "r2", "r3"]
    assert seen == [_WANT]
    seen.clear()
    router.run(_mixed_requests(), constrained=False)
    assert seen == [None]


def test_run_seq_routed_gives_each_request_its_own_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    router, _, _ = _router_with_fake_grammar(monkeypatch, SCHEMA)
    seen: list[list[Any] | None] = []
    _stub_decode(router, seen)
    router.backend = "seq_routed"
    router._seq_ctx = object()
    router._seq_index = {}
    router._seq_binding = type("Binding", (), {"set_seq_adapter": lambda self, *a: None})()
    router.run(_mixed_requests(), constrained=True)
    assert seen == [_WANT]


def test_run_stream_gives_the_request_its_own_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    router, _, _ = _router_with_fake_grammar(monkeypatch, SCHEMA)
    seen: list[list[Any] | None] = []

    def iter_wave(ctx: Any, prompts: Any, max_tokens: int, matchers: Any, samplers: Any) -> Any:
        seen.append(matchers)
        yield [(7, None)]

    _stub_decode(router, [])
    router._iter_wave = iter_wave
    router.backend = "context_pool"
    for request, want in zip(_mixed_requests(), _WANT, strict=True):
        assert list(router.run_stream(request, constrained=True)) == [(7, None)]
        assert seen[-1] == [want]


def test_run_cached_gives_each_request_its_own_grammar_in_both_legs(monkeypatch: pytest.MonkeyPatch) -> None:
    import inferference.router as router_module

    router, _, _ = _router_with_fake_grammar(monkeypatch, SCHEMA)
    seen: list[list[Any] | None] = []
    _stub_decode(router, seen)
    router.backend = "context_pool"
    router.n_vocab = 65
    router.C = type("C", (), {"llama_synchronize": lambda self, ctx: None})()
    router._seq_rm = lambda ctx, seq_id: None
    router._prefill = lambda ctx, tokens, seq_id: None
    router._baseline_generate = lambda adapter, tokens, max_tokens: ([1], "stop")
    restored: list[list[Any] | None] = []

    def generate_restored(
        ctx: Any, blob: Any, n_prefix: int, suffixes: Any, max_tokens: int, matchers: Any, bridge: Any
    ) -> Any:
        restored.append(matchers)
        return [([1], "stop")] * len(suffixes)

    router._generate_wave_restored = generate_restored

    class Bridge:
        def __init__(self, **_: Any) -> None:
            pass

        def capture_seq_state(self, ctx: Any, seq_id: int) -> bytes:
            return b""

    monkeypatch.setattr(router_module, "LlamaSeqStateBridge", Bridge)
    router.run_cached(_mixed_requests(), [1, 2], constrained=True)
    assert restored == [_WANT]  # the cached leg
    assert seen == [_WANT]  # the cold leg


# ---- real xgrammar compiler over a synthetic vocabulary ----


def _real_grammar(schema: dict[str, Any]) -> JsonSchemaGrammar:
    xgr = pytest.importorskip("xgrammar")
    info = xgr.TokenizerInfo(_ASCII_VOCAB, xgr.VocabType.RAW, vocab_size=len(_ASCII_VOCAB), stop_token_ids=[0])
    compiler = xgr.GrammarCompiler(info)
    compiled = compiler.compile_json_schema(schema, strict_mode=True, **json_schema_compile_options())
    return JsonSchemaGrammar(compiler, compiled, vocab_size=len(_ASCII_VOCAB))


# One token for each printable ASCII character, plus the stop token at index 0.
_ASCII_VOCAB = ["<eos>", "\n", *[chr(c) for c in range(32, 127)]]
_ASCII_ID = {tok: i for i, tok in enumerate(_ASCII_VOCAB)}


def _accepts(grammar: JsonSchemaGrammar, text: str, *, complete: bool = True) -> bool:
    matcher = grammar.new_matcher()
    for ch in text:
        if matcher.accept_token(_ASCII_ID[ch]) is False:
            return False
    return matcher.accept_token(0) is not False and matcher.is_terminated() if complete else True


def _router_over(grammar: JsonSchemaGrammar, schema: dict[str, Any]) -> MultiLoRARouter:
    router = object.__new__(MultiLoRARouter)
    router._grammar = grammar
    router._grammar_schema = schema
    router._grammar_whitespace = None
    return router


NESTED: dict[str, Any] = {
    "type": "object",
    "properties": {
        "a": {
            "type": "object",
            "properties": {"b": {"type": "array", "items": {"type": "integer"}}},
            "required": ["b"],
        },
        "s": {"type": "string"},
    },
    "required": ["a", "s"],
}


@pytest.mark.runtime
@pytest.mark.parametrize("schema", [NESTED, {}], ids=["nested", "empty-schema"])
def test_the_router_whitespace_grammar_for_a_nested_and_an_empty_schema(schema: dict[str, Any]) -> None:
    router = _router_over(_real_grammar(schema), schema)
    compact = router._grammar_for(RouteRequest(prompt="p"))
    loose = router._grammar_for(RouteRequest(prompt="p", allow_whitespace=True))
    assert loose is not compact and loose.compiler is compact.compiler
    # xgrammar 0.2.1 still wants ", " after a comma inside a free-form ``{}`` value, so the
    # empty-schema document has no comma.
    doc = '{"a":{"b":[1,2]},"s":"x"}' if schema is NESTED else '{"k":"v"}'
    spaced = '{ "a" : { "b" : [ 1 , 2 ] } ,\n "s" : "x" }' if schema is NESTED else '{ "k" : "v" }'
    assert _accepts(compact, doc) and _accepts(loose, doc)
    assert not _accepts(compact, spaced) and _accepts(loose, spaced)


@pytest.mark.runtime
def test_compact_output_keeps_the_spaces_inside_strings() -> None:
    router = _router_over(_real_grammar(NESTED), NESTED)
    compact = router._grammar_for(RouteRequest(prompt="p"))
    assert _accepts(compact, '{"a":{"b":[]},"s":"two words  and a gap"}')
    assert not _accepts(compact, '{"a":{"b":[]}, "s":"x"}')  # a space after the comma
    assert not _accepts(compact, '{"a": {"b":[]},"s":"x"}')  # a space after the colon


# ---- real xgrammar matcher over a synthetic vocabulary ----

_VOCAB = ["<eos>", "{", "}", '"', ":", ",", " ", "\n", "t", "r", "u", "e", "f", "a", "l", "s", "b"]
_ID = {tok: i for i, tok in enumerate(_VOCAB)}


def _matcher(allow_whitespace: bool) -> Any:
    import xgrammar as xgr

    info = xgr.TokenizerInfo(_VOCAB, xgr.VocabType.RAW, vocab_size=len(_VOCAB), stop_token_ids=[0])
    compiler = xgr.GrammarCompiler(info)
    grammar = compiler.compile_json_schema(
        {"type": "object", "properties": {"b": {"type": "boolean"}}, "required": ["b"]},
        strict_mode=True,
        **json_schema_compile_options(allow_whitespace=allow_whitespace),
    )
    return xgr.GrammarMatcher(grammar)


def _allowed(matcher: Any) -> set[str]:
    import numpy as np
    import xgrammar as xgr

    mask = np.zeros(xgr.get_bitmask_shape(1, len(_VOCAB)), dtype=np.int32)
    matcher.fill_next_token_bitmask(mask)
    return {tok for tok, i in _ID.items() if (int(mask[0, 0]) >> i) & 1}


def _feed(matcher: Any, text: str) -> bool:
    """Feed one single-character token at a time; False at the first rejection."""
    return all(matcher.accept_token(_ID[ch]) is not False for ch in text)


@pytest.mark.runtime
def test_compact_grammar_rejects_whitespace_after_key_separator() -> None:
    matcher = _matcher(allow_whitespace=False)
    assert _feed(matcher, '{"b":')
    assert not _allowed(matcher) & {" ", "\n"}
    assert matcher.accept_token(_ID[" "]) is False
    assert matcher.accept_token(_ID["\n"]) is False


@pytest.mark.runtime
def test_compact_grammar_rejects_whitespace_everywhere_and_accepts_compact_json() -> None:
    for text in ("{ ", '{"b" ', '{"b": ', '{"b":\n', '{"b":true ', '{"b":true\n'):
        assert not _feed(_matcher(allow_whitespace=False), text), text
    done = _matcher(allow_whitespace=False)
    assert _feed(done, '{"b":true}')
    assert done.accept_token(_ID["<eos>"]) is not False
    assert done.is_terminated()


@pytest.mark.runtime
def test_whitespace_option_allows_whitespace_after_key_separator() -> None:
    matcher = _matcher(allow_whitespace=True)
    assert _feed(matcher, '{"b":')
    assert {" ", "\n"} <= _allowed(matcher)
    assert _feed(matcher, " \n true \n}")
    assert _feed(_matcher(allow_whitespace=True), '{ \n"b" : false}')


@pytest.mark.runtime
def test_whitespace_option_rejects_a_run_above_the_bound() -> None:
    at_bound = _matcher(allow_whitespace=True)
    assert _feed(at_bound, '{"b":' + " " * MAX_WHITESPACE_CNT)
    assert " " not in _allowed(at_bound) and "\n" not in _allowed(at_bound)
    assert at_bound.accept_token(_ID[" "]) is False

    over = _matcher(allow_whitespace=True)
    assert not _feed(over, '{"b":' + " " * (MAX_WHITESPACE_CNT + 1))
    # The runaway failure: sixteen runs of newline plus two spaces after a key.
    runaway = _matcher(allow_whitespace=True)
    assert not _feed(runaway, '{"b":' + "\n  " * 16)
