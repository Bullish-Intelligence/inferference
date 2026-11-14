"""Boundary-model contracts for the multi-LoRA router surface.

No native library or GPU: this validates the Pydantic edges only, matching the
project rule that validation stops at the boundary and the hot path stays plain.
Stage 2 additions (llgym Phase 1): per-request sampler params, the sampler-chain
builder, and the streaming wave generator — all exercised GPU-free with test
doubles (the ``decode.py:221`` bypass-``__init__`` pattern).
"""

from __future__ import annotations

import types

import pytest
from pydantic import ValidationError

from inferference.gpu_env import GPU_BACKEND_LABEL
from inferference.models import EngineConfig, GenerationResult
from inferference.router import (
    BASE,
    FINISH_LENGTH,
    FINISH_STOP,
    AdapterSpec,
    MultiLoRARouter,
    RouterConfig,
    RouteRequest,
    RouteResult,
    build_sampler_chain,
    request_uses_sampling,
)


def _engine() -> EngineConfig:
    return EngineConfig(model_path="m.gguf", n_ctx=4096, n_batch=256, n_gpu_layers=-1, backend=GPU_BACKEND_LABEL)


def test_router_config_validates_and_forbids_extra() -> None:
    config = RouterConfig(
        engine=_engine(),
        adapters=(AdapterSpec(name="a", gguf_path="a.gguf"), AdapterSpec(name="b", gguf_path="b.gguf", scale=0.75)),
        n_seq_max=8,
    )
    assert [s.name for s in config.adapters] == ["a", "b"]
    assert config.adapters[1].scale == 0.75
    assert config.include_base is True
    with pytest.raises(ValidationError):
        RouterConfig(engine=_engine(), adapters=(), n_seq_max=8, bogus=1)  # type: ignore[call-arg]


def test_adapter_scale_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        AdapterSpec(name="a", gguf_path="a.gguf", scale=0.0)


def test_route_request_defaults_and_base_sentinel() -> None:
    req = RouteRequest(prompt="route me", request_id="r0")
    assert req.adapter is BASE  # None -> base context
    assert req.max_tokens == 64
    with pytest.raises(ValidationError):
        RouteRequest(prompt="x", max_tokens=0)


def test_route_request_sampling_fields_default_to_none_and_validate() -> None:
    req = RouteRequest(prompt="p")
    assert req.temperature is None
    assert req.top_p is None
    assert req.top_k is None
    assert req.seed is None
    req = RouteRequest(prompt="p", temperature=0.8, top_p=0.9, top_k=20, seed=42)
    assert req.temperature == 0.8
    assert req.top_p == 0.9
    assert req.top_k == 20
    assert req.seed == 42
    with pytest.raises(ValidationError):
        RouteRequest(prompt="p", temperature=-0.1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        RouteRequest(prompt="p", top_p=1.5)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        RouteRequest(prompt="p", top_k=0)  # type: ignore[call-arg]


def test_request_uses_sampling_decision_table() -> None:
    # Greedy defaults: all-None, and temperature == 0 alone.
    assert request_uses_sampling(temperature=None, top_p=None, top_k=None, seed=None) is False
    assert request_uses_sampling(temperature=0.0, top_p=None, top_k=None, seed=None) is False
    # Any real sampling signal builds a chain.
    assert request_uses_sampling(temperature=1.0, top_p=None, top_k=None, seed=None) is True
    assert request_uses_sampling(temperature=None, top_p=0.9, top_k=None, seed=None) is True
    assert request_uses_sampling(temperature=None, top_p=None, top_k=20, seed=None) is True
    assert request_uses_sampling(temperature=None, top_p=None, top_k=None, seed=42) is True


class _RecordingC:
    """llama.cpp surface recording which sampler init/add calls happen."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.chain = object()

    def llama_sampler_chain_default_params(self) -> object:
        return object()

    def llama_sampler_chain_init(self, params: object) -> object:
        self.calls.append(("init", ()))
        return self.chain

    def llama_sampler_chain_add(self, chain: object, sampler: object) -> None:
        self.calls.append(("add", (sampler,)))

    def llama_sampler_init_top_k(self, k: int) -> object:
        return ("top_k", k)

    def llama_sampler_init_top_p(self, p: float, min_keep: int) -> object:
        return ("top_p", p, min_keep)

    def llama_sampler_init_temp(self, t: float) -> object:
        return ("temp", t)

    def llama_sampler_init_dist(self, seed: int) -> object:
        return ("dist", seed)


def test_build_sampler_chain_greedy_defaults_unchanged() -> None:
    c = _RecordingC()
    assert build_sampler_chain(c, None, temperature=None, top_p=None, top_k=None, seed=None) is None
    assert build_sampler_chain(c, None, temperature=0.0, top_p=None, top_k=None, seed=None) is None
    assert c.calls == []


def test_build_sampler_chain_full_sampling_wires_temp_top_p_top_k_and_seed() -> None:
    c = _RecordingC()
    chain = build_sampler_chain(c, engine_seed=7, temperature=0.8, top_p=0.9, top_k=20, seed=42)
    assert chain is c.chain
    assert c.calls == [
        ("init", ()),
        ("add", (("top_k", 20),)),
        ("add", (("top_p", 0.9, 1),)),
        ("add", (("temp", 0.8),)),
        ("add", (("dist", 42),)),  # request seed wins
    ]


def test_build_sampler_chain_seed_falls_back_to_engine_then_random() -> None:
    c = _RecordingC()
    build_sampler_chain(c, engine_seed=7, temperature=1.0, top_p=None, top_k=None, seed=None)
    assert c.calls[-1] == ("add", (("dist", 7),))
    c2 = _RecordingC()
    build_sampler_chain(c2, engine_seed=None, temperature=1.0, top_p=None, top_k=None, seed=None)
    # unseeded sampling must be random per request (LLAMA_DEFAULT_SEED sentinel,
    # llama.cpp src/llama-sampler.cpp get_rng_seed) — never a fixed 0.
    assert c2.calls[-1] == ("add", (("dist", 0xFFFFFFFF),))


def test_build_sampler_chain_top_p_alone_means_sampling_without_temp() -> None:
    c = _RecordingC()
    chain = build_sampler_chain(c, engine_seed=None, temperature=None, top_p=0.9, top_k=None, seed=None)
    assert chain is c.chain
    assert c.calls == [
        ("init", ()),
        ("add", (("top_p", 0.9, 1),)),
        ("add", (("dist", 0xFFFFFFFF),)),
    ]


def _fake_batch(*_: object) -> types.SimpleNamespace:
    """A batch shaped like llama_batch (ctypes arrays) so the wave loop can
    populate token/pos/seq_id/logits rows without llama.cpp."""
    import ctypes

    return types.SimpleNamespace(
        token=(ctypes.c_int * 8)(),
        pos=(ctypes.c_int * 8)(),
        n_seq_id=(ctypes.c_int * 8)(),
        logits=(ctypes.c_int * 8)(),
        seq_id=[(ctypes.c_int * 1)() for _ in range(8)],
    )


def _fake_router(script: dict[int, list[int]], n_seq: int, eos: int = 99) -> MultiLoRARouter:
    """Router double bypassing __init__ (decode.py test-double pattern).

    ``script[row]`` is the token sequence row ``row`` will 'sample', so the wave
    loop's finish/stop/length logic runs with no llama.cpp present.
    """
    router = MultiLoRARouter.__new__(MultiLoRARouter)
    router._eos = eos
    router.n_vocab = 10
    router.n_batch = 512
    router._candidate_type = None
    router._candidates = None
    router._candidate_view = None
    router._candidate_ids = None
    router.C = types.SimpleNamespace(
        llama_sampler_free=lambda *_: None,
        llama_batch_init=_fake_batch,
        llama_decode=lambda *_: 0,
        llama_batch_free=lambda *_: None,
    )
    steps = {row: iter(tokens) for row, tokens in script.items()}
    state = {"last": 0}

    def _sample_row(
        self: MultiLoRARouter, ctx: object, row: int, matcher: object | None = None, sampler: object | None = None
    ) -> int:
        if matcher is not None:
            raise AssertionError("matchers are not part of the wave fake")
        # The wave samples the first token from row -1 (last logits row), which
        # after the just-completed prefill is the seq prefilled last.
        return next(steps[state["last"] if row == -1 else row])

    router._sample_row = types.MethodType(_sample_row, router)  # type: ignore[assignment]

    def _prefill(self: MultiLoRARouter, ctx: object, tokens: list[int], seq_id: int) -> None:
        state["last"] = seq_id

    router._prefill = types.MethodType(_prefill, router)  # type: ignore[assignment]
    router._seq_rm = types.MethodType(lambda self, ctx, seq_id: None, router)  # type: ignore[assignment]
    return router


def test_generate_wave_materialized_matches_streaming_accumulation() -> None:
    router = _fake_router(script={0: [11, 12, 13], 1: [21, 22, 23]}, n_seq=2)
    gen = router._generate_wave(object(), [[1], [2]], max_tokens=3, matchers=None)
    assert gen == [([11, 12, 13], FINISH_LENGTH), ([21, 22, 23], FINISH_LENGTH)]


def test_iter_wave_steps_carry_token_and_finish_reason_contract() -> None:
    router = _fake_router(script={0: [11, 12], 1: [21, 99]}, n_seq=2)
    steps = list(router._iter_wave(object(), [[1], [2]], max_tokens=2, matchers=None))
    # Prefill steps carry one row-aligned entry per seq: seq0 tok 11, seq1 tok 21.
    assert steps[0] == [(11, None), (None, None)]
    assert steps[1] == [(None, None), (21, None)]
    # Batched step: seq0 tok 12 (continues), seq1 sampled EOS 99 -> stop, no emit.
    assert steps[2] == [(12, None), (None, FINISH_STOP)]
    # Flush: seq0 ran out of tokens -> length.
    assert steps[3] == [(None, FINISH_LENGTH), (None, FINISH_STOP)]


def test_iter_wave_frees_samplers_on_early_close() -> None:
    router = _fake_router(script={0: [11, 12]}, n_seq=1)
    freed: list[object] = []
    router.C.llama_sampler_free = lambda smp: freed.append(smp)  # type: ignore[attr-defined]
    smp = object()
    gen = router._iter_wave(object(), [[1]], max_tokens=5, matchers=None, samplers=[smp])
    assert next(gen) == [(11, None)]
    gen.close()  # client disconnect
    assert freed == [smp]


def test_run_stream_yields_tokens_and_frees_samplers_on_close() -> None:
    router = _fake_router(script={0: [11, 12]}, n_seq=1)
    router._ctx = {None: object()}
    router.backend = "context_pool"
    router._adapter_ptr = {}
    router._grammar = None
    router.tokenize = types.MethodType(lambda self, text: [1, 2], router)  # type: ignore[assignment]
    freed: list[object] = []
    router.C.llama_sampler_free = lambda smp: freed.append(smp)  # type: ignore[attr-defined]
    router._build_sampler = types.MethodType(  # type: ignore[assignment]
        lambda self, request: object(), router
    )
    req = RouteRequest(prompt="p", max_tokens=5, temperature=1.0)
    gen = router.run_stream(req)
    assert next(gen) == (11, None)
    assert next(gen) == (12, None)
    gen.close()  # disconnect before the flush step
    assert len(freed) == 1  # the request's sampler chain was freed


def test_run_stream_greedy_default_is_back_compatible() -> None:
    """The public run() surface is unchanged: sampling params absent -> greedy."""
    router = _fake_router(script={0: [11, 12, 13]}, n_seq=1)
    router._ctx = {None: object()}
    router.backend = "context_pool"
    router._adapter_ptr = {}
    router._grammar = None
    router.tokenize = types.MethodType(lambda self, text: [1, 2], router)  # type: ignore[assignment]
    built: list[RouteRequest] = []

    def _build(self, request: RouteRequest) -> object | None:
        built.append(request)
        return None

    router._build_sampler = types.MethodType(_build, router)  # type: ignore[assignment]
    req = RouteRequest(prompt="p", max_tokens=3)
    assert [(tok, fin) for tok, fin in router.run_stream(req)] == [
        (11, None),
        (12, None),
        (13, None),
        (None, FINISH_LENGTH),
    ]
    assert built == [req]
    assert built[0].temperature is None


def test_run_cached_rejects_sampling_params_fail_closed() -> None:
    router = _fake_router(script={0: [11]}, n_seq=1)
    router.backend = "context_pool"
    router._adapter_ptr = {}
    with pytest.raises(ValueError, match="does not support sampling"):
        router.run_cached(
            [RouteRequest(prompt="suffix", temperature=0.9, request_id="r1")],
            [1, 2, 3],
        )
    # Greedy requests still pass the guard (fails later on missing prefix state).
    with pytest.raises((ValueError, KeyError, AttributeError)):
        router.run_cached([RouteRequest(prompt="suffix", request_id="r2")], [1, 2, 3])


def test_route_result_extends_generation_result() -> None:
    result = RouteResult(
        text='{"tool": "none"}',
        token_ids=(1, 2, 3),
        prompt_token_count=5,
        completion_token_count=3,
        finish_reason="stop",
        adapter="a",
        decision={"tool": "none"},
        validated=True,
    )
    assert isinstance(result, GenerationResult)
    assert result.adapter == "a"
    assert result.decision == {"tool": "none"}
    assert result.finish_reason == "stop"
