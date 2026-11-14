"""Context-pool router gates (run_context_pool_router validation half).

The validation half; throughput is measured by the benchkit runner. These are
fail-closed gates over the library's :class:`inferference.router.MultiLoRARouter` (``context_pool``
backend). The gate family follows the loaded backend
(``inferference.mixed_batch_gates.gate_mode``); an unknown backend fails.

  gate 1 — equivalence: every request's routed (batched, multiplexed)
           continuation against its isolated single-seq baseline on the same
           adapter. A token-exact backend: token-exact (batch-1 greedy ground
           truth). A logit-tolerance backend: teacher-forced, per-step log-probability
           MEASURED and recorded, never gated (the output says so), and the same
           composition twice bit-identical;
  gate 2 — routing distinctiveness: the SAME prompt under base / probe-a /
           probe-b yields distinct continuations (adapters actually applied).

Known-flip rule (token-exact backend): a batched-greedy FP near-tie flip
is classified and reported, never failed; only a real equivalence break fails.

GPU-only: skips (never fails) when the env lacks GPU-offload lib / model /
probe adapters.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.asset_paths import asset_path

from inferference.benchkit.trace_router import TraceRouter
from inferference.gpu_env import GPU_BACKEND_LABEL, gpu_env_ready
from inferference.mixed_batch_gates import (
    TOKEN_EXACT,
    collect_isolated,
    collect_mixed,
    gate_determinism,
    gate_mode,
    pairs_from_traces,
    report_measurement,
    require,
)
from inferference.models import EngineConfig
from inferference.router import AdapterSpec, MultiLoRARouter, RouterConfig, RouteRequest

_REPO = Path(__file__).resolve().parents[3]
_MODEL = asset_path("MODEL_DEFAULT")
_PROBE_DIR = asset_path("MODEL_PROBE_DIR")
_LORA_A = _PROBE_DIR / "probe-a.gguf"
_LORA_B = _PROBE_DIR / "probe-b.gguf"

PROMPTS = [
    "The capital of France is",
    "In a distant galaxy,",
    "def add(a, b):",
    "The three primary colors are",
    "Once upon a time",
    "The chemical symbol for gold is",
]


def _ready() -> bool:
    # Backend-agnostic: a vendor device variable means nothing to a build for another backend,
    # and requiring it here SKIPS this gate on a rig that can run it.
    if not gpu_env_ready():
        return False
    if not (_MODEL.exists() and _LORA_A.exists() and _LORA_B.exists()):
        return False
    try:
        import llama_cpp
    except Exception:
        return False
    return bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())


pytestmark = pytest.mark.skipif(
    not _ready(),
    reason="requires GPU-offload lib + MODEL_DEFAULT + probe adapters (GPU env)",
)


def _make_router(cls: type[MultiLoRARouter] = MultiLoRARouter) -> MultiLoRARouter:
    return cls(
        RouterConfig(
            engine=EngineConfig(
                model_path=str(_MODEL), n_ctx=2048, n_gpu_layers=-1, n_batch=128, seed=17018, backend=GPU_BACKEND_LABEL
            ),
            adapters=(
                AdapterSpec(name="probe-a", gguf_path=str(_LORA_A)),
                AdapterSpec(name="probe-b", gguf_path=str(_LORA_B)),
            ),
            n_seq_max=8,
            include_base=True,
            backend="context_pool",
        )
    )


def _requests() -> list[RouteRequest]:
    return [
        RouteRequest(
            request_id=f"r{i}",
            prompt=PROMPTS[i % len(PROMPTS)],
            adapter=("probe-a" if i % 3 == 0 else "probe-b" if i % 3 == 1 else None),
            max_tokens=16,
        )
        for i in range(9)
    ]


def _assert_distinct(router: MultiLoRARouter) -> None:
    # gate 2: distinctiveness (same prompt, three adapters)
    probe_prompt = "The capital of France is"
    distinct = router.run(
        [
            RouteRequest(request_id="d-base", prompt=probe_prompt, adapter=None, max_tokens=16),
            RouteRequest(request_id="d-a", prompt=probe_prompt, adapter="probe-a", max_tokens=16),
            RouteRequest(request_id="d-b", prompt=probe_prompt, adapter="probe-b", max_tokens=16),
        ]
    )
    d_base, d_a, d_b = (tuple(g.token_ids) for g in distinct)
    assert d_base != d_a and d_a != d_b and d_base != d_b, "routing distinctiveness broken"


def _token_exact_equivalence() -> None:
    router = _make_router()
    try:
        requests = _requests()
        routed = router.run(requests)
        # isolated single-seq baselines (batch-1, same adapter context)
        baselines = [
            router.run(
                [RouteRequest(request_id=r.request_id, prompt=r.prompt, adapter=r.adapter, max_tokens=r.max_tokens)]
            )[0]
            for r in requests
        ]

        flips: list[str] = []
        mismatches = 0
        for r, b in zip(routed, baselines, strict=True):
            if tuple(r.token_ids) != tuple(b.token_ids):
                first = next(
                    (i for i, (x, y) in enumerate(zip(r.token_ids, b.token_ids, strict=False)) if x != y),
                    None,
                )
                flips.append(f"r{r.request_id}@{first}:{r.adapter}")
                mismatches += 1
        # batch-1 ground truth + known-flip reporting: a divergence is reported;
        # the GATE fails only on a real mismatch (the routed path must equal the
        # isolated baseline; benign FP flips are classified, never silently kept)
        assert mismatches == 0, f"equivalence gate broken: {flips}"

        _assert_distinct(router)
    finally:
        router.close()


def _logit_equivalence() -> None:
    router = _make_router(TraceRouter)
    assert isinstance(router, TraceRouter)
    try:
        requests = _requests()
        require(gate_determinism(router, requests), "determinism")
        iso = collect_isolated(router, requests)
        mixed = collect_mixed(router, requests, iso)
        report_measurement(
            "router equivalence: mixed against isolated", pairs_from_traces(requests, iso, mixed, "mixed9")
        )
        _assert_distinct(router)
    finally:
        router.close()


def test_router_equivalence_and_distinctiveness() -> None:
    if gate_mode() == TOKEN_EXACT:
        _token_exact_equivalence()
    else:
        _logit_equivalence()
