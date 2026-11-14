"""Adapter-isolation gate (run_qwen35_task_loras mechanism, with rig assets).

An earlier harness drove three REAL fine-tuned task LoRAs (ner-json / dolly-qa /
acrouter) on a Qwen3.5-0.8B base; none of those assets are on this rig. This gate
keeps the same MECHANISM with the available probe adapters: a batch mixing adapters must keep each request on its own
adapter's context (no bleed-through), and the same prompt under base vs each
adapter must yield distinct continuations (each adapter is actually applied).

Gate (fail-closed, deterministic greedy). The family follows the loaded backend
(``inferference.mixed_batch_gates.gate_mode``); an unknown backend fails:

  token-exact backend:
    equivalence — routed (batched + multiplexed) continuation is token-exact vs
                  its isolated single-seq baseline on the same adapter;
  logit-tolerance backend (the logits depend on the batch composition):
    determinism — the same composition twice is bit-identical;
    isolation   — change only the requests of the OTHER adapters: the target
                  group's logits stay bit-identical;
    parity/equivalence — routed vs isolated, teacher-forced. MEASURED and recorded, never
                  gated: the distribution (p50, p99, max) and the worst cases go to the
                  measurement file and the output says so;
  both:
    distinctness — the SAME prompt under base vs each adapter yields distinct
                  output (adapter actually does its job).

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
    gate_isolation,
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


_POOL = ["probe-a", "probe-b"]


def _config() -> RouterConfig:
    return RouterConfig(
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


def _requests() -> list[RouteRequest]:
    # mixed workload: all three (base, probe-a, probe-b) interleaved in one submit
    return [
        RouteRequest(
            request_id=f"r{i}",
            prompt=PROMPTS[i % len(PROMPTS)],
            adapter=("probe-a" if i % 3 == 0 else "probe-b" if i % 3 == 1 else None),
            max_tokens=16,
        )
        for i in range(6)
    ]


def _assert_distinct(router: MultiLoRARouter) -> None:
    # isolation: same prompt under base vs each adapter must differ
    probe_prompt = "The capital of France is"
    iso = router.run(
        [
            RouteRequest(request_id="iso-base", prompt=probe_prompt, adapter=None, max_tokens=16),
            RouteRequest(request_id="iso-a", prompt=probe_prompt, adapter="probe-a", max_tokens=16),
            RouteRequest(request_id="iso-b", prompt=probe_prompt, adapter="probe-b", max_tokens=16),
        ]
    )
    b, a, bb = (tuple(g.token_ids) for g in iso)
    assert b != a and b != bb, "adapter isolation broken (adapter not applied)"


def _token_exact_isolation() -> None:
    router = MultiLoRARouter(_config())
    try:
        requests = _requests()
        routed = router.run(requests)
        baselines = [
            router.run(
                [RouteRequest(request_id=r.request_id, prompt=r.prompt, adapter=r.adapter, max_tokens=r.max_tokens)]
            )[0]
            for r in requests
        ]
        for r, b in zip(routed, baselines, strict=True):
            assert tuple(r.token_ids) == tuple(b.token_ids), (
                f"mixed-batch bleed-through: r{r.request_id} ({r.adapter}) diverged from isolated baseline"
            )
        _assert_distinct(router)
    finally:
        router.close()


def _logit_isolation() -> None:
    router = TraceRouter(_config())
    try:
        requests = _requests()
        require(gate_determinism(router, requests), "determinism")
        # one target group per adapter: probe-a (0), probe-b (1), base (2)
        for target in (0, 1, 2):
            adapter = requests[target].adapter or "base"
            require(gate_isolation(router, requests, pool=_POOL, target=target), f"bleed-through, group {adapter}")
        iso = collect_isolated(router, requests)
        mixed = collect_mixed(router, requests, iso)
        report_measurement(
            "router isolation: mixed against isolated", pairs_from_traces(requests, iso, mixed, "mixed6")
        )
        _assert_distinct(router)
    finally:
        router.close()


def test_mixed_batch_adapter_isolation() -> None:
    if gate_mode() == TOKEN_EXACT:
        _token_exact_isolation()
    else:
        _logit_isolation()
