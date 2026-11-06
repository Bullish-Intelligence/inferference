"""Cached shared-prefix restore validation (run_context_pool_router_cached gates).

Composes the per-sequence KV reuse into the context-pool router: a shared prefix
is prefilled ONCE per adapter, its seq blob captured, and restored into every seq
slot so each request only decodes its own suffix.

Gate (fail-closed, deterministic greedy), swept over base / probe-a / probe-b:
  cached-restore continuation is token-exact vs the full (prefix+suffix)
  baseline on the same adapter — for EVERY sequence in the wave (also proves
  neighbour isolation across restored slots). Adapter-specificity is respected
  implicitly: each adapter captures/restores in its OWN context.

GPU-only: skips (never fails) when the env lacks CUDA-offload lib / model /
probe adapters.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.asset_paths import asset_path

from inferference.gpu_env import gpu_env_ready
from inferference.models import EngineConfig
from inferference.prefix_cache_live import LlamaSeqStateBridge
from inferference.router import AdapterSpec, MultiLoRARouter, RouterConfig, RouteRequest

_REPO = Path(__file__).resolve().parents[3]
_MODEL = asset_path("MODEL_ORNITH")
_PROBE_DIR = asset_path("MODEL_PROBE_DIR")
_LORA_A = _PROBE_DIR / "probe-a.gguf"
_LORA_B = _PROBE_DIR / "probe-b.gguf"

# A long shared prefix (past the ~192-token restore break-even) so restore is
# expected to beat cold re-prefill. Deterministic, no external data.
PREFIX_TEXT = (
    "You are a routing assistant for a tool-calling agent fleet. Follow these "
    "rules exactly. Always return a single well-formed decision. Consider the "
    "user's intent, the available tools, and the conversation so far. Do not "
    "invent tools that were not provided. Prefer the most specific tool. When no "
    "tool applies, say so plainly. Be concise, deterministic, and consistent. "
) * 3

SUFFIXES = [
    " User request: book a flight to Paris. Decision:",
    " User request: what is 2 plus 2? Decision:",
    " User request: summarize the attached document. Decision:",
    " User request: turn off the living room lights. Decision:",
]


def _ready() -> bool:
    # Backend-agnostic: CUDA_VISIBLE_DEVICES means nothing to the Vulkan build,
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
    reason="requires CUDA-offload lib + Ornith model + probe adapters (GPU env)",
)


def test_cached_shared_prefix_restore_is_token_exact() -> None:
    router = MultiLoRARouter(
        RouterConfig(
            engine=EngineConfig(
                model_path=str(_MODEL), n_ctx=4096, n_gpu_layers=-1, n_batch=128, seed=17018, backend="cuda"
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
    bridge = LlamaSeqStateBridge(n_batch=128, n_vocab=router.n_vocab)
    try:
        prefix = router.tokenize(PREFIX_TEXT)
        n_ctx_seq = 4096 // 8  # per-seq budget
        assert len(prefix) + max(len(s) for s in SUFFIXES) + 8 < n_ctx_seq, "budget too small"

        all_ok = True
        failures: list[str] = []
        for adapter in (None, "probe-a", "probe-b"):
            name = "base" if adapter is None else adapter
            ctx = router._ctx[adapter]
            # capture the shared prefix once on this adapter's context
            bridge.decode_tokens(ctx, prefix, 0, 0)
            blob = bridge.capture_seq_state(ctx, 0)
            router._seq_rm(ctx, 0)

            suffix_lists = [router.tokenize(s) for s in SUFFIXES]
            # cold baselines: full prefix+suffix, one seq each
            cold: list[tuple[int, ...]] = []
            for i, suffix in enumerate(suffix_lists):
                router._seq_rm(ctx, i)
                bridge.decode_tokens(ctx, prefix + suffix, i, 0)
                cold.append((bridge.last_token(ctx),))
            # cached: restore blob into each slot, decode only the suffix
            cached: list[tuple[int, ...]] = []
            for i, suffix in enumerate(suffix_lists):
                router._seq_rm(ctx, i)
                code = bridge.restore_seq_state(ctx, blob, i)
                assert code > 0, f"{name}: set_data returned 0 (n_seq_max/portability reject)"
                bridge.decode_tokens(ctx, suffix, i, len(prefix))
                cached.append((bridge.last_token(ctx),))
            for i, (c, k) in enumerate(zip(cold, cached, strict=True)):
                if c != k:
                    all_ok = False
                    failures.append(f"{name}-seq{i}: cold={c} cached={k}")
        assert all_ok, f"cached restore equivalence broken: {failures}"
    finally:
        router.close()


def test_run_cached_api_equivalence_and_shape() -> None:
    """The public ``run_cached`` composition keeps the token-exact gate.

    Prefill-once per adapter + restore-into-every-slot + suffix-only decode,
    and the fail-closed equivalence: every cached continuation is token-exact
    vs the isolated batch-1 full-prompt baseline, for EVERY request/adapter.
    """
    router = MultiLoRARouter(
        RouterConfig(
            engine=EngineConfig(
                model_path=str(_MODEL), n_ctx=4096, n_gpu_layers=-1, n_batch=128, seed=17018, backend="cuda"
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
    try:
        prefix = router.tokenize(PREFIX_TEXT)
        n_ctx_seq = 4096 // 8  # per-seq budget
        assert len(prefix) + max(len(router.tokenize_suffix(s)) for s in SUFFIXES) + 24 < n_ctx_seq

        requests = [
            RouteRequest(
                request_id=f"{'base' if adapter is None else adapter}-{i}",
                prompt=SUFFIXES[i],
                adapter=adapter,
                max_tokens=24,
            )
            for adapter in (None, "probe-a", "probe-b")
            for i in range(len(SUFFIXES))
        ]
        res = router.run_cached(requests, prefix)

        assert len(res.cached) == len(res.cold) == len(res.baselines) == len(requests)
        assert res.cached_wall_ns > 0 and res.cold_wall_ns > 0
        for i, r in enumerate(requests):
            cached_toks = list(res.cached[i].token_ids)
            base_toks = list(res.baselines[i].token_ids)
            assert cached_toks, f"{r.request_id}: empty cached continuation"
            assert cached_toks == base_toks[: len(cached_toks)], (
                f"{r.request_id}: cached continuation diverges from the isolated baseline"
            )
            assert res.cold[i].request_id == r.request_id
            assert res.cached[i].adapter == r.adapter
    finally:
        router.close()
