"""Cached shared-prefix restore validation (run_context_pool_router_cached gates).

Composes the per-sequence KV reuse into the context-pool router: a shared prefix
is prefilled ONCE per adapter, its seq blob captured, and restored into every seq
slot so each request only decodes its own suffix.

Gate (fail-closed, deterministic greedy), swept over base / probe-a / probe-b. The
family follows the loaded backend (``inferference.mixed_batch_gates.gate_mode``);
an unknown backend fails.

  token-exact backend: the cached-restore continuation is token-exact vs the full
  (prefix+suffix) baseline on the same adapter — for EVERY sequence in the wave
  (also proves neighbour isolation across restored slots).

  logit-tolerance backend: a restore and a cold prefill sit in different KV cells
  with different neighbours, so their logits differ by batch-composition noise. That
  difference is MEASURED and recorded, never gated (the output says so). The gates are
  exact: the restore, run twice from a clean state, is bit-identical, and a replay of
  the reference leg changes nothing.

Adapter-specificity is respected implicitly: each adapter captures/restores in its
OWN context.

GPU-only: skips (never fails) when the env lacks GPU-offload lib / model /
probe adapters.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.asset_paths import asset_path

from inferference.benchkit.trace_router import TraceRouter, cached_legs, cached_state_pairs, pairs_from_state
from inferference.gpu_env import GPU_BACKEND_LABEL, gpu_env_ready
from inferference.mixed_batch_gates import (
    TOKEN_EXACT,
    check_exact,
    gate_mode,
    report_measurement,
    require,
)
from inferference.models import EngineConfig
from inferference.prefix_cache_live import LlamaSeqStateBridge
from inferference.router import AdapterSpec, MultiLoRARouter, RouterConfig, RouteRequest

_REPO = Path(__file__).resolve().parents[3]
_MODEL = asset_path("MODEL_DEFAULT")
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


def _config() -> RouterConfig:
    return RouterConfig(
        engine=EngineConfig(
            model_path=str(_MODEL), n_ctx=4096, n_gpu_layers=-1, n_batch=128, seed=17018, backend=GPU_BACKEND_LABEL
        ),
        adapters=(
            AdapterSpec(name="probe-a", gguf_path=str(_LORA_A)),
            AdapterSpec(name="probe-b", gguf_path=str(_LORA_B)),
        ),
        n_seq_max=8,
        include_base=True,
        backend="context_pool",
    )


def _token_exact_restore() -> None:
    router = MultiLoRARouter(_config())
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


def _logit_restore() -> None:
    router = TraceRouter(_config())
    bridge = LlamaSeqStateBridge(n_batch=128, n_vocab=router.n_vocab, native=router.C)
    try:
        prefix = router.tokenize(PREFIX_TEXT)
        n_ctx_seq = 4096 // 8  # per-seq budget
        suffix_lists = [router.tokenize(s) for s in SUFFIXES]
        assert len(prefix) + max(len(s) for s in suffix_lists) + 8 < n_ctx_seq, "budget too small"

        findings: list[str] = []
        measured = []
        for adapter in (None, "probe-a", "probe-b"):
            name = "base" if adapter is None else adapter
            first = cached_state_pairs(router, bridge, adapter, prefix, suffix_lists)
            again = cached_state_pairs(router, bridge, adapter, prefix, suffix_lists)
            for i, ((cold1, cached1), (cold2, cached2)) in enumerate(zip(first, again, strict=True)):
                findings += check_exact(f"cached restore {name} seq{i}: cold run twice", cold1, cold2)
                findings += check_exact(f"cached restore {name} seq{i}: cached run twice", cached1, cached2)
            measured += pairs_from_state(adapter, first, f"state/{name}")
        require(findings, "cached restore is not deterministic")
        report_measurement("cached restore: cached against cold", measured)
    finally:
        router.close()


def test_cached_shared_prefix_restore_is_token_exact() -> None:
    if gate_mode() == TOKEN_EXACT:
        _token_exact_restore()
    else:
        _logit_restore()


def test_run_cached_api_equivalence_and_shape() -> None:
    """The public ``run_cached`` composition keeps its equivalence gate.

    Prefill-once per adapter + restore-into-every-slot + suffix-only decode,
    and the fail-closed equivalence of every cached continuation against the
    isolated batch-1 full-prompt baseline, for EVERY request/adapter. A token-exact
    backend: token-exact. A logit-tolerance backend: the exact gates of ``cached_legs``
    (the reference leg replays bit-identical, the cached leg repeats bit-identical),
    and the difference to the baseline MEASURED, never gated. The shape checks hold on both.
    """
    mode = gate_mode()
    router = TraceRouter(_config()) if mode != TOKEN_EXACT else MultiLoRARouter(_config())
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
            if mode == TOKEN_EXACT:
                assert cached_toks == base_toks[: len(cached_toks)], (
                    f"{r.request_id}: cached continuation diverges from the isolated baseline"
                )
            assert res.cold[i].request_id == r.request_id
            assert res.cached[i].adapter == r.adapter
        if mode != TOKEN_EXACT:
            assert isinstance(router, TraceRouter)
            findings, pairs = cached_legs(router, requests, prefix)
            require(findings, "cached restore legs are not deterministic")
            report_measurement("run_cached: cached and cold against the baseline", pairs)
    finally:
        router.close()
