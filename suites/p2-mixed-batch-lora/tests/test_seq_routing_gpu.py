"""GPU-gated token-exact correctness gate for the P2-fork seq-routed backend.

Ports the spike's fail-closed gate (``benchmarks/project17/run_p2_mixed_batch.py`` +
``run_p2_correctness_check.py``) into the library test suite. Ground truth = each
sequence decoded ALONE on a single-adapter context (the proven uniform-LoRA path,
here the ``context_pool`` backend at batch-1). One mixed-batch fork decode must be
token-exact greedy per sequence; a benign batched-greedy FP tie-flip (present in the
no-fork path too) is classified as such, not reported as a routing failure.

Runs only under the CUDA facility (``project20-gpu-pytest``) pointed at a P2 fork
lib. On a stock lib the seq-routing capability is absent and the whole module skips
— fail-closed, never a hard failure.
"""

from __future__ import annotations

import pytest
from tests.asset_paths import asset_path

from inferference.gpu_env import gpu_env_ready
from inferference.router import Backend

_MODEL_PATH = asset_path("LLAMA_TEST_MODEL")
# The probe LoRA adapters live in the rig cache (copied from the read-only
# an earlier harness's probe runtime dir).
# MODEL_PROBE_DIR is wired by ci/runner/rig-env.sh.
_PROBE_DIR = asset_path("PROJECT20_PROBE_DIR")
_LORA_A = _PROBE_DIR / "probe-a.gguf"
_LORA_B = _PROBE_DIR / "probe-b.gguf"


def _fork_gpu_ready() -> bool:
    # Backend-agnostic: CUDA_VISIBLE_DEVICES means nothing to the Vulkan build,
    # and requiring it here SKIPS this gate on a rig that can run it.
    if not gpu_env_ready():
        return False
    if not (_MODEL_PATH.exists() and _LORA_A.exists() and _LORA_B.exists()):
        return False
    try:
        import llama_cpp

        from inferference.seq_routing import library_supports_seq_routing
    except Exception:
        return False
    return library_supports_seq_routing(getattr(llama_cpp.llama_cpp, "_lib", None))


pytestmark = pytest.mark.skipif(
    not _fork_gpu_ready(),
    reason="requires P2 fork GPU env (LLAMA_CPP_LIB_PATH -> fork lib, CUDA, model, probe adapters)",
)

_N_CTX, _N_SEQ_MAX, _MAX_TOKENS, _SEED = 2048, 4, 24, 17018


def _adapters():
    from inferference.router import AdapterSpec

    return (
        AdapterSpec(name="probe-a", gguf_path=str(_LORA_A)),
        AdapterSpec(name="probe-b", gguf_path=str(_LORA_B)),
    )


def _router(backend: Backend):
    from inferference.models import EngineConfig
    from inferference.router import MultiLoRARouter, RouterConfig

    return MultiLoRARouter(
        RouterConfig(
            engine=EngineConfig(
                model_path=str(_MODEL_PATH), n_ctx=_N_CTX, n_gpu_layers=-1, n_batch=128, seed=_SEED, backend="cuda"
            ),
            adapters=_adapters(),
            n_seq_max=_N_SEQ_MAX,
            include_base=True,
            backend=backend,
        )
    )


def _first_div(a: tuple[int, ...], b: tuple[int, ...]) -> int | None:
    """Index of the first differing token, or None when identical over the overlap."""
    pair = enumerate(zip(a, b, strict=False))
    return next((i for i, (x, y) in pair if x != y), None if len(a) == len(b) else min(len(a), len(b)))


# The mixed workload: each sequence tagged with its own adapter, including the base
# (-1) sentinel, exercised as one decode. Distinct prompts so a routing swap would
# show as a wrong-adapter continuation, not a benign near-tie.
_WORKLOAD = [
    ("probe-a", "The capital of France is"),
    ("probe-b", "In a distant galaxy,"),
    (None, "Two plus two equals"),  # base / -1 sentinel
    ("probe-a", "The opposite of hot is"),
]


def test_seq_routed_backend_is_selected_on_fork_lib() -> None:
    """auto must resolve to seq_routed on the fork lib (D2)."""
    router = _router("auto")
    try:
        assert router.backend == "seq_routed"
    finally:
        router.close()


def test_seq_routed_batch1_is_token_exact_vs_isolated_baseline() -> None:
    """E1 — the clean routing gate, free of batch-FP confound.

    Ground truth = each request decoded ALONE on a single-adapter context (context-
    pool backend, batch-1 = the proven uniform-LoRA path). The SAME request decoded
    alone through the seq_routed backend (also batch-1) must be TOKEN-EXACT greedy for
    every sequence — the ``-1`` base sentinel included. At batch-1 there is no
    batched-GEMM FP nondeterminism, so an exact match here isolates and proves correct
    per-sequence adapter routing + application, swept over probe-a / probe-b / base.
    """
    from inferference.router import RouteRequest

    reqs = [RouteRequest(prompt=p, adapter=a, max_tokens=_MAX_TOKENS) for a, p in _WORKLOAD]

    pool = _router("context_pool")
    try:
        baseline = [pool.run([r])[0].token_ids for r in reqs]
    finally:
        pool.close()

    fork = _router("seq_routed")
    try:
        assert fork.backend == "seq_routed"
        fork_b1 = [fork.run([r])[0].token_ids for r in reqs]  # batch-1 through the fork
    finally:
        fork.close()

    for i, (got, want) in enumerate(zip(fork_b1, baseline, strict=True)):
        adapter = _WORKLOAD[i][0] or "base(-1)"
        assert got == want, f"seq {i} ({adapter}) not token-exact vs isolated baseline: routing bug"


def test_mixed_batch_decode_routes_without_leakage() -> None:
    """E2/E5 — one mixed fork decode carries all adapters, no cross-seq leakage.

    A single ``llama_decode`` wave mixes probe-a / probe-b / base. Divergence from the
    batch-1 baseline is EXPECTED and benign (base-GEMM FP nondeterminism at larger
    batch — present in the no-fork router too; see 21-P2-THROUGHPUT). The routing
    invariant that MUST hold: no sequence takes another adapter's continuation. Reuses
    the spike's classifier — a divergence is a mis-route only if it matches a DIFFERENT
    adapter's isolated output; otherwise it is batched-greedy FP, not a routing fault.
    """
    from inferference.router import RouteRequest

    reqs = [RouteRequest(prompt=p, adapter=a, max_tokens=_MAX_TOKENS) for a, p in _WORKLOAD]
    other = {"probe-a": "probe-b", "probe-b": "probe-a"}

    pool = _router("context_pool")
    try:
        baseline = [pool.run([r])[0].token_ids for r in reqs]
        crossed = {
            i: pool.run([RouteRequest(prompt=w[1], adapter=other[w[0]], max_tokens=_MAX_TOKENS)])[0].token_ids
            for i, w in enumerate(_WORKLOAD)
            if w[0] in other
        }
    finally:
        pool.close()

    fork = _router("seq_routed")
    try:
        fork_out = [res.token_ids for res in fork.run(reqs)]
    finally:
        fork.close()

    flips: list[str] = []
    for i, (out, base) in enumerate(zip(fork_out, baseline, strict=True)):
        cross = crossed.get(i)
        # Leakage is only detectable where the two adapters actually diverge for this
        # prompt (the tiny probe adapters produce identical output for some prompts).
        if cross is not None and cross != base:
            assert out != cross, f"seq {i} routed to the wrong adapter (leakage)"
        if out != base:
            flips.append(f"seq{i}@{_first_div(out, base)}")
    if flips:
        # Benign per 21-P2-THROUGHPUT: batch-FP flips, proven by the batch-1 exact gate.
        print(f"[gate] benign batched-greedy FP flips (not routing bugs): {', '.join(flips)}")


def test_no_cross_sequence_adapter_leakage() -> None:
    """E5 — the two adapters produce distinct outputs, and base differs from both."""
    from inferference.router import RouteRequest

    prompt = "The capital of France is"
    fork = _router("seq_routed")
    try:
        outs = fork.run(
            [
                RouteRequest(prompt=prompt, adapter="probe-a", max_tokens=_MAX_TOKENS),
                RouteRequest(prompt=prompt, adapter="probe-b", max_tokens=_MAX_TOKENS),
                RouteRequest(prompt=prompt, adapter=None, max_tokens=_MAX_TOKENS),
            ]
        )
    finally:
        fork.close()
    a, b, base = (o.token_ids for o in outs)
    # Same prompt, three routings: adapters must actually differentiate the decode.
    assert a != b, "probe-a and probe-b produced identical output — adapters not applied per-seq"
    assert a != base and b != base, "an adapter output matched the base — adapter not applied"
