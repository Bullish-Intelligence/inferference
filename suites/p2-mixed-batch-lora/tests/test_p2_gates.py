"""P2 fork gate semantics as pytest — the run_p2_mixed_batch / run_p2_correctness_check
gates expressed against the
ported library's :class:`inferference.router.MultiLoRARouter`.

  * ``run_p2_mixed_batch`` semantics — one mixed-batch routed decode (4 sequences:
    probe-a / probe-b / base(-1) / probe-a) must be TOKEN-EXACT against each
    sequence's isolated single-adapter baseline (batch-1, the proven uniform-LoRA
    path). Verdict GO iff every sequence matches AND the outputs are not trivially
    uniform.
  * ``run_p2_correctness_check`` semantics — sweep S in {4, 8}: compare the
    seq_routed / context_pool (router) / sequential modes against the batch-1
    baseline and report match-count + first-divergence per request. A divergence
    is classified by the spike's rule: it is a routing bug only if it matches a
    DIFFERENT adapter's isolated output; otherwise it is batched-greedy FP
    nondeterminism and is REPORTED, never failed.

On a logit-tolerance backend (the logits depend on the batch composition, see
``inferference.mixed_batch_gates``) the token-exact mixed-batch gate cannot hold. The
mixed-batch verdict there is three exact gates: the same composition twice
bit-identical; the other sequences' adapters changed with the tokens held fixed leave each
target's logits bit-identical (bleed-through, and each neighbour must react to its adapter);
and the isolated ``seq_routed`` run bit-identical to the isolated ``context_pool`` run
(parity of the adapter path). The mixed batch against the isolated runs is MEASURED and
recorded, never gated (the output says so). An unknown backend fails.

GPU-gated like the reference gates (GPU env, P2 fork lib, model + probe
adapters); on a stock lib the whole module skips — fail-closed, never a hard
failure. Token-exact checks use batch-1 ground truth + known-flip reporting.
"""

from __future__ import annotations

import pytest
from tests.asset_paths import asset_path

from inferference.benchkit.trace_router import TraceRouter
from inferference.gpu_env import GPU_BACKEND_LABEL, gpu_env_ready
from inferference.mixed_batch_gates import (
    TOKEN_EXACT,
    check_parity,
    collect_isolated,
    collect_isolated_forced,
    collect_mixed,
    gate_determinism,
    gate_isolation,
    gate_mode,
    pairs_from_traces,
    report_measurement,
    require,
)
from inferference.router import Backend

_MODEL_PATH = asset_path("MODEL_DEFAULT")
_PROBE_DIR = asset_path("MODEL_PROBE_DIR")
_LORA_A = _PROBE_DIR / "probe-a.gguf"
_LORA_B = _PROBE_DIR / "probe-b.gguf"


def _fork_gpu_ready() -> bool:
    # Backend-agnostic: a vendor device variable means nothing to a build for another backend,
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
    reason=(
        "requires P2 fork GPU env (LLAMA_CPP_LIB_PATH -> fork lib, a GPU device, "
        "MODEL_DEFAULT, MODEL_PROBE_DIR adapters)"
    ),
)

_N_CTX, _N_SEQ_MAX, _MAX_TOKENS, _SEED = 2048, 8, 32, 17018
_PROMPTS = [
    "The capital of France is",
    "In a distant galaxy,",
    "def add(a, b):",
    "Once upon a time",
    "The three primary colors are",
    "Water boils at",
    "The opposite of hot is",
    "To make tea you first",
]
_POOL = ["probe-a", "probe-b"]


def _adapters():
    from inferference.router import AdapterSpec

    return (
        AdapterSpec(name="probe-a", gguf_path=str(_LORA_A)),
        AdapterSpec(name="probe-b", gguf_path=str(_LORA_B)),
    )


def _router(backend: Backend, cls: type | None = None):
    from inferference.models import EngineConfig
    from inferference.router import MultiLoRARouter, RouterConfig

    return (cls or MultiLoRARouter)(
        RouterConfig(
            engine=EngineConfig(
                model_path=str(_MODEL_PATH),
                n_ctx=_N_CTX,
                n_gpu_layers=-1,
                n_batch=128,
                seed=_SEED,
                backend=GPU_BACKEND_LABEL,
            ),
            adapters=_adapters(),
            n_seq_max=_N_SEQ_MAX,
            include_base=True,
            backend=backend,
        )
    )


def _first_div(a: tuple[int, ...], b: tuple[int, ...]) -> int | None:
    pair = enumerate(zip(a, b, strict=False))
    return next((i for i, (x, y) in pair if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def _summarize(
    mode: dict[int, tuple[int, ...]],
    base: dict[int, tuple[int, ...]],
    crossed: dict[int, tuple[int, ...]],
) -> tuple[int, list[str]]:
    """Match-count + first-divergence vs the batch-1 baseline; classify each
    divergence as a routing bug (matches the OTHER adapter) or batched-greedy FP."""
    match_count = sum(1 for i in base if mode[i] == base[i])
    divs: list[str] = []
    for i in base:
        if mode[i] != base[i]:
            d = _first_div(mode[i], base[i])
            if mode[i] == crossed[i] and crossed[i] != base[i]:
                divs.append(f"seq{i}@{d}:ROUTE-BUG")
            else:
                divs.append(f"seq{i}@{d}:fp")
    return match_count, divs


def _token_exact_mixed_batch_verdict() -> None:
    """run_p2_mixed_batch semantics: one mixed-batch routed decode, token-exact.

    The four-sequence mixed wave (probe-a / probe-b / base(-1) / probe-a) decoded
    through the fork's seq_routed backend must equal each request's isolated
    batch-1 baseline. Verdict GO iff all match AND outputs are distinct (the wave
    is not trivially uniform).
    """
    from inferference.router import BASE, RouteRequest

    prompts = _PROMPTS[:4]
    adapters: list[str | None] = ["probe-a", "probe-b", BASE, "probe-a"]
    requests = [RouteRequest(prompt=prompts[i], adapter=adapters[i], max_tokens=_MAX_TOKENS) for i in range(4)]

    pool = _router("context_pool")
    try:
        baselines = [pool.run([r])[0].token_ids for r in requests]
    finally:
        pool.close()

    fork = _router("seq_routed")
    try:
        assert fork.backend == "seq_routed"
        routed = [res.token_ids for res in fork.run(requests)]
    finally:
        fork.close()

    rows = []
    all_match = True
    for i, (got, want) in enumerate(zip(routed, baselines, strict=True)):
        match = got == want
        all_match = all_match and match
        rows.append(
            {
                "rid": f"r{i}",
                "adapter": "base" if adapters[i] is BASE else adapters[i],
                "match": match,
                "first_div": _first_div(got, want),
            }
        )
    distinct = len({tuple(r) for r in routed}) > 1

    for row in rows:
        print(f"[p2] {row['rid']} ({row['adapter']}): match={row['match']} first_div={row['first_div']}")
    assert distinct, "the mixed batch is trivially uniform — adapters not applied per-seq"
    assert all_match, f"mixed-batch routed decode not token-exact vs isolated baselines: {rows}"
    print("[p2] verdict GO — routed == isolated baseline, outputs distinct")


def _logit_mixed_batch_verdict() -> None:
    """The mixed-batch verdict on a logit-tolerance backend: three exact gates, all must hold."""
    from inferference.router import BASE, RouteRequest

    prompts = _PROMPTS[:4]
    adapters: list[str | None] = ["probe-a", "probe-b", BASE, "probe-a"]
    requests = [RouteRequest(prompt=prompts[i], adapter=adapters[i], max_tokens=_MAX_TOKENS) for i in range(4)]

    pool = _router("context_pool", TraceRouter)
    try:
        iso = collect_isolated(pool, requests)  # the reference: batch-1, clean state, proven adapter path
    finally:
        pool.close()

    fork = _router("seq_routed", TraceRouter)
    try:
        assert fork.backend == "seq_routed"
        assert isinstance(fork, TraceRouter)
        require(gate_determinism(fork, requests), "determinism")
        # bleed-through: each sequence is the target once, the others change adapter
        for target in range(len(requests)):
            require(gate_isolation(fork, requests, pool=_POOL, target=target), f"bleed-through, target {target}")
        mixed = collect_mixed(fork, requests, iso)
        report_measurement(
            "p2 mixed batch: seq_routed mixed against context_pool alone",
            pairs_from_traces(requests, iso, mixed, "mixed4"),
        )
        # adapter-path parity: one request alone, seq_routed against context_pool, bit-identical
        fork_iso = collect_isolated_forced(fork, requests, iso)
        parity = [
            f for r, a, b in zip(requests, iso, fork_iso, strict=True) for f in check_parity(f"{r.adapter}", a, b)
        ]
        require(parity, "adapter-path parity: seq_routed alone differs from context_pool alone")
        free = [t.tokens for t in collect_mixed(fork, requests)]
    finally:
        fork.close()
    assert len(set(free)) > 1, "the mixed batch is trivially uniform — adapters not applied per-seq"
    print("[p2] verdict GO — determinism, bleed-through and parity hold; outputs distinct; equivalence measured")


def test_p2_mixed_batch_verdict_go() -> None:
    if gate_mode() == TOKEN_EXACT:
        _token_exact_mixed_batch_verdict()
    else:
        _logit_mixed_batch_verdict()


def test_p2_correctness_check_modes_sweep() -> None:
    """run_p2_correctness_check semantics: S in {4,8} cross-mode comparison.

    Ground truth = batch-1 isolated baseline. Compare seq_routed (fork),
    context_pool (router), and sequential (one-at-a-time seq_routed) modes;
    report match-count + first divergence. Any mode divergence that equals a
    DIFFERENT adapter's isolated output is a routing bug (hard fail); anything
    else is batched-greedy FP nondeterminism and is reported as a known flip.
    """
    from inferference.router import RouteRequest

    flips: list[str] = []
    for S in (4, 8):
        reqs = [
            RouteRequest(
                prompt=_PROMPTS[i % len(_PROMPTS)],
                adapter=_POOL[i % len(_POOL)],
                max_tokens=_MAX_TOKENS,
            )
            for i in range(S)
        ]

        pool = _router("context_pool")
        try:
            # batch-1 isolated baselines (ground truth) + batched context_pool mode
            base = {i: pool.run([r])[0].token_ids for i, r in enumerate(reqs)}
            router_out = {i: res.token_ids for i, res in enumerate(pool.run(reqs))}
            # the other adapter's isolated output, to classify a divergence
            crossed = {
                i: pool.run(
                    [
                        RouteRequest(
                            prompt=_PROMPTS[i % len(_PROMPTS)], adapter=_POOL[(i + 1) % 2], max_tokens=_MAX_TOKENS
                        )
                    ]
                )[0].token_ids
                for i in range(S)
            }
        finally:
            pool.close()

        fork = _router("seq_routed")
        try:
            fork_out = {i: res.token_ids for i, res in enumerate(fork.run(reqs))}
            sequential = {}
            for i, r in enumerate(reqs):
                sequential[i] = fork.run([r])[0].token_ids
        finally:
            fork.close()

        f_match, f_divs = _summarize(fork_out, base, crossed)
        r_match, r_divs = _summarize(router_out, base, crossed)
        s_match, s_divs = _summarize(sequential, base, crossed)
        print(
            f"[chk] S={S}: fork {f_match}/{S} router {r_match}/{S} sequential {s_match}/{S} match baseline",
            flush=True,
        )
        bugs = [d for d in f_divs + r_divs + s_divs if "ROUTE-BUG" in d]
        flips.extend(f"S{S}:{d}" for d in f_divs + r_divs + s_divs if d.endswith(":fp"))
        assert not bugs, f"S={S} routing bugs: {bugs}"

    if flips:
        # Benign per 21-P2-THROUGHPUT: batched-greedy FP near-tie flips, proven by
        # the batch-1 exact gates. Reported, never failed.
        print(f"[gate] benign batched-greedy FP flips (not routing bugs): {', '.join(flips)}")
