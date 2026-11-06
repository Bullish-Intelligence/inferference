"""P2 composition-determinism control: does a sequence depend on its batch-mates?

**Question.** Does a sequence's output depend on who shares its batch, and if so,
does that depend on the fork's routing at all?

A measured `max_abs_delta` of 0.17 between a sequence decoded alone and the
same sequence decoded inside a mixed-adapter batch motivated the question, but
that measurement carried **no adapter-presence control**: every arm held a
LoRA. This runner adds that control and four more arms.

Five arms, one prompt set, greedy, alone vs inside a batch:

  A0   no adapter at all, no pool registered  — THE CONTROL
  A1   identity adapter on every sequence     — "the seq path is active"
  A2   one real adapter on every sequence     — the seq path without mixing
  A2S  the same adapter, STOCK uniform path   — the fork-free LoRA control
  A3   two real adapters, alternating         — the shipping configuration
  A4   `-1` seats interleaved with adapter A  — the realistic mixed workload

A2S is added because A0 alone cannot close the question: A0 controls for "is a
LoRA present", not for "does the FORK route it". A2S pins the same adapter with
the stock `llama_set_adapters_lora`, so no seq pool, no mask and no fork routing
code reach the graph. A2 ≈ A2S therefore
means the fork's seq path adds nothing to the effect.

**Method.** For each arm the *reference* is sequence `i` decoded ALONE on its own
seat, greedy, `--max-tokens` steps. The *batched* pass decodes sequences
`0..S-1` together and **forces** each sequence to follow its own reference
tokens, so the comparison stays aligned after a flip. Logits are captured at
every step in both passes. Reported per arm and per concurrency:

  max_abs_delta   max |logit_batched - logit_alone| over steps x vocab
  flips           steps where argmax(batched) != the reference token
  fraction_over   fraction of steps whose per-step max delta exceeds the
                  threshold max(3 * jitter, 1e-4) (the llgym metric shape)
  jitter          max |logit_rep - logit_rep0| across repeats of the batched pass
  route_bugs      a flip whose token matches the CROSSED route's isolated output
                  (the oracle of ``p2_throughput.py:196-199``, kept verbatim)
  oracle_separation
                  max |logit_correct_route - logit_crossed_route| with BOTH
                  decoded alone — the oracle's discriminating power. A route-bug
                  is counted as informative only when this exceeds the delta
                  being classified; two routes closer together than the noise
                  cannot be told apart by a token match.

**Fail-closed harness check.** At concurrency 1 the batched pass IS the alone
pass, so every arm must report exactly `0.0`. A non-zero value means the harness
is wrong and no arm may be believed.

**Shape sweep.** With the adapter set held fixed, `--n-ubatch` is varied at the
top concurrency. If the delta tracks the ubatch shape and ignores the adapter
count, composition — not routing — is the source.

This runner **sits beside** ``p2_throughput.py`` and never modifies it:
``p2_throughput`` is CI-wired at ``ci/workflows/bench.yml``. This runner
touches no code under ``src/inferference/``.

GPU-only: `CUDA_VISIBLE_DEVICES=0`, `n_gpu_layers=-1`, `LLAMA_CPP_LIB_PATH`
pinned to a fork lib. The GPU is asserted by measurement (VRAM floor and a
decode-throughput floor), never by reading a log line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from inferference.fingerprint import LlamaEngineFingerprint, register_artifact
from inferference.seq_routing import NO_ADAPTER, SeqRoutingBinding, library_supports_seq_routing

from ..env import gpu_snapshot, require_ci_device
from ..runenv import RunContext

REPO = Path(__file__).resolve().parents[3]

# The p2_throughput prompt set (p2_throughput.py:37-46) PLUS the three
# attention-sensitive probes drawn from a parity suite. These three are
# load-bearing: the probe adapters perturb layer-3 k/v/o_proj, and without
# coreference prompts the flips do not reproduce.
SENSITIVE_PROMPTS = [
    "Alice is looking at Bob. Bob is looking at ",
    "The key insight of the paper is that ",
    "In a surprising turn of events, the ",
]
THROUGHPUT_PROMPTS = [
    "The capital of France is",
    "In a distant galaxy,",
    "def add(a, b):",
    "Once upon a time",
    "The three primary colors are",
    "Water boils at",
    "The opposite of hot is",
    "To make tea you first",
]
PROMPTS = SENSITIVE_PROMPTS + THROUGHPUT_PROMPTS

# Route = the adapter key a seat carries, or None for the -1 (base) sentinel.
Route = Callable[[int], str | None]


class Arm:
    """One experimental arm: an ordered pool plus a seat -> route mapping.

    ``pool`` is the ordered adapter pool registered with
    ``llama_set_seq_adapters``. An empty pool means the call is never made: the
    context is a plain base-model context with no fork routing state at all.
    That is arm A0, and it is the control the whole spike exists for.

    ``crossed`` is the route used for the route-bug oracle — the *wrong* route
    for that seat. ``None`` for the whole arm means the oracle does not apply.
    """

    def __init__(
        self,
        name: str,
        pool: tuple[str, ...],
        route: Route,
        crossed: Route | None,
        note: str,
        pin: str | None = None,
    ) -> None:
        self.name = name
        self.pool = pool
        self.route = route
        self.crossed = crossed
        self.note = note
        # ``pin`` installs an adapter with the STOCK context-scope call
        # ``llama_set_adapters_lora`` — no seq pool, no mask, no fork routing
        # code on the graph. That is the fork-free LoRA control.
        self.pin = pin


ARMS: dict[str, Arm] = {
    "A0": Arm("A0", (), lambda i: None, None, "no adapter at all, no pool registered — the control"),
    "A1": Arm("A1", ("identity",), lambda i: "identity", lambda i: None, "identity adapter on every sequence"),
    "A2": Arm("A2", ("a",), lambda i: "a", lambda i: None, "one real adapter on every sequence (fork seq path)"),
    # The smoke run showed A0 << A2, which reads as "the seq LoRA path is
    # composition-sensitive" — but that reading assumes a LoRA-free control is
    # the only control needed. It is not: "a LoRA is present" and "the fork
    # routes it" are still confounded. A2S separates them by pinning the same
    # adapter through the STOCK uniform path, and it is the strongest control in
    # this record. It runs on logits.
    "A2S": Arm(
        "A2S",
        (),
        lambda i: None,
        None,
        "one real adapter on every sequence via the STOCK llama_set_adapters_lora — no seq pool",
        pin="a",
    ),
    "A3": Arm(
        "A3",
        ("a", "b"),
        lambda i: "a" if i % 2 == 0 else "b",
        lambda i: "b" if i % 2 == 0 else "a",
        "two real adapters, alternating — the shipping configuration",
    ),
    "A4": Arm(
        "A4",
        ("a",),
        lambda i: None if i % 2 == 0 else "a",
        lambda i: "a" if i % 2 == 0 else None,
        "interleaved -1 seats and adapter-A seats (4+4 at S=8)",
    ),
}


def require_gpu(snapshot: dict[str, Any], *, floor_mib: int) -> tuple[int, int]:
    """Fail-closed GPU assertion by MEASUREMENT, never by reading a log line.

    ``ci/runner/gpu-serialized.sh`` sources ``ci/runner/rig-env.sh``, which
    supplies the real CUDA driver and pins the CI device (``LLAMA_CI_GPU``);
    without rig-env.sh a built lib resolves ``libcuda.so.1`` to the nix stub,
    ggml_cuda_init fails, and llama.cpp then runs on the CPU anyway. That is a
    silent pass. The only honest check is the VRAM the model actually occupies.

    ``benchkit.env.require_gpu0_only`` hard-codes a 4000 MiB floor sized for
    a ~9B model; a ~3B is a 2.6 GB file, so the floor is a parameter here.
    The server-device idle rule is unchanged (the rig's server device must not
    also hold the bench's model).
    """
    lines = snapshot["stdout"].splitlines()
    if len(lines) < 2:
        raise RuntimeError("expected two GPUs in the nvidia-smi snapshot")
    used = [int(line.split(", ")[-1].split()[0]) for line in lines]
    ci = int(os.environ.get("LLAMA_CI_GPU", "0"))
    srv = int(os.environ.get("LLAMA_SERVER_GPU", "1"))
    if used[ci] < floor_mib:
        raise RuntimeError(f"CI device holds {used[ci]} MiB < floor {floor_mib} MiB — the model is not on the GPU")
    if used[srv] > 200:
        raise RuntimeError(f"GPU isolation rejected: server device={used[srv]} MiB busy")
    return used[ci], used[srv]


class Harness:
    """Base model loaded once; one context per arm; raw batches and logit rows."""

    def __init__(self, model_path: str, n_ctx: int, n_batch: int, seed: int) -> None:
        from llama_cpp import Llama, llama_cpp  # noqa: PLC0415

        self.C = llama_cpp
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_batch=n_batch, n_gpu_layers=-1, seed=seed, verbose=False)
        self.model = self.llm._model.model
        self.n_vocab = int(self.llm._n_vocab)
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.adapters: dict[str, Any] = {}
        self.binding = SeqRoutingBinding(self.C)

    def pin_adapter(self, ctx: Any, key: str) -> None:
        """Stock context-scope LoRA install (``router._pin_adapter``)."""
        import ctypes  # noqa: PLC0415

        arr = (self.C.llama_adapter_lora_p_ctypes * 1)(self.adapters[key])
        scales = (ctypes.c_float * 1)(1.0)
        if self.C.llama_set_adapters_lora(ctx, arr, 1, scales) != 0:
            raise RuntimeError("llama_set_adapters_lora failed")

    def load_adapter(self, key: str, path: str) -> None:
        ptr = self.C.llama_adapter_lora_init(self.model, path.encode("utf-8"))
        if not ptr:
            raise RuntimeError(f"failed to load adapter {key!r} from {path}")
        self.adapters[key] = ptr

    def tokenize(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True))

    def make_ctx(self, n_seq_max: int, n_ubatch: int) -> Any:
        p = self.C.llama_context_default_params()
        p.n_ctx = self.n_ctx
        p.n_batch = self.n_batch
        p.n_ubatch = n_ubatch
        p.n_seq_max = n_seq_max
        # Unified KV, matching router._make_ctx: the llama.cpp default partitions
        # n_ctx across n_seq_max, so a requested 2048 would silently become 256
        # per sequence at n_seq_max=8.
        p.kv_unified = True
        ctx = self.C.llama_new_context_with_model(self.model, p)
        if not ctx:
            raise RuntimeError("llama_new_context_with_model returned NULL (out of VRAM?)")
        return ctx

    def free(self, ctx: Any) -> None:
        self.C.llama_free(ctx)

    def sync(self, ctx: Any) -> None:
        self.C.llama_synchronize(ctx)

    def clear(self, ctx: Any, n_seq_max: int) -> None:
        mem = self.C.llama_get_memory(ctx)
        for sid in range(n_seq_max):
            self.C.llama_memory_seq_rm(mem, sid, -1, -1)

    def prefill(self, ctx: Any, tokens: list[int], seq_id: int) -> None:
        """Prefill one sequence, chunked at n_batch — the router's _prefill."""
        n = len(tokens)
        for off in range(0, n, self.n_batch):
            chunk = tokens[off : off + self.n_batch]
            m = len(chunk)
            batch = self.C.llama_batch_init(m, 0, 1)
            batch.n_tokens = m
            for i, t in enumerate(chunk):
                batch.token[i] = t
                batch.pos[i] = off + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq_id
                batch.logits[i] = 1 if off + m == n and i == m - 1 else 0
            rc = self.C.llama_decode(ctx, batch)
            self.C.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"prefill llama_decode rc={rc}")

    def row(self, ctx: Any, index: int) -> np.ndarray:
        """A COPY of one logit row (the native buffer is reused every decode)."""
        ptr = self.C.llama_get_logits_ith(ctx, index)
        return np.array(np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)), dtype=np.float32)

    def wave(
        self,
        ctx: Any,
        seats: list[int],
        prompts: list[list[int]],
        n_steps: int,
        *,
        forced: list[list[int]] | None = None,
        n_seq_max: int,
    ) -> tuple[np.ndarray, list[list[int]], float]:
        """Decode ``len(seats)`` sequences together; capture every logit row.

        Prefill is per sequence (one ``llama_decode`` each, exactly as
        ``router._iter_wave`` does), so the prefill compute shape is identical
        alone and batched. Only the generation steps carry the mix, which is
        what "who shares your batch" means here.

        ``forced`` replaces the sampled token with a fixed one, keeping the
        batched pass aligned with its reference after a flip. The returned
        greedy tokens are still the batched argmaxes, so a flip stays visible.

        Returns ``(logits[S, n_steps, n_vocab], greedy_tokens, decode_ms)``.
        """
        S = len(seats)
        self.clear(ctx, n_seq_max)
        logits = np.empty((S, n_steps, self.n_vocab), dtype=np.float32)
        greedy: list[list[int]] = [[] for _ in range(S)]
        cur = [0] * S
        pos = [0] * S
        for i, sid in enumerate(seats):
            self.prefill(ctx, prompts[i], sid)
            r = self.row(ctx, -1)
            logits[i, 0] = r
            g = int(np.argmax(r))
            greedy[i].append(g)
            cur[i] = forced[i][0] if forced else g
            pos[i] = len(prompts[i])
        self.sync(ctx)
        t0 = perf_counter_ns()
        for t in range(1, n_steps):
            batch = self.C.llama_batch_init(S, 0, 1)
            batch.n_tokens = S
            for i, sid in enumerate(seats):
                batch.token[i] = cur[i]
                batch.pos[i] = pos[i]
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = sid
                batch.logits[i] = 1
            rc = self.C.llama_decode(ctx, batch)
            self.C.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"batched llama_decode rc={rc}")
            for i in range(S):
                pos[i] += 1
                r = self.row(ctx, i)
                logits[i, t] = r
                g = int(np.argmax(r))
                greedy[i].append(g)
                cur[i] = forced[i][t] if forced else g
        self.sync(ctx)
        return logits, greedy, (perf_counter_ns() - t0) / 1e6


def logsoftmax(x: np.ndarray) -> np.ndarray:
    """Log-softmax over the last axis, in float32, numerically stabilized.

    The number this harness re-measures (0.17) is a **log-probability** delta,
    because that is what a parity suite scores. Raw logits carry an
    arbitrary per-row offset, so a raw-logit delta is not comparable with it.
    Both views are reported.
    """
    z = x - x.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def per_step_logprob_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-(sequence, step) max |log p_a - log p_b|, computed sequence by
    sequence so the temporaries stay at one sequence's size."""
    out = np.empty(a.shape[:2], dtype=np.float64)
    for i in range(a.shape[0]):
        out[i] = np.abs(logsoftmax(a[i]) - logsoftmax(b[i])).max(axis=-1)
    return out


def per_step_prob_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-(sequence, step) max |p_a - p_b|. Bounded in [0, 1].

    A raw-logit or log-prob delta scales with the model's logit spread, and the
    noise probe adapters widen that spread a lot. The probability delta does
    not, so it is the metric that compares honestly ACROSS arms.
    """
    out = np.empty(a.shape[:2], dtype=np.float64)
    for i in range(a.shape[0]):
        out[i] = np.abs(np.exp(logsoftmax(a[i])) - np.exp(logsoftmax(b[i]))).max(axis=-1)
    return out


def logit_spread(x: np.ndarray) -> float:
    """Mean per-step (max - min) of a reference logit block — the arm's scale."""
    return float((x.max(axis=-1) - x.min(axis=-1)).mean())


def route_index(arm: Arm, key: str | None) -> int:
    """Pool index for a route key; ``NO_ADAPTER`` (-1) for the base sentinel."""
    return NO_ADAPTER if key is None else arm.pool.index(key)


def apply_routes(h: Harness, ctx: Any, arm: Arm, n_seq_max: int, route: Route) -> list[str | None]:
    """Assign every seat its adapter. A pool-free arm sets nothing."""
    keys = [route(i) for i in range(n_seq_max)]
    if not arm.pool:
        return keys
    for i, key in enumerate(keys):
        h.binding.set_seq_adapter(ctx, i, route_index(arm, key))
    return keys


def measure_arm(
    h: Harness,
    arm: Arm,
    *,
    prompts: list[list[int]],
    concurrencies: list[int],
    n_seq_max: int,
    n_ubatch: int,
    max_tokens: int,
    reps: int,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Run one arm: alone references, crossed references, then every batch size."""
    ctx = h.make_ctx(n_seq_max, n_ubatch)
    try:
        if arm.pin is not None:
            h.pin_adapter(ctx, arm.pin)
        if arm.pool:
            h.binding.set_seq_adapters(ctx, [h.adapters[k] for k in arm.pool])
        routes = apply_routes(h, ctx, arm, n_seq_max, arm.route)

        # ---- alone references: seat i, on its own, free-running greedy ----
        ref_logits = np.empty((n_seq_max, max_tokens, h.n_vocab), dtype=np.float32)
        ref_tokens: list[list[int]] = []
        alone_ms = 0.0
        for i in range(n_seq_max):
            lg, gd, ms = h.wave(ctx, [i], [prompts[i]], max_tokens, n_seq_max=n_seq_max)
            ref_logits[i] = lg[0]
            ref_tokens.append(gd[0])
            alone_ms += ms
        alone_tok_s = (n_seq_max * (max_tokens - 1)) / (alone_ms / 1e3)
        log(f"[comp] {arm.name}: alone references done, {alone_tok_s:.1f} tok/s batch-1")

        # ---- crossed references: the route-bug oracle (p2_throughput.py:149-161) ----
        crossed_tokens: list[list[int]] | None = None
        crossed_keys: list[str | None] | None = None
        separation: list[float] | None = None
        if arm.crossed is not None and arm.pool:
            crossed_keys = apply_routes(h, ctx, arm, n_seq_max, arm.crossed)
            crossed_tokens = []
            separation = []
            for i in range(n_seq_max):
                lg, gd, _ = h.wave(ctx, [i], [prompts[i]], max_tokens, n_seq_max=n_seq_max)
                crossed_tokens.append(gd[0])
                # Oracle POWER: how far apart the correct and crossed routes are
                # when BOTH decode alone. If two routes sit closer together than
                # the batching delta, "this token matches the crossed output"
                # proves nothing — the two routes are the same model to within
                # the noise. Measured, never assumed.
                separation.append(float(np.abs(lg[0] - ref_logits[i]).max()))
            apply_routes(h, ctx, arm, n_seq_max, arm.route)  # restore

        # An oracle is degenerate when the crossed route produces the SAME tokens
        # as the correct route: then "matches the crossed output" carries no
        # information. Recorded, and excluded from the informative bug count.
        degenerate = (
            [crossed_tokens[i] == ref_tokens[i] for i in range(n_seq_max)] if crossed_tokens is not None else None
        )

        # ---- batched passes ----
        by_S: dict[str, Any] = {}
        for S in concurrencies:
            if S > n_seq_max:
                continue
            seats = list(range(S))
            # Only rep 0 is kept in full; later reps fold into the jitter scalar
            # and are released (8 x 32 x n_vocab float32 is ~155 MB per rep).
            first_logits: np.ndarray | None = None
            first_greedy: list[list[int]] = []
            batch_ms: list[float] = []
            jitter = 0.0
            jitter_lp = 0.0
            for rep in range(reps):
                lg, gd, ms = h.wave(
                    ctx,
                    seats,
                    prompts[:S],
                    max_tokens,
                    forced=[ref_tokens[i] for i in seats],
                    n_seq_max=n_seq_max,
                )
                batch_ms.append(ms)
                if rep == 0:
                    first_logits, first_greedy = lg, gd
                else:
                    assert first_logits is not None
                    jitter = max(jitter, float(np.max(np.abs(lg - first_logits))))
                    if jitter > 0.0:
                        jitter_lp = max(jitter_lp, float(per_step_logprob_delta(lg, first_logits).max()))
                    del lg
            assert first_logits is not None
            # A zero raw-logit jitter implies a zero log-prob jitter (identical
            # arrays), so the log-prob pass runs only when the raw pass moved.
            threshold = max(3.0 * jitter, 1e-4)
            threshold_lp = max(3.0 * jitter_lp, 1e-4)

            delta = np.abs(first_logits - ref_logits[:S])  # [S, steps, vocab]
            per_step = delta.max(axis=2)  # [S, steps]
            per_seq_max = per_step.max(axis=1)  # [S]
            per_step_lp = per_step_logprob_delta(first_logits, ref_logits[:S])
            per_seq_max_lp = per_step_lp.max(axis=1)
            per_step_p = per_step_prob_delta(first_logits, ref_logits[:S])
            per_seq_max_p = per_step_p.max(axis=1)
            spread = logit_spread(ref_logits[:S])

            flips: list[str] = []
            route_bugs = 0
            route_bugs_informative = 0
            for i in range(S):
                for t in range(max_tokens):
                    if first_greedy[i][t] == ref_tokens[i][t]:
                        continue
                    # Near-tie evidence: the top-1/top-2 gap in the ALONE
                    # reference at the flipped step. A gap at or below the
                    # measured delta is a coin flip, not a routing decision.
                    top2 = np.partition(ref_logits[i, t], -2)[-2:]
                    gap = float(abs(top2[1] - top2[0]))
                    tag = f"s{i}@tok{t}(gap={gap:.4g})"
                    is_bug = (
                        crossed_tokens is not None
                        and t < len(crossed_tokens[i])
                        and crossed_tokens[i][t] == first_greedy[i][t]
                    )
                    if is_bug:
                        route_bugs += 1
                        # Informative only when the oracle can actually tell the
                        # two routes apart on this sequence: distinct token
                        # streams AND a route separation larger than the delta
                        # being classified.
                        powered = (
                            degenerate is not None
                            and not degenerate[i]
                            and separation is not None
                            and separation[i] > float(per_seq_max[i])
                        )
                        if powered:
                            route_bugs_informative += 1
                        flips.append(f"{tag}:ROUTE-BUG" + ("" if powered else "(oracle-underpowered)"))
                    else:
                        flips.append(f"{tag}:fp")

            by_S[str(S)] = {
                "S": S,
                "seats": seats,
                "routes": routes[:S],
                "max_abs_delta": float(per_seq_max.max()),
                "max_abs_delta_logprob": float(per_seq_max_lp.max()),
                "max_prob_delta": float(per_seq_max_p.max()),
                "ref_logit_spread": spread,
                "relative_delta": float(per_seq_max.max() / spread) if spread else None,
                "per_seq_max_abs_delta": [float(x) for x in per_seq_max],
                "per_seq_max_abs_delta_logprob": [float(x) for x in per_seq_max_lp],
                "per_seq_max_prob_delta": [float(x) for x in per_seq_max_p],
                "mean_abs_delta": float(delta.mean()),
                "greedy_flips": len(flips),
                "flips": flips,
                "route_bugs": route_bugs,
                "route_bugs_informative": route_bugs_informative,
                "oracle_separation": separation,
                "oracle_power_ratio": (
                    [separation[i] / float(per_seq_max[i]) if per_seq_max[i] else None for i in range(S)]
                    if separation is not None
                    else None
                ),
                "jitter": jitter,
                "jitter_logprob": jitter_lp,
                "threshold": threshold,
                "threshold_logprob": threshold_lp,
                "fraction_over": float((per_step > threshold).mean()),
                "fraction_over_logprob": float((per_step_lp > threshold_lp).mean()),
                "steps_over_per_seq": [float((per_step[i] > threshold).mean()) for i in range(S)],
                "decode_ms": round(min(batch_ms), 2),
                "decode_tok_s": round((S * (max_tokens - 1)) / (min(batch_ms) / 1e3), 1),
            }
            log(
                f"[comp] {arm.name} S={S}: max_abs_delta={by_S[str(S)]['max_abs_delta']:.6g} "
                f"logprob={by_S[str(S)]['max_abs_delta_logprob']:.6g} "
                f"prob={by_S[str(S)]['max_prob_delta']:.4g} rel={by_S[str(S)]['relative_delta']:.4g} "
                f"flips={len(flips)} route_bugs={route_bugs} jitter={jitter:.3g} "
                f"frac_over={by_S[str(S)]['fraction_over']:.3f}"
            )
        return {
            "name": arm.name,
            "note": arm.note,
            "pool": list(arm.pool),
            "routes": routes,
            "crossed_routes": crossed_keys,
            "oracle_degenerate": degenerate,
            "oracle_separation_alone": separation,
            "n_ubatch": n_ubatch,
            "alone_tok_s": round(alone_tok_s, 1),
            "reference_tokens": ref_tokens,
            "concurrency": by_S,
        }
    finally:
        h.free(ctx)


def main() -> int:
    ap = argparse.ArgumentParser(description="P2 composition-determinism control")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--lora-a", required=True, type=Path)
    ap.add_argument("--lora-b", required=True, type=Path)
    ap.add_argument("--lora-identity", type=Path, default=None)
    ap.add_argument("--artifacts", required=True, type=Path)
    ap.add_argument("--n-ctx", default=2048, type=int)
    ap.add_argument("--n-batch", default=128, type=int)
    ap.add_argument("--n-seq-max", default=8, type=int)
    ap.add_argument("--max-tokens", default=32, type=int)
    ap.add_argument("--concurrencies", default="1,2,4,8")
    ap.add_argument("--reps", default=3, type=int)
    ap.add_argument("--arms", default="A0,A1,A2,A2S,A3,A4")
    ap.add_argument("--ubatch-sweep", default="128,8,4,2,1")
    ap.add_argument("--sweep-arms", default="A0,A2S,A3")
    ap.add_argument("--vram-floor-mib", default=1024, type=int)
    ap.add_argument("--tok-s-floor", default=20.0, type=float)
    ap.add_argument("--seed", default=17018, type=int)
    args = ap.parse_args()

    require_ci_device()
    lib_path = os.environ.get("LLAMA_CPP_LIB_PATH")
    if not lib_path:
        raise RuntimeError("LLAMA_CPP_LIB_PATH must point at the p2fork lib dir")

    run = RunContext(args.artifacts, name="p2-composition-s7", repo=REPO)
    run.write_command(sys.argv)
    run.write_git_status_before()
    run.write_runtime_environment()
    log = run.log

    gpu_before = gpu_snapshot()
    concurrencies = [int(x) for x in args.concurrencies.split(",")]
    arm_names = [x.strip() for x in args.arms.split(",") if x.strip()]
    sweep_arms = [x.strip() for x in args.sweep_arms.split(",") if x.strip()]
    ubatches = [int(x) for x in args.ubatch_sweep.split(",") if x.strip()]

    skips: list[str] = []
    if "A1" in arm_names and (args.lora_identity is None or not args.lora_identity.exists()):
        # Fail-closed: a missing capability is a LOUD skip with a reason.
        skips.append("A1 skipped: --lora-identity missing or not on disk (identity adapter unavailable)")
        arm_names = [n for n in arm_names if n != "A1"]
        log(f"[comp] SKIP {skips[-1]}")

    h = Harness(str(args.model), args.n_ctx, args.n_batch, args.seed)
    try:
        if not library_supports_seq_routing(h.C._lib):
            raise RuntimeError(f"loaded lib {lib_path} lacks the fork seq-routing symbols")
        gpu_loaded = gpu_snapshot()
        used0, used1 = require_gpu(gpu_loaded, floor_mib=args.vram_floor_mib)
        log(f"[comp] GPU asserted by measurement: GPU0={used0} MiB GPU1={used1} MiB")

        h.load_adapter("a", str(args.lora_a))
        h.load_adapter("b", str(args.lora_b))
        if args.lora_identity is not None and args.lora_identity.exists():
            h.load_adapter("identity", str(args.lora_identity))

        prompts = [h.tokenize(PROMPTS[i % len(PROMPTS)]) for i in range(args.n_seq_max)]

        arms: dict[str, Any] = {}
        for name in arm_names:
            arms[name] = measure_arm(
                h,
                ARMS[name],
                prompts=prompts,
                concurrencies=concurrencies,
                n_seq_max=args.n_seq_max,
                n_ubatch=args.n_batch,
                max_tokens=args.max_tokens,
                reps=args.reps,
                log=log,
            )

        # ---- shape sweep: adapter set fixed, ubatch shape varied ----
        sweep: dict[str, Any] = {}
        top = max(c for c in concurrencies if c <= args.n_seq_max)
        for name in sweep_arms:
            if name not in arm_names:
                continue
            sweep[name] = {}
            for u in ubatches:
                res = measure_arm(
                    h,
                    ARMS[name],
                    prompts=prompts,
                    concurrencies=[top],
                    n_seq_max=args.n_seq_max,
                    n_ubatch=u,
                    max_tokens=args.max_tokens,
                    reps=1,
                    log=log,
                )
                row = res["concurrency"][str(top)]
                sweep[name][str(u)] = {
                    "n_ubatch": u,
                    "S": top,
                    "max_abs_delta": row["max_abs_delta"],
                    "max_abs_delta_logprob": row["max_abs_delta_logprob"],
                    "max_prob_delta": row["max_prob_delta"],
                    "relative_delta": row["relative_delta"],
                    "greedy_flips": row["greedy_flips"],
                    "route_bugs": row["route_bugs"],
                    "decode_tok_s": row["decode_tok_s"],
                }
                log(
                    f"[comp] sweep {name} n_ubatch={u} S={top}: "
                    f"max_abs_delta={row['max_abs_delta']:.6g} flips={row['greedy_flips']}"
                )

        # ---- fail-closed harness check: concurrency 1 must be EXACTLY 0.0 ----
        c1 = {n: arms[n]["concurrency"].get("1", {}).get("max_abs_delta") for n in arms}
        harness_ok = all(v == 0.0 for v in c1.values() if v is not None)
        alone_tok_s = max(a["alone_tok_s"] for a in arms.values())
        gpu_ok = alone_tok_s >= args.tok_s_floor

        manifest_path = Path(lib_path).parent / "build-manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        model_id = register_artifact(args.model)
        lora_ids = {
            "a": register_artifact(args.lora_a).model_dump(),
            "b": register_artifact(args.lora_b).model_dump(),
        }
        if args.lora_identity is not None and args.lora_identity.exists():
            lora_ids["identity"] = register_artifact(args.lora_identity).model_dump()
        import llama_cpp  # noqa: PLC0415

        fp = LlamaEngineFingerprint(
            model=model_id,
            tokenizer=model_id,  # GGUF carries its own tokenizer
            llama_cpp_python_version=llama_cpp.__version__,
            llama_cpp_commit=(manifest or {}).get("llama_cpp_commit"),
            llama_cpp_build_id=(manifest or {}).get("build_id"),
            seq_adapter_routing=True,
            backend="cuda",
            n_ctx=args.n_ctx,
        )

        result = {
            "control": "p2_composition",
            "harness_check": {
                "rule": "concurrency 1 max_abs_delta must be exactly 0.0",
                "per_arm_c1_max_abs_delta": c1,
                "passed": harness_ok,
            },
            "gpu_check": {
                "rule": "asserted by measurement: VRAM floor + batch-1 decode throughput floor",
                "vram_used_mib_gpu0": used0,
                "vram_used_mib_gpu1": used1,
                "vram_floor_mib": args.vram_floor_mib,
                "alone_tok_s": alone_tok_s,
                "tok_s_floor": args.tok_s_floor,
                "passed": gpu_ok,
            },
            "config": {
                "n_ctx": args.n_ctx,
                "n_batch": args.n_batch,
                "n_ubatch_arms": args.n_batch,
                "n_seq_max": args.n_seq_max,
                "n_gpu_layers": -1,
                "kv_unified": True,
                "max_tokens": args.max_tokens,
                "concurrencies": concurrencies,
                "reps": args.reps,
                "seed": args.seed,
                "prompts": PROMPTS[: args.n_seq_max],
                "greedy": True,
            },
            "arms": arms,
            "shape_sweep": sweep,
            "skips": skips,
            "fingerprint": {
                "cache_key": fp.cache_key(),
                "engine": json.loads(fp.model_dump_json()),
                "adapters": lora_ids,
            },
            "build_manifest": manifest,
            "lib": lib_path,
            "model": str(args.model),
            "gpu_before": gpu_before,
            "gpu_after": gpu_snapshot(),
        }
        (run.dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        log(f"[comp] summary: {run.dir / 'summary.json'}")

        if not harness_ok:
            print(f"[comp] FAIL — harness wrong: concurrency-1 deltas {c1} (expected all 0.0)", flush=True)
            return 1
        if not gpu_ok:
            print(f"[comp] FAIL — {alone_tok_s} tok/s below the {args.tok_s_floor} floor (CPU fallback?)", flush=True)
            return 1
        bugs = {n: a["concurrency"][s]["route_bugs_informative"] for n, a in arms.items() for s in a["concurrency"]}
        total_bugs = sum(bugs.values())
        if total_bugs:
            print(f"[comp] FAIL — {total_bugs} informative route-bug(s): {bugs}", flush=True)
            return 1
        print(
            json.dumps(
                {
                    "harness_check": result["harness_check"],
                    "max_abs_delta": {
                        n: {s: a["concurrency"][s]["max_abs_delta"] for s in a["concurrency"]} for n, a in arms.items()
                    },
                    "max_abs_delta_logprob": {
                        n: {s: a["concurrency"][s]["max_abs_delta_logprob"] for s in a["concurrency"]}
                        for n, a in arms.items()
                    },
                    "max_prob_delta": {
                        n: {s: a["concurrency"][s]["max_prob_delta"] for s in a["concurrency"]} for n, a in arms.items()
                    },
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    finally:
        h.llm.close()


if __name__ == "__main__":
    sys.exit(main())
