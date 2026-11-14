"""Mixed-batch gates that hold on a backend whose logits depend on batch composition.

Why this module exists. On a Vulkan build, the same request gives different logits
when the batch composition changes: other sequences in the batch, the cell that the
unified KV cache gives the sequence, a different n_kv. A decode repeats itself bit
for bit, so the effect is not run-to-run noise. A gate that needs a batch to give
the same tokens as batch 1 (token-exact) cannot hold there. It fails on a correct
build and tells nobody what broke. The first measurement on a card also showed that
the composition noise is large (a tail of 3 to 19 log-probability units), so a
tolerance that covers it would pass almost any decode.

Decision (2026-11-13): on Vulkan, mixed-versus-isolated equivalence is NOT a
gate. It is a measured quantity. Three exact gates carry the claim.

``determinism``
    The same composition, run twice from a clean state, is bit-identical.

``isolation`` (bleed-through)
    Hold the composition fixed: the same sequences, the same tokens, the same shapes.
    Change only the adapters of the OTHER sequences. The target sequence's logits
    must be bit-identical. A correct per-row adapter cannot change another row.
    Each neighbour must also react to its own adapter, or the gate is vacuous for
    it. That catches an adapter that is dropped on one row.

``parity``
    One request alone, from a clean state, on two backends of the router
    (``context_pool`` and ``seq_routed``). Same shapes, same tokens. Bit-identical.
    A gap names a real difference in how the adapter is applied.

Equivalence is measured. :func:`measure_pairs` turns pairs of (isolated run, mixed run),
both teacher-forced with the same tokens, into a distribution per class (p50, p99, max)
and a list of worst cases. A worst case carries the forced token's log-probability in
both runs and the total-variation distance over the probe ids. That separates a large
difference on tokens that hold no mass (benign sensitivity) from a distribution that
differs broadly (a possible coupling). :func:`report_measurement` writes the record with
its provenance and warns that the quantity is measured, not gated. No test passes or
fails on it.

A gate never passes silently. A backend that this module does not know is a loud
failure (:class:`UnknownBackendError`). The CUDA path keeps token-exact gates.
:func:`gate_mode` picks the path from the loaded library (see
``gpu_env.lib_backend_label``).

Nothing here imports the native stack. The router side is
``inferference.benchkit.trace_router``.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from .gpu_env import lib_backend_label
from .router import RouteRequest

__all__ = [
    "LOGIT_TOLERANCE",
    "TOKEN_EXACT",
    "GateError",
    "Pair",
    "Step",
    "Trace",
    "TracingEngine",
    "UnknownBackendError",
    "check_exact",
    "check_parity",
    "collect_isolated",
    "collect_isolated_forced",
    "collect_mixed",
    "current_build",
    "gate_cross_context_isolation",
    "gate_determinism",
    "gate_isolation",
    "gate_mixed_adapter_isolation",
    "gate_mode",
    "measure_pairs",
    "measurements_path",
    "pairs_from_traces",
    "report_measurement",
    "require",
    "row_step",
]

TOKEN_EXACT: Literal["token_exact"] = "token_exact"
# The name stays. It means: the logits depend on the composition, so the gates read logits and none is token-exact.
LOGIT_TOLERANCE: Literal["logit_tolerance"] = "logit_tolerance"
GateMode = Literal["token_exact", "logit_tolerance"]

# How many ids each step records. The reference keeps its own top ids. A run that is
# compared to it reads the logits at the SAME ids.
TOP_K = 16

# A worst case counts as "tail" when the probability mass that moved is under this and the
# top-1 agrees. The label describes a case. No test reads it.
TAIL_TV = 0.05

# How many lines a failure message carries. More lines go to the run artifact.
_MAX_FINDINGS = 12


class GateError(AssertionError):
    """A gate failed. The message lists every finding."""


class UnknownBackendError(RuntimeError):
    """The loaded library is a backend that no gate knows. A gate never guesses."""


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #


def gate_mode(label: str | None = None) -> GateMode:
    """Pick the gate family for the loaded backend.

    ``label`` defaults to ``gpu_env.lib_backend_label()``, which reads the loaded
    library. CUDA keeps the token-exact gates. Vulkan takes the logit gates. Every
    other label raises: a gate that falls back would run a check that the backend
    may not support, or skip a check that it needs.
    """
    backend = lib_backend_label() if label is None else label
    if backend == "cuda":
        return TOKEN_EXACT
    if backend == "vulkan":
        return LOGIT_TOLERANCE
    raise UnknownBackendError(
        f"no mixed-batch gate is defined for backend {backend!r} (known: 'cuda', 'vulkan'). "
        "'gpu' means the library gave no evidence, 'cpu' is not a serving backend, and a joined label "
        "('a+b') names more than one backend. Fix the library path, or add the backend with its own gates."
    )


# --------------------------------------------------------------------------- #
# Traces
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Step:
    """One sampled step of one sequence.

    ``token`` is the token that the run took (the forced one, or its own argmax).
    ``argmax`` and ``argmax_logit`` come from the full raw row. ``lse`` is the
    log-sum-exp of the full raw row, so a difference anywhere in the row changes it
    and the exact gates see it. ``ids`` and ``logits`` are the raw logits at the
    probe ids.
    """

    token: int
    argmax: int
    argmax_logit: float
    lse: float
    ids: tuple[int, ...]
    logits: tuple[float, ...]

    def logprobs(self) -> tuple[float, ...]:
        return tuple(x - self.lse for x in self.logits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "argmax": self.argmax,
            "argmax_logit": self.argmax_logit,
            "lse": self.lse,
            "ids": list(self.ids),
            "logits": list(self.logits),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        return cls(
            token=int(data["token"]),
            argmax=int(data["argmax"]),
            argmax_logit=float(data["argmax_logit"]),
            lse=float(data["lse"]),
            ids=tuple(int(i) for i in data["ids"]),
            logits=tuple(float(x) for x in data["logits"]),
        )


@dataclass(frozen=True, slots=True)
class Trace:
    """The steps of one sequence, in order."""

    steps: tuple[Step, ...]

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(s.token for s in self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        return cls(steps=tuple(Step.from_dict(s) for s in data["steps"]))


def row_step(logits: Any, token: int | None = None, probe_ids: Sequence[int] | None = None, k: int = TOP_K) -> Step:
    """Record one logits row. Read only: the row is not changed.

    ``token`` is the token the run takes. ``None`` means the argmax. ``probe_ids`` are
    the ids to read. ``None`` means this row's own top ``k``, sorted by logit
    descending and id ascending on a tie.
    """
    import numpy as np

    row = np.asarray(logits, dtype=np.float32)
    argmax = int(np.argmax(row))
    peak = float(row[argmax])
    # float64 sum of the float32 row: deterministic for a given row
    total = float(np.exp(row.astype(np.float64) - peak).sum())
    lse = peak + math.log(total)
    if probe_ids is None:
        top = np.argpartition(row, -k)[-k:]
        ids = tuple(int(i) for i in top[np.lexsort((top, -row[top]))])
    else:
        ids = tuple(int(i) for i in probe_ids)
    return Step(
        token=argmax if token is None else int(token),
        argmax=argmax,
        argmax_logit=peak,
        lse=lse,
        ids=ids,
        logits=tuple(float(row[i]) for i in ids),
    )


# --------------------------------------------------------------------------- #
# Exact checks. Each returns findings. An empty list is a pass.
# --------------------------------------------------------------------------- #


def _finite(label: str, trace: Trace) -> list[str]:
    for t, s in enumerate(trace.steps):
        values = (s.argmax_logit, s.lse, *s.logits)
        if not all(math.isfinite(v) for v in values):
            return [
                f"{label} step {t}: non-finite logit or log-sum-exp (a NaN or inf in one row can reach its neighbours)"
            ]
    return []


def _same_step(a: Step, b: Step) -> bool:
    return a == b


def check_exact(label: str, ref: Trace, got: Trace) -> list[str]:
    """The two traces are bit-identical, step for step, over the full length of ``ref``."""
    findings = _finite(label, ref) + _finite(label, got)
    if len(got) < len(ref):
        findings.append(f"{label}: trace has {len(got)} steps, the reference has {len(ref)}")
    for t, (a, b) in enumerate(zip(ref.steps, got.steps, strict=False)):
        if _same_step(a, b):
            continue
        if a.token != b.token:
            findings.append(f"{label} step {t}: token {b.token} against reference {a.token} (forcing broke)")
        else:
            lp = _max_dlogprob(a, b) if a.ids == b.ids else float("nan")
            findings.append(
                f"{label} step {t}: logits differ; max |dlogprob| over the probe ids {lp:.6g}; "
                f"lse {a.lse!r} against {b.lse!r}"
            )
        break
    return findings


def _max_dlogprob(ref: Step, got: Step) -> float:
    return max((abs(g - r) for r, g in zip(ref.logprobs(), got.logprobs(), strict=True)), default=0.0)


def check_parity(label: str, ref: Trace, got: Trace) -> list[str]:
    """Same request, alone, two router backends. Exact. The text names the claim that fails.

    The two traces must have the same length. ``check_exact`` reports a short ``got``;
    this check reports a longer one too, because extra steps are a difference.
    """
    findings = check_exact(label, ref, got)
    if len(got) > len(ref):
        findings.append(f"{label}: trace has {len(got)} steps, the reference has {len(ref)}")
    return [f"parity {m}" for m in findings]


def require(findings: Sequence[str], title: str) -> None:
    """Raise :class:`GateError` when there is any finding."""
    if not findings:
        return
    shown = list(findings[:_MAX_FINDINGS])
    more = len(findings) - len(shown)
    tail = f"\n  ... and {more} more" if more > 0 else ""
    raise GateError(f"{title}: {len(findings)} finding(s)\n  " + "\n  ".join(shown) + tail)


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


class TracingEngine(Protocol):
    """What a gate needs from a router. ``benchkit.trace_router.TraceRouter`` is the real one."""

    @property
    def mixes_adapters(self) -> bool:
        """True when one decode carries sequences of several adapters (``seq_routed``)."""
        ...

    def reset_state(self) -> None:
        """Remove all sequence state. Every run then starts from the same state."""
        ...

    def trace(self, requests: Sequence[RouteRequest], *, reference: Sequence[Trace] | None = None) -> list[Trace]:
        """Run ``requests`` as one workload. Return one trace per request.

        With ``reference``, request ``i`` is teacher-forced with ``reference[i]``'s tokens
        and reads the logits at its probe ids.
        """
        ...


def collect_isolated(engine: TracingEngine, requests: Sequence[RouteRequest]) -> list[Trace]:
    """Each request alone, free-running, from a clean state."""
    out: list[Trace] = []
    for r in requests:
        engine.reset_state()
        out.append(engine.trace([r])[0])
    return out


def collect_isolated_forced(
    engine: TracingEngine, requests: Sequence[RouteRequest], reference: Sequence[Trace]
) -> list[Trace]:
    """Each request alone, forced by ``reference``, from a clean state."""
    out: list[Trace] = []
    for r, ref in zip(requests, reference, strict=True):
        engine.reset_state()
        out.append(engine.trace([r], reference=[ref])[0])
    return out


def collect_mixed(
    engine: TracingEngine, requests: Sequence[RouteRequest], reference: Sequence[Trace] | None = None
) -> list[Trace]:
    """All requests in one workload from a clean state, forced by ``reference`` if given."""
    engine.reset_state()
    return engine.trace(list(requests), reference=reference)


# --------------------------------------------------------------------------- #
# Gates that run on one engine
# --------------------------------------------------------------------------- #


def gate_determinism(engine: TracingEngine, requests: Sequence[RouteRequest]) -> list[str]:
    """Same composition twice, with other work in between: bit-identical.

    This also proves that ``reset_state`` removes the history. A history effect that
    survives the reset makes the second run differ, and this gate names it.
    """
    first = collect_mixed(engine, requests)
    # other work: a different composition leaves its own sequence state behind
    collect_mixed(engine, list(reversed(requests)))
    second = collect_mixed(engine, requests, reference=first)
    findings: list[str] = []
    for i, (a, b) in enumerate(zip(first, second, strict=True)):
        findings += check_exact(f"determinism request {i}", a, b)
    return findings


def _next_adapter(current: str | None, options: Sequence[str | None]) -> str | None:
    """The option after ``current``, wrapping. The first option when ``current`` is not one."""
    return options[(options.index(current) + 1) % len(options)] if current in options else options[0]


def _with_adapters(requests: Sequence[RouteRequest], adapters: dict[int, str | None]) -> list[RouteRequest]:
    return [r.model_copy(update={"adapter": adapters[i]}) if i in adapters else r for i, r in enumerate(requests)]


def gate_mixed_adapter_isolation(
    engine: TracingEngine,
    requests: Sequence[RouteRequest],
    *,
    target: int,
    pool: Sequence[str],
) -> list[str]:
    """Bleed-through gate for an engine that mixes adapters in one decode.

    Free-run the composition once. Every variant then runs the SAME tokens (forced)
    through the same shapes, and only the adapters of the other sequences change:
    one neighbour at a time to each other adapter and to none, then all neighbours
    to none, then all neighbours rotated to the next adapter. The target's trace must
    be bit-identical in every variant.

    The gate also fails when a neighbour never reacts to its own adapter. A neighbour
    whose logits do not change when its adapter changes has no adapter applied. That
    is an adapter drop on that row. A gate whose variation changes nothing proves
    nothing, so each neighbour must move in at least one of its variants.
    """
    base = collect_mixed(engine, requests)
    options: list[str | None] = [None, *pool]
    others = [i for i in range(len(requests)) if i != target]
    variants: list[tuple[str, dict[int, str | None]]] = []
    for j in others:
        for a in options:
            if a != requests[j].adapter:
                variants.append((f"neighbour {j} -> {a or 'none'}", {j: a}))
    variants.append(("all neighbours -> none", {j: None for j in others}))
    variants.append(("all neighbours rotated", {j: _next_adapter(requests[j].adapter, list(pool)) for j in others}))

    findings = _finite(f"isolation target {target}", base[target])
    reacted: set[int] = set()
    for name, change in variants:
        got = collect_mixed(engine, _with_adapters(requests, change), reference=base)
        findings += check_exact(f"isolation target {target} ({name})", base[target], got[target])
        reacted |= {j for j in change if not _same_step_list(base[j], got[j])}
    for j in others:
        if j not in reacted:
            findings.append(
                f"isolation is vacuous for neighbour {j}: its logits did not change in any variant, "
                "so its adapter is not applied (an adapter drop on that row)"
            )
    return findings


def _same_step_list(a: Trace, b: Trace) -> bool:
    return a.steps == b.steps


def gate_cross_context_isolation(
    engine: TracingEngine,
    requests: Sequence[RouteRequest],
    *,
    target_adapter: str | None,
    pool: Sequence[str],
) -> list[str]:
    """Bleed-through gate for the ``context_pool`` engine.

    Each adapter has its own context, so a sequence has no same-batch neighbour of
    another adapter. The claim is narrower and still worth a gate: the sequences of
    one adapter do not change when the requests of the OTHER adapters change. Variants:
    the other requests removed, and the other requests moved to a different other
    adapter. The target group runs the same tokens (forced) and its traces must be
    bit-identical.
    """
    target_idx = [i for i, r in enumerate(requests) if r.adapter == target_adapter]
    other_idx = [i for i in range(len(requests)) if i not in target_idx]
    if not target_idx or not other_idx:
        return ["cross-context isolation needs a target group and at least one request of another adapter"]
    base = collect_mixed(engine, requests)
    options: list[str | None] = [a for a in [None, *pool] if a != target_adapter]
    findings: list[str] = []

    # variant 1: the other requests removed
    only_target = [requests[i] for i in target_idx]
    got = collect_mixed(engine, only_target, reference=[base[i] for i in target_idx])
    for slot, i in enumerate(target_idx):
        findings += check_exact(f"cross-context target {i} (others removed)", base[i], got[slot])

    # variant 2: each other request moves to the next other adapter
    rotate = {i: _next_adapter(requests[i].adapter, options) for i in other_idx}
    got2 = collect_mixed(engine, _with_adapters(requests, rotate), reference=base)
    for i in target_idx:
        findings += check_exact(f"cross-context target {i} (others moved)", base[i], got2[i])
    for i in other_idx:
        if _same_step_list(base[i], got2[i]):
            findings.append(
                f"cross-context isolation is vacuous for request {i}: its logits did not change when its "
                "adapter changed, so its adapter is not applied"
            )
    return findings


def gate_isolation(
    engine: TracingEngine,
    requests: Sequence[RouteRequest],
    *,
    pool: Sequence[str],
    target: int = 0,
) -> list[str]:
    """Pick the bleed-through gate that fits the engine."""
    if engine.mixes_adapters:
        return gate_mixed_adapter_isolation(engine, requests, target=target, pool=pool)
    return gate_cross_context_isolation(engine, requests, target_adapter=requests[target].adapter, pool=pool)


# --------------------------------------------------------------------------- #
# Equivalence as a measured quantity. No gate reads it.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Pair:
    """An isolated run and the same request in a composition, both teacher-forced with the same tokens."""

    prompt: str
    composition: str
    adapter: str | None
    ref: Trace
    got: Trace


def pairs_from_traces(
    requests: Sequence[RouteRequest], iso: Sequence[Trace], mixed: Sequence[Trace], composition: str
) -> list[Pair]:
    """One pair per request of a workload."""
    return [Pair(r.prompt, composition, r.adapter, a, b) for r, a, b in zip(requests, iso, mixed, strict=True)]


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    return sorted_values[min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))]


def _lp_at(step: Step, token: int) -> float | None:
    """The log-probability of ``token`` when it is one of the probe ids, else None (below the probe)."""
    for i, x in zip(step.ids, step.logprobs(), strict=True):
        if i == token:
            return x
    return None


def _step_case(pair: Pair, t: int) -> dict[str, Any]:
    a, b = pair.ref.steps[t], pair.got.steps[t]
    lp_a, lp_b = a.logprobs(), b.logprobs()
    diffs = [abs(y - x) for x, y in zip(lp_a, lp_b, strict=True)]
    k = max(range(len(diffs)), key=diffs.__getitem__) if diffs else 0
    tv = 0.5 * sum(abs(math.exp(x) - math.exp(y)) for x, y in zip(lp_a, lp_b, strict=True))
    top = sorted(zip(a.ids, lp_a, lp_b, strict=True), key=lambda row: -row[1])[:8]
    return {
        "prompt": pair.prompt,
        "composition": pair.composition,
        "adapter": pair.adapter,
        "step": t,
        "max_abs_dlogprob": diffs[k] if diffs else 0.0,
        "id_at_max": a.ids[k] if diffs else None,
        "ref_lp_at_max": lp_a[k] if diffs else None,
        "got_lp_at_max": lp_b[k] if diffs else None,
        "forced_token": a.token,
        "forced_lp_ref": _lp_at(a, a.token),
        "forced_lp_got": _lp_at(b, a.token),
        "tv_over_probe_ids": tv,
        "top1_ref": a.argmax,
        "top1_got": b.argmax,
        "top1_agree": a.argmax == b.argmax,
        "ref_top8_with_got_lp": [{"id": i, "ref_lp": x, "got_lp": y} for i, x, y in top],
        "kind": "tail" if tv < TAIL_TV and a.argmax == b.argmax else "broad",
    }


def measure_pairs(pairs: Iterable[Pair], *, n_worst: int = 8) -> dict[str, dict[str, Any]]:
    """Distribution and worst cases per class (``base`` and ``adapter``).

    The sample of a class is the per-step largest |dlogprob| over the probe ids. Also counted:
    steps where the top-1 differs, and how many of the worst cases are tail-only.
    """
    by_class: dict[str, list[Pair]] = {}
    for p in pairs:
        by_class.setdefault("base" if p.adapter is None else "adapter", []).append(p)
    out: dict[str, dict[str, Any]] = {}
    for name, group in sorted(by_class.items()):
        cases: list[dict[str, Any]] = []
        for p in group:
            if p.ref.tokens != p.got.tokens[: len(p.ref)]:
                raise GateError(f"measurement pair {p.prompt!r}/{p.composition!r}: teacher forcing broke")
            cases += [_step_case(p, t) for t in range(len(p.ref))]
        deltas = sorted(c["max_abs_dlogprob"] for c in cases)
        worst = sorted(cases, key=lambda c: -c["max_abs_dlogprob"])[:n_worst]
        out[name] = {
            "n_pairs": len(group),
            "n_steps": len(cases),
            "n_prompts": len({p.prompt for p in group}),
            "n_compositions": len({p.composition for p in group}),
            "max": deltas[-1] if deltas else 0.0,
            "p99": _percentile(deltas, 0.99),
            "p50": _percentile(deltas, 0.50),
            "top1_disagree_steps": sum(1 for c in cases if not c["top1_agree"]),
            "broad_steps": sum(1 for c in cases if c["kind"] == "broad"),
            "worst": worst,
            "worst_tail": sum(1 for c in worst if c["kind"] == "tail"),
        }
    return out


def current_build(env: dict[str, str] | None = None) -> str:
    """The build id of the loaded library: the name of the directory above ``lib``, or ``unknown``."""
    src = os.environ if env is None else env
    raw = src.get("LLAMA_CPP_LIB_PATH", "")
    if not raw:
        return "unknown"
    path = Path(raw)
    return path.parent.name if path.name == "lib" else path.name


def measurements_path(root: Path | None = None) -> Path:
    """``$INFERFERENCE_MEASUREMENTS`` or ``ci/runner/measurements/equivalence.jsonl`` in the repository."""
    env = os.environ.get("INFERFERENCE_MEASUREMENTS")
    if env:
        return Path(env)
    base = root if root is not None else Path(__file__).resolve().parents[2]
    return base / "ci" / "runner" / "measurements" / "equivalence.jsonl"


def _device() -> str:
    try:
        from .benchkit.env import visible_gpu_uuid

        return visible_gpu_uuid() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def report_measurement(
    title: str, pairs: Iterable[Pair], *, path: Path | None = None, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Measure, append the record (with provenance) to the measurement file, and warn.

    The warning is the line that every suite output carries: equivalence is measured, not gated.
    Returns the record. Nothing here raises on the size of a difference.
    """
    classes = measure_pairs(pairs)
    record: dict[str, Any] = {
        "title": title,
        "provenance": {
            "backend": lib_backend_label(),
            "build": current_build(),
            "device": _device(),
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "classes": classes,
        **(extra or {}),
    }
    where = measurements_path() if path is None else path
    where.parent.mkdir(parents=True, exist_ok=True)
    with where.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    summary = "; ".join(
        f"{name}: p50 {c['p50']:.3g}, p99 {c['p99']:.3g}, max {c['max']:.3g}, "
        f"{c['top1_disagree_steps']}/{c['n_steps']} top-1 steps differ"
        for name, c in classes.items()
    )
    warnings.warn(
        f"equivalence is MEASURED, not gated [{title}]: {summary or 'no pairs'} (record: {where})",
        stacklevel=2,
    )
    return record
