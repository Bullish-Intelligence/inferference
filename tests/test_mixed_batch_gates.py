"""CPU tests for the mixed-batch gates: no GPU, no native library.

A fake engine stands in for a router on a backend whose logits depend on the batch
composition (Vulkan-like noise) and on the history of the KV cache. The fake can carry a
planted fault: a bleed-through, a dropped adapter, a scaled adapter, a reset that does not
reset. Each test shows what the gates do with it. A gate that passes on a planted fault is a
defect of the gate, and these tests fail then.
"""

from __future__ import annotations

import json
import math
import warnings
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from inferference.benchkit.runners.batch_noise_measurement import batch_pairs
from inferference.mixed_batch_gates import (
    LOGIT_TOLERANCE,
    TOKEN_EXACT,
    GateError,
    Pair,
    Step,
    Trace,
    UnknownBackendError,
    check_exact,
    check_parity,
    collect_isolated,
    collect_isolated_forced,
    collect_mixed,
    gate_cross_context_isolation,
    gate_determinism,
    gate_isolation,
    gate_mode,
    measure_pairs,
    pairs_from_traces,
    report_measurement,
    require,
    row_step,
)
from inferference.router import RouteRequest

V = 48
# the label of the library that keeps the token-exact gates
_TOKEN_EXACT_LABEL = "cuda"
POOL = ["probe-a", "probe-b"]


def _rng(*key: object) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


class FakeEngine:
    """Deterministic logits with composition noise, history noise and optional planted faults.

    ``mixes`` is True for a ``seq_routed``-like engine (one batch carries every adapter) and
    False for a ``context_pool``-like engine (one batch per adapter). The noise of a row
    depends on the prompts of its batch and its slot. It never depends on the adapters of the
    neighbours, as in a correct engine.
    """

    def __init__(
        self,
        *,
        mixes: bool,
        noise: float = 0.04,
        adapter_noise: float = 0.25,
        history_noise: float = 0.0,
        reset_clears: bool = True,
        leak: float = 0.0,
        drop_rows: frozenset[int] = frozenset(),
        drop_always: bool = False,
        scale: float = 1.0,
    ) -> None:
        self._mixes = mixes
        self.noise = noise
        self.adapter_noise = adapter_noise
        self.history_noise = history_noise
        self.reset_clears = reset_clears
        self.leak = leak
        self.drop_rows = drop_rows
        self.drop_always = drop_always
        self.scale = scale
        self.history = 0
        self.resets = 0

    @property
    def mixes_adapters(self) -> bool:
        return self._mixes

    def reset_state(self) -> None:
        self.resets += 1
        if self.reset_clears:
            self.history = 0

    def _delta(self, adapter: str | None, prompt: str, tokens: tuple[int, ...]) -> np.ndarray:
        if adapter is None:
            return np.zeros(V)
        return _rng("delta", adapter, prompt, tokens).normal(0.0, 2.0, V)

    def _groups(self, requests: Sequence[RouteRequest]) -> list[list[int]]:
        if self._mixes:
            return [list(range(len(requests)))]
        by: dict[str | None, list[int]] = {}
        for i, r in enumerate(requests):
            by.setdefault(r.adapter, []).append(i)
        return list(by.values())

    def trace(self, requests: Sequence[RouteRequest], *, reference: Sequence[Trace] | None = None) -> list[Trace]:
        reqs = list(requests)
        out: list[Trace | None] = [None] * len(reqs)
        for group in self._groups(reqs):
            prompts = tuple(reqs[j].prompt for j in group)
            for slot, j in enumerate(group):
                r = reqs[j]
                tokens: list[int] = []
                steps: list[Step] = []
                for t in range(r.max_tokens):
                    ref = None if reference is None else reference[j]
                    ref_step = ref.steps[t] if ref is not None and t < len(ref) else None
                    logits = self._row(reqs, group, j, slot, prompts, tuple(tokens), t)
                    step = row_step(
                        logits,
                        token=None if ref_step is None else ref_step.token,
                        probe_ids=None if ref_step is None else ref_step.ids,
                    )
                    steps.append(step)
                    tokens.append(step.token)
                out[j] = Trace(tuple(steps))
        self.history = zlib.crc32(repr((self.history, tuple(r.prompt for r in reqs))).encode())
        return [t for t in out if t is not None]

    def _row(
        self,
        reqs: list[RouteRequest],
        group: list[int],
        j: int,
        slot: int,
        prompts: tuple[str, ...],
        tokens: tuple[int, ...],
        t: int,
    ) -> np.ndarray:
        r = reqs[j]
        row = _rng("base", r.prompt, tokens).normal(0.0, 3.0, V)
        adapter = r.adapter
        if self.drop_always or (len(reqs) > 1 and slot in self.drop_rows):
            adapter = None
        row = row + self.scale * self._delta(adapter, r.prompt, tokens)
        if self.leak:
            # planted bleed-through: a part of every neighbour's adapter delta reaches this row
            for k, other in enumerate(reqs):
                if k != j and other.adapter is not None:
                    row = row + self.leak * self._delta(other.adapter, other.prompt, ())
        eps = self.adapter_noise if adapter is not None else self.noise
        row = row + eps * _rng("noise", prompts, slot, t, tokens).normal(0.0, 1.0, V)
        if self.history and self.history_noise:
            row = row + self.history_noise * _rng("history", self.history, t).normal(0.0, 1.0, V)
        return row


def _requests(
    adapters: Sequence[str | None] = ("probe-a", "probe-b", None, "probe-a"), n: int = 12
) -> list[RouteRequest]:
    prompts = ["The capital of France is", "In a distant galaxy,", "def add(a, b):", "Once upon a time"]
    return [
        RouteRequest(request_id=f"r{i}", prompt=prompts[i], adapter=a, max_tokens=n) for i, a in enumerate(adapters)
    ]


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #


def test_gate_mode_names_the_family_of_each_known_backend() -> None:
    assert gate_mode("cuda") == TOKEN_EXACT
    assert gate_mode("vulkan") == LOGIT_TOLERANCE


@pytest.mark.parametrize("label", ["gpu", "cpu", "vulkan+cuda", "", "metal", "rocm"])
def test_gate_mode_fails_on_a_backend_it_does_not_know(label: str) -> None:
    with pytest.raises(UnknownBackendError, match="no mixed-batch gate"):
        gate_mode(label)


def test_gate_mode_reads_the_loaded_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib = tmp_path / "build" / "lib"
    lib.mkdir(parents=True)
    (lib / "libggml.so.0").write_text("")
    (lib / "libggml-vulkan.so.0").write_text("")
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", str(lib))
    assert gate_mode() == LOGIT_TOLERANCE
    (lib / "libggml-vulkan.so.0").unlink()
    (lib / "libggml-cuda.so.0").write_text("")
    assert gate_mode() == TOKEN_EXACT
    (lib / "libggml-cuda.so.0").unlink()
    with pytest.raises(UnknownBackendError):  # a ggml library with no GPU evidence reads as cpu
        gate_mode()
    monkeypatch.delenv("LLAMA_CPP_LIB_PATH")
    with pytest.raises(UnknownBackendError):  # no library at all reads as 'gpu', which names no backend
        gate_mode()


# --------------------------------------------------------------------------- #
# Steps and checks
# --------------------------------------------------------------------------- #


def test_row_step_reads_the_row_without_changing_it() -> None:
    row = np.array([1.0, 5.0, 3.0, 5.0, -2.0] + [0.0] * 30, dtype=np.float32)
    before = row.copy()
    step = row_step(row, k=3)
    assert np.array_equal(row, before)
    assert step.argmax == 1  # first maximum, as np.argmax in the router
    assert step.ids[:2] == (1, 3)  # logit descending, id ascending on a tie
    assert step.lse == pytest.approx(math.log(float(np.exp(row.astype(np.float64)).sum())))
    probed = row_step(row, probe_ids=[4, 2])
    assert probed.ids == (4, 2) and probed.logits == (-2.0, 3.0)


def _trace(rows: list[np.ndarray], probe: Trace | None = None) -> Trace:
    steps = []
    for t, row in enumerate(rows):
        ref = None if probe is None else probe.steps[t]
        steps.append(
            row_step(row, token=None if ref is None else ref.token, probe_ids=None if ref is None else ref.ids)
        )
    return Trace(tuple(steps))


def _rows(n: int = 5, seed: int = 0) -> list[np.ndarray]:
    return [_rng("rows", seed, t).normal(0.0, 3.0, V).astype(np.float32) for t in range(n)]


def test_check_exact_accepts_identical_and_names_a_one_bit_difference() -> None:
    rows = _rows()
    ref = _trace(rows)
    assert check_exact("same", ref, _trace(rows, ref)) == []
    bumped = [r.copy() for r in rows]
    bumped[3][17] = np.nextafter(bumped[3][17], np.float32(np.inf))  # one ulp in one entry of one row
    findings = check_exact("bump", ref, _trace(bumped, ref))
    assert len(findings) == 1 and "step 3" in findings[0]


def test_check_exact_fails_on_a_short_trace_and_on_a_nan() -> None:
    rows = _rows()
    ref = _trace(rows)
    assert check_exact("short", ref, _trace(rows[:3], ref))
    nan = [r.copy() for r in rows]
    nan[1][0] = np.nan
    assert any("non-finite" in f for f in check_exact("nan", ref, _trace(nan, ref)))


def test_check_parity_needs_equal_length_in_both_directions() -> None:
    rows = _rows()
    ref = _trace(rows)
    assert check_parity("same", ref, _trace(rows, ref)) == []
    longer = check_parity("long", ref, _extended(ref))
    assert len(longer) == 1 and longer[0].startswith("parity ") and "6 steps, the reference has 5" in longer[0]
    shorter = check_parity("short", ref, _trace(rows[:3], ref))
    assert len(shorter) == 1 and shorter[0].startswith("parity ") and "3 steps, the reference has 5" in shorter[0]


def _extended(ref: Trace) -> Trace:
    """``ref`` with one more step on the end: every step of ``ref`` is the same."""
    extra = row_step(_rows(1, seed=9)[0])
    return Trace((*ref.steps, extra))


def test_require_lists_every_finding_and_caps_the_message() -> None:
    require([], "nothing")
    with pytest.raises(GateError) as err:
        require([f"f{i}" for i in range(30)], "title")
    assert "30 finding(s)" in str(err.value) and "and 18 more" in str(err.value)


# --------------------------------------------------------------------------- #
# Equivalence is measured, never gated
# --------------------------------------------------------------------------- #


def test_measure_pairs_gives_a_distribution_and_worst_cases_per_class() -> None:
    pairs, _ = batch_pairs(FakeEngine(mixes=False), True, 12)
    pairs += batch_pairs(FakeEngine(mixes=False), False, 12)[0]
    stats = measure_pairs(pairs)
    assert set(stats) == {"base", "adapter"}
    for c in stats.values():
        assert c["p50"] <= c["p99"] <= c["max"]
        assert c["worst"][0]["max_abs_dlogprob"] == c["max"]
        assert {"forced_lp_ref", "forced_lp_got", "tv_over_probe_ids", "ref_top8_with_got_lp"} <= set(c["worst"][0])
    # the adapter path amplifies noise in this fake, as the first card measurement suggests for the real one
    assert stats["adapter"]["max"] > stats["base"]["max"]


def _one_pair(change) -> Pair:
    rows = _rows(3)
    ref = _trace(rows)
    got = _trace([change(t, r.copy()) for t, r in enumerate(rows)], ref)
    return Pair("p", "c", None, ref, got)


def test_a_difference_on_tokens_without_mass_is_labelled_tail() -> None:
    def tail(t: int, row: np.ndarray) -> np.ndarray:
        if t == 1:
            row[int(np.argsort(row)[-12])] -= 15.0  # a token far below the top moves by 15 logits and holds no mass
        return row

    case = measure_pairs([_one_pair(tail)])["base"]["worst"][0]
    assert case["max_abs_dlogprob"] > 10 and case["kind"] == "tail" and case["top1_agree"]


def test_a_distribution_that_differs_broadly_is_labelled_broad() -> None:
    def broad(t: int, row: np.ndarray) -> np.ndarray:
        if t == 1:
            row[int(np.argmax(row))] -= 6.0  # the top token loses its mass
        return row

    case = measure_pairs([_one_pair(broad)])["base"]["worst"][0]
    assert case["kind"] == "broad" and case["tv_over_probe_ids"] > 0.05


def test_measurement_with_a_broken_teacher_forcing_is_refused() -> None:
    a, b = _trace(_rows(3, 1)), _trace(_rows(3, 2))  # each took its own argmax
    with pytest.raises(GateError, match="teacher forcing broke"):
        measure_pairs([Pair("p", "c", None, a, b)])


def test_report_measurement_writes_the_record_and_says_it_is_not_gated(tmp_path: Path) -> None:
    pairs, _ = batch_pairs(FakeEngine(mixes=False, noise=5.0, adapter_noise=5.0), True, 12)  # huge noise
    path = tmp_path / "m" / "equivalence.jsonl"
    with pytest.warns(UserWarning, match="MEASURED, not gated"):
        record = report_measurement("t", pairs, path=path)  # a huge difference raises nothing
    line = json.loads(path.read_text().splitlines()[0])
    assert line["title"] == "t" and set(line["provenance"]) == {"backend", "build", "device", "date"}
    assert line["classes"]["adapter"]["max"] == record["classes"]["adapter"]["max"] > 5


# --------------------------------------------------------------------------- #
# The exact gates on a correct engine with Vulkan-like noise: they pass
# --------------------------------------------------------------------------- #


def test_a_correct_mixed_engine_passes_every_exact_gate() -> None:
    pool = FakeEngine(mixes=False)
    fork = FakeEngine(mixes=True)
    requests = _requests()
    iso = collect_isolated(pool, requests)
    require(gate_determinism(fork, requests), "determinism")
    for target in range(len(requests)):
        require(gate_isolation(fork, requests, pool=POOL, target=target), f"isolation {target}")
    fork_iso = collect_isolated_forced(fork, requests, iso)
    require([f for a, b in zip(iso, fork_iso, strict=True) for f in check_parity("p", a, b)], "parity")
    # the noise is real: a mixed batch is NOT bit-identical to the isolated run, which is why token-exact fails
    mixed = collect_mixed(fork, requests, iso)
    assert any(check_exact("mix", a, b) for a, b in zip(iso, mixed, strict=True))
    # and it is measured, not gated: reporting it raises nothing
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report_measurement("t", pairs_from_traces(requests, iso, mixed, "mixed4"), path=Path("/dev/null"))
    assert any("MEASURED, not gated" in str(w.message) for w in caught)


def test_a_correct_context_pool_engine_passes_the_cross_context_gate() -> None:
    pool = FakeEngine(mixes=False)
    requests = _requests()
    for target in (0, 1, 2):
        require(gate_isolation(pool, requests, pool=POOL, target=target), f"cross-context {target}")


# --------------------------------------------------------------------------- #
# Planted faults: an exact gate fails
# --------------------------------------------------------------------------- #


def test_a_planted_bleed_through_fails_isolation_even_when_it_is_tiny() -> None:
    leaky = FakeEngine(mixes=True, leak=1e-3)
    findings = gate_isolation(leaky, _requests(), pool=POOL, target=0)
    assert findings and any("isolation target 0" in f for f in findings)


def test_a_planted_bleed_through_between_contexts_fails_the_cross_context_gate() -> None:
    leaky_pool = FakeEngine(mixes=False, leak=1e-3)
    findings = gate_isolation(leaky_pool, _requests(), pool=POOL, target=0)
    assert any("cross-context target" in f for f in findings)


def test_a_planted_adapter_drop_on_every_neighbour_fails_isolation_as_vacuous() -> None:
    dropping = FakeEngine(mixes=True, drop_rows=frozenset({1, 2, 3}))  # rows 1.. lose their adapter in a mixed batch
    findings = gate_isolation(dropping, _requests(), pool=POOL, target=0)
    assert any("vacuous for neighbour" in f for f in findings)


def test_a_planted_adapter_drop_on_one_row_fails_isolation_for_the_other_targets() -> None:
    # Only row 1 (probe-b) loses its adapter in a mixed batch. Target 1 sees a constant row, so
    # it passes. Every other target sees row 1 as a neighbour that never reacts. That fails.
    dropping = FakeEngine(mixes=True, drop_rows=frozenset({1}))
    requests = _requests()
    assert gate_isolation(dropping, requests, pool=POOL, target=1) == []
    findings = gate_isolation(dropping, requests, pool=POOL, target=0)
    assert any("vacuous for neighbour 1" in f for f in findings)


def test_a_planted_adapter_drop_that_reaches_a_row_alone_fails_parity() -> None:
    requests = _requests()
    iso = collect_isolated(FakeEngine(mixes=False), requests)
    dropping = FakeEngine(mixes=True, drop_always=True)  # seq_routed never applies the adapter
    got = collect_isolated_forced(dropping, requests, iso)
    findings = [f for a, b in zip(iso, got, strict=True) for f in check_parity("p", a, b)]
    assert findings and all(f.startswith("parity ") for f in findings)


def test_a_planted_adapter_scale_error_fails_parity() -> None:
    requests = _requests()
    iso = collect_isolated(FakeEngine(mixes=False), requests)
    off = FakeEngine(mixes=True, scale=0.98)  # the adapter delta is 2 % low on every row
    off_iso = collect_isolated_forced(off, requests, iso)
    findings = [f for a, b in zip(iso, off_iso, strict=True) for f in check_parity("p", a, b)]
    assert findings and all(f.startswith("parity ") for f in findings)


def test_a_reset_that_does_not_reset_fails_determinism() -> None:
    sticky = FakeEngine(mixes=True, history_noise=0.2, reset_clears=False)
    findings = gate_determinism(sticky, _requests())
    assert findings and "determinism request" in findings[0]
    # the same engine with a working reset passes
    assert gate_determinism(FakeEngine(mixes=True, history_noise=0.2), _requests()) == []


def test_isolation_needs_a_neighbour_of_another_adapter() -> None:
    only = _requests(adapters=("probe-a", "probe-a", "probe-a", "probe-a"))
    findings = gate_isolation(FakeEngine(mixes=False), only, pool=POOL, target=0)
    assert findings and "needs a target group" in findings[0]
    # the direct call too
    assert gate_cross_context_isolation(FakeEngine(mixes=False), only, target_adapter="probe-a", pool=POOL)
