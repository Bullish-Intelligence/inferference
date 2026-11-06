"""Run one experiment on two cards at once: one whole model instance on each card.

A dual-card spec declares `fixed.placement` (two distinct DRM cards). For each
scheduled run, `run_pair` starts two threads in this one process. Each thread
calls the same `_run_one` that a single-card run uses, with its own card, port,
and artifact root. The two threads meet at named rendezvous points through
`PairSync`. The rendezvous points make the two runs share one workload clock.

The order of one pair run is:

1. `run`: both runs write their manifest, then start preflight together.
2. `preflight`: both preflights pass before any server loads.
3. The runs load one at a time (`load_section`). The second run's "before"
   snapshot then holds the first card's loaded VRAM, so the residency check
   "other cards undisturbed" can pass.
4. `loaded`: both servers are healthy and both residency proofs passed.
5. `gate`: both start gates passed. The release time of this barrier is the
   zero of the pair's workload clock.
6. `stage:<name>`: both runs enter each workload stage together.

A run that leaves the pair is *departed*. Before the `gate` release, a departed
peer ends the waiting run as `aborted` with failure boundary `pair:peer_lost`.
After the release, a departed peer counts as arrived at every later barrier
(`on_arm_bound: continue_peer`). A peer that hits its arm bound then does not
hold up the surviving run.

Each run keeps the single-card run-directory layout, under
`<artifact_root>/card<N>/`. After each pair, `<artifact_root>/pair/<idx>-pair.json`
names both run directories and records the release time.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .artifacts import write_pair_record
from .conditions import ScheduledRun, schedule
from .control import FanLease, FanLeaseLike, SharedFanLease
from .preflight import HostPaths, RunTarget
from .runner import (
    PairPeerLost,
    PairStopped,
    PairTimeout,
    RunnerError,
    RunResult,
    _load_spec_bytes,
    _run_one,
    _wait_for_next_run_directory_tick,
)

__all__ = [
    "GATE_BARRIER",
    "PairResult",
    "PairSync",
    "run_pair",
]

# The barrier whose completion releases the pair. Its monotonic time is the
# zero of the shared workload clock.
GATE_BARRIER = "gate"
_DEFAULT_RENDEZVOUS_TIMEOUT_S = 1800.0
_DEFAULT_POLL_S = 0.05
_DEFAULT_HOST_PATHS = HostPaths()


class PairSync:
    """Named rendezvous points for the two runs of a pair, in one process.

    One `PairSync` serves one scheduled pair run. Each run gets a handle from
    `handle(card)`, and the handle implements the runner's `RunSync` protocol.
    Every wait polls the caller's stop event, so a stopped run never blocks.
    """

    def __init__(
        self,
        cards: tuple[int, int],
        *,
        timeout_s: float = _DEFAULT_RENDEZVOUS_TIMEOUT_S,
        poll_s: float = _DEFAULT_POLL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if len(set(cards)) != 2:
            raise RunnerError(f"a pair needs two distinct cards, got {list(cards)}")
        self.cards = cards
        self._timeout_s = timeout_s
        self._poll_s = poll_s
        self._clock = clock
        self._cond = threading.Condition()
        self._arrived: dict[str, set[int]] = {}
        self._departed: set[int] = set()
        self._release_mono: float | None = None
        self._abort_reason: str | None = None
        self._load_lock = threading.Lock()

    def handle(self, card: int) -> PairHandle:
        """The `RunSync` handle for one card of the pair."""
        if card not in self.cards:
            raise RunnerError(f"card {card} is not in the pair {list(self.cards)}")
        return PairHandle(self, card)

    @property
    def release_mono(self) -> float | None:
        """The monotonic time the `gate` barrier released, or `None` before it."""
        with self._cond:
            return self._release_mono

    def abort(self, reason: str) -> None:
        """End every wait of both runs with `PairPeerLost`. The first reason stays."""
        with self._cond:
            if self._abort_reason is None:
                self._abort_reason = reason
            self._cond.notify_all()

    @property
    def aborted(self) -> str | None:
        """The reason the pair was aborted, or `None`. A running stage polls this."""
        with self._cond:
            return self._abort_reason

    def depart(self, card: int) -> None:
        """Mark one run as gone. A second call changes nothing."""
        with self._cond:
            self._departed.add(card)
            self._cond.notify_all()

    def _peer(self, card: int) -> int:
        return self.cards[1] if card == self.cards[0] else self.cards[0]

    def _check_alive(self, card: int, stop: threading.Event | None, what: str) -> None:
        # The caller holds the condition lock.
        if self._abort_reason is not None:
            raise PairPeerLost(f"the pair was aborted: {self._abort_reason}")
        peer = self._peer(card)
        if peer in self._departed and self._release_mono is None:
            raise PairPeerLost(
                f"peer card {peer} left the pair before the release, while card {card} waited for {what}"
            )
        if stop is not None and stop.is_set():
            raise PairStopped(f"card {card} was stopped while it waited for {what}")

    def arrive(self, card: int, name: str, stop: threading.Event | None = None) -> None:
        """Block until the peer also arrives at `name`.

        Raises `PairPeerLost` when the peer left before the release,
        `PairStopped` when `stop` is set, and `PairTimeout` when the peer does
        not arrive in time. After the release, a departed peer counts as arrived.
        """
        peer = self._peer(card)
        deadline = self._clock() + self._timeout_s
        with self._cond:
            self._arrived.setdefault(name, set()).add(card)
            self._cond.notify_all()
            while True:
                if self._abort_reason is not None:
                    raise PairPeerLost(f"the pair was aborted: {self._abort_reason}")
                # A peer that left before the release ends the wait, even when it
                # had already arrived here. Its arrival is then stale: it stopped
                # while it waited, and it will not leave the gate.
                if peer in self._departed and self._release_mono is None:
                    raise PairPeerLost(
                        f"peer card {peer} left the pair before the release, while card {card} waited for {name!r}"
                    )
                arrived = peer in self._arrived[name]
                gone_after_release = peer in self._departed and self._release_mono is not None
                if arrived or gone_after_release:
                    if name == GATE_BARRIER and self._release_mono is None:
                        self._release_mono = self._clock()
                        self._cond.notify_all()
                    return
                self._check_alive(card, stop, f"{name!r}")
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise PairTimeout(
                        f"peer card {peer} did not reach {name!r} within {self._timeout_s:.0f} s (card {card} waited)"
                    )
                self._cond.wait(timeout=min(self._poll_s, remaining))

    @contextlib.contextmanager
    def load_section(self, card: int, stop: threading.Event | None = None) -> Iterator[None]:
        """Hold the pair's one load slot. Only one run loads a server at a time."""
        deadline = self._clock() + self._timeout_s
        while True:
            with self._cond:
                self._check_alive(card, stop, "the load slot")
            if self._load_lock.acquire(timeout=self._poll_s):
                break
            if self._clock() >= deadline:
                raise PairTimeout(f"card {card} did not get the load slot within {self._timeout_s:.0f} s")
        try:
            # The peer may have left while this run waited for the slot.
            with self._cond:
                self._check_alive(card, stop, "the load slot")
            yield
        finally:
            self._load_lock.release()


class PairHandle:
    """One run's view of a `PairSync`. It implements the runner's `RunSync` protocol."""

    def __init__(self, sync: PairSync, card: int) -> None:
        self._sync = sync
        self.card = card

    @property
    def peer_card(self) -> int:
        return self._sync._peer(self.card)

    @property
    def release_mono(self) -> float | None:
        return self._sync.release_mono

    def arrive(self, name: str, stop: threading.Event | None = None) -> None:
        self._sync.arrive(self.card, name, stop)

    def load_section(self, stop: threading.Event | None = None) -> contextlib.AbstractContextManager[None]:
        return self._sync.load_section(self.card, stop)

    def depart(self) -> None:
        self._sync.depart(self.card)

    def abort(self, reason: str) -> None:
        self._sync.abort(reason)

    @property
    def aborted(self) -> str | None:
        return self._sync.aborted


@dataclass(frozen=True, slots=True)
class PairResult:
    """One scheduled pair run: both outcomes, the record file, and the release time."""

    index: int
    scheduled_run: ScheduledRun
    runs: Mapping[int, RunResult]
    release_mono: float | None
    first_bound_mono: float | None
    record_path: Path

    @property
    def ok(self) -> bool:
        """Both runs ended `completed` or `bounded`."""
        return all(result.state in {"completed", "bounded"} for result in self.runs.values())


def _first_bound_mono(runs: Mapping[int, RunResult]) -> float | None:
    times = [result.stop_mono for result in runs.values() if result.state == "bounded" and result.stop_mono is not None]
    return min(times) if times else None


def run_pair(
    spec_path: Path,
    targets: Sequence[RunTarget],
    *,
    artifact_root: Path,
    paths: HostPaths = _DEFAULT_HOST_PATHS,
    env: Mapping[str, str] | None = None,
    card_envs: Mapping[int, Mapping[str, str]] | None = None,
    unit_active: Callable[[], bool],
    sample_interval_s: float = 0.25,
    fan_lease_factory: Callable[[str], FanLeaseLike] = FanLease,
    http_timeout_s: float = 900,
    rendezvous_timeout_s: float = _DEFAULT_RENDEZVOUS_TIMEOUT_S,
) -> tuple[PairResult, ...]:
    """Run every scheduled pair until the first pair with a run that is not completed or bounded.

    `targets` holds one `RunTarget` per card of `fixed.placement`. The cards
    need distinct ports. `card_envs` adds environment values for one card's
    server. The two runs share one fan lease and one model-hash cache.
    """
    spec, spec_sha256 = _load_spec_bytes(spec_path)
    placement = spec.fixed.placement
    if placement is None:
        raise RunnerError("run_pair needs a spec with fixed.placement")
    by_card = {target.card: target for target in targets}
    if len(by_card) != len(targets) or set(by_card) != set(placement.cards):
        raise RunnerError(
            f"targets name cards {sorted(target.card for target in targets)}, "
            f"but fixed.placement.cards is {list(placement.cards)}"
        )
    ports = [target.port for target in targets]
    if len(set(ports)) != len(ports):
        raise RunnerError(f"both cards use one server port: {ports}")
    environment = dict(os.environ if env is None else env)
    model_hashes: dict[Path, str] = {}
    model_hash_lock = threading.Lock()
    results: list[PairResult] = []
    scheduled_runs = schedule(spec)
    for index, scheduled_run in enumerate(scheduled_runs):
        sync = PairSync(placement.cards, timeout_s=rendezvous_timeout_s)
        # One shared lease for both runs. The last release restores the fan.
        shared_lease = SharedFanLease(fan_lease_factory(spec.fixed.control.fan_profile))

        def work(
            target: RunTarget,
            sync: PairSync = sync,
            shared_lease: SharedFanLease = shared_lease,
            current_run: ScheduledRun = scheduled_run,
        ) -> RunResult:
            handle = sync.handle(target.card)
            run_env: dict[str, str] = {**environment, **(card_envs or {}).get(target.card, {})}
            try:
                return _run_one(
                    spec,
                    spec_sha256,
                    target,
                    current_run,
                    artifact_root=artifact_root / f"card{target.card}",
                    paths=paths,
                    env=run_env,
                    unit_active=unit_active,
                    sample_interval_s=sample_interval_s,
                    fan_lease_factory=lambda _profile: shared_lease.member(),
                    http_timeout_s=http_timeout_s,
                    model_hashes=model_hashes,
                    sync=handle,
                    model_hash_lock=model_hash_lock,
                )
            finally:
                # Also covers a failure before `_run_one` can depart on its own.
                handle.depart()

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="experiment-pair") as pool:
            futures = {card: pool.submit(work, by_card[card]) for card in placement.cards}
        errors = [exc for future in futures.values() if (exc := future.exception()) is not None]
        if errors:
            raise RunnerError(f"a pair run raised before it could record its outcome: {errors[0]}") from errors[0]
        runs = {card: future.result() for card, future in futures.items()}

        first_bound = _first_bound_mono(runs)
        record_path = write_pair_record(
            artifact_root / "pair",
            index,
            {
                "experiment": spec.experiment,
                "condition": scheduled_run.condition.condition_id,
                "factor_name": scheduled_run.condition.factor_name,
                "factor_value": scheduled_run.condition.factor_value,
                "replicate": scheduled_run.replicate,
                "spec_sha256": spec_sha256,
                "cards": list(placement.cards),
                "release_mono": sync.release_mono,
                "first_bound_mono": first_bound,
                "runs": {
                    str(card): {
                        "card": card,
                        "port": by_card[card].port,
                        "run_dir": str(result.run_dir),
                        "state": result.state,
                        "failure_boundary": result.failure_boundary,
                        "stop_mono": result.stop_mono,
                    }
                    for card, result in runs.items()
                },
            },
        )
        pair_result = PairResult(
            index=index,
            scheduled_run=scheduled_run,
            runs=runs,
            release_mono=sync.release_mono,
            first_bound_mono=first_bound,
            record_path=record_path,
        )
        results.append(pair_result)
        if not pair_result.ok:
            break
        if index + 1 < len(scheduled_runs):
            for result in runs.values():
                _wait_for_next_run_directory_tick(result.run_dir)
    return tuple(results)
