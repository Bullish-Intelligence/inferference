"""Tests for the dual-card pair runner: `PairSync` ordering and `run_pair` on fakes.

`PairSync` tests drive two threads against the rendezvous object alone. The
`run_pair` tests reuse the runner tests' fabricated host and fake server, then
add card 1: a second resident model, a second server port, and a pair spec.
Nothing here touches hardware or a real server.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

from inferference.experiment import cli
from inferference.experiment.pair import GATE_BARRIER, PairResult, PairSync, run_pair
from inferference.experiment.runner import PairPeerLost, PairStopped, PairTimeout, RunnerError

from .test_experiment_preflight import _build_pair_host as _build_preflight_pair_host
from .test_experiment_runner import (
    _PCI,
    _atomic_text,
    _free_port,
    _Host,
    _make_host,
    _wait_for,
)

_MIB = 1024 * 1024
# Larger than the spec's 512 MiB `gtt_limit_mib`. Two servers that load at the
# same time then fail the residency check "other cards undisturbed".
_LOAD_MIB = 600


# ---- PairSync: rendezvous rules --------------------------------------------


def _in_thread(target: Callable[[], Any]) -> tuple[threading.Thread, list[BaseException | None]]:
    outcome: list[BaseException | None] = [None]

    def run() -> None:
        try:
            target()
        except BaseException as exc:  # the test inspects every kind
            outcome[0] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def test_both_runs_leave_a_barrier_together_and_the_gate_records_one_release_time() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    first, second = sync.handle(0), sync.handle(1)
    order: list[str] = []

    def run_second() -> None:
        time.sleep(0.05)
        order.append("second arrives")
        second.arrive(GATE_BARRIER)

    thread, outcome = _in_thread(run_second)
    first.arrive(GATE_BARRIER)
    order.append("first released")
    thread.join(timeout=2)

    assert outcome == [None]
    assert order == ["second arrives", "first released"]
    assert first.release_mono is not None
    assert first.release_mono == second.release_mono == sync.release_mono
    assert first.peer_card == 1
    assert second.peer_card == 0


def test_a_barrier_before_the_gate_does_not_set_the_release_time() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    thread, _ = _in_thread(lambda: sync.handle(1).arrive("loaded"))
    sync.handle(0).arrive("loaded")
    thread.join(timeout=2)

    assert sync.release_mono is None


def test_a_peer_that_departs_before_the_release_ends_the_wait_with_peer_lost() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    waiter = sync.handle(0)
    thread, outcome = _in_thread(lambda: waiter.arrive("loaded"))
    time.sleep(0.05)

    sync.handle(1).depart()
    thread.join(timeout=2)

    assert isinstance(outcome[0], PairPeerLost)
    assert "peer card 1" in str(outcome[0])
    assert sync.release_mono is None


def test_a_peer_that_departed_earlier_ends_a_later_arrival_with_peer_lost() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    sync.handle(1).depart()

    with pytest.raises(PairPeerLost):
        sync.handle(0).arrive("preflight")


def test_the_load_section_refuses_a_departed_peer_before_the_release() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    sync.handle(1).depart()

    with pytest.raises(PairPeerLost), sync.handle(0).load_section():
        pytest.fail("a lost pair must not start a load")


def test_after_the_release_a_departed_peer_counts_as_arrived_at_every_barrier() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    survivor, leaver = sync.handle(0), sync.handle(1)
    thread, outcome = _in_thread(lambda: leaver.arrive(GATE_BARRIER))
    survivor.arrive(GATE_BARRIER)
    thread.join(timeout=2)
    assert outcome == [None]

    leaver.depart()
    started = time.monotonic()
    survivor.arrive("stage:soak_300s")
    survivor.arrive("stage:lane_sweep")

    assert time.monotonic() - started < 1.0


def test_a_peer_that_departs_while_the_survivor_waits_after_the_release_frees_the_survivor() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    survivor, leaver = sync.handle(0), sync.handle(1)
    thread, _ = _in_thread(lambda: leaver.arrive(GATE_BARRIER))
    survivor.arrive(GATE_BARRIER)
    thread.join(timeout=2)

    waiter, outcome = _in_thread(lambda: survivor.arrive("stage:soak_300s"))
    time.sleep(0.05)
    leaver.depart()
    waiter.join(timeout=2)

    assert outcome == [None]


def test_depart_is_idempotent() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    handle = sync.handle(0)
    handle.depart()
    handle.depart()

    with pytest.raises(PairPeerLost):
        sync.handle(1).arrive("run")


def test_abort_ends_every_wait_even_after_the_release() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    thread, _ = _in_thread(lambda: sync.handle(1).arrive(GATE_BARRIER))
    sync.handle(0).arrive(GATE_BARRIER)
    thread.join(timeout=2)

    waiter, outcome = _in_thread(lambda: sync.handle(0).arrive("stage:soak_300s"))
    time.sleep(0.05)
    sync.abort("operator stop")
    waiter.join(timeout=2)

    assert isinstance(outcome[0], PairPeerLost)
    assert "operator stop" in str(outcome[0])


def test_a_peer_that_arrived_and_then_left_before_the_release_does_not_free_the_gate() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    stop = threading.Event()
    leaver, survivor = sync.handle(1), sync.handle(0)
    thread, outcome = _in_thread(lambda: leaver.arrive(GATE_BARRIER, stop))
    time.sleep(0.05)
    # The peer is stopped while it waits. Its arrival is then stale.
    stop.set()
    thread.join(timeout=2)
    assert isinstance(outcome[0], PairStopped)
    leaver.depart()

    with pytest.raises(PairPeerLost):
        survivor.arrive(GATE_BARRIER)

    assert sync.release_mono is None


def test_the_aborted_reason_is_readable_and_the_first_reason_stays() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    assert sync.aborted is None

    sync.handle(0).abort("first")
    sync.handle(1).abort("second")

    assert sync.aborted == "first"
    assert sync.handle(1).aborted == "first"


def test_a_stop_event_ends_the_wait() -> None:
    sync = PairSync((0, 1), timeout_s=30, poll_s=0.005)
    stop = threading.Event()
    thread, outcome = _in_thread(lambda: sync.handle(0).arrive("loaded", stop))
    time.sleep(0.05)

    stop.set()
    thread.join(timeout=2)

    assert isinstance(outcome[0], PairStopped)


def test_a_missing_peer_times_the_wait_out() -> None:
    sync = PairSync((0, 1), timeout_s=0.1, poll_s=0.005)

    with pytest.raises(PairTimeout, match="did not reach 'run'"):
        sync.handle(0).arrive("run")


def test_the_load_section_lets_one_run_load_at_a_time() -> None:
    sync = PairSync((0, 1), timeout_s=5, poll_s=0.005)
    inside = 0
    peak = 0
    guard = threading.Lock()

    def load(card: int) -> None:
        nonlocal inside, peak
        with sync.handle(card).load_section():
            with guard:
                inside += 1
                peak = max(peak, inside)
            time.sleep(0.05)
            with guard:
                inside -= 1

    threads = [_in_thread(lambda card=card: load(card))[0] for card in (0, 1)]
    for thread in threads:
        thread.join(timeout=3)

    assert peak == 1


def test_a_waiter_for_the_load_slot_sees_a_stop_event() -> None:
    sync = PairSync((0, 1), timeout_s=30, poll_s=0.005)
    stop = threading.Event()
    holder_in = threading.Event()
    holder_out = threading.Event()

    def hold() -> None:
        with sync.handle(0).load_section():
            holder_in.set()
            holder_out.wait(timeout=5)

    holder, _ = _in_thread(hold)
    assert holder_in.wait(timeout=2)
    waiter, outcome = _in_thread(lambda: sync.handle(1).load_section(stop).__enter__())
    time.sleep(0.05)
    stop.set()
    waiter.join(timeout=2)
    holder_out.set()
    holder.join(timeout=2)

    assert isinstance(outcome[0], PairStopped)


def test_a_pair_needs_two_distinct_cards() -> None:
    with pytest.raises(RunnerError, match="two distinct cards"):
        PairSync((1, 1))
    with pytest.raises(RunnerError, match="not in the pair"):
        PairSync((0, 1)).handle(2)


# ---- run_pair on the fabricated host -----------------------------------------


@dataclasses.dataclass
class _PairHost:
    host: _Host
    targets: list[Any]
    card_envs: dict[int, dict[str, str]]

    @property
    def artifact_root(self) -> Path:
        return self.host.artifact_root

    def device(self, card: int) -> Path:
        return self.host.paths.drm_root / "pci-devices" / _PCI[card]

    def hwmon(self, card: int) -> Path:
        return self.device(card) / "hwmon" / f"hwmon{card + 1}"


@contextmanager
def _make_pair_host(
    tmp_path: Path,
    *,
    workloads: tuple[str, ...] = ("lane_sweep",),
    power_cap_w: int | None = None,
) -> Iterator[_PairHost]:
    """The runner tests' host, extended with card 1 and a `fixed.placement` spec."""
    with _make_host(tmp_path, workloads=workloads, power_cap_w=power_cap_w) as host:
        config_root = host.target.config_root
        (config_root / "placement.yaml").write_text(
            yaml.safe_dump(
                {
                    "devices": [
                        {"id": "Vulkan0", "card": 0, "reserve_mib": 100},
                        {"id": "Vulkan1", "card": 1, "reserve_mib": 100},
                    ],
                    "resident": [{"model": "m", "device": "Vulkan0"}, {"model": "m1", "device": "Vulkan1"}],
                    "router": {"host": "127.0.0.1", "port": 8100, "models_max": 2},
                }
            )
        )
        shutil.copyfile(config_root / "models" / "m.yaml", config_root / "models" / "m1.yaml")
        raw = yaml.safe_load(host.spec_path.read_text())
        raw["fixed"]["placement"] = {"mode": "per-card", "cards": [0, 1], "on_arm_bound": "continue_peer"}
        host.spec_path.write_text(yaml.safe_dump(raw, sort_keys=False))
        targets = [
            host.target,
            dataclasses.replace(host.target, card=1, port=_free_port(), expect_pci=_PCI[1]),
        ]
        card_envs = {card: {"FAKE_SERVER_PCI": _PCI[card]} for card in (0, 1)}
        yield _PairHost(host=host, targets=targets, card_envs=card_envs)


class _PairLoader:
    """Raise a card's fake VRAM while its fake server runs, for both cards."""

    def __init__(self, pair: _PairHost) -> None:
        self.pair = pair
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="fake-pair-loader", daemon=True)
        self.loaded: set[Path] = set()

    def __enter__(self) -> _PairLoader:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1)

    def _run(self) -> None:
        baseline = 100 * _MIB
        while not self.stop_event.is_set():
            for card in (0, 1):
                root = self.pair.artifact_root / f"card{card}"
                if not root.exists():
                    continue
                usage_file = self.pair.device(card) / "mem_info_vram_used"
                for run_dir in root.iterdir():
                    if (run_dir / ".server-output").exists():
                        if run_dir not in self.loaded:
                            _atomic_text(usage_file, f"{baseline + _LOAD_MIB * _MIB}\n")
                            self.loaded.add(run_dir)
                    elif run_dir in self.loaded and (run_dir / "server.log").exists():
                        _atomic_text(usage_file, f"{baseline}\n")
                        self.loaded.remove(run_dir)
            self.stop_event.wait(0.002)


def _run_pair(pair: _PairHost, *, card_envs: dict[int, dict[str, str]] | None = None, **kwargs: Any):
    host = pair.host
    with _PairLoader(pair):
        return run_pair(
            host.spec_path,
            pair.targets,
            artifact_root=host.artifact_root,
            paths=host.paths,
            env=host.env(),
            card_envs=pair.card_envs if card_envs is None else card_envs,
            unit_active=lambda: True,
            sample_interval_s=0.01,
            fan_lease_factory=host.lease_factory,
            http_timeout_s=3,
            rendezvous_timeout_s=20,
            **kwargs,
        )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def test_a_pair_completes_with_one_run_directory_per_card(tmp_path: Path) -> None:
    with _make_pair_host(tmp_path) as pair:
        results = _run_pair(pair)

        assert len(results) == 1
        result = results[0]
        assert result.ok
        assert {card: run.state for card, run in result.runs.items()} == {0: "completed", 1: "completed"}
        for card, run in result.runs.items():
            assert run.run_dir.parent == pair.artifact_root / f"card{card}"
            assert (run.run_dir / "outcome.json").is_file()
            assert (run.run_dir / "server.log").is_file()
        # Each run measured its own card with its own resident model.
        assert _json(result.runs[0].run_dir / "preflight.json")["device"]["card"] == 0
        assert _json(result.runs[1].run_dir / "preflight.json")["device"]["card"] == 1
        assert _json(result.runs[1].run_dir / "identity.json")["card_pci"] == _PCI[1]
        # The residency proof passed on both cards, so the loads did not overlap.
        for run in result.runs.values():
            checks = _json(run.run_dir / "residency.json")["checks"]
            assert all(check["passed"] for check in checks)
        # One fan lease: the first release was deferred, the last one real.
        fans = [_json(run.run_dir / "control.json")["fan"] for run in result.runs.values()]
        assert sorted("deferred" in fan for fan in fans) == [False, True]
        assert all(fan["restore_confirmed"] is True for fan in fans)
        assert not pair.host.request_file.exists()


def test_the_pair_shares_one_workload_clock(tmp_path: Path) -> None:
    with _make_pair_host(tmp_path, workloads=("soak_1s", "lane_sweep")) as pair:
        result = _run_pair(pair)[0]

        stages = {card: _json(run.run_dir / "stages.json") for card, run in result.runs.items()}
        release = result.release_mono
        assert release is not None
        for card, payload in stages.items():
            assert payload["pair"]["release_mono"] == release
            assert payload["pair"]["peer_card"] == 1 - card
            assert payload["start_gate"]["passed"] is True
            workload_stages = [item for item in payload["stages"] if item["name"] != "residency_probe"]
            assert [item["name"] for item in workload_stages] == ["soak_1s", "lane_sweep"]
            assert all(item["started_mono"] >= release for item in workload_stages)
        for index in (0, 1):
            started = [
                [item for item in stages[card]["stages"] if item["name"] != "residency_probe"][index]["started_mono"]
                for card in (0, 1)
            ]
            assert abs(started[0] - started[1]) < 0.5


def test_the_pair_record_names_both_run_directories(tmp_path: Path) -> None:
    with _make_pair_host(tmp_path) as pair:
        result = _run_pair(pair)[0]

        record = _json(pair.artifact_root / "pair" / "00-pair.json")
        assert record["schema_version"] == 1
        assert record["cards"] == [0, 1]
        assert record["release_mono"] == result.release_mono
        assert record["first_bound_mono"] is None
        for card in (0, 1):
            entry = record["runs"][str(card)]
            assert entry["run_dir"] == str(result.runs[card].run_dir)
            assert entry["state"] == "completed"
            assert entry["port"] == pair.targets[card].port
        assert result.record_path == pair.artifact_root / "pair" / "00-pair.json"


def test_a_cap_pair_checks_the_cap_on_both_cards(tmp_path: Path) -> None:
    with _make_pair_host(tmp_path, power_cap_w=180) as pair:
        results = _run_pair(pair)

        assert results[0].ok
        for run in results[0].runs.values():
            identity = _json(run.run_dir / "identity.json")
            assert identity["pp_table"]["live_power1_cap_uw"] == 180_000_000


def test_drift_on_the_peer_cap_invalidates_both_runs(tmp_path: Path) -> None:
    with _make_pair_host(tmp_path, workloads=("soak_2s",), power_cap_w=180) as pair:

        def reset_peer_cap() -> None:
            if not _wait_for(
                lambda: any(
                    len(tuple((directory / "requests").glob("*.request.json"))) >= 2
                    for directory in (pair.artifact_root / "card0").glob("*")
                )
            ):
                return
            # A GPU reset puts the stock cap back on card 1 only.
            _atomic_text(pair.hwmon(1) / "power1_cap", "250000000\n")

        thread, _ = _in_thread(reset_peer_cap)
        results = _run_pair(pair)
        thread.join(timeout=2)

        assert not results[0].ok
        for run in results[0].runs.values():
            assert run.state == "invalid"
            assert run.failure_boundary == "drift:power_cap"
            outcome = _json(run.run_dir / "outcome.json")
            assert "card 1" in outcome["detail"]


def test_a_peer_that_hits_its_arm_bound_does_not_stop_the_survivor(tmp_path: Path) -> None:
    with _make_pair_host(tmp_path, workloads=("soak_3s", "lane_sweep")) as pair:

        def heat_card_one() -> None:
            if not _wait_for(
                lambda: any(
                    len(tuple((directory / "requests").glob("*.request.json"))) >= 2
                    for directory in (pair.artifact_root / "card1").glob("*")
                )
            ):
                return
            for sensor in ("temp1_input", "temp2_input", "temp3_input"):
                _atomic_text(pair.hwmon(1) / sensor, "76000\n")

        thread, _ = _in_thread(heat_card_one)
        results = _run_pair(pair)
        thread.join(timeout=2)

        result = results[0]
        assert result.ok
        assert result.runs[1].state == "bounded"
        assert result.runs[1].failure_boundary == "safety:arm_bound"
        assert result.runs[0].state == "completed"
        # The survivor went on to the stage after the peer left.
        survivor_stages = [item["name"] for item in _json(result.runs[0].run_dir / "stages.json")["stages"]]
        assert survivor_stages == ["residency_probe", "soak_3s", "lane_sweep"]
        assert result.first_bound_mono is not None
        assert result.release_mono is not None
        assert result.first_bound_mono > result.release_mono
        record = _json(pair.artifact_root / "pair" / "00-pair.json")
        assert record["first_bound_mono"] == result.first_bound_mono
        assert _json(result.runs[1].run_dir / "stages.json")["pair"]["stop_mono"] == result.first_bound_mono


def test_a_machine_limit_event_on_one_run_stops_the_other_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferference.experiment.safety import SafetyEvent, SafetyMonitor

    original = SafetyMonitor._check_machine_limit
    tripped = threading.Event()

    def check_machine_limit(self: SafetyMonitor, sample: Any) -> SafetyEvent | None:
        # Only the monitor of card 1 sees the event. Card 0's own sampler misses it.
        if tripped.is_set() and self._arms[0].card == 1:
            return SafetyEvent(verdict="machine_limit", card=1, observed="forced 81 C", mono=sample.monotonic)
        return original(self, sample)

    monkeypatch.setattr(SafetyMonitor, "_check_machine_limit", check_machine_limit)
    with _make_pair_host(tmp_path, workloads=("soak_3s", "lane_sweep")) as pair:

        def trip() -> None:
            if _wait_for(
                lambda: any(
                    len(tuple((directory / "requests").glob("*.request.json"))) >= 2
                    for directory in (pair.artifact_root / "card1").glob("*")
                )
            ):
                tripped.set()

        thread, _ = _in_thread(trip)
        results = _run_pair(pair)
        thread.join(timeout=2)

        result = results[0]
        assert not result.ok
        assert result.runs[1].state == "aborted"
        assert result.runs[1].failure_boundary == "safety:machine_limit"
        assert result.runs[0].state == "aborted"
        assert result.runs[0].failure_boundary == "pair:peer_lost"
        assert "safety:machine_limit" in _json(result.runs[0].run_dir / "outcome.json")["detail"]
        survivor_stages = [item["name"] for item in _json(result.runs[0].run_dir / "stages.json")["stages"]]
        assert "lane_sweep" not in survivor_stages


def test_a_peer_that_fails_to_load_ends_the_other_run_as_peer_lost(tmp_path: Path) -> None:
    with _make_pair_host(tmp_path) as pair:
        broken = {**pair.card_envs, 1: {**pair.card_envs[1], "FAKE_SERVER_EXIT_BEFORE_BIND": "1"}}

        results = _run_pair(pair, card_envs=broken)

        result = results[0]
        assert not result.ok
        assert result.runs[1].state == "aborted"
        assert result.runs[1].failure_boundary == "server:health"
        assert result.runs[0].state == "aborted"
        assert result.runs[0].failure_boundary == "pair:peer_lost"
        assert result.release_mono is None
        # The survivor never reached a workload stage.
        stages = _json(result.runs[0].run_dir / "stages.json")
        assert [item["name"] for item in stages["stages"]] in ([], ["residency_probe"])


def test_a_refused_preflight_on_one_card_ends_the_pair_before_any_load(tmp_path: Path) -> None:
    with _make_pair_host(tmp_path) as pair:
        _atomic_text(pair.device(1) / "mem_info_vram_used", f"{2048 * _MIB}\n")

        results = _run_pair(pair)

        result = results[0]
        assert not result.ok
        # Card 0 refuses its busy peer. Card 1 passes its own preflight, then
        # finds that its peer left before the "preflight" barrier.
        assert result.runs[0].failure_boundary == "preflight:peer_idle"
        assert result.runs[1].failure_boundary == "pair:peer_lost"
        for run in result.runs.values():
            assert not (run.run_dir / "server.log").exists()


def test_the_schedule_stops_at_the_first_pair_that_is_not_completed_or_bounded(tmp_path: Path) -> None:
    with _make_pair_host(tmp_path) as pair:
        raw = yaml.safe_load(pair.host.spec_path.read_text())
        raw["factor"]["values"] = ["1", "2"]
        pair.host.spec_path.write_text(yaml.safe_dump(raw, sort_keys=False))
        _atomic_text(pair.device(1) / "mem_info_vram_used", f"{2048 * _MIB}\n")

        results = _run_pair(pair)

        assert len(results) == 1
        assert not results[0].ok
        assert sorted(path.name for path in (pair.artifact_root / "pair").iterdir()) == ["00-pair.json"]


def test_two_scheduled_pairs_run_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The second pair reuses both ports, which can sit in TIME_WAIT.
    monkeypatch.setattr("inferference.experiment.runner.wait_for_bindable_port", lambda _port, **_kwargs: None)
    with _make_pair_host(tmp_path) as pair:
        raw = yaml.safe_load(pair.host.spec_path.read_text())
        raw["factor"]["values"] = ["1", "2"]
        pair.host.spec_path.write_text(yaml.safe_dump(raw, sort_keys=False))

        results = _run_pair(pair)

        assert [result.index for result in results] == [0, 1]
        assert all(result.ok for result in results)
        assert [result.scheduled_run.condition.factor_value for result in results] == ["1", "2"]
        assert sorted(path.name for path in (pair.artifact_root / "pair").iterdir()) == ["00-pair.json", "01-pair.json"]
        assert not pair.host.request_file.exists()


def test_run_pair_hashes_the_model_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferference.experiment import runner

    hashed: list[Path] = []
    original = runner._hash_file

    def counting_hash(path: Path, check_stop: Callable[[], None] | None = None) -> str:
        hashed.append(path)
        time.sleep(0.05)
        return original(path, check_stop)

    monkeypatch.setattr(runner, "_hash_file", counting_hash)
    with _make_pair_host(tmp_path) as pair:
        results = _run_pair(pair)

        assert results[0].ok
        assert len(hashed) == 1


def test_run_pair_refuses_targets_that_do_not_match_the_spec(tmp_path: Path) -> None:
    with _make_pair_host(tmp_path) as pair:
        pair.targets = pair.targets[:1]
        with pytest.raises(RunnerError, match="targets name cards"):
            _run_pair(pair)

        pair.targets = [pair.host.target, dataclasses.replace(pair.host.target, card=1)]
        with pytest.raises(RunnerError, match="one server port"):
            _run_pair(pair)


def test_run_pair_refuses_a_single_card_spec(tmp_path: Path) -> None:
    with _make_pair_host(tmp_path) as pair:
        raw = yaml.safe_load(pair.host.spec_path.read_text())
        del raw["fixed"]["placement"]
        pair.host.spec_path.write_text(yaml.safe_dump(raw, sort_keys=False))

        with pytest.raises(RunnerError, match="fixed.placement"):
            _run_pair(pair)


# ---- command interface -------------------------------------------------------


def _pair_argv(command: str, spec: Path, **overrides: str | None) -> list[str]:
    values: dict[str, str | None] = {
        "--cards": "0,1",
        "--ports": "18080,18081",
        "--server-bin": "/bin/true",
        "--config-root": "/tmp/config",
        "--stock-pp-table": "/tmp/stock.pp_table",
        "--artifact-root": "/tmp/artifacts",
        **overrides,
    }
    argv = [command, str(spec)]
    for flag, value in values.items():
        if value is not None:
            argv.extend((flag, value))
    argv.extend(("--expect-pci", "0=0000:04:00.0", "--expect-pci", "1=0000:03:00.0", "--stall-floor", "2=1000"))
    return argv


@pytest.mark.parametrize("command", ("preflight-pair", "run-pair", "analyze-pair"))
def test_each_pair_subcommand_help_returns_zero(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([command, "--help"]) == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"--cards": "1,1"}, "two distinct cards"),
        ({"--cards": "0"}, "two comma-separated integers"),
        ({"--cards": "a,b"}, "two comma-separated integers"),
        ({"--ports": "5000,5000"}, "two distinct ports"),
        ({"--ports": "0,5000"}, "between 1 and 65535"),
        ({"--server-bin": None}, "--server-bin"),
    ],
)
def test_pair_arguments_are_checked(
    override: dict[str, str | None], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_pair_argv("run-pair", Path("spec.yaml"), **override)) == 2
    assert message in capsys.readouterr().err


def test_a_pair_expect_pci_must_name_a_listed_card(capsys: pytest.CaptureFixture[str]) -> None:
    argv = _pair_argv("run-pair", Path("spec.yaml"), **{"--cards": "0,2"})

    assert cli.main(argv) == 2
    assert "not in --cards" in capsys.readouterr().err


def test_run_pair_builds_one_target_per_card_and_reports_each_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from inferference.experiment.conditions import Condition, ScheduledRun
    from inferference.experiment.runner import RunResult

    seen: dict[str, Any] = {}

    def fake_run_pair(spec_path: Path, targets: Any, **kwargs: Any) -> tuple[PairResult, ...]:
        seen.update(spec=spec_path, targets=list(targets), kwargs=kwargs)
        scheduled = ScheduledRun(Condition("exp", "lanes", "8", "d" * 64), 1)
        runs = {
            card: RunResult(run_dir=tmp_path / f"card{card}" / "run", state=state, failure_boundary=boundary)
            for card, state, boundary in ((0, "completed", None), (1, "bounded", "safety:arm_bound"))
        }
        return (PairResult(0, scheduled, runs, 123.0, 130.0, tmp_path / "pair" / "00-pair.json"),)

    monkeypatch.setattr(cli, "run_pair", fake_run_pair)

    code = cli.main(_pair_argv("run-pair", tmp_path / "spec.yaml"))

    assert code == 1  # a bounded run is not a completed run, as with `run`
    targets = seen["targets"]
    assert [(target.card, target.port, target.expect_pci) for target in targets] == [
        (0, 18080, "0000:04:00.0"),
        (1, 18081, "0000:03:00.0"),
    ]
    assert targets[0].stall_floors_rpm == {2: 1000.0}
    assert seen["kwargs"]["artifact_root"] == Path("/tmp/artifacts")
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is False
    assert result["pairs"][0]["release_mono"] == 123.0
    assert result["pairs"][0]["first_bound_mono"] == 130.0
    assert result["pairs"][0]["runs"]["1"]["failure_boundary"] == "safety:arm_bound"


def test_preflight_pair_reports_a_passing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths, target, peer_target, spec = _build_preflight_pair_host(tmp_path)
    spec_path = tmp_path / "pair.yaml"
    spec_path.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    monkeypatch.setattr(cli, "_host_paths", lambda: paths)
    monkeypatch.setattr(cli, "unit_active", lambda: True)
    monkeypatch.setenv("GPU_THERMAL_MAX_C", "80")

    argv = [
        "preflight-pair",
        str(spec_path),
        "--cards",
        "0,1",
        "--ports",
        f"{target.port},{peer_target.port}",
        "--server-bin",
        str(target.server_bin),
        "--config-root",
        str(target.config_root),
        "--stock-pp-table",
        str(target.stock_pp_table),
        "--thermal-guard",
        str(target.thermal_guard),
        "--gpu-lease",
        str(target.gpu_lease),
        "--expect-pci",
        f"0={target.expect_pci}",
        "--expect-pci",
        f"1={peer_target.expect_pci}",
        "--stall-floor",
        "2=1000",
    ]
    code = cli.main(argv)

    result = json.loads(capsys.readouterr().out)
    assert code == 0, result
    assert result["passed"] is True
    assert sorted(result["cards"]) == ["0", "1"]
    names = [check["name"] for check in result["cards"]["1"]["checks"]]
    assert names[-3:] == ["placement_pair", "peer_cap", "peer_idle"]


def test_preflight_pair_reports_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths, target, peer_target, spec = _build_preflight_pair_host(tmp_path)
    spec_path = tmp_path / "pair.yaml"
    spec_path.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    monkeypatch.setattr(cli, "_host_paths", lambda: paths)
    monkeypatch.setattr(cli, "unit_active", lambda: True)
    monkeypatch.setenv("GPU_THERMAL_MAX_C", "80")

    code = cli.main(
        [
            "preflight-pair",
            str(spec_path),
            "--cards",
            "0,1",
            "--ports",
            "18080,18081",
            "--server-bin",
            str(target.server_bin),
            "--config-root",
            str(target.config_root),
            "--stock-pp-table",
            str(tmp_path / "missing.pp_table"),
            "--thermal-guard",
            str(target.thermal_guard),
            "--gpu-lease",
            str(target.gpu_lease),
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["passed"] is False
    assert result["check"] == "pp_table"


# ---- the dual spec set ---------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("watts", (160, 180, 200, 220, 250))
def test_each_dual_spec_matches_its_single_card_arm_plus_a_placement(watts: int) -> None:
    from inferference.experiment.conditions import hash_fixed
    from inferference.experiment.spec import load_spec

    specs = Path(
        os.environ.get("INFERFERENCE_EXPERIMENT_SPECS", "~/.local/share/inferference/experiment-specs")
    ).expanduser()
    dual_path = specs / "033-v620-cooling-v2-dual" / f"arm-{watts}w.yaml"
    single_path = specs / "032-v620-board-power-cap" / f"arm-{watts}w.yaml"
    if not (dual_path.is_file() and single_path.is_file()):
        pytest.skip(
            f"experiment specs not present (looked in {specs}): they are not redistributed with "
            "this repository; set INFERFERENCE_EXPERIMENT_SPECS to the directory that holds them"
        )
    dual = load_spec(dual_path)
    single = load_spec(single_path)

    assert dual.experiment == f"v620-cooling-v2-dual-cap-{watts}w"
    assert dual.replication.n == 1
    assert dual.fixed.placement is not None
    assert dual.fixed.placement.cards == (0, 1)
    assert dual.fixed.placement.on_arm_bound == "continue_peer"
    without_placement = dual.fixed.model_copy(update={"placement": None})
    assert without_placement == single.fixed
    assert hash_fixed(dual.fixed) != hash_fixed(single.fixed)
    assert (dual.factor, dual.workloads, dual.endpoints) == (single.factor, single.workloads, single.endpoints)
