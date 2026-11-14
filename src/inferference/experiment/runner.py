"""Run one declared experiment schedule and write its raw artifacts.

The runner connects the experiment modules. It checks preconditions, requests
fan control through the lease, samples telemetry, proves residency, dispatches
workloads, and records the outcome. It does not calculate measurements.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any, Protocol

import httpx

from inferference.telemetry import CardSample, TelemetryError, TelemetrySample, TelemetrySampler

from .artifacts import (
    OutcomeState,
    RunDirectoryError,
    TelemetryWriter,
    create_run_directory,
    write_blob,
    write_control,
    write_declaration,
    write_identity,
    write_outcome,
    write_preflight,
    write_request_body,
    write_residency,
    write_resolved_command,
    write_response_body,
    write_run_manifest,
    write_stages,
    write_text_artifact,
)
from .conditions import ScheduledRun, schedule
from .control import (
    ControlError,
    FanLease,
    FanLeaseLike,
    normalize_power_profile,
    read_cpu_state,
    read_power_state,
    verify_cpu_epp,
    verify_power_profile,
)
from .preflight import (
    HostPaths,
    PreflightRefusal,
    RunTarget,
    _find_amdgpu_hwmon_dir,
    _read_git_revision,
    _read_hwmon_int,
    preflight,
    start_gate,
)
from .residency import (
    MemorySnapshot,
    ResidencyError,
    ResidencyThresholds,
    first_failure,
    memory_snapshot,
    prove_residency,
    server_baseline,
    sysfs_pci_address,
)
from .safety import ArmBound, SafetyError, SafetyEvent, SafetyMonitor
from .server import (
    ServerError,
    ServerProcess,
    build_command,
    list_devices,
    merge_extra_args,
    wait_for_bindable_port,
)
from .spec import ExperimentSpec, SpecError, load_spec
from .stop import StopRequest
from .workload import (
    AgentSession,
    CorpusLane,
    LaneScript,
    WorkloadError,
    WorkloadPlan,
    assign_corpus_prompts,
    calibrate_prefill,
    chat_payload,
    corpus_prompts,
    decode_payload,
    parse_workload,
    records_prompt,
    validate_response,
)

_CARD_DIR = re.compile(r"^card([0-9]+)$")
_SERVER_TIMING_KEYS = ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n", "draft_n", "draft_n_accepted")
_DEFAULT_HOST_PATHS = HostPaths()


class RunnerError(RuntimeError):
    """The experiment runner could not create or complete a run."""


class PairPeerLost(RuntimeError):
    """The peer run of a pair left before the pair was released, or the pair was aborted."""


class PairStopped(RuntimeError):
    """This run was stopped while it waited for its peer."""


class PairTimeout(RuntimeError):
    """The peer run did not arrive at a rendezvous in time."""


class RunSync(Protocol):
    """What `_run_one` needs from a pair coordinator (see `pair.PairSync`).

    Each run holds one handle. Every method raises `PairPeerLost`,
    `PairStopped`, or `PairTimeout` when the wait cannot end well. A run with
    no handle does no pair coordination at all.
    """

    @property
    def peer_card(self) -> int: ...

    @property
    def release_mono(self) -> float | None: ...

    def arrive(self, name: str, stop: threading.Event | None = None) -> None: ...

    def load_section(self, stop: threading.Event | None = None) -> contextlib.AbstractContextManager[None]: ...

    def depart(self) -> None: ...

    def abort(self, reason: str) -> None: ...

    @property
    def aborted(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class RunResult:
    """The terminal state and artifact directory for one scheduled run."""

    run_dir: Path
    state: OutcomeState
    failure_boundary: str | None
    # Monotonic time of the first terminal event, or `None` for a clean run.
    stop_mono: float | None = None


@dataclass(frozen=True, slots=True)
class _ChatResult:
    record: dict[str, Any]
    validation_error: str | None
    content: str = ""


def gate_junctions(cards: Sequence[int], sample_cards: Iterable[Any]) -> dict[int, float]:
    """Return the junction readings of the cards that the run uses.

    The start gate waits for these cards only. A busy neighbour card does not
    hold it. The thermal guard and the machine limit still watch every card.
    A used card that the sample lacks returns an empty mapping, and the gate
    refuses.
    """
    readings = {item.card: item.temperature_hotspot_c for item in sample_cards}
    if any(card not in readings for card in cards):
        return {}
    return {card: readings[card] for card in cards}


class _RunStopped(Exception):
    """Stop one run after the first terminal event."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _hash_file(path: Path, check_stop: Callable[[], None] | None = None) -> str:
    """Hash a file in 1 MiB blocks. `check_stop` runs before each block and may raise."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if check_stop is not None:
                check_stop()
            digest.update(block)
    return digest.hexdigest()


def _card_indices(drm_root: Path) -> tuple[int, ...]:
    cards = []
    try:
        entries = tuple(drm_root.iterdir())
    except OSError as exc:
        raise RunnerError(f"could not list DRM cards under {drm_root}: {exc}") from exc
    for entry in entries:
        match = _CARD_DIR.fullmatch(entry.name)
        if match is not None and (entry / "device").exists():
            cards.append(int(match.group(1)))
    if not cards:
        raise RunnerError(f"no DRM cards found under {drm_root}")
    return tuple(sorted(cards))


def _snapshot_cards(cards: Sequence[int], drm_root: Path) -> dict[int, MemorySnapshot]:
    snapshots = {}
    for card in cards:
        snapshots[card] = memory_snapshot(card, drm_root=drm_root)
    return snapshots


def _sample_card(sample: TelemetrySample, card: int) -> CardSample | None:
    return next((item for item in sample.cards if item.card == card), None)


def _parse_sse_event(data: bytes) -> dict[str, Any] | None:
    if not data or data == b"[DONE]":
        return None
    try:
        parsed = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _send_chat(
    client: httpx.Client,
    *,
    run_dir: Path,
    request_id: str,
    lane: int,
    sequence: int,
    workload: str,
    validation_kind: str,
    payload: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None = None,
) -> _ChatResult:
    payload_dict = dict(payload)
    payload_sha = write_request_body(run_dir, payload_dict)
    dispatch_utc = _utc_now()
    dispatch_mono = time.monotonic()
    first_token_mono: float | None = None
    status_code = 0
    raw_body = bytearray()
    buffer = bytearray()
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    finish_reason: str | None = None

    try:
        with client.stream("POST", "/v1/chat/completions", json=payload_dict) as response:
            status_code = response.status_code
            for chunk in response.iter_bytes():
                raw_body.extend(chunk)
                buffer.extend(chunk)
                while b"\n" in buffer:
                    line, _, remainder = buffer.partition(b"\n")
                    buffer = bytearray(remainder)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    event = _parse_sse_event(bytes(line[5:].strip()))
                    if event is None:
                        continue
                    choices = event.get("choices")
                    if isinstance(choices, list) and choices:
                        choice = choices[0]
                        if isinstance(choice, dict):
                            delta = choice.get("delta")
                            if isinstance(delta, dict):
                                token = delta.get("content")
                                thought = delta.get("reasoning_content")
                                if isinstance(token, str) and token:
                                    content.append(token)
                                    if first_token_mono is None:
                                        first_token_mono = time.monotonic()
                                if isinstance(thought, str) and thought:
                                    reasoning.append(thought)
                                    if first_token_mono is None:
                                        first_token_mono = time.monotonic()
                            candidate_finish = choice.get("finish_reason")
                            if isinstance(candidate_finish, str):
                                finish_reason = candidate_finish
                    candidate_usage = event.get("usage")
                    if isinstance(candidate_usage, dict):
                        usage = candidate_usage
                    candidate_timings = event.get("timings")
                    if isinstance(candidate_timings, dict):
                        timings = candidate_timings
            if buffer.strip().startswith(b"data:"):
                event = _parse_sse_event(bytes(buffer.strip()[5:].strip()))
                if event is not None:
                    choices = event.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        choice = choices[0]
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            token = delta.get("content")
                            thought = delta.get("reasoning_content")
                            if isinstance(token, str) and token:
                                content.append(token)
                                if first_token_mono is None:
                                    first_token_mono = time.monotonic()
                            if isinstance(thought, str) and thought:
                                reasoning.append(thought)
                                if first_token_mono is None:
                                    first_token_mono = time.monotonic()
                        candidate_finish = choice.get("finish_reason")
                        if isinstance(candidate_finish, str):
                            finish_reason = candidate_finish
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    if isinstance(event.get("timings"), dict):
                        timings = event["timings"]
    except httpx.HTTPError as exc:
        raise RunnerError(f"request {request_id} failed: {exc}") from exc

    response_sha = write_response_body(run_dir, bytes(raw_body))
    end_mono = time.monotonic()
    result = {
        "http_status": status_code,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
    }
    validation_error = validate_response(validation_kind, result, expected)
    server_timings = {key: timings[key] for key in _SERVER_TIMING_KEYS if key in timings}
    record = {
        "request_id": request_id,
        "lane": lane,
        "seq": sequence,
        "payload_sha256": payload_sha,
        "response_sha256": response_sha,
        # The raw body hash includes timing fields. This hash covers the
        # generated text only, so two runs can compare output byte for byte.
        "content_sha256": hashlib.sha256(result["content"].encode()).hexdigest(),
        "dispatch_utc": dispatch_utc,
        "dispatch_mono": dispatch_mono,
        "first_token_mono": first_token_mono,
        "end_mono": end_mono,
        "http_status": status_code,
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "finish_reason": finish_reason,
        "valid": validation_error is None,
        "validation_error": validation_error,
        "server_timings": server_timings,
        "workload": workload,
    }
    if extra:
        record.update(extra)
    return _ChatResult(record=record, validation_error=validation_error, content=result["content"])


def _tokenize_content(client: httpx.Client, run_dir: Path, content: str) -> tuple[int, str, str]:
    request_payload = {"content": content}
    request_sha = write_request_body(run_dir, request_payload)
    try:
        response = client.post("/tokenize", json=request_payload)
    except httpx.HTTPError as exc:
        raise RunnerError(f"tokenize request failed: {exc}") from exc
    response_sha = write_response_body(run_dir, response.content)
    if response.status_code != 200:
        raise RunnerError(f"tokenize request returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise RunnerError("tokenize response was not JSON") from exc
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list):
        raise RunnerError("tokenize response did not contain a token list")
    return len(tokens), request_sha, response_sha


def _terminal_boundary(event: SafetyEvent) -> tuple[OutcomeState, str]:
    if event.verdict == "arm_bound":
        return "bounded", "safety:arm_bound"
    if event.verdict == "machine_limit":
        return "aborted", "safety:machine_limit"
    if event.verdict == "fan_fault":
        return "aborted", "safety:fan"
    if event.verdict == "gtt_spill":
        return "aborted", "safety:gtt"
    raise RunnerError(f"unknown safety verdict {event.verdict!r}")


def _wait_for_next_run_directory_tick(run_dir: Path) -> None:
    """Wait for the next UTC second before a scheduled run can reuse a name.

    The existing run-directory contract has one-second timestamps and keys
    the directory by the fixed-block digest. Conditions in one schedule share
    that digest, so a bounded run can finish before its next condition gets a
    distinct timestamp. Wait only for that timestamp tick before continuing.
    """
    run_stamp = run_dir.name[:16]
    while datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") == run_stamp:
        time.sleep(0.01)


def _load_spec_bytes(spec_path: Path) -> tuple[ExperimentSpec, str]:
    try:
        data = spec_path.read_bytes()
        spec = load_spec(spec_path)
    except (OSError, SpecError) as exc:
        raise RunnerError(f"could not load experiment spec {spec_path}: {exc}") from exc
    return spec, hashlib.sha256(data).hexdigest()


def _run_one(
    spec: ExperimentSpec,
    spec_sha256: str,
    target: RunTarget,
    scheduled_run: ScheduledRun,
    *,
    artifact_root: Path,
    paths: HostPaths,
    env: Mapping[str, str],
    unit_active: Callable[[], bool],
    sample_interval_s: float,
    fan_lease_factory: Callable[[str], FanLeaseLike],
    http_timeout_s: float,
    model_hashes: dict[Path, str],
    sync: RunSync | None = None,
    model_hash_lock: threading.Lock | None = None,
    stop_request: StopRequest | None = None,
) -> RunResult:
    if sample_interval_s <= 0:
        raise RunnerError(f"sample_interval_s must be positive, got {sample_interval_s}")
    started = datetime.now(UTC)
    try:
        run_dir = create_run_directory(
            artifact_root,
            spec.experiment,
            scheduled_run.condition,
            scheduled_run.replicate,
            utc=started,
        )
    except (OSError, RunDirectoryError) as exc:
        raise RunnerError(f"could not create run directory: {exc}") from exc

    result_state: OutcomeState = "completed"
    result_boundary: str | None = None
    result_detail: str | None = None
    result_lock = threading.Lock()
    first_event: list[tuple[OutcomeState, str, str] | None] = [None]
    first_event_mono: list[float | None] = [None]
    stop_event = threading.Event()
    collector_stop_event = threading.Event()
    first_sample = threading.Event()
    latest_lock = threading.Lock()
    latest_sample: list[TelemetrySample | None] = [None]
    collector: threading.Thread | None = None
    collector_error: list[str] = []
    telemetry_writer: TelemetryWriter | None = None
    fan_lease: FanLeaseLike | None = None
    server: ServerProcess | None = None
    server_temp_log = run_dir / ".server-output"
    server_log_written = False
    server_stopped = False
    # One lock covers the whole stop. The stop watcher and the `finally` block
    # can both ask. The second caller waits, and it never sends a second SIGINT.
    server_stop_lock = threading.Lock()
    watcher_done = threading.Event()
    stop_watcher: threading.Thread | None = None
    preflight_snapshot = None
    model_path: Path | None = None
    model_signature: tuple[int, int] | None = None
    card_ids: tuple[int, ...] | None = None
    host_payload: dict[str, Any] = {
        "cpu_epp_declared": spec.fixed.host.cpu_epp,
        "cpu": None,
        "server_threads": None,
    }
    control_payload: dict[str, Any] = {
        "fan": {
            "requested_profile": spec.fixed.control.fan_profile,
            "request_written_utc": None,
            "readback_profile": None,
            "readback_utc": None,
            "released_utc": None,
            "restore_confirmed": False,
            "restore_profile": None,
        },
        "power": {
            "declared": normalize_power_profile(spec.fixed.control.power_profile),
            "readback": {},
            "performance_level": {},
        },
        "host": host_payload,
        "pp_table": {
            "declared_name": spec.fixed.control.pp_table.name if spec.fixed.control.pp_table is not None else "stock",
            "declared_sha256": spec.fixed.control.pp_table.sha256 if spec.fixed.control.pp_table is not None else None,
            "live_sha256": None,
            "stock_sha256": None,
            "deltas_verified": False,
            "control_mode": "power_cap" if spec.fixed.control.power_cap_w is not None else None,
            "declared_power_cap_w": spec.fixed.control.power_cap_w,
            "live_power1_cap_uw": None,
            "live_power1_cap_min_uw": None,
            "final_power1_cap_uw": None,
        },
    }
    stages_payload: dict[str, Any] = {"start_gate": None, "stages": []}

    # One stack holds the pair's load section from the first snapshot until the
    # residency proof ends. `close()` is safe to call twice.
    load_stack = contextlib.ExitStack()

    def set_result(state: OutcomeState, boundary: str, detail: str) -> None:
        nonlocal result_state, result_boundary, result_detail
        with result_lock:
            if first_event[0] is None:
                first_event[0] = (state, boundary, detail)
                first_event_mono[0] = time.monotonic()
                stop_event.set()
            severity = {"completed": 0, "bounded": 1, "invalid": 2, "aborted": 3}
            if severity[state] > severity[result_state]:
                result_state = state
                result_boundary = boundary
                result_detail = detail
            elif result_boundary is None:
                result_state = state
                result_boundary = boundary
                result_detail = detail

    def stop_on_safety(event: SafetyEvent) -> None:
        state, boundary = _terminal_boundary(event)
        if sync is not None and event.verdict in {"machine_limit", "fan_fault"}:
            # These events cover the whole box. The peer run stops too, even
            # when its own sampler misses the reading.
            sync.abort(f"{boundary} on card {target.card}: {event.observed}")
        set_result(state, boundary, event.observed)

    def raise_if_stopped() -> None:
        with result_lock:
            current = first_event[0]
        if current is not None:
            raise _RunStopped(current[2])

    @contextlib.contextmanager
    def pair_errors() -> Iterator[None]:
        """Turn a failed pair wait into this run's terminal event."""
        try:
            yield
        except PairPeerLost as exc:
            set_result("aborted", "pair:peer_lost", str(exc))
            raise _RunStopped(str(exc)) from exc
        except PairTimeout as exc:
            set_result("aborted", "pair:timeout", str(exc))
            raise _RunStopped(str(exc)) from exc
        except PairStopped as exc:
            raise_if_stopped()
            raise _RunStopped(str(exc)) from exc

    def pair_arrive(name: str) -> None:
        if sync is None:
            return
        with pair_errors():
            sync.arrive(name, stop_event)

    def stop_server() -> None:
        nonlocal server_stopped
        with server_stop_lock:
            if server is None or server_stopped:
                return
            try:
                if server.alive():
                    # Mark the stop only when a process exists to stop. A call
                    # that comes before `start` must leave a later call free to act.
                    server_stopped = True
                    server.stop()
            except (OSError, ServerError) as exc:
                server_stopped = True
                set_result("aborted", "server:stop", str(exc))

    def watch_stop_request(request: StopRequest) -> None:
        """Turn an outside stop request into this run's first event, then stop the server.

        The watcher keeps asking until the server stops, so a server that starts
        after the request still gets its one SIGINT.
        """
        while not watcher_done.is_set():
            if request.wait(0.05):
                break
        else:
            return
        set_result("aborted", f"signal:{request.reason}", f"{request.reason} asked the runner to stop")
        while not watcher_done.is_set() and not server_stopped:
            stop_server()
            watcher_done.wait(0.05)

    def write_server_log() -> None:
        nonlocal server_log_written
        if server is None or server_log_written:
            return
        stop_server()
        try:
            write_text_artifact(run_dir, "server.log", server.log_text())
            server_temp_log.unlink(missing_ok=True)
            server_log_written = True
        except (OSError, RunDirectoryError) as exc:
            set_result("aborted", "server:log", str(exc))

    def collector_main(
        sampler: TelemetrySampler,
        writer: TelemetryWriter,
        monitor: SafetyMonitor,
        lease: FanLeaseLike,
        card: int,
    ) -> None:
        try:
            while not collector_stop_event.is_set():
                tick_started = time.monotonic()
                sample = sampler.sample()
                writer.write(sample)
                with latest_lock:
                    latest_sample[0] = sample
                first_sample.set()
                arm_card = _sample_card(sample, card)
                if arm_card is None:
                    raise SafetyError(f"telemetry sample carries no card {card}")
                events = monitor.check(sample)
                if events:
                    stop_on_safety(events[0])
                drift = monitor.profile_drift(sample)
                if drift is not None:
                    set_result("invalid", "drift:fan_profile", drift)
                lease.renew_if_due()
                if sync is not None:
                    pair_abort = sync.aborted
                    if pair_abort is not None:
                        set_result("aborted", "pair:peer_lost", f"the pair was aborted: {pair_abort}")
                remaining = sample_interval_s - (time.monotonic() - tick_started)
                if remaining > 0 and collector_stop_event.wait(remaining):
                    break
        except (TelemetryError, SafetyError, ControlError, OSError, ValueError) as exc:
            collector_error.append(str(exc))
            set_result("aborted", "telemetry", str(exc))
            first_sample.set()
        except Exception as exc:  # fail closed on an unexpected collector failure
            collector_error.append(str(exc))
            set_result("aborted", "telemetry", f"unexpected collector error: {exc}")
            first_sample.set()

    def check_drift(
        snapshot: Any, model_path: Path, model_signature: tuple[int, int], card_ids: tuple[int, ...]
    ) -> None:
        observed: list[tuple[str, str]] = []
        expected_table_hash = snapshot.identity["pp_table"]["live_sha256"]
        table_path = paths.drm_root / f"card{target.card}" / "device" / "pp_table"
        try:
            table_hash = hashlib.sha256(table_path.read_bytes()).hexdigest()
            if table_hash != expected_table_hash:
                observed.append(("pp_table", f"expected {expected_table_hash}, found {table_hash}"))
        except OSError as exc:
            observed.append(("pp_table", str(exc)))

        declared_cap_w = spec.fixed.control.power_cap_w
        if declared_cap_w is not None:
            # A GPU reset silently restores the stock 250 W cap, so the
            # declared cap is read again here. A pair spec declares the cap
            # for every listed card, so each one is read.
            placement = spec.fixed.placement
            cap_cards = (target.card,) if placement is None else placement.cards
            for cap_card in cap_cards:
                label = "" if placement is None else f"card {cap_card}: "
                try:
                    cap_uw = _read_hwmon_int(_find_amdgpu_hwmon_dir(paths.drm_root, cap_card) / "power1_cap")
                    if cap_card == target.card:
                        control_payload["pp_table"]["final_power1_cap_uw"] = cap_uw
                    if cap_uw != declared_cap_w * 1_000_000:
                        observed.append(
                            (
                                "power_cap",
                                f"{label}declared {declared_cap_w * 1_000_000} uW, found {cap_uw} uW (power1_cap)",
                            )
                        )
                except OSError as exc:
                    observed.append(("power_cap", f"{label}{exc}"))

        try:
            current_power = read_power_state(card_ids, drm_root=paths.drm_root)
            verify_power_profile(spec.fixed.control.power_profile, current_power)
            initial_power = snapshot.identity["power_profile"]["state"]
            for card in card_ids:
                initial = initial_power.get(card, initial_power.get(str(card), {}))
                current = current_power[card]
                if initial.get("profile") != current["profile"]:
                    observed.append(("power_profile", f"card {card}: {initial.get('profile')} -> {current['profile']}"))
        except (ControlError, OSError) as exc:
            observed.append(("power_profile", str(exc)))

        declared_cpu_epp = spec.fixed.host.cpu_epp
        if declared_cpu_epp is not None:
            try:
                verify_cpu_epp(declared_cpu_epp, read_cpu_state(paths.cpu_root))
            except ControlError as exc:
                observed.append(("cpu_epp", str(exc)))

        try:
            generation = os.readlink(paths.current_system)
            if generation != snapshot.identity["nixos_generation"]:
                observed.append(("nixos_generation", f"{snapshot.identity['nixos_generation']} -> {generation}"))
        except OSError as exc:
            observed.append(("nixos_generation", str(exc)))

        try:
            revision = _read_git_revision(paths.repo_root)
            if revision != snapshot.identity["repo_revision"]:
                observed.append(("repo_revision", f"{snapshot.identity['repo_revision']} -> {revision}"))
        except OSError as exc:
            observed.append(("repo_revision", str(exc)))

        try:
            stat = model_path.stat()
            current_signature = (stat.st_size, stat.st_mtime_ns)
            if current_signature != model_signature:
                observed.append(("model_size_or_mtime", f"{model_signature} -> {current_signature}"))
        except OSError as exc:
            observed.append(("model_size_or_mtime", str(exc)))

        if observed:
            field, detail = observed[0]
            set_result("invalid", f"drift:{field}", detail)

    try:
        if stop_request is not None:
            stop_watcher = threading.Thread(
                target=watch_stop_request,
                args=(stop_request,),
                name="experiment-stop-watcher",
                daemon=True,
            )
            stop_watcher.start()
        write_run_manifest(
            run_dir,
            experiment=spec.experiment,
            condition=scheduled_run.condition,
            replicate=scheduled_run.replicate,
            spec_sha256=spec_sha256,
        )
        write_declaration(run_dir, spec=spec, condition=scheduled_run.condition, spec_sha256=spec_sha256)
        # Both runs of a pair start preflight together, so the peer card is
        # still idle for the `peer_idle` check.
        pair_arrive("run")
        try:
            cached_model_hash = model_hashes

            def model_sha256(path: Path) -> str:
                resolved = path.resolve()
                # A pair shares one lock, so the second run reads the first
                # run's hash and the large model file is hashed once.
                with model_hash_lock if model_hash_lock is not None else contextlib.nullcontext():
                    if resolved not in cached_model_hash:
                        cached_model_hash[resolved] = _hash_file(path, raise_if_stopped)
                    return cached_model_hash[resolved]

            preflight_snapshot = preflight(
                spec,
                target,
                paths=paths,
                env=env,
                unit_active=unit_active,
                model_sha256=model_sha256,
            )
        except PreflightRefusal as exc:
            write_preflight(
                run_dir,
                {
                    "refused": True,
                    "check": exc.check,
                    "detail": exc.detail,
                    "passed": [{"name": item.name, "observed": item.observed} for item in exc.passed],
                },
            )
            set_result("aborted", f"preflight:{exc.check}", exc.detail)
            raise _RunStopped(exc.detail) from exc

        write_preflight(run_dir, preflight_snapshot.to_json())
        # No run loads a server until both preflights pass.
        pair_arrive("preflight")
        model_path = preflight_snapshot.resolved_model.model
        model_digest = model_sha256(model_path)
        model_stat = model_path.stat()
        model_signature = (model_stat.st_size, model_stat.st_mtime_ns)
        identity = dict(preflight_snapshot.identity)
        identity.update(
            {
                "model_sha256": model_digest,
                "model_path": str(model_path),
                "model_size_bytes": model_stat.st_size,
                "model_mtime_ns": model_stat.st_mtime_ns,
                "build_manifest": identity.get("build"),
            }
        )
        write_identity(run_dir, identity)
        table_sha = write_blob(run_dir, "pp_table.bin", preflight_snapshot.pp_table_bytes)
        control_payload["pp_table"].update(preflight_snapshot.identity["pp_table"])
        control_payload["pp_table"]["live_sha256"] = table_sha

        card_ids = _card_indices(paths.drm_root)
        try:
            fan_lease = fan_lease_factory(spec.fixed.control.fan_profile)
            fan_readback = fan_lease.acquire()
            control_payload["fan"].update(fan_readback)
        except ControlError as exc:
            set_result("aborted", "control:fan_lease", str(exc))
            raise _RunStopped(str(exc)) from exc

        try:
            power_state = read_power_state(card_ids, drm_root=paths.drm_root)
            verify_power_profile(spec.fixed.control.power_profile, power_state)
        except ControlError as exc:
            set_result("aborted", "control:power_profile", str(exc))
            raise _RunStopped(str(exc)) from exc
        control_payload["power"]["readback"] = {str(card): state["profile"] for card, state in power_state.items()}
        control_payload["power"]["performance_level"] = {
            str(card): state["performance_level"] for card, state in power_state.items()
        }
        try:
            cpu_state = read_cpu_state(paths.cpu_root)
            control_payload["host"]["cpu"] = cpu_state
            if spec.fixed.host.cpu_epp is not None:
                verify_cpu_epp(spec.fixed.host.cpu_epp, cpu_state)
        except ControlError as exc:
            set_result("aborted", "control:cpu_epp", str(exc))
            raise _RunStopped(str(exc)) from exc

        # GTT monitoring starts with the server's post-initialization baseline.
        monitor = SafetyMonitor(
            machine_limit_c=spec.fixed.safety.machine_limit_c,
            arms=(ArmBound(card=target.card, bound_c=spec.fixed.safety.arm_bound_c, gtt_baseline_mib=None),),
            unit_active=unit_active,
            requested_fan_profile=spec.fixed.control.fan_profile,
        )
        try:
            telemetry_writer = TelemetryWriter(run_dir)
            sampler = TelemetrySampler(
                cards=card_ids,
                drm_root=paths.drm_root,
                hwmon_root=paths.hwmon_root,
                watchdog_status_file=paths.fan_status_file,
            )
        except (TelemetryError, OSError, RunDirectoryError) as exc:
            set_result("aborted", "telemetry", str(exc))
            raise _RunStopped(str(exc)) from exc
        collector = threading.Thread(
            target=collector_main,
            args=(sampler, telemetry_writer, monitor, fan_lease, target.card),
            name="experiment-telemetry",
            daemon=True,
        )
        collector.start()
        if not first_sample.wait(timeout=max(5.0, sample_interval_s * 10)):
            set_result("aborted", "telemetry", "collector did not produce its first sample")
            raise _RunStopped("collector did not produce its first sample")
        raise_if_stopped()

        if sync is not None:
            # One server loads at a time. The second run's `before` snapshot
            # then holds the first card's loaded VRAM, so the residency check
            # "other cards undisturbed" can pass.
            with pair_errors():
                load_stack.enter_context(sync.load_section(stop_event))
            raise_if_stopped()

        try:
            before = _snapshot_cards(card_ids, paths.drm_root)
        except ResidencyError as exc:
            set_result("aborted", "residency:1", str(exc))
            raise _RunStopped(str(exc)) from exc

        try:
            device_list = list_devices(target.server_bin, card=target.card, base_env=env, stop=stop_event)
            write_text_artifact(run_dir, "list-devices.txt", device_list)
        except (ServerError, OSError, RunDirectoryError) as exc:
            set_result("aborted", "server:list_devices", str(exc))
            raise _RunStopped(str(exc)) from exc

        raise_if_stopped()
        try:
            wait_for_bindable_port(target.port, stop=stop_event)
        except ServerError as exc:
            set_result("aborted", "server:port", str(exc))
            raise _RunStopped(str(exc)) from exc

        slots = spec.fixed.context.slots
        lease_mib = ceil(preflight_snapshot.footprint.total_mib(slots))
        wrapper = (
            str(target.gpu_lease),
            str(lease_mib),
            "--",
            str(target.thermal_guard),
            "run",
            "--",
        )
        spec_extra_args = {} if spec.fixed.server is None else spec.fixed.server.extra_args
        try:
            extra_args = merge_extra_args(preflight_snapshot.resolved_model.extra_args, spec_extra_args)
            control_payload["host"]["server_threads"] = extra_args.get("threads")
            command = build_command(
                server_bin=target.server_bin,
                model=model_path,
                alias=preflight_snapshot.resolved_model.name,
                card=target.card,
                port=target.port,
                slots=slots,
                tokens_per_slot=spec.fixed.context.tokens_per_slot,
                batch=spec.fixed.batch.size,
                ubatch=spec.fixed.batch.ubatch,
                cache_k=spec.fixed.cache.k,
                cache_v=spec.fixed.cache.v,
                extra_args=extra_args,
                wrapper=wrapper,
                base_env=env,
                fan_status_file=paths.fan_status_file,
            )
        except ServerError as exc:
            set_result("aborted", "server:extra_args", str(exc))
            raise _RunStopped(str(exc)) from exc
        server = ServerProcess(command, server_temp_log)
        try:
            server.start()
            write_resolved_command(run_dir, argv=command.argv, env=command.env, cwd=str(Path.cwd()))
            server.wait_healthy(timeout_s=min(600.0, http_timeout_s))
        except (OSError, ServerError, RunDirectoryError) as exc:
            set_result("aborted", "server:health", str(exc))
            raise _RunStopped(str(exc)) from exc

        raise_if_stopped()
        try:
            live_pci = sysfs_pci_address(target.card, drm_root=paths.drm_root)
            baseline = server_baseline(server.log_text(), card=target.card, pci=live_pci)
            before[target.card] = MemorySnapshot(
                card=target.card,
                vram_used_mib=before[target.card].vram_used_mib,
                gtt_used_mib=baseline.gtt_used_mib,
            )
            monitor.set_gtt_baseline(target.card, baseline.gtt_used_mib)
            after = _snapshot_cards(card_ids, paths.drm_root)
            if after[target.card].gtt_used_mib > baseline.gtt_used_mib:
                raise ResidencyError("GTT rose during model load")
        except ResidencyError as exc:
            set_result("aborted", "residency:6", str(exc))
            raise _RunStopped(str(exc)) from exc
        alias = preflight_snapshot.resolved_model.name
        probe_payload = chat_payload(
            ({"role": "user", "content": "Reply with one short word."},),
            seed=0,
            max_tokens=4,
            model_alias=alias,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{target.port}", timeout=http_timeout_s) as client:
                raise_if_stopped()
                probe = _send_chat(
                    client,
                    run_dir=run_dir,
                    request_id="residency-probe",
                    lane=0,
                    sequence=0,
                    workload="residency_probe",
                    validation_kind="decode",
                    payload=probe_payload,
                    expected=None,
                )
        except RunnerError as exc:
            set_result("aborted", "server:probe", str(exc))
            raise _RunStopped(str(exc)) from exc

        probe_stage = {
            "name": "residency_probe",
            "workload": "residency_probe",
            "concurrency": 1,
            "started_utc": probe.record["dispatch_utc"],
            "started_mono": probe.record["dispatch_mono"],
            "ended_utc": _utc_now(),
            "ended_mono": probe.record["end_mono"],
            "completed": probe.validation_error is None,
            "stop_reason": probe.validation_error,
            "requests": [probe.record],
        }
        stages_payload["stages"].append(probe_stage)

        try:
            after = _snapshot_cards(card_ids, paths.drm_root)
            if after[target.card].gtt_used_mib > baseline.gtt_used_mib:
                raise ResidencyError("GTT rose during the residency probe")
            live_pci = sysfs_pci_address(target.card, drm_root=paths.drm_root)
            expected_pci = target.expect_pci or live_pci
            checks = prove_residency(
                expected_pci=expected_pci,
                live_pci=live_pci,
                list_devices=device_list,
                log=server.log_text(),
                before=before,
                after=after,
                measured_card=target.card,
                thresholds=ResidencyThresholds(
                    min_vram_delta_mib=preflight_snapshot.footprint.base_mib,
                    gtt_limit_mib=spec.fixed.safety.gtt_limit_mib,
                ),
                output_valid=probe.validation_error is None,
            )
        except (ResidencyError, OSError) as exc:
            set_result("aborted", "residency:1", str(exc))
            raise _RunStopped(str(exc)) from exc
        write_residency(
            run_dir,
            {
                "checks": [
                    {"number": item.number, "name": item.name, "passed": item.passed, "observed": item.observed}
                    for item in checks
                ]
            },
        )
        failed_check = first_failure(checks)
        if failed_check is not None:
            set_result("aborted", f"residency:{failed_check.number}", failed_check.observed)
            raise _RunStopped(failed_check.observed)

        load_stack.close()
        raise_if_stopped()
        pair_arrive("loaded")
        raise_if_stopped()

        gate_cards = (target.card,) if spec.fixed.placement is None else spec.fixed.placement.cards

        def read_junctions() -> Mapping[int, float]:
            if stop_event.is_set():
                raise _RunStopped("a safety or telemetry event stopped the run during the start gate")
            with latest_lock:
                sample = latest_sample[0]
            if sample is None:
                return {}
            return gate_junctions(gate_cards, sample.cards)

        try:
            gate_result = start_gate(
                spec,
                read_junctions=read_junctions,
                poll_s=min(1.0, sample_interval_s),
            )
        except _RunStopped:
            raise_if_stopped()
            raise
        except PreflightRefusal as exc:
            set_result("aborted", "start_gate", exc.detail)
            raise _RunStopped(exc.detail) from exc
        stages_payload["start_gate"] = gate_result
        if sync is not None:
            # Both runs leave the gate at the same moment. This time is the
            # zero of the shared workload clock.
            pair_arrive("gate")
            stages_payload["pair"] = {
                "peer_card": sync.peer_card,
                "release_mono": sync.release_mono,
                "stop_mono": None,
            }
            raise_if_stopped()

        factor_lanes = 1
        if spec.factor.name == "lanes":
            try:
                factor_lanes = int(scheduled_run.condition.factor_value)
            except ValueError as exc:
                set_result("aborted", "preflight:workloads", "lanes factor must be an integer")
                raise _RunStopped("lanes factor must be an integer") from exc
        if factor_lanes < 1:
            set_result("aborted", "preflight:workloads", f"lane count must be positive, got {factor_lanes}")
            raise _RunStopped("lane count must be positive")

        for workload_name in spec.workloads:
            raise_if_stopped()
            plan = parse_workload(workload_name)
            concurrency = factor_lanes if plan.concurrency_from_factor else (plan.sessions or 1)
            expected: Mapping[str, Any] | None = None
            messages: tuple[dict[str, str], ...] | None = None
            calibration: dict[str, Any] | None = None
            calibration_hashes: list[dict[str, str]] = []
            if plan.kind in ("prefill", "agent_turns"):
                assert plan.target_prompt_tokens is not None
                try:
                    with httpx.Client(base_url=f"http://127.0.0.1:{target.port}", timeout=http_timeout_s) as client:

                        def tokenize(
                            content: str,
                            request_client: httpx.Client = client,
                            hashes: list[dict[str, str]] = calibration_hashes,
                        ) -> int:
                            raise_if_stopped()
                            count, request_sha, response_sha = _tokenize_content(request_client, run_dir, content)
                            hashes.append({"request": request_sha, "response": response_sha})
                            return count

                        content, calibration = calibrate_prefill(
                            plan.target_prompt_tokens,
                            tokenize,
                        )
                        # The tokenization bodies are content-addressed. Store
                        # their hashes with the calibration metadata too.
                        calibration["request_sha256"] = [item["request"] for item in calibration_hashes]
                        calibration["response_sha256"] = [item["response"] for item in calibration_hashes]
                    if plan.kind == "prefill":
                        _, expected = records_prompt(calibration["record_count"])
                        messages = ({"role": "user", "content": content},)
                except (WorkloadError, RunnerError, httpx.HTTPError) as exc:
                    set_result("aborted", "server:tokenize", str(exc))
                    raise _RunStopped(str(exc)) from exc

            lane_scripts: dict[int, LaneScript] = {}
            if plan.kind == "agent_turns":
                assert calibration is not None and plan.sessions is not None and plan.turns is not None
                lane_scripts = {
                    session: AgentSession(
                        session,
                        records_prompt(calibration["record_count"], salt=session)[0],
                        turns=plan.turns,
                        model_alias=alias,
                    )
                    for session in range(plan.sessions)
                }
            if plan.corpus_set is not None:
                try:
                    prompts = corpus_prompts(plan.corpus_set)
                except WorkloadError as exc:
                    set_result("aborted", "preflight:workloads", str(exc))
                    raise _RunStopped(str(exc)) from exc
                lane_scripts = {
                    lane: CorpusLane(items, model_alias=alias)
                    for lane, items in enumerate(assign_corpus_prompts(prompts, concurrency))
                }

            pair_arrive(f"stage:{workload_name}")
            raise_if_stopped()
            stage_started_utc = _utc_now()
            stage_started_mono = time.monotonic()
            stage_deadline = stage_started_mono + (plan.duration_s or 0.0)
            stage_requests: list[dict[str, Any]] = []

            def run_lane(
                lane: int,
                current_plan: WorkloadPlan = plan,
                current_concurrency: int = concurrency,
                current_workload_name: str = workload_name,
                current_deadline: float = stage_deadline,
                current_messages: tuple[dict[str, str], ...] | None = messages,
                current_expected: Mapping[str, Any] | None = expected,
                current_scripts: dict[int, LaneScript] = lane_scripts,
            ) -> list[dict[str, Any]]:
                records: list[dict[str, Any]] = []
                script = current_scripts.get(lane)
                try:
                    with httpx.Client(base_url=f"http://127.0.0.1:{target.port}", timeout=http_timeout_s) as client:
                        sequence = 0
                        while not stop_event.is_set():
                            if current_plan.kind == "soak" and time.monotonic() >= current_deadline:
                                break
                            seed = 1000 + lane + sequence * current_concurrency
                            extra: Mapping[str, Any] | None = None
                            if script is not None:
                                step = script.next_step(sequence)
                                if step is None:
                                    break
                                payload = step.payload
                                extra = step.extra
                            elif current_plan.kind == "prefill":
                                request_messages = current_messages or ()
                                payload = chat_payload(
                                    request_messages,
                                    seed=seed,
                                    max_tokens=256,
                                    model_alias=alias,
                                )
                            else:
                                payload = decode_payload(seed=seed, model_alias=alias)
                            request_id = f"{current_workload_name}-l{lane}-s{sequence}"
                            if stop_event.is_set():
                                break
                            sent = _send_chat(
                                client,
                                run_dir=run_dir,
                                request_id=request_id,
                                lane=lane,
                                sequence=sequence,
                                workload=current_workload_name,
                                validation_kind=current_plan.kind,
                                payload=payload,
                                expected=current_expected,
                                extra=extra,
                            )
                            if script is not None:
                                script.annotate(sent.record)
                            records.append(sent.record)
                            if sent.validation_error is not None:
                                set_result("aborted", f"output:{request_id}", sent.validation_error)
                                break
                            sequence += 1
                            if script is not None:
                                script.observe(sent.content)
                            elif current_plan.kind != "soak":
                                break
                except RunnerError as exc:
                    with result_lock:
                        already_stopped = first_event[0] is not None
                    if already_stopped:
                        # A safety stop (or an earlier recorded outcome) already
                        # stopped this server. Losing the in-flight stream is a
                        # consequence of that deliberate shutdown, not a new
                        # root cause: swallow it so the first verdict stands.
                        return records
                    set_result("aborted", "server:request", str(exc))
                except httpx.HTTPError as exc:
                    with result_lock:
                        already_stopped = first_event[0] is not None
                    if already_stopped:
                        return records
                    set_result("aborted", "server:request", str(exc))
                return records

            futures: set[Future[list[dict[str, Any]]]] = set()
            with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="experiment-lane") as pool:
                futures = {pool.submit(run_lane, lane) for lane in range(concurrency)}
                pending = set(futures)
                while pending:
                    if stop_event.is_set():
                        stop_server()
                    done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                    for future in done:
                        try:
                            stage_requests.extend(future.result())
                        except Exception as exc:
                            set_result("aborted", "server:request", str(exc))
            stage_requests.sort(key=lambda item: (item["lane"], item["seq"]))
            stage_ended_mono = time.monotonic()
            with result_lock:
                terminal = first_event[0]
            stop_reason = terminal[1] if terminal is not None else None
            stages_payload["stages"].append(
                {
                    "name": workload_name,
                    "workload": workload_name,
                    "concurrency": concurrency,
                    "started_utc": stage_started_utc,
                    "started_mono": stage_started_mono,
                    "ended_utc": _utc_now(),
                    "ended_mono": stage_ended_mono,
                    "completed": terminal is None,
                    "stop_reason": stop_reason,
                    "requests": stage_requests,
                    **({"calibration": calibration} if calibration is not None else {}),
                }
            )
            if terminal is not None:
                raise _RunStopped(terminal[2])

        if server is not None and not server.alive():
            set_result("aborted", "server:crash", "server exited before the runner requested shutdown")
        write_server_log()
        raise_if_stopped()
    except _RunStopped:
        pass
    except (SpecError, WorkloadError, RunnerError) as exc:
        set_result("aborted", "server:runner", str(exc))
    except Exception as exc:  # preserve artifacts when an unexpected stage fails
        set_result("aborted", "server:runner", f"unexpected runner error: {exc}")
    finally:
        stop_event.set()
        watcher_done.set()
        if stop_watcher is not None:
            # The watcher may be inside `server.stop`. Wait for it, so the
            # server has one stop and the log below is complete.
            stop_watcher.join()
        if sync is not None:
            # A departed peer no longer holds up this run's later barriers.
            # Depart before the load slot opens, so a waiting peer sees a lost
            # pair and does not start a load.
            sync.depart()
        load_stack.close()
        if server is not None and not server_log_written:
            write_server_log()
        if (
            preflight_snapshot is not None
            and model_path is not None
            and model_signature is not None
            and card_ids is not None
        ):
            check_drift(preflight_snapshot, model_path, model_signature, card_ids)
        collector_stop_event.set()
        if collector is not None:
            collector.join()
        if telemetry_writer is not None:
            telemetry_writer.close()
        if fan_lease is not None:
            try:
                release = fan_lease.release()
                control_payload["fan"].update(release)
                # A deferred release leaves the lease to the peer run. The peer
                # confirms the real restore when it makes the last release.
                if not (release.get("restore_confirmed", False) or release.get("deferred", False)):
                    set_result("aborted", "control:fan_release", "fan profile restore was not confirmed")
            except ControlError as exc:
                set_result("aborted", "control:fan_release", str(exc))
        if collector_error and result_boundary is None:
            set_result("aborted", "telemetry", collector_error[0])
        if isinstance(stages_payload.get("pair"), dict):
            stages_payload["pair"]["stop_mono"] = first_event_mono[0]
        write_control(run_dir, control_payload)
        write_stages(run_dir, stages_payload)
        ended = _utc_now()
        write_outcome(
            run_dir,
            state=result_state,
            failure_boundary=result_boundary,
            detail=result_detail,
            started_utc=started.isoformat(),
            ended_utc=ended,
        )

    return RunResult(
        run_dir=run_dir,
        state=result_state,
        failure_boundary=result_boundary,
        stop_mono=first_event_mono[0],
    )


def run_one(
    spec_path: Path,
    target: RunTarget,
    scheduled_run: ScheduledRun,
    *,
    artifact_root: Path,
    paths: HostPaths = _DEFAULT_HOST_PATHS,
    env: Mapping[str, str] | None = None,
    unit_active: Callable[[], bool],
    sample_interval_s: float = 0.25,
    fan_lease_factory: Callable[[str], FanLeaseLike] = FanLease,
    http_timeout_s: float = 900,
    stop_request: StopRequest | None = None,
) -> RunResult:
    """Run one item from `schedule(spec)` and return its artifact directory."""
    spec, spec_sha256 = _load_spec_bytes(spec_path)
    if scheduled_run not in schedule(spec):
        raise RunnerError("scheduled_run is not declared by this spec")
    environment = dict(os.environ if env is None else env)
    return _run_one(
        spec,
        spec_sha256,
        target,
        scheduled_run,
        artifact_root=artifact_root,
        paths=paths,
        env=environment,
        unit_active=unit_active,
        sample_interval_s=sample_interval_s,
        fan_lease_factory=fan_lease_factory,
        http_timeout_s=http_timeout_s,
        model_hashes={},
        stop_request=stop_request,
    )


def run_experiment(
    spec_path: Path,
    target: RunTarget,
    *,
    artifact_root: Path,
    paths: HostPaths = _DEFAULT_HOST_PATHS,
    env: Mapping[str, str] | None = None,
    unit_active: Callable[[], bool],
    sample_interval_s: float = 0.25,
    fan_lease_factory: Callable[[str], FanLeaseLike] = FanLease,
    http_timeout_s: float = 900,
    stop_request: StopRequest | None = None,
) -> tuple[RunResult, ...]:
    """Run scheduled conditions until the first refusal or machine stop.

    A set `stop_request` stops the run that is active and starts no later run.
    """
    spec, spec_sha256 = _load_spec_bytes(spec_path)
    environment = dict(os.environ if env is None else env)
    model_hashes: dict[Path, str] = {}
    results = []
    scheduled_runs = schedule(spec)
    for index, scheduled_run in enumerate(scheduled_runs):
        if stop_request is not None and stop_request.is_set():
            break
        result = _run_one(
            spec,
            spec_sha256,
            target,
            scheduled_run,
            artifact_root=artifact_root,
            paths=paths,
            env=environment,
            unit_active=unit_active,
            sample_interval_s=sample_interval_s,
            fan_lease_factory=fan_lease_factory,
            http_timeout_s=http_timeout_s,
            model_hashes=model_hashes,
            stop_request=stop_request,
        )
        results.append(result)
        if result.state not in {"completed", "bounded"}:
            break
        if index + 1 < len(scheduled_runs):
            _wait_for_next_run_directory_tick(result.run_dir)
    return tuple(results)
