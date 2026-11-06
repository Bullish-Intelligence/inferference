"""Run experiment preflight, collection, analysis, and comparison commands."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from .analysis.compare import ComparisonError, compare
from .analysis.estimators import EstimatorError, mean, ols_slope, peak, transit_time
from .analysis.load import LoadError, RunRecord, load_run
from .analysis.paired import PairedAnalysisError, analyze_pair_root
from .analysis.plateau import plateau_test
from .analysis.series import Point, band_samples, card_series, episodes, window
from .control import FanLease
from .pair import run_pair
from .preflight import HostPaths, PreflightRefusal, RunTarget, preflight, preflight_pair
from .report import ReportError, render_analysis, render_comparison
from .runner import RunnerError, run_experiment
from .spec import SpecError, load_spec

_DEFAULT_THERMAL_GUARD = Path("ci/runner/gpu-thermal-guard.sh")
_DEFAULT_GPU_LEASE = Path("ci/runner/gpu-lease.sh")
_TRANSIT_ENDPOINT = re.compile(r"^transit_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")
_KNOWN_ENDPOINTS = frozenset(
    {"band_slope", "band_power", "peak", "peak_junction", "transit_time", "ttft", "decode_rate", "prefill_rate"}
)


class ExperimentCLIError(RuntimeError):
    """A command could not use its input or complete its requested operation."""


class _UsageError(ValueError):
    """A parsed argument has invalid command-specific meaning."""


def unit_active() -> bool:
    """Return whether the ARCTIC fan watchdog service is active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", "arctic-fan-watchdog.service"],
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _host_paths() -> HostPaths:
    """Return the real host roots, with a patch point for fabricated tests."""
    return HostPaths()


def _fan_lease_factory(profile: str) -> FanLease:
    """Build a lease with the default request and status paths."""
    return FanLease(profile)


def _card_number(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("card must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("card must be zero or greater")
    return value


def _port_number(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return value


def _load_threshold(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("load threshold must be a number") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("load threshold must be finite")
    return value


def _stall_floor(raw: str) -> tuple[int, float]:
    channel_text, separator, rpm_text = raw.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("stall floor must use CH=RPM")
    try:
        channel = int(channel_text)
        rpm = float(rpm_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("stall floor must use an integer channel and numeric RPM") from exc
    if channel < 0 or rpm < 0 or not math.isfinite(rpm):
        raise argparse.ArgumentTypeError("stall floor channel and RPM must be finite and non-negative")
    return channel, rpm


def _int_pair(raw: str, label: str) -> tuple[int, int]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{label} must be two comma-separated integers, such as 0,1")
    try:
        first, second = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be two comma-separated integers, such as 0,1") from exc
    return first, second


def _card_pair(raw: str) -> tuple[int, int]:
    first, second = _int_pair(raw, "cards")
    if first < 0 or second < 0:
        raise argparse.ArgumentTypeError("cards must be zero or greater")
    if first == second:
        raise argparse.ArgumentTypeError("cards must be two distinct cards")
    return first, second


def _port_pair(raw: str) -> tuple[int, int]:
    first, second = _int_pair(raw, "ports")
    for port in (first, second):
        if not 1 <= port <= 65535:
            raise argparse.ArgumentTypeError("ports must be between 1 and 65535")
    if first == second:
        raise argparse.ArgumentTypeError("ports must be two distinct ports")
    return first, second


def _card_pci(raw: str) -> tuple[int, str]:
    card_text, separator, address = raw.partition("=")
    if not separator or not address:
        raise argparse.ArgumentTypeError("expect-pci must use CARD=ADDRESS, such as 0=0000:04:00.0")
    try:
        card = int(card_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expect-pci must use an integer card") from exc
    if card < 0:
        raise argparse.ArgumentTypeError("card must be zero or greater")
    return card, address


def _add_stall_floor_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stall-floor",
        action="append",
        type=_stall_floor,
        default=[],
        metavar="CH=RPM",
        help="minimum tachometer RPM for a fan channel; repeat for each channel",
    )


def _add_pair_arguments(parser: argparse.ArgumentParser, *, run: bool) -> None:
    parser.add_argument("spec", type=Path, help="dual-card experiment YAML spec (declares fixed.placement)")
    parser.add_argument("--cards", required=True, type=_card_pair, metavar="C0,C1", help="the two DRM card indices")
    parser.add_argument("--ports", required=True, type=_port_pair, metavar="P0,P1", help="one server port per card")
    parser.add_argument("--server-bin", required=True, type=Path, help="llama-server executable")
    parser.add_argument("--config-root", required=True, type=Path, help="serving configuration directory")
    parser.add_argument("--stock-pp-table", required=True, type=Path, help="stock pp_table reference blob")
    parser.add_argument("--thermal-guard", type=Path, default=_DEFAULT_THERMAL_GUARD)
    parser.add_argument("--gpu-lease", type=Path, default=_DEFAULT_GPU_LEASE)
    parser.add_argument(
        "--expect-pci",
        action="append",
        type=_card_pci,
        default=[],
        metavar="CARD=ADDRESS",
        help="expected PCI address for one card; repeat for each card",
    )
    _add_stall_floor_argument(parser)
    if run:
        parser.add_argument("--artifact-root", required=True, type=Path, help="directory for write-once run artifacts")


def _add_target_arguments(parser: argparse.ArgumentParser, *, run: bool) -> None:
    parser.add_argument("spec", type=Path, help="experiment YAML spec")
    parser.add_argument("--card", required=True, type=_card_number, help="physical DRM card index")
    parser.add_argument("--port", required=True, type=_port_number, help="local server port")
    parser.add_argument("--server-bin", required=True, type=Path, help="llama-server executable")
    parser.add_argument("--config-root", required=True, type=Path, help="serving configuration directory")
    parser.add_argument("--stock-pp-table", required=True, type=Path, help="stock pp_table reference blob")
    parser.add_argument("--thermal-guard", type=Path, default=_DEFAULT_THERMAL_GUARD)
    parser.add_argument("--gpu-lease", type=Path, default=_DEFAULT_GPU_LEASE)
    parser.add_argument("--expect-pci", help="expected PCI address for the selected card")
    _add_stall_floor_argument(parser)
    if run:
        parser.add_argument("--artifact-root", required=True, type=Path, help="directory for write-once run artifacts")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m inferference.experiment")
    commands = parser.add_subparsers(dest="command", required=True)

    preflight_parser = commands.add_parser("preflight", help="check whether a spec may run")
    _add_target_arguments(preflight_parser, run=False)

    run_parser = commands.add_parser("run", help="run the scheduled experiment")
    _add_target_arguments(run_parser, run=True)

    pair_preflight_parser = commands.add_parser("preflight-pair", help="check whether a dual-card spec may run")
    _add_pair_arguments(pair_preflight_parser, run=False)

    pair_run_parser = commands.add_parser("run-pair", help="run the scheduled experiment on two cards at once")
    _add_pair_arguments(pair_run_parser, run=True)

    pair_analyze_parser = commands.add_parser("analyze-pair", help="analyze the pair records of a dual-card run")
    pair_analyze_parser.add_argument(
        "path", type=Path, help="a <idx>-pair.json file, a pair/ directory, or an artifact root"
    )

    analyze_parser = commands.add_parser("analyze", help="analyze a stored run directory")
    analyze_parser.add_argument("run_dir", type=Path)
    analyze_parser.add_argument("--card", type=_card_number, help="physical DRM card index")
    analyze_parser.add_argument("--load-threshold-w", type=_load_threshold, default=150.0)
    analyze_parser.add_argument("--report", action="store_true", help="render Markdown instead of JSON")

    compare_parser = commands.add_parser("compare", help="compare one endpoint across two run groups")
    compare_parser.add_argument("dir_a", type=Path)
    compare_parser.add_argument("dir_b", type=Path)
    compare_parser.add_argument("--endpoint", required=True, help="endpoint name to compare")
    compare_parser.add_argument("--condition-a", help="select one condition ID from the first group")
    compare_parser.add_argument("--condition-b", help="select one condition ID from the second group")
    compare_parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="DOTTED.PATH",
        help="fixed-block path allowed to differ; repeat for each path",
    )
    compare_parser.add_argument("--report", action="store_true", help="render Markdown instead of JSON")
    return parser


def _target_from_args(args: argparse.Namespace) -> RunTarget:
    stall_floors: dict[int, float] = {}
    for channel, rpm in args.stall_floor:
        if channel in stall_floors:
            raise _UsageError(f"--stall-floor names channel {channel} more than once")
        stall_floors[channel] = rpm
    return RunTarget(
        card=args.card,
        port=args.port,
        server_bin=args.server_bin,
        config_root=args.config_root,
        stock_pp_table=args.stock_pp_table,
        thermal_guard=args.thermal_guard,
        gpu_lease=args.gpu_lease,
        expect_pci=args.expect_pci,
        stall_floors_rpm=stall_floors,
    )


def _pair_targets_from_args(args: argparse.Namespace) -> list[RunTarget]:
    stall_floors: dict[int, float] = {}
    for channel, rpm in args.stall_floor:
        if channel in stall_floors:
            raise _UsageError(f"--stall-floor names channel {channel} more than once")
        stall_floors[channel] = rpm
    expect_pci: dict[int, str] = {}
    for card, address in args.expect_pci:
        if card in expect_pci:
            raise _UsageError(f"--expect-pci names card {card} more than once")
        if card not in args.cards:
            raise _UsageError(f"--expect-pci names card {card}, which is not in --cards {list(args.cards)}")
        expect_pci[card] = address
    return [
        RunTarget(
            card=card,
            port=port,
            server_bin=args.server_bin,
            config_root=args.config_root,
            stock_pp_table=args.stock_pp_table,
            thermal_guard=args.thermal_guard,
            gpu_lease=args.gpu_lease,
            expect_pci=expect_pci.get(card),
            stall_floors_rpm=stall_floors,
        )
        for card, port in zip(args.cards, args.ports, strict=True)
    ]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    return value


def _print_json(value: Any) -> None:
    print(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False))


def _preflight_command(args: argparse.Namespace) -> int:
    target = _target_from_args(args)
    spec = load_spec(args.spec)
    try:
        snapshot = preflight(spec, target, paths=_host_paths(), unit_active=unit_active)
    except PreflightRefusal as exc:
        _print_json(
            {
                "passed": False,
                "refused": True,
                "check": exc.check,
                "detail": exc.detail,
                "checks_passed": exc.passed,
            }
        )
        return 1
    _print_json({"passed": True, **snapshot.to_json()})
    return 0


def _preflight_pair_command(args: argparse.Namespace) -> int:
    targets = _pair_targets_from_args(args)
    spec = load_spec(args.spec)
    try:
        snapshots = preflight_pair(spec, targets, paths=_host_paths(), unit_active=unit_active)
    except PreflightRefusal as exc:
        _print_json(
            {
                "passed": False,
                "refused": True,
                "check": exc.check,
                "detail": exc.detail,
                "checks_passed": exc.passed,
            }
        )
        return 1
    _print_json(
        {
            "passed": True,
            "cards": {
                str(target.card): snapshot.to_json() for target, snapshot in zip(targets, snapshots, strict=True)
            },
        }
    )
    return 0


def _run_pair_command(args: argparse.Namespace) -> int:
    targets = _pair_targets_from_args(args)
    results = run_pair(
        args.spec,
        targets,
        artifact_root=args.artifact_root,
        paths=_host_paths(),
        unit_active=unit_active,
        fan_lease_factory=_fan_lease_factory,
    )
    payload = {
        "passed": bool(results) and all(run.state == "completed" for result in results for run in result.runs.values()),
        "pairs": [
            {
                "index": result.index,
                "record": result.record_path,
                "release_mono": result.release_mono,
                "first_bound_mono": result.first_bound_mono,
                "runs": {
                    str(card): {
                        "run_dir": run.run_dir,
                        "state": run.state,
                        "failure_boundary": run.failure_boundary,
                    }
                    for card, run in result.runs.items()
                },
            }
            for result in results
        ],
    }
    _print_json(payload)
    return 0 if payload["passed"] else 1


def _analyze_pair_command(args: argparse.Namespace) -> int:
    _print_json(analyze_pair_root(args.path))
    return 0


def _run_command(args: argparse.Namespace) -> int:
    target = _target_from_args(args)
    results = run_experiment(
        args.spec,
        target,
        artifact_root=args.artifact_root,
        paths=_host_paths(),
        unit_active=unit_active,
        fan_lease_factory=_fan_lease_factory,
    )
    payload = {
        "passed": bool(results) and all(result.state == "completed" for result in results),
        "runs": [
            {
                "run_dir": result.run_dir,
                "state": result.state,
                "failure_boundary": result.failure_boundary,
            }
            for result in results
        ],
    }
    _print_json(payload)
    return 0 if payload["passed"] else 1


def _select_card(record: RunRecord, requested: int | None) -> int:
    cards = sorted({card for observation in record.observations for card in observation.cards})
    if not cards:
        raise ExperimentCLIError(f"{record.path} has no card observations")
    if requested is not None:
        if requested not in cards:
            raise ExperimentCLIError(f"card {requested} has no telemetry in {record.path}; available cards: {cards}")
        return requested
    identity_path = record.path / "identity.json"
    if identity_path.is_file():
        try:
            identity = json.loads(identity_path.read_text())
            expected_pci = identity.get("card_pci")
            telemetry_path = record.path / "telemetry.jsonl"
            with telemetry_path.open() as handle:
                first_line = handle.readline()
            first_sample = json.loads(first_line)
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise ExperimentCLIError(f"could not identify the measured card in {record.path}: {exc}") from exc
        if expected_pci is not None:
            matches = [
                int(item["card"])
                for item in first_sample.get("cards", ())
                if item.get("pci") == expected_pci and "card" in item
            ]
            if len(matches) != 1:
                raise ExperimentCLIError(
                    "identity.json names PCI "
                    f"{expected_pci!r}, but the first telemetry sample has "
                    f"{len(matches)} matches"
                )
            if matches[0] not in cards:
                raise ExperimentCLIError(f"identity.json maps to card {matches[0]}, which has no loaded observations")
            return matches[0]
    return 0 if 0 in cards else cards[0]


def _band_from_record(record: RunRecord) -> tuple[float, float] | None:
    if record.declaration is None:
        return None
    endpoints = record.declaration.get("endpoints")
    band = endpoints.get("band_c") if isinstance(endpoints, Mapping) else None
    if not isinstance(band, (list, tuple)) or len(band) != 2:
        return None
    low, high = float(band[0]), float(band[1])
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ExperimentCLIError(f"{record.path} has an invalid endpoints.band_c: {band!r}")
    return low, high


def _episode_json(item: Any) -> dict[str, Any]:
    return {
        "start_mono": item.start_mono,
        "end_mono": item.end_mono,
        "duration_s": None if item.end_mono is None else item.end_mono - item.start_mono,
    }


def _sum_tokens(requests: Sequence[Mapping[str, Any]], key: str) -> tuple[int, bool]:
    total = 0
    complete = bool(requests)
    for request in requests:
        value = request.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            complete = False
            continue
        total += value
    return total, complete


def _stage_rates(record: RunRecord) -> dict[str, Any]:
    stage_rows: list[dict[str, Any]] = []
    prefill_group: list[dict[str, Any]] = []
    decode_group: list[dict[str, Any]] = []
    ttft_values: list[float] = []
    stages = () if record.stages is None else record.stages.get("stages", ())

    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        requests = [request for request in stage.get("requests", ()) if isinstance(request, Mapping)]
        try:
            duration_s = float(stage["ended_mono"]) - float(stage["started_mono"])
        except (KeyError, TypeError, ValueError):
            duration_s = 0.0
        if not math.isfinite(duration_s) or duration_s <= 0:
            duration_s = 0.0
        prompt_tokens, prompt_complete = _sum_tokens(requests, "prompt_tokens")
        completion_tokens, completion_complete = _sum_tokens(requests, "completion_tokens")
        prompt_rate = prompt_tokens / duration_s if duration_s and prompt_complete else None
        decode_rate = completion_tokens / duration_s if duration_s and completion_complete else None
        workload = str(stage.get("workload", stage.get("name", "")))
        kind = "prefill" if workload.startswith("prefill_") else "decode"
        row = {
            "name": stage.get("name"),
            "workload": workload,
            "duration_s": duration_s or None,
            "prompt_tokens": prompt_tokens if prompt_complete else None,
            "completion_tokens": completion_tokens if completion_complete else None,
            "prefill_tokens_per_second": prompt_rate,
            "decode_tokens_per_second": decode_rate,
        }
        stage_rows.append(row)
        if kind == "prefill":
            prefill_group.append({"duration_s": duration_s, "tokens": prompt_tokens, "complete": prompt_complete})
        else:
            decode_group.append(
                {"duration_s": duration_s, "tokens": completion_tokens, "complete": completion_complete}
            )
        for request in requests:
            dispatch = request.get("dispatch_mono")
            first_token = request.get("first_token_mono")
            if request.get("valid") is False or dispatch is None or first_token is None:
                continue
            try:
                elapsed = float(first_token) - float(dispatch)
            except (TypeError, ValueError):
                continue
            if math.isfinite(elapsed) and elapsed >= 0:
                ttft_values.append(elapsed)

    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        duration = sum(float(item["duration_s"]) for item in group if item["duration_s"] > 0)
        complete = bool(group) and all(item["complete"] and item["duration_s"] > 0 for item in group)
        tokens = sum(int(item["tokens"]) for item in group)
        return {
            "tokens": tokens if complete else None,
            "duration_s": duration if complete else None,
            "tokens_per_second": tokens / duration if complete and duration > 0 else None,
        }

    return {
        "stages": stage_rows,
        "decode": summarize(decode_group),
        "prefill": summarize(prefill_group),
        "ttft_mean_s": mean(ttft_values) if ttft_values else None,
    }


def _transit_endpoint_names(record: RunRecord, extra: Sequence[str]) -> tuple[str, ...]:
    names = list(extra)
    if record.declaration is not None:
        endpoints = record.declaration.get("endpoints")
        if isinstance(endpoints, Mapping):
            for name in (endpoints.get("primary"), *(endpoints.get("secondary", ()) or ())):
                if isinstance(name, str):
                    names.append(name)
    return tuple(sorted({name for name in names if _TRANSIT_ENDPOINT.fullmatch(name)}))


def _analyze_record(
    record: RunRecord,
    *,
    card: int | None,
    load_threshold_w: float,
    extra_endpoints: Sequence[str] = (),
) -> dict[str, Any]:
    selected_card = _select_card(record, card)
    power = card_series(record, selected_card, "power_w")
    junction = card_series(record, selected_card, "junction_c")
    found_episodes = episodes(power, threshold_w=load_threshold_w)
    first_episode = found_episodes[0] if found_episodes else None
    band_c = _band_from_record(record)

    episode_temperatures: tuple[Point, ...] = ()
    if first_episode is not None:
        end = first_episode.end_mono if first_episode.end_mono is not None else math.inf
        episode_temperatures = window(junction, first_episode.start_mono, end)

    slope = None
    slope_error = None
    transit = None
    band_power_w = None
    if first_episode is not None and band_c is not None:
        selected_band = band_samples(junction, first_episode, band_c=band_c)
        if selected_band:
            selected_times = {point.mono for point in selected_band}
            matched_power = [point.value for point in power if point.mono in selected_times]
            if matched_power:
                band_power_w = mean(matched_power)
            try:
                slope = ols_slope(
                    [point.mono for point in selected_band],
                    [point.value for point in selected_band],
                    band_c=band_c,
                )
            except EstimatorError as exc:
                slope = None
                slope_error = str(exc)
        transit = transit_time(
            [point.mono for point in episode_temperatures],
            [point.value for point in episode_temperatures],
            band_c=band_c,
        )

    peak_value = None
    if episode_temperatures:
        peak_value = peak(
            [point.mono for point in episode_temperatures],
            [point.value for point in episode_temperatures],
            sensor="junction",
        )

    transit_times: dict[str, Any] = {}
    if first_episode is not None:
        for name in _transit_endpoint_names(record, extra_endpoints):
            match = _TRANSIT_ENDPOINT.fullmatch(name)
            if match is None:
                continue
            endpoint_band = (float(match.group(1)), float(match.group(2)))
            if endpoint_band[0] >= endpoint_band[1]:
                raise ExperimentCLIError(f"invalid transit endpoint band in {name!r}")
            transit_times[name] = transit_time(
                [point.mono for point in episode_temperatures],
                [point.value for point in episode_temperatures],
                band_c=endpoint_band,
            )

    rates = _stage_rates(record)
    plateau: dict[str, Any] = {}
    if record.stages is not None:
        declaration_fixed = record.declaration.get("fixed", {}) if record.declaration is not None else {}
        safety = declaration_fixed.get("safety", {}) if isinstance(declaration_fixed, Mapping) else {}
        arm_bound = safety.get("arm_bound_c") if isinstance(safety, Mapping) else None
        for stage in record.stages.get("stages", ()):
            if not isinstance(stage, Mapping):
                continue
            stage_name = str(stage.get("name", ""))
            workload = str(stage.get("workload", stage_name))
            if not workload.startswith("soak_") and not stage_name.startswith("soak_"):
                continue
            if arm_bound is None:
                plateau[stage_name] = {"available": False, "reason": "declaration.fixed.safety.arm_bound_c is missing"}
                continue
            try:
                result = plateau_test(
                    record,
                    stage_name=stage_name,
                    card=selected_card,
                    arm_bound_c=float(arm_bound),
                )
            except (EstimatorError, KeyError, TypeError, ValueError) as exc:
                plateau[stage_name] = {"available": False, "reason": str(exc)}
            else:
                plateau[stage_name] = {"available": True, **_jsonable(result)}

    endpoint_values = {
        "band_slope": None if slope is None else slope.slope_c_per_s,
        "band_power": band_power_w,
        "peak": None if peak_value is None else peak_value.value_c,
        "peak_junction": None if peak_value is None else peak_value.value_c,
        "transit_time": None if transit is None else transit.seconds,
        "ttft": rates["ttft_mean_s"],
        "decode_rate": rates["decode"]["tokens_per_second"],
        "prefill_rate": rates["prefill"]["tokens_per_second"],
    }
    endpoint_values.update({name: item.seconds for name, item in transit_times.items()})
    first_episode_json = None if first_episode is None else _episode_json(first_episode)
    decode_stages = [
        row
        for row in rates["stages"]
        if row["workload"].startswith("decode")
        or row["workload"].startswith("lane_sweep")
        or row["workload"].startswith("soak_")
    ]
    prefill_stages = [row for row in rates["stages"] if row["workload"].startswith("prefill_")]
    analysis = {
        "run_dir": record.path,
        "layout": record.layout,
        "card": selected_card,
        "load_threshold_w": load_threshold_w,
        "outcome": record.outcome,
        "band_c": band_c,
        "episodes": [_episode_json(item) for item in found_episodes],
        "episode": first_episode_json,
        "band_slope": slope,
        "band_fit_error": slope_error,
        "transit_time": transit,
        "transit_times": transit_times,
        "peak": peak_value,
        "band_power_w": band_power_w,
        "plateau": plateau,
        "rates": rates,
        "decode": {
            "rate_tokens_per_second": rates["decode"]["tokens_per_second"],
            "ttft_mean_s": rates["ttft_mean_s"],
            "stages": decode_stages,
        },
        "prefill": {
            "rate_tokens_per_second": rates["prefill"]["tokens_per_second"],
            "stages": prefill_stages,
        },
        "endpoints": endpoint_values,
    }
    return _jsonable(analysis)


def _analyze_command(args: argparse.Namespace) -> int:
    record = load_run(args.run_dir)
    analysis = _analyze_record(record, card=args.card, load_threshold_w=args.load_threshold_w)
    if args.report:

        def report_values(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: report_values(child)
                    for key, child in value.items()
                    if not ("slope" in str(key).casefold() and child is None)
                }
            if isinstance(value, list):
                return [report_values(child) for child in value]
            return value

        print(render_analysis(report_values(analysis)), end="")
    else:
        _print_json(analysis)
    return 0


def _run_directories(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise ExperimentCLIError(f"run group is not a directory: {root}")
    if (root / "run.json").is_file() or (root / "summary.json").is_file():
        return (root,)
    return tuple(sorted(path.parent for path in root.rglob("run.json")))


def _load_group(root: Path, label: str, requested_condition: str | None) -> tuple[tuple[RunRecord, ...], str]:
    paths = _run_directories(root)
    if not paths:
        raise ExperimentCLIError(f"group {label} at {root} contains no run directories")
    records = tuple(load_run(path) for path in paths)
    missing_ids = [str(record.path) for record in records if record.run is None]
    if missing_ids:
        raise ExperimentCLIError(f"group {label} has runs without condition IDs: {missing_ids}")
    ids = {str(record.run["condition"]) for record in records if record.run is not None}
    if requested_condition is None:
        if len(ids) != 1:
            raise ExperimentCLIError(
                f"group {label} contains mixed condition IDs {sorted(ids)}; specify --condition-{label.lower()}"
            )
        selected = records
        condition_id = next(iter(ids))
    else:
        selected = tuple(
            record for record in records if record.run is not None and record.run["condition"] == requested_condition
        )
        if not selected:
            raise ExperimentCLIError(f"group {label} contains no runs for condition ID {requested_condition!r}")
        condition_id = requested_condition
    ordered = tuple(sorted(selected, key=lambda record: (int((record.run or {})["replicate"]), str(record.path))))
    return ordered, condition_id


def _endpoint_value(name: str, analysis: Mapping[str, Any]) -> float:
    value = analysis.get("endpoints", {}).get(name)
    if value is None:
        raise ExperimentCLIError(f"endpoint {name!r} is unavailable for {analysis.get('run_dir')}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentCLIError(f"endpoint {name!r} is not numeric for {analysis.get('run_dir')}") from exc
    if not math.isfinite(numeric):
        raise ExperimentCLIError(f"endpoint {name!r} is not finite for {analysis.get('run_dir')}")
    return numeric


def _compare_command(args: argparse.Namespace) -> int:
    records_a, condition_a = _load_group(args.dir_a, "A", args.condition_a)
    records_b, condition_b = _load_group(args.dir_b, "B", args.condition_b)
    values_a = [
        _endpoint_value(
            args.endpoint,
            _analyze_record(record, card=None, load_threshold_w=150.0, extra_endpoints=(args.endpoint,)),
        )
        for record in records_a
    ]
    values_b = [
        _endpoint_value(
            args.endpoint,
            _analyze_record(record, card=None, load_threshold_w=150.0, extra_endpoints=(args.endpoint,)),
        )
        for record in records_b
    ]
    if len(values_a) != len(records_a) or len(values_b) != len(records_b):
        raise ExperimentCLIError("endpoint values do not match the selected run counts")
    result = compare(
        records_a,
        records_b,
        endpoint=args.endpoint,
        values_a=values_a,
        values_b=values_b,
        allowed_differences=tuple(args.allow),
    )
    if args.report:
        print(render_comparison(result), end="")
    else:
        _print_json(
            {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "comparison": result,
            }
        )
    return 0


def _valid_endpoint_name(name: str) -> bool:
    if name in _KNOWN_ENDPOINTS:
        return True
    match = _TRANSIT_ENDPOINT.fullmatch(name)
    return match is not None and float(match.group(1)) < float(match.group(2))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one experiment command and return its process status."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "compare" and not _valid_endpoint_name(args.endpoint):
            raise _UsageError(f"unsupported endpoint name: {args.endpoint!r}")
        if args.command == "preflight":
            return _preflight_command(args)
        if args.command == "run":
            return _run_command(args)
        if args.command == "preflight-pair":
            return _preflight_pair_command(args)
        if args.command == "run-pair":
            return _run_pair_command(args)
        if args.command == "analyze-pair":
            return _analyze_pair_command(args)
        if args.command == "analyze":
            return _analyze_command(args)
        if args.command == "compare":
            return _compare_command(args)
        raise _UsageError(f"unknown command: {args.command!r}")
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except _UsageError as exc:
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2
    except (
        ComparisonError,
        EstimatorError,
        LoadError,
        PairedAnalysisError,
        PreflightRefusal,
        ReportError,
        RunnerError,
        SpecError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"inferference.experiment: {exc}", file=sys.stderr)
        return 1
    return 1
