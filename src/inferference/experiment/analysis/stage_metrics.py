"""Validate experiment stages and calculate token rates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class StageMetricsError(ValueError):
    """A stage is missing or has invalid measurements."""


def _mapping(value: Any, label: str, run_dir: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageMetricsError(f"{run_dir}: {label} must be an object")
    return value


def _positive_number(value: Any, label: str, run_dir: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageMetricsError(f"{run_dir}: {label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise StageMetricsError(f"{run_dir}: {label} must be finite and positive")
    return number


def _single_stage(stages: Mapping[str, Any], name: str, run_dir: Path) -> Mapping[str, Any]:
    """Return the one named stage, without asserting it completed."""
    found = [item for item in stages.get("stages", ()) if isinstance(item, Mapping) and item.get("name") == name]
    if len(found) != 1:
        raise StageMetricsError(f"{run_dir}: expected one {name!r} stage, found {len(found)}")
    return found[0]


def _stage(stages: Mapping[str, Any], name: str, run_dir: Path) -> Mapping[str, Any]:
    item = _single_stage(stages, name, run_dir)
    if item.get("completed") is not True or item.get("stop_reason") is not None:
        raise StageMetricsError(f"{run_dir}: stage {name!r} did not complete cleanly")
    start = _positive_number(item.get("started_mono"), f"{name}.started_mono", run_dir)
    end = _positive_number(item.get("ended_mono"), f"{name}.ended_mono", run_dir)
    if end <= start:
        raise StageMetricsError(f"{run_dir}: stage {name!r} has non-positive duration")
    return item


def _valid_requests(stage: Mapping[str, Any], name: str, run_dir: Path) -> tuple[Mapping[str, Any], ...]:
    requests = stage.get("requests")
    if not isinstance(requests, list) or not requests:
        raise StageMetricsError(f"{run_dir}: stage {name!r} has no request records")
    rows = tuple(_mapping(item, f"{name}.request", run_dir) for item in requests)
    invalid = [item.get("request_id", "unknown") for item in rows if item.get("valid") is not True]
    if invalid:
        raise StageMetricsError(f"{run_dir}: stage {name!r} has invalid request(s): {invalid}")
    return rows


def _timing_rate(stage: Mapping[str, Any], *, workload: str, count_key: str, duration_key: str, run_dir: Path) -> float:
    requests = _valid_requests(stage, workload, run_dir)
    token_count = 0.0
    duration_ms = 0.0
    for request in requests:
        timings = _mapping(request.get("server_timings"), f"{workload}.server_timings", run_dir)
        token_count += _positive_number(timings.get(count_key), f"{workload}.{count_key}", run_dir)
        duration_ms += _positive_number(timings.get(duration_key), f"{workload}.{duration_key}", run_dir)
    return token_count * 1000.0 / duration_ms


def _stage_requests_throughput(stage: Mapping[str, Any], run_dir: Path) -> float:
    requests = _valid_requests(stage, "lane_sweep", run_dir)
    tokens = 0
    for request in requests:
        value = request.get("completion_tokens")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise StageMetricsError(f"{run_dir}: lane_sweep completion_tokens must be a positive integer")
        tokens += value
    start = float(stage["started_mono"])
    end = float(stage["ended_mono"])
    return tokens / (end - start)
