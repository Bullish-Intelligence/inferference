"""Load a run directory into typed records, for either telemetry layout.

Two shapes exist. The **current** shape is a run directory with `run.json` at
its root and a `telemetry.jsonl` of `TelemetrySample.to_json_line()` records,
each schema-versioned. The **legacy** shape predates `run.json` and
`schema_version` entirely: it has `summary.json` beside a `telemetry.jsonl` of
a different, unversioned record shape, and no monotonic clock at all.

`load_run` reads either shape into one `RunRecord`, so an analysis function
never has to know which layout it was handed. It never guesses: an unknown or
missing `schema_version`, a directory that carries markers of both shapes,
or a directory that matches neither, is a refusal (`LoadError`), never a
best-effort parse.

This module is read-only. It never writes into a run directory, least of all
a legacy one — the legacy shape is historical record, not a target for
migration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from inferference.telemetry import load_jsonl

Layout = Literal["030-v1", "pre030-v1"]


class LoadError(ValueError):
    """A run directory did not match a known layout, or a record in it refused to parse."""


@dataclass(frozen=True, slots=True)
class CardObservation:
    """One card's reading at one tick, named the same way across both layouts."""

    junction_c: float
    edge_c: float
    mem_c: float
    power_w: float
    gpu_busy_percent: int | None
    throttled: bool | None
    gtt_used_mib: float | None


@dataclass(frozen=True, slots=True)
class Observation:
    """One telemetry tick: both clocks it could have, and every card it names."""

    utc: str
    mono: float
    clock: Literal["monotonic", "utc-derived"]
    cards: Mapping[int, CardObservation]


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One run directory, loaded: its artifacts as plain mappings, and its series."""

    path: Path
    layout: Layout
    run: Mapping[str, Any] | None
    declaration: Mapping[str, Any] | None
    stages: Mapping[str, Any] | None
    outcome: Mapping[str, Any] | None
    observations: tuple[Observation, ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise LoadError(f"could not read {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LoadError(f"{path} is not valid JSON: {exc}") from exc


def _require_schema_version_1(payload: dict[str, Any], path: Path) -> None:
    version = payload.get("schema_version")
    if version != 1:
        raise LoadError(f"{path} has schema_version {version!r}, expected 1")


def _load_required_json(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    _require_schema_version_1(payload, path)
    return payload


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_required_json(path)


def _parse_utc(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise LoadError(f"could not parse UTC timestamp {value!r}: {exc}") from exc


def _card_observation_from_030(card: dict[str, Any]) -> CardObservation:
    return CardObservation(
        junction_c=float(card["temperature_hotspot_c"]),
        edge_c=float(card["temperature_edge_c"]),
        mem_c=float(card["temperature_mem_c"]),
        power_w=float(card["power_w"]),
        gpu_busy_percent=card.get("gpu_busy_percent"),
        throttled=card.get("throttled"),
        gtt_used_mib=card.get("gtt_used_mib"),
    )


def _load_030_observations(telemetry_path: Path) -> tuple[Observation, ...]:
    lines = load_jsonl(telemetry_path)
    observations = []
    for line in lines:
        version = line.get("schema_version")
        if version != 1:
            raise LoadError(f"{telemetry_path} has a telemetry line with schema_version {version!r}, expected 1")
        try:
            cards = {int(card["card"]): _card_observation_from_030(card) for card in line["cards"]}
            observations.append(
                Observation(utc=line["utc"], mono=float(line["monotonic"]), clock="monotonic", cards=cards)
            )
        except KeyError as exc:
            raise LoadError(f"{telemetry_path} has a malformed 030-v1 telemetry line, missing key {exc}") from exc
    return tuple(observations)


def _load_030_v1(path: Path) -> RunRecord:
    run_payload = _load_required_json(path / "run.json")
    declaration = _load_optional_json(path / "declaration.json")
    stages = _load_optional_json(path / "stages.json")
    outcome = _load_optional_json(path / "outcome.json")

    telemetry_path = path / "telemetry.jsonl"
    if not telemetry_path.is_file():
        raise LoadError(f"{path} is a 030-v1 run but has no telemetry.jsonl")
    observations = _load_030_observations(telemetry_path)

    return RunRecord(
        path=path,
        layout="030-v1",
        run=run_payload,
        declaration=declaration,
        stages=stages,
        outcome=outcome,
        observations=observations,
    )


def _card_observation_from_legacy(card: dict[str, Any]) -> CardObservation:
    temperatures = card["temperatures"]
    gtt_used_bytes = card.get("gtt_used_bytes")
    return CardObservation(
        junction_c=float(temperatures["junction"]["celsius"]),
        edge_c=float(temperatures["edge"]["celsius"]),
        mem_c=float(temperatures["mem"]["celsius"]),
        power_w=float(card["power_w"]),
        gpu_busy_percent=None,
        throttled=None,
        gtt_used_mib=None if gtt_used_bytes is None else gtt_used_bytes / 2**20,
    )


def _load_legacy(path: Path) -> RunRecord:
    telemetry_path = path / "telemetry.jsonl"
    lines = load_jsonl(telemetry_path)
    if not lines:
        raise LoadError(f"{telemetry_path} has no samples; a legacy run needs at least one")

    try:
        first_utc = _parse_utc(lines[0]["utc"])
        observations = []
        for line in lines:
            utc_str = line["utc"]
            mono = (_parse_utc(utc_str) - first_utc).total_seconds()
            cards = {int(card_id): _card_observation_from_legacy(card) for card_id, card in line["cards"].items()}
            observations.append(Observation(utc=utc_str, mono=mono, clock="utc-derived", cards=cards))
    except KeyError as exc:
        raise LoadError(f"{telemetry_path} has a malformed legacy telemetry line, missing key {exc}") from exc

    return RunRecord(
        path=path,
        layout="pre030-v1",
        run=None,
        declaration=None,
        stages=None,
        outcome=None,
        observations=tuple(observations),
    )


def load_run(path: Path) -> RunRecord:
    """Load a run directory at `path`, detecting its layout.

    A `run.json` at the root means the current shape. No `run.json`, but a
    `summary.json` beside a `telemetry.jsonl`, means the legacy shape. Both
    markers together is a mixed
    directory, and neither pairing is present is an unknown layout — both
    refuse with `LoadError`, naming the problem, never a guess.
    """
    has_run = (path / "run.json").is_file()
    has_summary = (path / "summary.json").is_file()
    has_telemetry = (path / "telemetry.jsonl").is_file()

    if has_run and has_summary:
        raise LoadError(f"{path} is a mixed layout: both run.json (030-v1) and summary.json (legacy) are present")
    if has_run:
        return _load_030_v1(path)
    if has_summary and has_telemetry:
        return _load_legacy(path)

    missing = [
        name
        for name, present in (("run.json", has_run), ("summary.json", has_summary), ("telemetry.jsonl", has_telemetry))
        if not present
    ]
    raise LoadError(
        f"{path} matches neither the 030-v1 layout (needs run.json) nor the legacy layout "
        f"(needs summary.json and telemetry.jsonl); missing: {', '.join(missing)}"
    )
