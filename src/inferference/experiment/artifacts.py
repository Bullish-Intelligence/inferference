"""The run directory: write-once, schema-versioned, named by condition.

A run directory is write-once. Nothing rewrites it, including this module.
The module holds the whole artifact contract: `run.json` and `requests/`,
plus `declaration.json`, `preflight.json`, `resolved-command.json`,
`identity.json`, `residency.json`, `stages.json`, `control.json`,
`outcome.json`, `telemetry.jsonl`, and the two blob kinds (`pp_table.bin`,
`server.log`, `list-devices.txt`).

Every writer added here is write-once, the same way `write_json_artifact` is:
a second write to the same name is a `RunDirectoryError`, never a silent
overwrite. The write itself goes through a temp file in the run directory,
linked onto the target with `os.link` — which fails if the target already
exists — then the temp file is unlinked. This makes "does the target already
have content" and "create the target" one atomic step, so two writers racing
for the same name can never both believe they won.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, get_args
from uuid import uuid4

from inferference.telemetry import TelemetrySample

from .conditions import Condition
from .spec import ExperimentSpec

RUN_DIRECTORY_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1

OutcomeState = Literal["completed", "bounded", "aborted", "invalid"]
_OUTCOME_STATES: frozenset[str] = frozenset(get_args(OutcomeState))


class RunDirectoryError(RuntimeError):
    """A run directory, or a file inside it, was about to be overwritten."""


def run_directory_name(utc: datetime, experiment: str, condition: Condition, replicate: int) -> str:
    """`<utc>-<experiment>-<condition-hash>-r<rep>`, the one naming rule."""
    timestamp = utc.strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{experiment}-{condition.fixed_digest[:12]}-r{replicate}"


def create_run_directory(
    artifact_root: Path,
    experiment: str,
    condition: Condition,
    replicate: int,
    utc: datetime | None = None,
) -> Path:
    """Create a fresh, empty run directory. Refuses if one already exists."""
    resolved_utc = utc if utc is not None else datetime.now(UTC)
    run_dir = artifact_root / run_directory_name(resolved_utc, experiment, condition, replicate)
    if run_dir.exists():
        raise RunDirectoryError(f"{run_dir} already exists; a run directory is write-once")
    run_dir.mkdir(parents=True)
    (run_dir / "requests").mkdir()
    return run_dir


def write_json_artifact(run_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write one JSON artifact into a run directory. Refuses to overwrite it."""
    path = run_dir / name
    if path.exists():
        raise RunDirectoryError(f"{path} already exists; a run directory is write-once")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def write_run_manifest(
    run_dir: Path,
    *,
    experiment: str,
    condition: Condition,
    replicate: int,
    spec_sha256: str,
) -> Path:
    """Write `run.json`: the one file every reader checks before trusting the rest."""
    return write_json_artifact(
        run_dir,
        "run.json",
        {
            "schema_version": RUN_DIRECTORY_SCHEMA_VERSION,
            "experiment": experiment,
            "condition": condition.condition_id,
            "factor_name": condition.factor_name,
            "factor_value": condition.factor_value,
            "fixed_digest": condition.fixed_digest,
            "replicate": replicate,
            "spec_sha256": spec_sha256,
        },
    )


def _written_utc() -> str:
    return datetime.now(UTC).isoformat()


def _write_once_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` exactly once; refuse a second write.

    Writes to a temp file beside `path`, links the temp file onto `path`
    (`os.link` fails if `path` already exists), then unlinks the temp file.
    Linking is the one atomic step that both creates `path` and checks it did
    not already exist, so there is no window between a check and a write.
    """
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp_path.write_bytes(data)
    try:
        os.link(temp_path, path)
    except FileExistsError as exc:
        raise RunDirectoryError(f"{path} already exists; a run directory is write-once") from exc
    finally:
        temp_path.unlink()


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write one schema-versioned JSON artifact, once."""
    body: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "written_utc": _written_utc(),
        **payload,
    }
    data = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()
    _write_once_bytes(path, data)
    return path


def _write_content_addressed(path: Path, data: bytes) -> None:
    """Write `data` to `path`, keyed by its own content.

    A file already at `path` with the same bytes is a no-op: two writers for
    the same content agree and neither needs to win. Different bytes under
    the same name is a `RunDirectoryError` — the name is supposed to be the
    content's own hash, so this can only mean the caller computed the name
    from different bytes than the ones it is trying to write.

    Two lanes can write the same content at the same moment: both see no file,
    and the second link loses. `_write_once_bytes` links atomically, so the
    winner's file is whole. The loser compares bytes, like the first check.
    """
    if path.exists():
        if path.read_bytes() == data:
            return
        raise RunDirectoryError(f"{path} already exists with different content; requests/ is content-addressed")
    try:
        _write_once_bytes(path, data)
    except RunDirectoryError:
        if path.exists() and path.read_bytes() == data:
            return
        raise


def write_declaration(run_dir: Path, *, spec: ExperimentSpec, condition: Condition, spec_sha256: str) -> Path:
    """Write `declaration.json`: the one place the arm bound and machine limit live.

    `fixed` is the spec's whole fixed block, so `arm_bound_c` and
    `machine_limit_c` are recorded here, inside `fixed.safety`, and nowhere
    else — no other writer in this module copies them out.
    """
    payload = {
        "experiment": spec.experiment,
        "spec_sha256": spec_sha256,
        "fixed": spec.fixed.model_dump(mode="json"),
        "factor": {"name": condition.factor_name, "value": condition.factor_value},
        "workloads": list(spec.workloads),
        "replication": spec.replication.model_dump(mode="json"),
        "endpoints": spec.endpoints.model_dump(mode="json"),
    }
    return _write_json_once(run_dir / "declaration.json", payload)


def write_preflight(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Write `preflight.json`: the preflight snapshot, or a refusal's passed checks."""
    return _write_json_once(run_dir / "preflight.json", payload)


def write_resolved_command(run_dir: Path, *, argv: Sequence[str], env: Mapping[str, str], cwd: str) -> Path:
    """Write `resolved-command.json`: the exact argv, env, and cwd the server ran with."""
    payload = {"argv": list(argv), "env": dict(env), "cwd": cwd}
    return _write_json_once(run_dir / "resolved-command.json", payload)


def write_identity(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Write `identity.json`: model, build, repo, and pp_table identity, as read."""
    return _write_json_once(run_dir / "identity.json", payload)


def write_residency(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Write `residency.json`: the ten placement checks, in order, every one recorded."""
    return _write_json_once(run_dir / "residency.json", payload)


def write_stages(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Write `stages.json`: the start gate and every workload stage's requests."""
    return _write_json_once(run_dir / "stages.json", payload)


def write_pair_record(pair_dir: Path, index: int, payload: Mapping[str, Any]) -> Path:
    """Write `<index>-pair.json`: both run directories of one pair run, and the release time.

    `pair_dir` is created when missing. The write is once per index, like every
    other artifact.
    """
    pair_dir.mkdir(parents=True, exist_ok=True)
    return _write_json_once(pair_dir / f"{index:02d}-pair.json", payload)


def write_control(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Write `control.json`: the fan lease and power profile request and readback."""
    return _write_json_once(run_dir / "control.json", payload)


def write_outcome(
    run_dir: Path,
    *,
    state: OutcomeState,
    failure_boundary: str | None,
    detail: str | None,
    started_utc: str,
    ended_utc: str,
) -> Path:
    """Write `outcome.json`: the one verdict every reader checks first.

    Refuses an unknown `state`. Refuses `failure_boundary=None` paired with
    any state other than `"completed"` — a non-completed run always names
    where it stopped.
    """
    if state not in _OUTCOME_STATES:
        raise RunDirectoryError(f"outcome state {state!r} is not one of {sorted(_OUTCOME_STATES)}")
    if failure_boundary is None and state != "completed":
        raise RunDirectoryError(f"outcome state {state!r} needs a failure_boundary; only 'completed' may omit one")
    payload = {
        "state": state,
        "failure_boundary": failure_boundary,
        "detail": detail,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
    }
    return _write_json_once(run_dir / "outcome.json", payload)


def write_text_artifact(run_dir: Path, name: str, text: str) -> Path:
    """Write one verbatim text artifact (`server.log`, `list-devices.txt`), once."""
    path = run_dir / name
    _write_once_bytes(path, text.encode())
    return path


def write_blob(run_dir: Path, name: str, data: bytes) -> str:
    """Write one verbatim binary artifact (`pp_table.bin`), once. Returns its sha256."""
    path = run_dir / name
    _write_once_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


def write_request_body(run_dir: Path, payload: Mapping[str, Any]) -> str:
    """Write one request payload under `requests/`, content-addressed. Returns its sha256."""
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(data).hexdigest()
    _write_content_addressed(run_dir / "requests" / f"{digest}.request.json", data)
    return digest


def write_response_body(run_dir: Path, raw: bytes) -> str:
    """Write one raw response body under `requests/`, content-addressed. Returns its sha256."""
    digest = hashlib.sha256(raw).hexdigest()
    _write_content_addressed(run_dir / "requests" / f"{digest}.response.sse", raw)
    return digest


class TelemetryWriter:
    """A write-once, append-only `telemetry.jsonl` writer.

    Opens `telemetry.jsonl` with mode `"x"` (exclusive create) at
    construction, so a second writer for the same run directory refuses
    immediately, before any tick is lost to a race between two writers.
    """

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / "telemetry.jsonl"
        try:
            self._handle = self._path.open("x")
        except FileExistsError as exc:
            raise RunDirectoryError(f"{self._path} already exists; telemetry.jsonl is write-once") from exc

    def write(self, sample: TelemetrySample) -> None:
        """Append one tick and flush, so a reader never sees a partial line."""
        self._handle.write(sample.to_json_line())
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> TelemetryWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
