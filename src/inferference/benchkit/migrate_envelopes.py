"""Envelope v2 migration stamp.

Stamps ``envelope.json`` onto pre-existing v1 artifact dirs (the ~40 legacy
``project17-*`` dirs under an older artifacts root) so the baseline store can
reference them. Each run dir predates the envelope contract;
the stamp reconstructs the best-effort envelope from what the dir records:

  * workload — derived from the dir name (the ``project17-<name>-<ts>`` shape)
    with the corpus_sha256 from ``summary.json`` when present;
  * environment — parsed from ``runtime-environment.txt`` (KEY=VALUE lines) +
    the ``git-commit.txt`` / ``git-status-before.txt`` head;
  * gate/summary — a "migrated" marker with the end_to_end summary carried
    over verbatim (so the baseline gate can read the aggregate metric);
  * baseline_ref — null (unknown provenance).

Idempotent: skips dirs that already carry ``envelope.json``.

Usage:
    ci/library/.venv/bin/python src/inferference/benchkit/migrate_envelopes.py \
        --artifacts <reference artifacts root> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .envelope import RunEnvelope

# The env keys the envelope records.
_ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "LLAMA_CPP_LIB_PATH", "LD_LIBRARY_PATH", "PYTHONPATH")


def _parse_runtime_environment(path: Path) -> dict[str, str]:
    facts: dict[str, str] = {}
    if not path.is_file():
        return facts
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            if key in _ENV_KEYS or key.strip().endswith("GPU"):
                facts[key.strip()] = value.strip()
    return facts


def _workload_from_dirname(name: str) -> dict[str, object]:
    # project17-json-baseline-20260724T174322Z -> json-baseline
    rest = name.removeprefix("project17-")
    parts = rest.split("-")
    # drop the trailing timestamp chunk (e.g. 20260724T174322Z)
    while parts and (parts[-1].endswith("Z") or parts[-1][0].isdigit()):
        parts.pop()
    return {"name": "-".join(parts) or name, "version": "legacy-v1"}


def _summary_carryover(dir_path: Path) -> dict[str, object]:
    summary_path = dir_path / "summary.json"
    if not summary_path.is_file():
        return {}
    try:
        data = json.loads(summary_path.read_text())
    except Exception:  # noqa: BLE001 — a corrupt legacy summary is recorded, not fatal
        return {"corrupt": True}
    return {"end_to_end": data.get("end_to_end", {}), "migrated": True}


def stamp_one(dir_path: Path, dry: bool) -> Path | None:
    if (dir_path / "envelope.json").is_file():
        return None
    name = dir_path.name
    env = _parse_runtime_environment(dir_path / "runtime-environment.txt")
    git_head = ""
    if (dir_path / "git-commit.txt").is_file():
        git_head = (dir_path / "git-commit.txt").read_text(errors="replace").strip()
    elif (dir_path / "git-status-before.txt").is_file():
        first = (dir_path / "git-status-before.txt").read_text(errors="replace").splitlines()
        git_head = first[0].strip() if first else ""
    envelope = RunEnvelope(
        run_id=f"legacy-{name}",
        workload=_workload_from_dirname(name),
        environment={
            "fingerprint": {"git_head": git_head},
            "env": env,
            "nvidia_smi": {},
        },
        baseline_ref=None,
        gate={"kind": "perf", "verdict": "pass", "migrated": True},
        summary=_summary_carryover(dir_path),
        records="records/",
        command=[],
    )
    if not dry:
        (dir_path / "envelope.json").write_text(envelope.model_dump_json(indent=2) + "\n")
    return dir_path / "envelope.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="the legacy artifacts root that holds the project17-* run dirs",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dirs = sorted(p for p in args.artifacts.glob("project17-*") if p.is_dir())
    stamped = 0
    skipped = 0
    for d in dirs:
        target = stamp_one(d, args.dry_run)
        if target is None:
            skipped += 1
        else:
            stamped += 1
            print(f"  {'would stamp' if args.dry_run else 'stamped'} {target}")
    print(f">> {stamped} dir(s) {'to stamp' if args.dry_run else 'stamped'}; {skipped} already have envelope.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
