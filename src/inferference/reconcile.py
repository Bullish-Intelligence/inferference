"""Reading the running router, and reading the config that overrides it.

This is the half of a reconcile that talks to the outside world: the YAML on
disk, the router's ``/models``, and the cards' own sysfs accounting. It is split
out of ``ci/runner/inferference-reconcile.py`` so it can be tested on a CPU with
no router and no GPU — the CLI script is now argument parsing and printing.

The rules it enforces are the ones that cannot be recovered from later:

* **A missing thing is loud.** No config, no model file, no footprint,
  unreachable router, unknown device — every one is a refusal. A reconcile that
  quietly does less than it was asked is worse than one that stops.
* **A crashed child is not an unloaded one.** They report the same ``status``
  value and differ only in ``exit_code``. That difference is the only health
  signal the router gives, so it is read here rather than eyeballed in a table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from .placement import (
    ModelSpec,
    Placement,
    PlacementError,
    parse_preset_sections,
    resolve_auto_devices,
)

__all__ = [
    "ChildHealth",
    "ConfigError",
    "child_health",
    "crashed_children",
    "load_config",
    "loaded_names",
    "models_with_matching_presets",
    "read_config_dir",
    "router_models_max",
    "running_presets",
]


class ConfigError(PlacementError):
    """The declared configuration cannot be used. Never a warning."""


class ChildHealth(BaseModel):
    """One router child, classified."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: str
    exit_code: int | None = None

    @property
    def crashed(self) -> bool:
        """A child that died on its own.

        The router reports a crashed child as ``unloaded`` WITH an exit code; a
        child we unloaded on purpose is ``unloaded`` with ``exit_code`` null.
        Nothing else distinguishes them, so a non-null exit code is the whole
        signal — and a zero exit code still counts, because a declared-resident
        model is not supposed to exit at all.
        """
        return self.status == "unloaded" and self.exit_code is not None

    @property
    def loaded(self) -> bool:
        return self.status == "loaded"


def child_health(models: list[dict[str, Any]]) -> list[ChildHealth]:
    """Classify every child the router reports."""
    out: list[ChildHealth] = []
    for m in models:
        status = m.get("status") or {}
        exit_code = status.get("exit_code")
        out.append(
            ChildHealth(
                name=m.get("id", "?"),
                status=status.get("value", "?"),
                exit_code=exit_code if isinstance(exit_code, int) else None,
            )
        )
    return out


def crashed_children(models: list[dict[str, Any]]) -> list[ChildHealth]:
    return [c for c in child_health(models) if c.crashed]


def loaded_names(models: list[dict[str, Any]]) -> list[str]:
    return [c.name for c in child_health(models) if c.loaded]


def models_with_matching_presets(models: list[dict[str, Any]], names: list[str], expected_ini: str) -> set[str]:
    """Return requested children that are loaded with their declared preset."""
    expected = parse_preset_sections(expected_ini)
    matches: set[str] = set()
    for model in models:
        name = model.get("id")
        status = model.get("status") or {}
        preset = status.get("preset")
        if (
            name in names
            and status.get("value") == "loaded"
            and isinstance(preset, str)
            and parse_preset_sections(preset).get(name) == expected.get(name)
            and name in expected
        ):
            matches.add(name)
    return matches


def router_models_max(models: list[dict[str, Any]]) -> int | None:
    """Read ``--models-max`` off any child's rendered argv.

    Returns None when no child is running, because the flag is only visible
    through a child. None means UNKNOWN, never "fine": the caller cross-checks
    it against the value declared in placement.yaml rather than proceeding.
    """
    for m in models:
        args = (m.get("status") or {}).get("args") or []
        if "--models-max" in args:
            try:
                return int(args[args.index("--models-max") + 1])
            except (IndexError, ValueError, TypeError):
                return None
    return None


def running_presets(models: list[dict[str, Any]]) -> str | None:
    """The preset text the router says each LOADED child is actually running.

    This is the honest answer to "what is running now", and it is what a changed
    model is detected against. The generated INI on disk is a statement of
    INTENT — this tool wrote it — so a router restarted against an older file,
    or a preset changed through the router's own endpoints, would compare equal
    to it and the change would be missed.

    Returns None when no child reports a preset, which the planner reads as
    "unknown" and answers by reloading everything.

    MEASURED 2026-09-05: llama.cpp reports this as ``status.preset``, verbatim,
    including the section header.
    """
    parts = [(m.get("status") or {}).get("preset") for m in models if (m.get("status") or {}).get("value") == "loaded"]
    real = [p for p in parts if isinstance(p, str) and p.strip()]
    return "\n".join(real) if real else None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if raw is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping at the top level, got {type(raw).__name__}")
    return raw


def read_config_dir(root: Path, *, require_model_files: bool = True) -> tuple[Placement, dict[str, ModelSpec]]:
    """Load ``placement.yaml`` and every ``models/*.yaml`` under ``root``.

    ``require_model_files`` checks that each declared GGUF is really there. A
    missing file would otherwise fail at load time, on the card, AFTER the budget
    said yes — so it is caught here instead.
    """
    pfile = root / "placement.yaml"
    if not pfile.is_file():
        raise ConfigError(f"no placement at {pfile}. See ci/runner/placement.example.yaml.")
    try:
        placement = Placement.model_validate(_read_yaml(pfile))
    except ValueError as exc:
        raise ConfigError(f"{pfile}: {exc}") from exc

    mdir = root / "models"
    if not mdir.is_dir():
        raise ConfigError(f"no model configs at {mdir}. See ci/runner/model.example.yaml.")

    specs: dict[str, ModelSpec] = {}
    for f in sorted(mdir.glob("*.yaml")):
        data = _read_yaml(f)
        data.setdefault("name", f.stem)
        try:
            spec = ModelSpec.model_validate(data)
        except ValueError as exc:
            raise ConfigError(f"{f}: {exc}") from exc
        if spec.name != f.stem:
            raise ConfigError(
                f"{f}: declares name '{spec.name}' but the file is named '{f.stem}'. The file "
                f"name is the model's name; two names for one model is ambiguous."
            )
        if require_model_files and not spec.model.expanduser().is_file():
            raise ConfigError(f"{f}: model file not found: {spec.model}")
        specs[spec.name] = spec

    if not specs:
        raise ConfigError(f"{mdir} defines no models")

    # A placement naming a model nobody defined is caught in budget_devices too,
    # but saying it here names the FILE that is missing rather than the symptom.
    declared = {r.model for r in placement.resident}
    missing = sorted(declared - set(specs))
    if missing:
        raise ConfigError(
            f"{pfile} declares model(s) {missing} resident, but no file defines them. "
            f"Expected {', '.join(str(mdir / f'{m}.yaml') for m in missing)}."
        )
    # Resolve automatic placements HERE, so every consumer -- render, plan, apply,
    # the launcher -- sees a placement whose every model names a concrete device.
    # Doing it once at the read boundary is what keeps `device:` optional from
    # leaking a None into the preset writer or the budget check.
    placement = resolve_auto_devices(placement, specs)
    return placement, specs


def load_config(root: Path, **kw: Any) -> tuple[Placement, dict[str, ModelSpec]]:
    """``read_config_dir`` under its historical name."""
    return read_config_dir(root, **kw)
