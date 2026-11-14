"""Resolve where the router is, so a consumer does not learn the config layout.

YAML is the only source of runtime configuration; the environment names only
*where* the YAML lives. ``config_dir()`` is the one definition of that location,
shared by the reconciler and by consumers of the upward contract.

Fail closed. A missing ``placement.yaml`` is a loud refusal naming the path it
looked at, never a silent default to port 8100.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .placement import Placement, PlacementError


class ConfigError(PlacementError):
    """The router configuration cannot be read. Never a silent default."""


#: The system dir of the router service. It wins over the home dir when it exists.
SYSTEM_CONFIG_DIR = Path("/etc/inferference")
HOME_CONFIG_DIR = "~/.config/inferference"


def resolve_config_dir(system_dir: Path | None = None, home_dir: str | None = None) -> tuple[Path, str]:
    """Return the config directory and the reason it was chosen.

    This is the one definition of the order. The reconcile CLI and the library
    both call it. The order is:

    1. ``$INFERFERENCE_CONFIG_DIR``, when set and not empty (reason ``env``).
    2. ``system_dir`` (``/etc/inferference``), when it exists (reason ``system``).
    3. ``home_dir`` (``~/.config/inferference``) (reason ``home-fallback``).

    The arguments exist so that a test can point at a fake system dir. ``None``
    means the module constant, read at call time.
    """
    system_dir = SYSTEM_CONFIG_DIR if system_dir is None else system_dir
    home_dir = HOME_CONFIG_DIR if home_dir is None else home_dir
    env = os.environ.get("INFERFERENCE_CONFIG_DIR")
    if env:
        return Path(env).expanduser(), "env"
    if system_dir.is_dir():
        return system_dir, "system"
    return Path(home_dir).expanduser(), "home-fallback"


def _config_dir() -> Path:
    return resolve_config_dir()[0]


def config_dir() -> Path:
    """Return the inferference config directory.

    The order is ``$INFERFERENCE_CONFIG_DIR``, then ``/etc/inferference`` when
    it exists, then ``~/.config/inferference``. This is the order of the
    reconcile CLI. The directory is not required to exist here.
    """
    return _config_dir()


def resolve_router_url(config_dir: Path | None = None) -> str:
    """Return the router's OpenAI-compatible base URL as ``placement.yaml`` declares it.

    The ``/v1`` form is returned because :class:`InferferenceClient` accepts it
    and strips the suffix itself. Raises :class:`ConfigError` when
    ``placement.yaml`` is missing or malformed, naming the path it looked at.
    """
    root = Path(config_dir) if config_dir is not None else _config_dir()
    placement_path = root / "placement.yaml"
    if not placement_path.is_file():
        raise ConfigError(f"no placement at {placement_path}")
    try:
        raw = yaml.safe_load(placement_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read {placement_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"cannot parse {placement_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{placement_path}: placement must be a mapping")
    try:
        placement = Placement.model_validate(raw)
    except ValueError as error:
        raise ConfigError(f"{placement_path}: {error}") from error
    return f"{placement.router.url}/v1"


__all__ = ["ConfigError", "config_dir", "resolve_config_dir", "resolve_router_url"]
