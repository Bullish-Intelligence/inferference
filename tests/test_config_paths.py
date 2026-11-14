"""`inferference.config_dir()` follows the order of the reconcile CLI.

The order is $INFERFERENCE_CONFIG_DIR, then /etc/inferference when it exists,
then ~/.config/inferference. Before, the library skipped the system dir, so a
consumer read a home copy while the router served /etc.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import inferference
from inferference import config_paths


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("INFERFERENCE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _system(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, exists: bool) -> Path:
    system = tmp_path / "etc-inferference"
    if exists:
        system.mkdir()
    monkeypatch.setattr(config_paths, "SYSTEM_CONFIG_DIR", system)
    return system


def test_the_env_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _system(monkeypatch, tmp_path, exists=True)
    monkeypatch.setenv("INFERFERENCE_CONFIG_DIR", str(tmp_path / "explicit"))
    assert config_paths.config_dir() == tmp_path / "explicit"
    assert inferference.config_dir() == tmp_path / "explicit"


def test_a_tilde_in_the_env_expands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _system(monkeypatch, tmp_path, exists=True)
    monkeypatch.setenv("INFERFERENCE_CONFIG_DIR", "~/cfg")
    assert config_paths.config_dir() == tmp_path / "home" / "cfg"


def test_the_system_dir_wins_over_home_when_it_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    system = _system(monkeypatch, tmp_path, exists=True)
    (tmp_path / "home" / ".config" / "inferference").mkdir(parents=True)
    assert config_paths.config_dir() == system


def test_home_is_the_last_resort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _system(monkeypatch, tmp_path, exists=False)
    assert config_paths.config_dir() == tmp_path / "home" / ".config" / "inferference"


def test_the_default_system_dir_is_etc_inferference() -> None:
    assert config_paths.SYSTEM_CONFIG_DIR == Path("/etc/inferference")
    assert config_paths.HOME_CONFIG_DIR == "~/.config/inferference"


def test_resolve_router_url_reads_the_system_dir_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    system = _system(monkeypatch, tmp_path, exists=True)
    with pytest.raises(config_paths.ConfigError, match=str(system / "placement.yaml")):
        config_paths.resolve_router_url()
