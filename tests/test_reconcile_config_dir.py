"""Which config the reconcile CLI reads, and which preset file `apply` may trust.

No router and no GPU. The fault guarded here: the CLI read a stale home config
while the router served another one, so the budget used the wrong footprint.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci/runner/inferference-reconcile.py"
SPEC = importlib.util.spec_from_file_location("inferference_reconcile_cli_config_dir", SCRIPT)
assert SPEC and SPEC.loader
reconciler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reconciler)

WANTED = "[a]\nmodel = /m/a.gguf\nparallel = 8\n"
STALE = "[a]\nmodel = /m/a.gguf\nparallel = 4\n"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERFERENCE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("INFERFERENCE_PRESET_INI", raising=False)
    reconciler._ANNOUNCED.clear()


def _dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, system: bool, home: bool) -> tuple[Path, Path]:
    system_dir = tmp_path / "etc-inferference"
    home_dir = tmp_path / "home-inferference"
    if system:
        system_dir.mkdir()
    if home:
        home_dir.mkdir()
    monkeypatch.setattr(reconciler, "SYSTEM_CONFIG_DIR", system_dir)
    monkeypatch.setattr(reconciler, "HOME_CONFIG_DIR", str(home_dir))
    return system_dir, home_dir


def test_unset_env_uses_the_system_dir_when_it_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    system_dir, _home = _dirs(monkeypatch, tmp_path, system=True, home=True)
    assert reconciler.resolve_config_dir() == (system_dir, "system")
    assert reconciler.config_dir() == system_dir


def test_an_explicit_env_wins_over_the_system_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _dirs(monkeypatch, tmp_path, system=True, home=True)
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("INFERFERENCE_CONFIG_DIR", str(explicit))
    assert reconciler.resolve_config_dir() == (explicit, "env")


def test_the_home_fallback_is_never_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _system, home_dir = _dirs(monkeypatch, tmp_path, system=False, home=True)
    assert reconciler.resolve_config_dir() == (home_dir, "home-fallback")
    reconciler.announce_config_dir()
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert str(home_dir) in err


def test_a_stale_home_copy_is_not_read_when_the_system_dir_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    system_dir, home_dir = _dirs(monkeypatch, tmp_path, system=True, home=True)
    seen: list[Path] = []

    def fake_read(path: Path) -> tuple[str, dict]:
        seen.append(path)
        return "placement", {}

    monkeypatch.setattr(reconciler, "read_config_dir", fake_read)
    assert reconciler.load_config() == ("placement", {})
    assert seen == [system_dir]
    assert home_dir not in seen
    err = capsys.readouterr().err
    assert str(system_dir) in err
    assert "WARNING" not in err


def test_the_preset_path_is_the_system_router_path_with_the_system_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    system_dir, home_dir = _dirs(monkeypatch, tmp_path, system=True, home=True)
    runtime = tmp_path / "run-inferference"
    monkeypatch.setattr(reconciler, "SYSTEM_PRESET_INI", runtime / "models-preset.ini")
    # No runtime dir: the file sits beside the config.
    assert reconciler.ini_path() == system_dir / "models-preset.ini"
    runtime.mkdir()
    assert reconciler.ini_path() == runtime / "models-preset.ini"
    # An explicit config dir keeps its own preset file beside it.
    monkeypatch.setenv("INFERFERENCE_CONFIG_DIR", str(home_dir))
    assert reconciler.ini_path() == home_dir / "models-preset.ini"
    override = tmp_path / "run" / "preset.ini"
    monkeypatch.setenv("INFERFERENCE_PRESET_INI", str(override))
    assert reconciler.ini_path() == override


# ------------------------------------------------------------ preset comparison --
def _read_only_router_preset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str | None) -> Path:
    router_ini = tmp_path / "run" / "models-preset.ini"
    router_ini.parent.mkdir()
    if text is not None:
        router_ini.write_text(text)
    monkeypatch.setenv("INFERFERENCE_PRESET_INI", str(router_ini))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    # A test run as root can write anything, so state the permission outright.
    monkeypatch.setattr(reconciler, "_can_write", lambda _path: False)
    return router_ini


def _stub_apply(monkeypatch: pytest.MonkeyPatch, requests: list) -> None:
    placement = SimpleNamespace(resident=[SimpleNamespace(model="a")])
    plan = SimpleNamespace(needs_router_restart=False, to_unload=["old"], to_load=["a"])
    monkeypatch.setattr(reconciler, "load_config", lambda: (placement, {}))
    monkeypatch.setattr(reconciler, "router_models", lambda _router: [])
    monkeypatch.setattr(reconciler, "plan_reconcile", lambda *_a, **_k: plan)
    monkeypatch.setattr(reconciler, "crashed_children", lambda _m: [])
    monkeypatch.setattr(reconciler, "device_snapshot", lambda _p: {})
    monkeypatch.setattr(reconciler, "render_preset_ini", lambda *_a: WANTED)
    monkeypatch.setattr(reconciler, "models_with_matching_presets", lambda *_a: {"a"})
    monkeypatch.setattr(reconciler, "_wait_loaded", lambda *_a, **_k: None)
    monkeypatch.setattr(reconciler, "_verify", lambda *_a: 0)
    monkeypatch.setattr(reconciler, "_residency_evidence", lambda *_a: {})
    monkeypatch.setattr(reconciler, "_req", lambda method, url, body=None, **_k: requests.append((method, url)))


def test_apply_refuses_when_the_read_only_router_preset_differs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router_ini = _read_only_router_preset(monkeypatch, tmp_path, STALE)
    requests: list = []
    _stub_apply(monkeypatch, requests)
    with pytest.raises(SystemExit) as exc:
        reconciler.cmd_apply(SimpleNamespace(router="http://router", timeout=1))
    assert exc.value.code == reconciler.EX_CONFIG
    assert requests == []  # nothing unloaded, nothing reloaded
    assert router_ini.read_text() == STALE
    assert (tmp_path / "state" / "inferference" / "models-preset.ini").read_text() == WANTED


def test_apply_refuses_when_the_read_only_router_preset_is_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _read_only_router_preset(monkeypatch, tmp_path, None)
    requests: list = []
    _stub_apply(monkeypatch, requests)
    with pytest.raises(SystemExit) as exc:
        reconciler.cmd_apply(SimpleNamespace(router="http://router", timeout=1))
    assert exc.value.code == reconciler.EX_CONFIG
    assert requests == []


def test_apply_goes_on_when_the_read_only_router_preset_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Same content, different comment, order and spacing: not a difference.
    router_ini = _read_only_router_preset(monkeypatch, tmp_path, "# note\n[a]\nparallel=8\nmodel=/m/a.gguf\n")
    before = router_ini.read_text()
    requests: list = []
    _stub_apply(monkeypatch, requests)
    assert reconciler.cmd_apply(SimpleNamespace(router="http://router", timeout=1)) == 0
    assert router_ini.read_text() == before  # not rewritten
    assert ("GET", "http://router/models?reload=1") in requests


def test_apply_writes_the_router_preset_when_it_is_writable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    router_ini = tmp_path / "preset.ini"
    router_ini.write_text(STALE)
    monkeypatch.setenv("INFERFERENCE_PRESET_INI", str(router_ini))
    requests: list = []
    _stub_apply(monkeypatch, requests)
    assert reconciler.cmd_apply(SimpleNamespace(router="http://router", timeout=1)) == 0
    assert router_ini.read_text() == WANTED


def test_plan_reports_a_differing_read_only_router_preset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _read_only_router_preset(monkeypatch, tmp_path, STALE)
    plan = SimpleNamespace(
        budgets=[],
        models_max=1,
        needs_router_restart=False,
        to_unload=[],
        to_load=[],
        changed=[],
        unchanged=[],
    )
    monkeypatch.setattr(reconciler, "load_config", lambda: (SimpleNamespace(), {}))
    monkeypatch.setattr(reconciler, "plan_reconcile", lambda *_a, **_k: plan)
    monkeypatch.setattr(reconciler, "render_preset_ini", lambda *_a: WANTED)
    assert reconciler.cmd_plan(SimpleNamespace(router="http://router", router_reachable=False)) == 0
    captured = capsys.readouterr()
    assert "DIFFERS" in captured.out
    assert "apply would REFUSE" in captured.err


# ------------------------------------------------ the one default preset path --
def test_the_default_preset_path_is_never_inside_etc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INFERFERENCE_CONFIG_DIR", "/etc/inferference")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert reconciler.resolve_ini_path() == (tmp_path / "state" / "inferference" / "models-preset.ini", "state")


def test_the_default_preset_path_leaves_a_read_only_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("INFERFERENCE_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert reconciler.resolve_ini_path() == (cfg / "models-preset.ini", "beside-config")
    monkeypatch.setattr(reconciler.os, "access", lambda _path, _mode: False)
    assert reconciler.resolve_ini_path()[1] == "state"
    # A config dir that does not exist yet is not a place to write.
    monkeypatch.setattr(reconciler.os, "access", lambda _path, _mode: True)
    monkeypatch.setenv("INFERFERENCE_CONFIG_DIR", str(tmp_path / "absent"))
    assert reconciler.resolve_ini_path()[1] == "state"


def test_preset_ini_command_prints_the_path_on_stdout_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INFERFERENCE_CONFIG_DIR", "/etc/inferference")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert reconciler.main(["preset-ini"]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"{tmp_path / 'state' / 'inferference' / 'models-preset.ini'}\n"
    assert "goes to the caller's state dir" in captured.err


def test_the_cli_and_the_library_share_one_config_dir_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from inferference import config_paths

    system_dir, home_dir = _dirs(monkeypatch, tmp_path, system=True, home=True)
    monkeypatch.setattr(config_paths, "SYSTEM_CONFIG_DIR", system_dir)
    assert reconciler.config_dir() == system_dir
    # The library function, with the same dirs, returns the same answer as the CLI.
    assert config_paths.resolve_config_dir(system_dir, str(home_dir)) == reconciler.resolve_config_dir()
