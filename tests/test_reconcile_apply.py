"""What `apply` unloads, when it does nothing, and which deadline it keeps.

No router and no GPU. The fault guarded here: a failed reload unloaded every
resident model, even on a reload that changed nothing. The rules now are:

1. A plan with no change and a matching preset touches nothing: no write, no
   reload, no verify, no unload.
2. A failed apply unloads the models that this run changed, and no other.
3. Both load waits of one apply share one deadline.
4. `_can_write` needs the dir to be writable, because the write makes a temp file.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci/runner/inferference-reconcile.py"
SPEC = importlib.util.spec_from_file_location("inferference_reconcile_cli_apply", SCRIPT)
assert SPEC and SPEC.loader
reconciler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reconciler)

WANTED = "[a]\nmodel = /m/a.gguf\n\n[b]\nmodel = /m/b.gguf\n\n[c]\nmodel = /m/c.gguf\n"
STALE = "[a]\nmodel = /m/a.gguf\n\n[b]\nmodel = /m/OLD.gguf\n\n[c]\nmodel = /m/c.gguf\n"


class Scenario:
    """A stubbed router with three declared models: a, b, c."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        to_load: list[str],
        to_unload: list[str] | None = None,
        preset_on_disk: str = WANTED,
        verify: int = 0,
    ) -> None:
        self.requests: list[tuple[str, str, Any]] = []
        self.verify_calls = 0
        self.waits: list[Any] = []
        self.ini = tmp_path / "models-preset.ini"
        self.ini.write_text(preset_on_disk)
        monkeypatch.setenv("INFERFERENCE_PRESET_INI", str(self.ini))
        placement = SimpleNamespace(resident=[SimpleNamespace(model=n) for n in "abc"])
        plan = SimpleNamespace(needs_router_restart=False, to_unload=to_unload or [], to_load=to_load)
        monkeypatch.setattr(reconciler, "load_config", lambda: (placement, {}))
        monkeypatch.setattr(reconciler, "router_models", lambda _router: [])
        monkeypatch.setattr(reconciler, "plan_reconcile", lambda *_a, **_k: plan)
        monkeypatch.setattr(reconciler, "crashed_children", lambda _m: [])
        monkeypatch.setattr(reconciler, "device_snapshot", lambda _p: {})
        monkeypatch.setattr(reconciler, "render_preset_ini", lambda *_a: WANTED)
        monkeypatch.setattr(reconciler, "models_with_matching_presets", lambda _m, names, _i: set(names))
        monkeypatch.setattr(reconciler, "_wait_loaded", lambda *_a, **k: self.waits.append(k.get("deadline")))
        monkeypatch.setattr(reconciler, "_verify", self._verify(verify))
        monkeypatch.setattr(reconciler, "_residency_evidence", lambda *_a: {})
        monkeypatch.setattr(reconciler, "_req", self._req)

    def _verify(self, result: int) -> Any:
        def verify(*_args: object) -> int:
            self.verify_calls += 1
            return result

        return verify

    def _req(self, method: str, url: str, body: Any = None, **_kw: Any) -> Any:
        self.requests.append((method, url.removeprefix("http://router"), body))
        return {"success": True}

    def apply(self) -> int:
        return reconciler.cmd_apply(SimpleNamespace(router="http://router", timeout=5))

    @property
    def unloaded(self) -> list[str]:
        return [b["model"] for m, u, b in self.requests if u == "/models/unload"]


def test_a_reload_with_no_change_touches_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = Scenario(monkeypatch, tmp_path, to_load=[], verify=1)  # a verify would fail
    assert run.apply() == 0
    assert run.requests == []  # no ?reload=1, no unload, no load
    assert run.verify_calls == 0
    assert run.ini.read_text() == WANTED
    assert "nothing to change" in capsys.readouterr().out


def test_a_reload_with_no_change_but_a_stale_preset_file_runs_the_full_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = Scenario(monkeypatch, tmp_path, to_load=[], preset_on_disk=STALE, verify=1)
    assert run.apply() == 1
    assert run.ini.read_text() == WANTED  # the file was rewritten
    assert ("GET", "/models?reload=1", None) in run.requests
    assert run.verify_calls == 1
    assert run.unloaded == []  # this run changed no model, so it unloads none


def test_a_failure_after_a_partial_change_unloads_only_the_changed_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = Scenario(monkeypatch, tmp_path, to_load=["b"], preset_on_disk=STALE, verify=1)
    assert run.apply() == 1
    assert run.unloaded == ["b"]  # a and c stay resident
    err = capsys.readouterr().err
    assert "the models that this run changed: ['b']" in err
    assert "placement was not proven" in err


def test_a_load_failure_unloads_only_the_changed_models_and_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = Scenario(monkeypatch, tmp_path, to_load=["a", "c"], preset_on_disk=STALE)

    def boom(*_a: object, **_k: object) -> None:
        raise SystemExit(reconciler.EX_CONFIG)

    monkeypatch.setattr(reconciler, "_verify", boom)
    with pytest.raises(SystemExit):
        run.apply()
    assert sorted(run.unloaded) == ["a", "c"]  # b stays resident
    assert "failed before the placement was proven" in capsys.readouterr().err


def test_undeclared_models_leave_before_the_write_and_are_not_a_failure_unload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = Scenario(monkeypatch, tmp_path, to_load=[], to_unload=["old"], verify=1)
    assert run.apply() == 1
    assert run.unloaded == ["old"]  # once, at the start; no resident model is unloaded after it


def test_the_unload_after_a_failure_survives_an_unload_that_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = Scenario(monkeypatch, tmp_path, to_load=["a", "b"], preset_on_disk=STALE, verify=1)

    def req(method: str, url: str, body: Any = None, **_kw: Any) -> Any:
        run.requests.append((method, url.removeprefix("http://router"), body))
        if body == {"model": "a"} and url.endswith("/models/unload"):
            raise SystemExit(reconciler.EX_CONFIG)
        return {"success": True}

    monkeypatch.setattr(reconciler, "_req", req)
    assert run.apply() == 1
    assert run.unloaded == ["a", "b"]  # b is tried after a fails
    assert "could not unload a" in capsys.readouterr().err


def test_both_waits_of_one_apply_share_one_deadline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = Scenario(monkeypatch, tmp_path, to_load=["a", "b"], preset_on_disk=STALE)
    state = {"calls": 0}

    def models(_router: str) -> list[dict[str, Any]]:
        state["calls"] += 1
        # The first read after the reload shows `a` loading, so the first wait runs.
        loading = state["calls"] == 2
        return [{"id": "a", "status": {"value": "loading" if loading else "loaded"}}]

    monkeypatch.setattr(reconciler, "router_models", models)
    monkeypatch.setattr(reconciler, "models_with_matching_presets", lambda *_a: set())  # b needs an explicit load
    assert run.apply() == 0
    assert len(run.waits) == 2
    assert run.waits[0] is run.waits[1]
    assert isinstance(run.waits[0], reconciler.Deadline)
    assert run.waits[0].seconds == 5


# -------------------------------------------------------------- _can_write ----
def test_a_writable_file_in_a_read_only_dir_is_not_writable(tmp_path: Path) -> None:
    sub = tmp_path / "ro"
    sub.mkdir()
    target = sub / "models-preset.ini"
    target.write_text("x")
    target.chmod(0o644)
    sub.chmod(0o555)
    try:
        if os.access(sub, os.W_OK):
            pytest.skip("the caller can write a 0555 dir (root)")
        assert os.access(target, os.W_OK)  # the file itself is writable
        assert reconciler._can_write(target) is False
    finally:
        sub.chmod(0o755)
    assert reconciler._can_write(target) is True


def test_can_write_asks_the_dir_even_when_the_file_is_writable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The same rule, with a fake os.access, so that it holds for root too."""
    target = tmp_path / "models-preset.ini"
    target.write_text("x")
    monkeypatch.setattr(reconciler.os, "access", lambda path, _mode: Path(path) == target)
    assert reconciler._can_write(target) is False  # file yes, dir no
    monkeypatch.setattr(reconciler.os, "access", lambda path, _mode: Path(path) == tmp_path)
    assert reconciler._can_write(target) is False  # dir yes, file no
    monkeypatch.setattr(reconciler.os, "access", lambda _path, _mode: True)
    assert reconciler._can_write(target) is True
    assert reconciler._can_write(tmp_path / "new" / "deep" / "x.ini") is True  # checks the nearest dir that exists


def test_apply_refuses_before_any_unload_when_the_dir_is_not_writable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = Scenario(monkeypatch, tmp_path, to_load=["b"], to_unload=["old"], preset_on_disk=STALE)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    # The file is writable and the dir is not: the temp file could not be made.
    monkeypatch.setattr(reconciler.os, "access", lambda path, _mode: Path(path) == run.ini)
    with pytest.raises(SystemExit) as exc:
        run.apply()
    assert exc.value.code == reconciler.EX_CONFIG
    assert run.requests == []  # nothing was unloaded, nothing reloaded
    assert run.ini.read_text() == STALE
