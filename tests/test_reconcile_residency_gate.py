import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci/runner/inferference-reconcile.py"
SPEC = importlib.util.spec_from_file_location("inferference_reconcile_cli", SCRIPT)
assert SPEC and SPEC.loader
reconciler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reconciler)


def test_snapshot_refuses_unreadable_counters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reconciler, "Path", lambda _path: tmp_path / "missing")
    placement = SimpleNamespace(
        devices=[SimpleNamespace(id="Vulkan0", card=0)],
        resident=[SimpleNamespace(model="example", devices=["Vulkan0"])],
    )
    with pytest.raises(SystemExit) as exc:
        reconciler.device_snapshot(placement)
    assert exc.value.code == reconciler.EX_CONFIG


@pytest.mark.parametrize(("gtt_after", "expected"), [(0.0, 0), (1.0, 1), (float("nan"), 1)])
def test_residency_requires_valid_unchanged_gtt(
    gtt_after: float, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    placement = SimpleNamespace(
        devices=[SimpleNamespace(id="Vulkan0", card=0)],
        resident=[SimpleNamespace(model="example", devices=["Vulkan0"])],
    )
    plan = SimpleNamespace(budgets=[SimpleNamespace(device="Vulkan0", requested_mib=100.0)])
    monkeypatch.setattr(reconciler.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(reconciler, "device_snapshot", lambda _placement: {"Vulkan0": (100.0, gtt_after)})
    monkeypatch.setattr(reconciler, "_pci_address", lambda _card: "0000:03:00.0")
    models = [
        {
            "id": "example",
            "status": {"value": "loaded"},
            "inferference_residency": {
                "layer_count": 1,
                "verified_mono_us": 1,
                "output_device": "Vulkan0",
                "cards": [
                    {
                        "device": "Vulkan0",
                        "pci": "0000:03:00.0",
                        "layers": 1,
                        "gtt_before": 0,
                        "gtt_after": 0,
                        "vram_before": 0,
                        "vram_after": 1,
                    }
                ],
            },
        }
    ]
    assert reconciler._verify(placement, {"Vulkan0": (0.0, 0.0)}, plan, models) == expected


def test_apply_unloads_unverified_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    placement = SimpleNamespace(resident=[SimpleNamespace(model="new"), SimpleNamespace(model="kept")])
    plan = SimpleNamespace(needs_router_restart=False, to_unload=[], to_load=["new", "kept"])
    monkeypatch.setattr(reconciler, "load_config", lambda: (placement, {}))
    monkeypatch.setattr(reconciler, "router_models", lambda _router: [])
    monkeypatch.setattr(reconciler, "plan_reconcile", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(reconciler, "crashed_children", lambda _models: [])
    monkeypatch.setattr(reconciler, "device_snapshot", lambda _placement: {})
    monkeypatch.setattr(reconciler, "ini_path", lambda: tmp_path / "presets.ini")
    monkeypatch.setattr(reconciler, "render_preset_ini", lambda *_args: "")
    monkeypatch.setattr(reconciler, "models_with_matching_presets", lambda *_args: {"kept"})
    monkeypatch.setattr(reconciler, "_wait_loaded", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(reconciler, "_verify", lambda *_args: 1)
    requests = []
    monkeypatch.setattr(reconciler, "_req", lambda method, url, body=None, **_kwargs: requests.append((url, body)))

    assert reconciler.cmd_apply(SimpleNamespace(router="http://router", timeout=1)) == 1
    assert ("http://router/models/unload", {"model": "new"}) in requests
    assert ("http://router/models/unload", {"model": "kept"}) in requests


def test_apply_refuses_changed_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    placement = SimpleNamespace(resident=[SimpleNamespace(model="example")])
    plan = SimpleNamespace(needs_router_restart=False, to_unload=[], to_load=["example"])
    calls = iter(
        [
            [],
            [],
            [{"id": "example", "inferference_residency": {"stamp": 1}}],
            [{"id": "example", "inferference_residency": {"stamp": 2}}],
        ]
    )
    monkeypatch.setattr(reconciler, "load_config", lambda: (placement, {}))
    monkeypatch.setattr(reconciler, "router_models", lambda _router: next(calls))
    monkeypatch.setattr(reconciler, "plan_reconcile", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(reconciler, "crashed_children", lambda _models: [])
    monkeypatch.setattr(reconciler, "device_snapshot", lambda _placement: {})
    monkeypatch.setattr(reconciler, "ini_path", lambda: tmp_path / "presets.ini")
    monkeypatch.setattr(reconciler, "render_preset_ini", lambda *_args: "")
    monkeypatch.setattr(reconciler, "models_with_matching_presets", lambda *_args: {"example"})
    monkeypatch.setattr(reconciler, "_verify", lambda *_args: 0)
    requests = []
    monkeypatch.setattr(reconciler, "_req", lambda method, url, body=None, **_kwargs: requests.append((url, body)))
    assert reconciler.cmd_apply(SimpleNamespace(router="http://router", timeout=1)) == 1
    assert ("http://router/models/unload", {"model": "example"}) in requests


@pytest.mark.parametrize("change", ["missing", "pci", "gtt", "vram", "layers", "device", "duplicate"])
def test_invalid_server_evidence_is_refused(change: str, monkeypatch: pytest.MonkeyPatch) -> None:
    placement = SimpleNamespace(
        devices=[SimpleNamespace(id="Vulkan0", card=0)],
        resident=[SimpleNamespace(model="example", devices=["Vulkan0"])],
    )
    monkeypatch.setattr(reconciler, "_pci_address", lambda _card: "0000:03:00.0")
    card = {
        "device": "Vulkan0",
        "pci": "0000:03:00.0",
        "layers": 24,
        "gtt_before": 2,
        "gtt_after": 2,
        "vram_before": 0,
        "vram_after": 1,
    }
    evidence = {"layer_count": 24, "output_device": "Vulkan0", "verified_mono_us": 1, "cards": [card]}
    model = {"id": "example", "status": {"value": "loaded"}, "inferference_residency": evidence}
    if change == "missing":
        del model["inferference_residency"]
    elif change == "pci":
        card["pci"] = "0000:04:00.0"
    elif change == "gtt":
        card["gtt_after"] = 3
    elif change == "vram":
        card["vram_after"] = 0
    elif change == "layers":
        evidence["layer_count"] = 25
    elif change == "device":
        card["device"] = "Vulkan1"
    else:
        evidence["cards"].append(card)
    with pytest.raises(ValueError):
        reconciler._residency_evidence(placement, [model])


def test_split_model_requires_evidence_for_each_card(monkeypatch: pytest.MonkeyPatch) -> None:
    placement = SimpleNamespace(
        devices=[SimpleNamespace(id="Vulkan0", card=0), SimpleNamespace(id="Vulkan1", card=1)],
        resident=[SimpleNamespace(model="example", devices=["Vulkan0", "Vulkan1"])],
    )
    monkeypatch.setattr(reconciler, "_pci_address", lambda card: f"0000:0{card + 3}:00.0")
    cards = [
        {
            "device": f"Vulkan{i}",
            "pci": f"0000:0{i + 3}:00.0",
            "layers": 12,
            "gtt_before": 10,
            "gtt_after": 10,
            "vram_before": 0,
            "vram_after": 1,
        }
        for i in range(2)
    ]
    evidence = {"layer_count": 24, "output_device": "Vulkan0", "verified_mono_us": 1, "cards": cards}
    models = [{"id": "example", "status": {"value": "loaded"}, "inferference_residency": evidence}]
    assert reconciler._residency_evidence(placement, models) == {"Vulkan0": 10, "Vulkan1": 10}
    evidence["cards"].pop()
    with pytest.raises(ValueError):
        reconciler._residency_evidence(placement, models)


def test_gtt_limit_comes_from_latest_model(monkeypatch: pytest.MonkeyPatch) -> None:
    placement = SimpleNamespace(
        devices=[SimpleNamespace(id="Vulkan0", card=0)],
        resident=[SimpleNamespace(model="one", devices=["Vulkan0"]), SimpleNamespace(model="two", devices=["Vulkan0"])],
    )
    monkeypatch.setattr(reconciler, "_pci_address", lambda _card: "0000:03:00.0")
    models = []
    for name, timestamp, gtt in (("one", 1, 200), ("two", 2, 100)):
        card = {
            "device": "Vulkan0",
            "pci": "0000:03:00.0",
            "layers": 1,
            "gtt_before": gtt,
            "gtt_after": gtt,
            "vram_before": 0,
            "vram_after": 1,
        }
        models.append(
            {
                "id": name,
                "status": {"value": "loaded"},
                "inferference_residency": {
                    "layer_count": 1,
                    "output_device": "Vulkan0",
                    "verified_mono_us": timestamp,
                    "cards": [card],
                },
            }
        )
    assert reconciler._residency_evidence(placement, models) == {"Vulkan0": 100}
