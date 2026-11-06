"""The reconciler's decisions, on a CPU with no router and no GPU.

These cover the parts that cannot be checked by looking at a live run: what the
reconciler decides to do BEFORE it touches anything, and how it reads a router
that is lying to it by omission.

The two failures being guarded against here both look like success at the time:

* a model whose preset changed remains in the reconcile plan until apply has
  confirmed that the router is running the declared preset;
* a router launched with a smaller ``--models-max`` than the placement declares
  accepts every load and silently evicts the least recently used model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inferference.placement import (
    Footprint,
    ModelSpec,
    Placement,
    parse_preset_sections,
    plan_reconcile,
    render_preset_ini,
)
from inferference.reconcile import (
    ConfigError,
    child_health,
    crashed_children,
    loaded_names,
    models_with_matching_presets,
    read_config_dir,
    router_models_max,
)

FP = Footprint(ctx_per_lane=4096, base_mib=2317.7, per_lane_mib=704.3)
BIG = Footprint(ctx_per_lane=4096, base_mib=8322.0, per_lane_mib=544.0)

# A whole card, so a budget never accidentally decides these tests.
ROOMY = lambda card: (16368.0, 16368.0)  # noqa: E731


def spec(name: str, *, lanes: int = 1, fp: Footprint = FP, ctx: int = 4096, **kw) -> ModelSpec:
    return ModelSpec(name=name, model=Path(f"/models/{name}.gguf"), lanes=lanes, ctx_per_lane=ctx, footprint=fp, **kw)


def placement(resident: list[dict], models_max: int = 4) -> Placement:
    return Placement.model_validate(
        {
            "devices": [{"id": "Vulkan0", "card": 0}, {"id": "Vulkan1", "card": 1}],
            "resident": resident,
            "router": {"models_max": models_max},
        }
    )


def child(name: str, status: str, exit_code: int | None = None, args: list[str] | None = None) -> dict:
    st: dict = {"value": status}
    if exit_code is not None:
        st["exit_code"] = exit_code
    if args is not None:
        st["args"] = args
    return {"id": name, "status": st}


# ------------------------------------------------------- changed vs unchanged --
def test_a_changed_preset_is_reloaded_not_left_down() -> None:
    """A changed model stays in the plan so apply can verify or load the preset."""
    p = placement([{"model": "a", "device": "Vulkan0"}])
    old_ini = render_preset_ini(p, {"a": spec("a", lanes=1)})

    # The operator raises the lane count. Same model, different preset.
    specs = {"a": spec("a", lanes=2)}
    plan = plan_reconcile(p, specs, loaded_now=["a"], router_models_max=4, free_mib=ROOMY, previous_ini=old_ini)

    assert plan.changed == ["a"]
    assert "a" in plan.to_load, "a changed model must be checked after preset reload"
    assert plan.unchanged == []


def test_an_unchanged_preset_is_left_running() -> None:
    p = placement([{"model": "a", "device": "Vulkan0"}])
    specs = {"a": spec("a")}
    ini = render_preset_ini(p, specs)
    plan = plan_reconcile(p, specs, loaded_now=["a"], router_models_max=4, free_mib=ROOMY, previous_ini=ini)
    assert plan.changed == []
    assert plan.to_load == []
    assert plan.unchanged == ["a"]
    assert plan.is_noop


def test_no_previous_ini_treats_everything_as_changed() -> None:
    """Unknown must fail toward reloading. Over-reloading costs time; under-
    reloading loses a model that reload=1 already took down."""
    p = placement([{"model": "a", "device": "Vulkan0"}])
    plan = plan_reconcile(p, {"a": spec("a")}, loaded_now=["a"], router_models_max=4, free_mib=ROOMY, previous_ini=None)
    assert plan.to_load == ["a"]


def test_comments_and_ordering_are_not_a_change() -> None:
    p = placement([{"model": "a", "device": "Vulkan0"}])
    specs = {"a": spec("a")}
    ini = render_preset_ini(p, specs)
    noisy = "; a new banner\n\n" + "\n".join(reversed(ini.splitlines()))
    # Reversing puts the section header last, so rebuild it honestly instead:
    body = [ln for ln in ini.splitlines() if ln.strip() and not ln.startswith(";")]
    header, rest = body[0], body[1:]
    noisy = "; a new banner\n" + header + "\n" + "\n".join(reversed(rest)) + "\n"
    assert parse_preset_sections(noisy) == parse_preset_sections(ini)


def test_a_removed_model_is_unloaded() -> None:
    p = placement([{"model": "a", "device": "Vulkan0"}])
    specs = {"a": spec("a")}
    plan = plan_reconcile(
        p,
        specs,
        loaded_now=["a", "gone"],
        router_models_max=4,
        free_mib=ROOMY,
        previous_ini=render_preset_ini(p, specs),
    )
    assert plan.to_unload == ["gone"]


# --------------------------------------------------------------- --models-max --
def test_declared_models_max_below_resident_count_demands_a_restart() -> None:
    p = placement(
        [{"model": "a", "device": "Vulkan0"}, {"model": "b", "device": "Vulkan1"}],
        models_max=1,
    )
    plan = plan_reconcile(
        p,
        {"a": spec("a"), "b": spec("b")},
        loaded_now=[],
        router_models_max=None,
        free_mib=ROOMY,
    )
    assert plan.needs_router_restart
    assert "silently evict" in (plan.restart_reason or "").lower()


def test_an_empty_router_still_catches_a_too_small_models_max() -> None:
    """The fail-open case. With no child running, --models-max cannot be scraped
    off any argv, so the declared value is what protects the resident set."""
    p = placement(
        [{"model": "a", "device": "Vulkan0"}, {"model": "b", "device": "Vulkan1"}],
        models_max=1,
    )
    plan = plan_reconcile(p, {"a": spec("a"), "b": spec("b")}, loaded_now=[], router_models_max=None, free_mib=ROOMY)
    assert plan.needs_router_restart


def test_a_running_router_that_disagrees_with_the_config_is_drift() -> None:
    p = placement([{"model": "a", "device": "Vulkan0"}], models_max=4)
    plan = plan_reconcile(p, {"a": spec("a")}, loaded_now=["a"], router_models_max=2, free_mib=ROOMY, previous_ini=None)
    assert plan.needs_router_restart
    assert "DRIFT" in (plan.restart_reason or "")


def test_models_max_zero_means_unlimited() -> None:
    p = placement([{"model": f"m{i}", "device": "Vulkan0"} for i in range(6)], models_max=0)
    specs = {f"m{i}": spec(f"m{i}", fp=Footprint(ctx_per_lane=4096, base_mib=100, per_lane_mib=10)) for i in range(6)}
    plan = plan_reconcile(p, specs, loaded_now=[], router_models_max=0, free_mib=ROOMY)
    assert not plan.needs_router_restart


# ------------------------------------------------------------- router reading --
def test_a_crashed_child_is_not_a_deliberate_unload() -> None:
    models = [
        child("deliberate", "unloaded"),
        child("died", "unloaded", exit_code=1),
        child("died_clean", "unloaded", exit_code=0),
        child("fine", "loaded"),
    ]
    crashed = {c.name for c in crashed_children(models)}
    assert crashed == {"died", "died_clean"}, "exit_code 0 is still an exit a resident model must not make"
    assert loaded_names(models) == ["fine"]


def test_child_health_survives_a_router_that_omits_status() -> None:
    assert child_health([{"id": "x"}])[0].status == "?"


def test_router_autoload_after_reload_is_reused_only_for_the_current_preset() -> None:
    p = placement([{"model": "a", "device": "Vulkan0"}])
    wanted = render_preset_ini(p, {"a": spec("a")})
    keys = ("model =", "device =", "ctx-size =", "parallel =", "n-gpu-layers =", "load-on-startup =")
    expected_section = "[a]\n" + "\n".join(line for line in wanted.splitlines() if line.startswith(keys))
    old_section = expected_section.replace("ctx-size = 4096", "ctx-size = 8192")
    models = [
        {"id": "a", "status": {"value": "loaded", "preset": expected_section}},
        {"id": "b", "status": {"value": "loaded", "preset": expected_section.replace("[a]", "[b]")}},
        {"id": "c", "status": {"value": "loaded", "preset": old_section}},
        {"id": "d", "status": {"value": "loading"}},
    ]

    assert models_with_matching_presets(models, ["a", "b", "c", "d"], wanted) == {"a"}


def test_models_max_is_scraped_off_a_child_argv() -> None:
    assert router_models_max([child("a", "loaded", args=["llama-server", "--models-max", "7"])]) == 7


def test_models_max_is_unknown_not_zero_when_no_child_runs() -> None:
    assert router_models_max([]) is None
    assert router_models_max([child("a", "unloaded")]) is None


def test_a_malformed_models_max_is_unknown_not_a_crash() -> None:
    assert router_models_max([child("a", "loaded", args=["--models-max", "banana"])]) is None
    assert router_models_max([child("a", "loaded", args=["--models-max"])]) is None


# --------------------------------------------------------------- config on disk --
def _write_config(root: Path, *, models: dict[str, dict], placement_doc: dict) -> None:
    (root / "models").mkdir(parents=True, exist_ok=True)
    for name, doc in models.items():
        (root / "models" / f"{name}.yaml").write_text(yaml.safe_dump(doc))
    (root / "placement.yaml").write_text(yaml.safe_dump(placement_doc))


def _model_doc(path: Path, **kw) -> dict:
    doc = {
        "model": str(path),
        "ctx_per_lane": 4096,
        "lanes": 1,
        "footprint": {"ctx_per_lane": 4096, "base_mib": 100.0, "per_lane_mib": 10.0},
    }
    doc.update(kw)
    return doc


def test_config_loads(tmp_path: Path) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    _write_config(
        tmp_path / "cfg",
        models={"a": _model_doc(gguf)},
        placement_doc={"devices": [{"id": "Vulkan0", "card": 0}], "resident": [{"model": "a", "device": "Vulkan0"}]},
    )
    p, specs = read_config_dir(tmp_path / "cfg")
    assert list(specs) == ["a"]
    assert p.router.models_max == 4


def test_a_missing_gguf_is_caught_before_the_budget_says_yes(tmp_path: Path) -> None:
    _write_config(
        tmp_path / "cfg",
        models={"a": _model_doc(tmp_path / "absent.gguf")},
        placement_doc={"devices": [{"id": "Vulkan0", "card": 0}], "resident": [{"model": "a", "device": "Vulkan0"}]},
    )
    with pytest.raises(ConfigError, match="model file not found"):
        read_config_dir(tmp_path / "cfg")


def test_a_placement_naming_an_undefined_model_names_the_missing_file(tmp_path: Path) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    _write_config(
        tmp_path / "cfg",
        models={"a": _model_doc(gguf)},
        placement_doc={
            "devices": [{"id": "Vulkan0", "card": 0}],
            "resident": [{"model": "ghost", "device": "Vulkan0"}],
        },
    )
    with pytest.raises(ConfigError, match="ghost.yaml"):
        read_config_dir(tmp_path / "cfg")


def test_a_model_file_whose_name_disagrees_with_its_filename_is_refused(tmp_path: Path) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    _write_config(
        tmp_path / "cfg",
        models={"a": _model_doc(gguf, name="something-else")},
        placement_doc={"devices": [{"id": "Vulkan0", "card": 0}], "resident": []},
    )
    with pytest.raises(ConfigError, match="two names for one model"):
        read_config_dir(tmp_path / "cfg")


def test_missing_placement_is_loud(tmp_path: Path) -> None:
    (tmp_path / "cfg" / "models").mkdir(parents=True)
    with pytest.raises(ConfigError, match="no placement"):
        read_config_dir(tmp_path / "cfg")


def test_an_empty_models_dir_is_loud(tmp_path: Path) -> None:
    _write_config(tmp_path / "cfg", models={}, placement_doc={"devices": [{"id": "Vulkan0", "card": 0}]})
    with pytest.raises(ConfigError, match="defines no models"):
        read_config_dir(tmp_path / "cfg")


def test_invalid_yaml_is_loud(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    (root / "models").mkdir(parents=True)
    (root / "placement.yaml").write_text("devices: [oh: no: yes]")
    with pytest.raises(ConfigError, match="not valid YAML"):
        read_config_dir(root)
