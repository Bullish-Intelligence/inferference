"""Contract tests for declarative placement and the VRAM budget.

The footprint cases below are REAL MEASUREMENTS from this rig
(the router-serving measurement record), not
invented numbers. They are here so the budget formula cannot drift away from the
hardware without a test going red.
"""

from __future__ import annotations

import pytest

from inferference.placement import (
    BudgetError,
    Device,
    Footprint,
    ModelSpec,
    NoSingleCardFitsError,
    Placement,
    PlacementError,
    Resident,
    RouterConfig,
    budget_devices,
    plan_reconcile,
    render_preset_ini,
    resolve_auto_devices,
)

# MEASURED on one MI25 at ctx_per_lane = 4096, from sysfs VRAM delta (the card's
# own accounting), NOT from summing the server log's buffer sizes. The log
# understates by an amount that grows per lane; budgeting from it declared a
# 16-lane Gemma placement that then spilled to host memory.
DENSE_3B = Footprint(ctx_per_lane=4096, base_mib=2317.7, per_lane_mib=704.3)
SWA_12B = Footprint(ctx_per_lane=4096, base_mib=8322.0, per_lane_mib=544.0)

MI25_TOTAL = 16368.0


def _card(_: int) -> tuple[float, float]:
    return MI25_TOTAL, MI25_TOTAL


@pytest.mark.parametrize(
    ("fp", "lanes", "expected"),
    [
        # 3B dense: sysfs deltas measured at 1 and 4 lanes.
        (DENSE_3B, 1, 3022),
        (DENSE_3B, 4, 5135),
        # Gemma-4-12B sliding window: sysfs deltas at 1, 2, 4 lanes...
        (SWA_12B, 1, 8866),
        (SWA_12B, 2, 9410),
        (SWA_12B, 4, 10498),
        # ...and an INDEPENDENT 8-lane observation from a separate sweep that was
        # not used to fit the line. This is the case that proves the model
        # extrapolates in lanes rather than merely interpolating.
        (SWA_12B, 8, 12682),
    ],
)
def test_footprint_matches_measured_hardware(fp: Footprint, lanes: int, expected: float) -> None:
    assert fp.total_mib(lanes) == pytest.approx(expected, abs=10.0)


def test_footprint_predicts_the_observed_16_lane_spill() -> None:
    """The regression that motivates measuring from sysfs at all.

    A 16-lane Gemma placement was observed to spill into host memory. Budgeting
    from the server log's buffer sums said it would fit; the card's own accounting
    says it needs ~17 GiB on a 16368 MiB card. The budget must land on the side
    that refuses.
    """
    assert SWA_12B.total_mib(16) > MI25_TOTAL


def test_both_architectures_cost_kv_per_lane() -> None:
    """The correction: neither architecture gets extra lanes for free.

    Held at a constant per-lane context, KV is linear in lanes for BOTH. Only the
    shared model weights are paid once.
    """
    for fp in (DENSE_3B, SWA_12B):
        one = fp.total_mib(1)
        four = fp.total_mib(4)
        assert four - one == pytest.approx(3 * fp.per_lane_mib, abs=0.01)


def test_swa_model_is_cheaper_per_lane_than_the_smaller_dense_one() -> None:
    """Counter-intuitive and measured: do not rank models by parameter count."""
    assert SWA_12B.per_lane_mib < DENSE_3B.per_lane_mib


def _spec(name: str, fp: Footprint | None, lanes: int = 1, ctx: int = 4096) -> ModelSpec:
    return ModelSpec(name=name, model=f"/models/{name}.gguf", ctx_per_lane=ctx, lanes=lanes, footprint=fp)


def test_missing_footprint_is_refused_not_guessed() -> None:
    with pytest.raises(PlacementError, match="no measured footprint"):
        _spec("x", None).require_footprint()


def test_footprint_measured_at_another_ctx_is_refused() -> None:
    """Extrapolation is unsafe: KV scaling is per-architecture."""
    spec = _spec("x", DENSE_3B, ctx=8192)
    with pytest.raises(PlacementError, match="re-measure|Re-measure"):
        spec.require_footprint()


def test_total_ctx_is_per_lane_times_lanes() -> None:
    """--ctx-size is the TOTAL cache; n_ctx_slot = ctx / parallel."""
    assert _spec("x", DENSE_3B, lanes=4).total_ctx == 16384


def test_budget_refuses_overcommit_and_lists_what_is_resident() -> None:
    placement = Placement(
        devices=[Device(id="Vulkan0", card=0, reserve_mib=512)],
        resident=[Resident(model="big", device="Vulkan0"), Resident(model="small", device="Vulkan0")],
    )
    specs = {"big": _spec("big", SWA_12B, lanes=8), "small": _spec("small", DENSE_3B, lanes=8)}
    with pytest.raises(BudgetError) as exc:
        plan_reconcile(placement, specs, loaded_now=[], router_models_max=8, free_mib=_card)
    msg = str(exc.value)
    assert "REFUSED" in msg
    # The owner has to be able to choose a victim, so both must be listed with sizes.
    assert "big" in msg and "small" in msg
    assert "OVER BY" in msg
    assert "nothing was evicted" in msg.lower()


def test_budget_accepts_a_placement_that_fits() -> None:
    placement = Placement(
        devices=[Device(id="Vulkan0", card=0), Device(id="Vulkan1", card=1)],
        resident=[Resident(model="big", device="Vulkan0"), Resident(model="small", device="Vulkan1")],
    )
    specs = {"big": _spec("big", SWA_12B, lanes=8), "small": _spec("small", DENSE_3B, lanes=8)}
    budgets = budget_devices(placement, specs, free_mib=_card)
    assert all(b.fits for b in budgets)
    assert budgets[0].requested_mib == pytest.approx(12674.0, abs=10.0)


def test_split_model_is_charged_to_both_devices_by_tensor_split() -> None:
    placement = Placement(
        devices=[Device(id="Vulkan0", card=0), Device(id="Vulkan1", card=1)],
        resident=[Resident(model="wide", device=["Vulkan0", "Vulkan1"], tensor_split=[75.0, 25.0])],
    )
    specs = {"wide": _spec("wide", SWA_12B)}
    b0, b1 = budget_devices(placement, specs, free_mib=_card)
    assert b0.requested_mib == pytest.approx(SWA_12B.total_mib(1) * 0.75, abs=0.01)
    assert b1.requested_mib == pytest.approx(SWA_12B.total_mib(1) * 0.25, abs=0.01)


def test_models_max_is_the_resident_count() -> None:
    """Below the resident count, a load silently evicts the LRU."""
    placement = Placement(
        devices=[Device(id="Vulkan0", card=0)],
        resident=[Resident(model="a", device="Vulkan0"), Resident(model="b", device="Vulkan0")],
    )
    assert placement.models_max == 2


def test_router_config_defaults_host_and_port() -> None:
    cfg = RouterConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8100
    assert cfg.url == "http://127.0.0.1:8100"


def test_router_config_url_reflects_declared_host_and_port() -> None:
    cfg = RouterConfig(host="0.0.0.0", port=9000)
    assert cfg.url == "http://0.0.0.0:9000"


def test_plan_flags_router_restart_when_models_max_is_too_low() -> None:
    placement = Placement(
        devices=[Device(id="Vulkan0", card=0)],
        resident=[Resident(model="a", device="Vulkan0"), Resident(model="b", device="Vulkan0")],
        router=RouterConfig(models_max=1),
    )
    specs = {"a": _spec("a", DENSE_3B), "b": _spec("b", DENSE_3B)}
    plan = plan_reconcile(placement, specs, loaded_now=[], router_models_max=1, free_mib=_card)
    assert plan.needs_router_restart
    assert "silently evict" in (plan.restart_reason or "").lower()


def test_plan_unloads_what_yaml_no_longer_declares() -> None:
    """YAML wins: anything resident but undeclared comes out, however it got there."""
    placement = Placement(
        devices=[Device(id="Vulkan0", card=0)],
        resident=[Resident(model="keep", device="Vulkan0")],
    )
    specs = {"keep": _spec("keep", DENSE_3B)}
    plan = plan_reconcile(
        placement,
        specs,
        loaded_now=["keep", "snuck-in-via-http"],
        router_models_max=4,
        free_mib=_card,
        previous_ini=render_preset_ini(placement, specs),
    )
    assert plan.to_unload == ["snuck-in-via-http"]
    assert plan.to_load == []
    assert plan.unchanged == ["keep"]


def test_without_the_running_ini_every_model_is_reloaded() -> None:
    """Fail-safe direction. Not knowing which presets changed must mean
    "reload it", because ?reload=1 has already taken a changed model down and
    will not put it back."""
    placement = Placement(
        devices=[Device(id="Vulkan0", card=0)],
        resident=[Resident(model="keep", device="Vulkan0")],
    )
    specs = {"keep": _spec("keep", DENSE_3B)}
    plan = plan_reconcile(placement, specs, loaded_now=["keep"], router_models_max=4, free_mib=_card)
    assert plan.to_load == ["keep"]


def test_placement_refuses_a_model_it_does_not_define() -> None:
    placement = Placement(devices=[Device(id="Vulkan0", card=0)], resident=[Resident(model="ghost", device="Vulkan0")])
    with pytest.raises(PlacementError, match="no model config"):
        budget_devices(placement, {}, free_mib=_card)


def test_placement_refuses_an_unknown_device() -> None:
    with pytest.raises(ValueError, match="not in"):
        Placement(devices=[Device(id="Vulkan0", card=0)], resident=[Resident(model="a", device="Vulkan9")])


def test_placement_refuses_a_model_declared_twice() -> None:
    with pytest.raises(ValueError, match="resident twice"):
        Placement(
            devices=[Device(id="Vulkan0", card=0)],
            resident=[Resident(model="a", device="Vulkan0"), Resident(model="a", device="Vulkan0")],
        )


def test_unknown_key_is_refused_not_ignored() -> None:
    """A typo silently ignored is a server running a config nobody wrote.

    Built through ``model_validate`` rather than a direct call so this exercises
    the RUNTIME refusal without asking the type checker to accept bad arguments.
    """
    with pytest.raises(ValueError):
        ModelSpec.model_validate({"name": "a", "model": "/m.gguf", "typo_key": 1})


def test_rendered_ini_carries_the_load_bearing_flags() -> None:
    placement = Placement(
        devices=[Device(id="Vulkan0", card=0), Device(id="Vulkan1", card=1)],
        resident=[Resident(model="wide", device=["Vulkan0", "Vulkan1"], tensor_split=[50.0, 50.0])],
    )
    specs = {"wide": _spec("wide", SWA_12B, lanes=4)}
    ini = render_preset_ini(placement, specs)
    assert "[wide]" in ini
    assert "device = Vulkan0,Vulkan1" in ini
    assert "tensor-split = 50.0,50.0" in ini
    # ctx-size is TOTAL across lanes, not the per-lane context.
    assert "ctx-size = 16384" in ini
    assert "parallel = 4" in ini
    # Explicit ngl stops -fit from silently reducing offload.
    assert "n-gpu-layers = 999" in ini
    # Declared resident set, never demand-loaded.
    assert "load-on-startup = true" in ini
    assert "GENERATED" in ini


# --------------------------------------------------------------------------- #
# a LoRA pool in the preset
# --------------------------------------------------------------------------- #


def _routed_placement(loras: list[str]) -> tuple[Placement, dict[str, ModelSpec]]:
    placement = Placement(
        devices=[Device(id="Vulkan1", card=1, reserve_mib=512)],
        resident=[Resident(model="routed", device="Vulkan1")],
        router=RouterConfig(models_max=1),
    )
    spec = ModelSpec(
        name="routed",
        model="/models/routed.gguf",
        ctx_per_lane=4096,
        lanes=4,
        loras=loras,
        extra_args={"lora-seq-routing": "true", "cache-ram": "0"},
        footprint=DENSE_3B,
    )
    return placement, {"routed": spec}


def test_a_lora_pool_renders_as_ONE_comma_separated_line() -> None:
    """A pool must not become one `lora =` line per adapter.

    The router's preset parser keeps options in a std::map and ASSIGNS per key
    (common/preset.cpp:322, "overwrite existing options"), so repeated `lora =`
    lines collapse to the last one and a declared pool of N silently becomes 1 —
    with no error anywhere. `--lora` takes a comma-separated list for exactly
    this case (common/arg.cpp:2859-2864).
    """
    placement, specs = _routed_placement(["/a/one.gguf", "/a/two.gguf", "/a/three.gguf"])
    ini = render_preset_ini(placement, specs)
    lora_lines = [ln for ln in ini.splitlines() if ln.startswith("lora =")]
    assert lora_lines == ["lora = /a/one.gguf,/a/two.gguf,/a/three.gguf"]
    # and the routing flag still comes through extra_args untouched
    assert "lora-seq-routing = true" in ini
    assert "cache-ram = 0" in ini


def test_a_single_adapter_still_renders_without_a_trailing_comma() -> None:
    placement, specs = _routed_placement(["/a/only.gguf"])
    assert "lora = /a/only.gguf" in render_preset_ini(placement, specs).splitlines()


def test_no_lora_line_when_the_model_declares_no_pool() -> None:
    placement, specs = _routed_placement([])
    lines = render_preset_ini(placement, specs).splitlines()
    # `lora =` exactly -- "lora-seq-routing" also starts with "lora".
    assert not [ln for ln in lines if ln.startswith("lora =")]


def test_a_comma_in_an_adapter_path_is_refused_not_silently_truncated() -> None:
    """The separator is a comma, so a comma in a path cannot be expressed.

    Refusing is the only safe answer: the alternative is a pool that loads
    different adapters than the ones declared.
    """
    placement, specs = _routed_placement(["/a/one,two.gguf"])
    with pytest.raises(PlacementError, match="comma"):
        render_preset_ini(placement, specs)


# --------------------------------------------------------------------------- #
# automatic placement: ONE CARD unless the operator says otherwise
# --------------------------------------------------------------------------- #

_TWO_MI25 = [Device(id="Vulkan0", card=0, reserve_mib=512), Device(id="Vulkan1", card=1, reserve_mib=512)]


def _free(_: int) -> tuple[float, float]:
    return MI25_TOTAL, MI25_TOTAL


def _auto(models: list[str], explicit: list[Resident] | None = None) -> Placement:
    return Placement(
        devices=_TWO_MI25,
        resident=[Resident(model=m) for m in models] + (explicit or []),
        router=RouterConfig(models_max=8),
    )


def test_an_auto_model_lands_on_one_card_not_split() -> None:
    specs = {"a": _spec("a", DENSE_3B)}
    out = resolve_auto_devices(_auto(["a"]), specs, free_mib=_free)
    assert out.resident[0].devices == ["Vulkan0"]
    assert out.resident[0].tensor_split is None


def test_auto_packs_by_best_fit_so_a_big_model_is_not_blocked() -> None:
    """Two 12B models must not both land on one card and strand the other."""
    specs = {"big1": _spec("big1", SWA_12B, lanes=8), "big2": _spec("big2", SWA_12B, lanes=8)}
    out = resolve_auto_devices(_auto(["big1", "big2"]), specs, free_mib=_free)
    assert sorted(r.devices[0] for r in out.resident) == ["Vulkan0", "Vulkan1"]


def test_auto_is_independent_of_the_order_models_are_listed() -> None:
    specs = {"big": _spec("big", SWA_12B, lanes=8), "small": _spec("small", DENSE_3B)}
    a = resolve_auto_devices(_auto(["big", "small"]), specs, free_mib=_free)
    b = resolve_auto_devices(_auto(["small", "big"]), specs, free_mib=_free)
    assert {r.model: r.devices for r in a.resident} == {r.model: r.devices for r in b.resident}


def test_auto_respects_a_card_the_operator_already_claimed() -> None:
    """An explicit placement is charged first; auto works around it."""
    specs = {"pinned": _spec("pinned", SWA_12B, lanes=8), "auto": _spec("auto", SWA_12B, lanes=8)}
    placement = Placement(
        devices=_TWO_MI25,
        resident=[Resident(model="pinned", device="Vulkan1"), Resident(model="auto")],
        router=RouterConfig(models_max=8),
    )
    out = resolve_auto_devices(placement, specs, free_mib=_free)
    assert {r.model: r.devices for r in out.resident} == {"pinned": ["Vulkan1"], "auto": ["Vulkan0"]}


def test_a_model_too_big_for_any_card_is_REFUSED_not_split() -> None:
    """The split ratio is a decision about the hardware; the tool must not invent it."""
    huge = Footprint(ctx_per_lane=4096, base_mib=20000.0, per_lane_mib=0.0)
    specs = {"huge": _spec("huge", huge)}
    with pytest.raises(NoSingleCardFitsError, match="no single card has room"):
        resolve_auto_devices(_auto(["huge"]), specs, free_mib=_free)


def test_the_refusal_says_how_to_span_cards_on_purpose() -> None:
    huge = Footprint(ctx_per_lane=4096, base_mib=20000.0, per_lane_mib=0.0)
    specs = {"huge": _spec("huge", huge)}
    with pytest.raises(NoSingleCardFitsError) as exc:
        resolve_auto_devices(_auto(["huge"]), specs, free_mib=_free)
    assert "tensor_split" in str(exc.value)
    assert "Vulkan0" in str(exc.value)


def test_explicit_placements_are_left_exactly_alone() -> None:
    specs = {"m": _spec("m", DENSE_3B)}
    placement = Placement(
        devices=_TWO_MI25,
        resident=[Resident(model="m", device=["Vulkan0", "Vulkan1"], tensor_split=[50, 50])],
        router=RouterConfig(models_max=1),
    )
    assert resolve_auto_devices(placement, specs, free_mib=_free) is placement


def test_tensor_split_without_a_device_is_refused() -> None:
    """A split is a deliberate placement, so it cannot ride on automatic."""
    with pytest.raises(ValueError, match="explicit device list"):
        Resident(model="m", tensor_split=[50, 50])
