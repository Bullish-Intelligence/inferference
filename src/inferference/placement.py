"""Declarative placement of models onto GPUs, and the budget that refuses overcommit.

This is the admission control llama.cpp does not do. Measured directly:
loading a model that does not fit does **not** fail. It returns success, reports every
layer on the GPU, and silently spills into host memory over PCIe, where it runs
about **14x slower** with nothing in any log to say so. So the refusal has to
happen HERE, before anything is loaded.

Two rules from the owner shape this module:

* **Refuse, never evict.** A placement that would exceed a card's budget is a
  loud failure that LISTS what is already resident there, with footprints, so
  the owner decides what comes out. Choosing a victim is not this code's call.
* **YAML wins.** The declared placement is the truth. Whatever is currently on
  the card — including anything done through the router's own
  ``/models/load`` — is overridden by a reconcile.

FOOTPRINTS ARE MEASURED, NEVER DERIVED. KV scaling is per-architecture and the
two architectures on this rig behave in opposite ways: a dense model's KV is
linear in context, a sliding-window model's is flat above its window. KV does
not track parameter count either — the 3B here costs 5.5x the 9B per token. So a
footprint is a measurement at a stated per-lane context, and using it at any
OTHER per-lane context is refused rather than extrapolated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "BudgetError",
    "Device",
    "DeviceBudget",
    "Footprint",
    "ModelSpec",
    "Placement",
    "PlacementError",
    "Resident",
    "ReconcilePlan",
    "RouterConfig",
    "budget_devices",
    "parse_preset_sections",
    "plan_reconcile",
    "NoSingleCardFitsError",
    "render_preset_ini",
    "resolve_auto_devices",
    "sysfs_free_mib",
]


class PlacementError(Exception):
    """A declared placement is unusable. Always loud, never a warning."""


class _Strict(BaseModel):
    # An unknown key is a typo, and a typo silently ignored is a server running a
    # configuration nobody wrote.
    model_config = ConfigDict(extra="forbid")


class Footprint(_Strict):
    """A model's MEASURED VRAM cost on one card, as a line in the lane count.

        total = base_mib + lanes * per_lane_mib

    MEASURE FROM SYSFS, NOT FROM THE SERVER LOG. llama.cpp reports its model / KV
    / compute buffer sizes, and summing them UNDERSTATES what the card actually
    commits — by an amount that grows per lane. Measured 2026-09-05 on Gemma-4-12B
    at ctx_per_lane 4096, log-sum versus the card's own accounting:

        lanes 1: log 8799, sysfs 8866   (+67)
        lanes 2: log 9279, sysfs 9410   (+131)
        lanes 4: log 10239, sysfs 10498 (+259)

    Budgeting from the log figures would have declared a 16-lane placement to fit;
    it did not, and it spilled into host memory at about 1/14th the speed. The
    difference is allocator and per-slot overhead the log never names, so the only
    honest source is ``mem_info_vram_used`` before and after the load.

    Fitted on sysfs numbers the model is essentially exact — Gemma-4-12B gives
    base 8322 / per-lane 544, which reproduces the fitted points to 0 MiB, predicts
    an independently measured 8-lane load to within 8 MiB, and correctly predicts
    that 16 lanes (17026 MiB) will not fit a 16368 MiB card.

    ``base_mib`` is everything that does not scale with lanes (weights, compute
    buffer, fixed overhead). ``per_lane_mib`` is the marginal cost of one lane (KV
    plus per-slot overhead). Both are properties of ONE (model, quant,
    ctx_per_lane), and using them at a different ctx_per_lane is refused rather
    than extrapolated: KV scaling is per-architecture, and the two architectures
    on this rig behave in opposite ways.
    """

    ctx_per_lane: int = Field(gt=0, description="the per-lane context these numbers were measured at")
    base_mib: float = Field(ge=0, description="cost that does not scale with lanes")
    per_lane_mib: float = Field(ge=0, description="marginal cost of one lane")
    measured_by: str = Field(
        default="sysfs-vram-delta",
        description="how it was measured; a log-derived figure is not trustworthy",
    )
    note: str = ""

    def total_mib(self, lanes: int) -> float:
        if lanes < 1:
            raise PlacementError(f"lanes must be >= 1, got {lanes}")
        return self.base_mib + lanes * self.per_lane_mib


class ModelSpec(_Strict):
    """What a model IS. One file per model; the platform names no model itself."""

    name: str
    model: Path
    ctx_per_lane: int = Field(default=4096, gt=0)
    lanes: int = Field(default=1, ge=1, description="--parallel; each lane costs its own KV")
    ngl: int = Field(default=999, description="always set explicitly: it stops -fit reducing offload")
    loras: list[Path] = Field(default_factory=list)
    extra_args: dict[str, str] = Field(default_factory=dict)
    footprint: Footprint | None = None

    @field_validator("name")
    @classmethod
    def _name_is_a_section(cls, v: str) -> str:
        # The name becomes an INI section and a request's "model" field.
        if not v or not all(c.isalnum() or c in "-_." for c in v):
            raise ValueError(f"model name {v!r} must be alphanumeric with - _ . only")
        return v

    def require_footprint(self) -> Footprint:
        if self.footprint is None:
            raise PlacementError(
                f"model '{self.name}' has no measured footprint. Measure it at "
                f"ctx_per_lane={self.ctx_per_lane} and record it under `footprint:`. Footprints are never "
                f"guessed: KV scaling is per-architecture, and an over-budget load does not "
                f"fail, it silently spills to host memory."
            )
        if self.footprint.ctx_per_lane != self.ctx_per_lane:
            raise PlacementError(
                f"model '{self.name}' is configured for ctx_per_lane={self.ctx_per_lane} but its "
                f"footprint was measured at {self.footprint.ctx_per_lane}. Re-measure it. "
                f"Extrapolating is not safe: a dense model's KV is linear in context while a "
                f"sliding-window model's is flat above its window."
            )
        return self.footprint

    @property
    def total_ctx(self) -> int:
        """``--ctx-size`` is the TOTAL cache; per-lane context is ctx/parallel."""
        return self.ctx_per_lane * self.lanes


class Device(_Strict):
    """One GPU, named the way llama.cpp names it, plus how to read its VRAM."""

    id: str = Field(description="llama.cpp device id, e.g. Vulkan0")
    card: int = Field(ge=0, description="DRM index: /sys/class/drm/card<N>")
    reserve_mib: int = Field(
        default=512,
        ge=0,
        description="headroom left unallocated; a card driven to its last MiB spills",
    )


class Resident(_Strict):
    """One model, declared resident on one or more devices."""

    model: str
    # OPTIONAL. Omitted means AUTO: resolve_auto_devices() binds it to the single
    # card that fits, and REFUSES if none does. Splitting a model across cards
    # stays an explicit, human decision -- see that function for why.
    device: str | list[str] | None = None
    tensor_split: list[float] | None = None

    @property
    def is_auto(self) -> bool:
        return self.device is None

    @model_validator(mode="after")
    def _split_matches_devices(self) -> Resident:
        if self.device is None:
            if self.tensor_split is not None:
                raise ValueError(
                    f"'{self.model}': tensor_split needs an explicit device list. A split is a "
                    f"deliberate placement, so it cannot be combined with automatic placement."
                )
            return self
        devs = self.devices
        if self.tensor_split is not None:
            if len(devs) < 2:
                raise ValueError(f"'{self.model}': tensor_split needs two or more devices, got {devs}")
            if len(self.tensor_split) != len(devs):
                raise ValueError(
                    f"'{self.model}': tensor_split has {len(self.tensor_split)} entries for {len(devs)} devices"
                )
            if sum(self.tensor_split) <= 0:
                raise ValueError(f"'{self.model}': tensor_split must sum to more than zero")
        return self

    @property
    def devices(self) -> list[str]:
        if self.device is None:
            raise PlacementError(
                f"'{self.model}' has no device yet. Automatic placements must be resolved with "
                f"resolve_auto_devices() before anything asks where the model lives."
            )
        return [self.device] if isinstance(self.device, str) else list(self.device)

    def share_for(self, device: str) -> float:
        """Fraction of this model's footprint charged to ``device``."""
        devs = self.devices
        if device not in devs:
            return 0.0
        if len(devs) == 1:
            return 1.0
        weights = self.tensor_split or [1.0] * len(devs)
        return weights[devs.index(device)] / sum(weights)


class RouterConfig(_Strict):
    """How the router itself was launched. Declared, because it is not readable.

    ``--models-max`` is the one router flag the reconciler MUST know: below the
    resident count, an explicit load returns success and silently evicts the
    least recently used model (007 report 02, U3). It can only be scraped off a
    RUNNING child's argv, so on an empty router it is unknowable — and treating
    "unknown" as "fine" is exactly the silent eviction the rule forbids.

    So it is declared here, next to the placement the launcher reads, and the
    reconciler cross-checks the declaration against any child it can see. A
    disagreement means the running router is not the one this file describes,
    which is drift, and drift is refused rather than assumed benign.

    ``host``/``port`` are here for the same reason ``port`` used to live in the
    single-role server's YAML: two files agreeing on a port by coincidence is
    exactly the failure mode this project refuses to ship. The router launcher
    (``ci/runner/inferference-router-launch.sh``) reads them from here, and
    ``inferference-reconcile.py`` defaults its ``--router`` URL to them.
    """

    host: str = Field(default="127.0.0.1", description="the router's --host")
    port: int = Field(default=8100, gt=0, le=65535, description="the router's --port")
    models_max: int = Field(
        default=4,
        ge=0,
        description="the --models-max the router was launched with; 0 means unlimited",
    )

    def allows(self, resident_count: int) -> bool:
        return self.models_max == 0 or self.models_max >= resident_count

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class Placement(_Strict):
    """The router-level config: which models are resident, and where."""

    devices: list[Device]
    resident: list[Resident] = Field(default_factory=list)
    router: RouterConfig = Field(default_factory=RouterConfig)

    @model_validator(mode="after")
    def _coherent(self) -> Placement:
        ids = [d.id for d in self.devices]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate device id in {ids}")
        cards = [d.card for d in self.devices]
        if len(set(cards)) != len(cards):
            raise ValueError(f"two devices share a DRM card index: {cards}")
        seen: set[str] = set()
        for r in self.resident:
            if r.model in seen:
                raise ValueError(f"model '{r.model}' is declared resident twice")
            seen.add(r.model)
            if r.is_auto:
                continue  # nothing to check yet; resolve_auto_devices() binds it
            for d in r.devices:
                if d not in ids:
                    raise ValueError(f"'{r.model}' names device '{d}', not in {ids}")
        return self

    @property
    def models_max(self) -> int:
        """``--models-max`` must be >= the resident count.

        Below it, an explicit load returns success and SILENTLY evicts the least
        recently used model (007 report 02, U3) — a declared-resident model goes
        down with no error to the caller.
        """
        return max(1, len(self.resident))


class ModelCharge(_Strict):
    """One model's share of one device."""

    model: str
    lanes: int
    share: float
    mib: float


class DeviceBudget(_Strict):
    """What a device is being asked to hold, against what it has."""

    device: str
    card: int
    total_mib: float
    free_mib: float
    reserve_mib: float
    charges: list[ModelCharge]

    @property
    def requested_mib(self) -> float:
        return sum(c.mib for c in self.charges)

    @property
    def available_mib(self) -> float:
        return self.free_mib - self.reserve_mib

    @property
    def fits(self) -> bool:
        return self.requested_mib <= self.available_mib

    @property
    def overage_mib(self) -> float:
        return max(0.0, self.requested_mib - self.available_mib)


class NoSingleCardFitsError(PlacementError):
    """An automatic placement found no single card with room.

    Its own class, not BudgetError, because BudgetError carries a per-device
    charge listing and this refusal happens BEFORE any such listing exists.
    """


class BudgetError(PlacementError):
    """A placement exceeds a device's budget. Carries the listing for the owner."""

    def __init__(self, budgets: Sequence[DeviceBudget]) -> None:
        self.budgets = list(budgets)
        super().__init__(self.render())

    def render(self) -> str:
        lines = ["placement REFUSED — it would exceed VRAM on:"]
        for b in self.budgets:
            if b.fits:
                continue
            lines.append(
                f"\n  {b.device} (card {b.card}): needs {b.requested_mib:.0f} MiB, "
                f"has {b.available_mib:.0f} MiB usable "
                f"({b.free_mib:.0f} free - {b.reserve_mib:.0f} reserved) "
                f"— OVER BY {b.overage_mib:.0f} MiB"
            )
            lines.append("    would hold:")
            for c in sorted(b.charges, key=lambda c: -c.mib):
                share = "" if c.share == 1.0 else f", {c.share:.0%} of a split model"
                lines.append(f"      {c.model:<24} {c.mib:>9.0f} MiB  ({c.lanes} lane(s){share})")
        lines.append(
            "\nNothing was loaded and nothing was evicted. Remove a model from the placement, "
            "lower its lanes or ctx_per_lane, or move it to another device."
        )
        lines.append(
            "Refusing rather than loading is deliberate: llama.cpp does NOT fail on an "
            "overcommit, it spills into host memory and runs ~14x slower while reporting success."
        )
        return "\n".join(lines)


def sysfs_free_mib(card: int) -> tuple[float, float]:
    """(total, free) VRAM in MiB for a DRM card, from amdgpu sysfs.

    ROCm is not installed on this rig by design (gfx900 was dropped after ROCm
    5.7), so there is no rocm-smi to ask; sysfs is the vendor-truthful source and
    is what ``ci/runner/gpu-lease.sh`` reads too.
    """
    base = Path(f"/sys/class/drm/card{card}/device")
    try:
        total = int((base / "mem_info_vram_total").read_text().strip())
        used = int((base / "mem_info_vram_used").read_text().strip())
    except (OSError, ValueError) as exc:
        raise PlacementError(
            f"cannot read VRAM for card {card} at {base}: {exc}. A budget cannot be "
            f"checked without a real free-memory figure; refusing rather than guessing."
        ) from exc
    return total / 1048576, (total - used) / 1048576


def resolve_auto_devices(
    placement: Placement,
    specs: Mapping[str, ModelSpec],
    *,
    free_mib: Callable[[int], tuple[float, float]] = sysfs_free_mib,
) -> Placement:
    """Bind every ``device``-less resident to ONE card. Refuse if none fits.

    **ONE CARD IS THE DEFAULT, and spanning cards is never automatic.** Splitting
    costs about 10% of decode throughput because every token crosses PCIe, and
    it needs a ``tensor_split`` whose ratio is a judgement
    about the two cards. AGENTS.md already says the reconciler does not guess
    placement -- "guessing which model matters least is not the reconciler's
    call" -- and choosing a split ratio is the same kind of guess. So a model
    that fits nowhere is a LOUD REFUSAL naming the sizes, not a split invented
    here. The refusal says exactly what to write to get a split on purpose.

    Explicit placements are left ALONE, including explicit multi-card ones: this
    only fills in what the operator did not say.

    **Best fit, not first fit.** Each model goes on the card with the LEAST
    remaining room that still holds it, so a big model is not blocked by a small
    one that landed on the only card with space. Models are placed largest
    first, for the same reason. Both are needed to make the result independent of
    the order the YAML happens to list models in.

    Ties break on device id so a reconcile is reproducible: the same config must
    always produce the same placement, or the generated preset churns and
    ``?reload=1`` restarts children for no reason.
    """
    auto = [r for r in placement.resident if r.is_auto]
    if not auto:
        return placement

    # Remaining room per device, after everything the operator placed BY HAND.
    # Explicit placements are charged first because they are not negotiable.
    room: dict[str, float] = {}
    for dev in placement.devices:
        total, _free = free_mib(dev.card)
        used = 0.0
        for r in placement.resident:
            if r.is_auto:
                continue
            share = r.share_for(dev.id)
            if share:
                used += specs[r.model].require_footprint().total_mib(specs[r.model].lanes) * share
        room[dev.id] = total - float(dev.reserve_mib) - used

    def _need(r: Resident) -> float:
        spec = specs[r.model]
        return spec.require_footprint().total_mib(spec.lanes)

    resolved: dict[str, str] = {}
    for r in sorted(auto, key=lambda x: (-_need(x), x.model)):
        need = _need(r)
        fits = sorted(
            (d for d in placement.devices if room[d.id] >= need),
            key=lambda d: (room[d.id], d.id),
        )
        if not fits:
            biggest = max(placement.devices, key=lambda d: room[d.id])
            raise NoSingleCardFitsError(
                f"model '{r.model}' needs {need:.1f} MiB and no single card has room. "
                f"Remaining after the placements you declared by hand: "
                + ", ".join(f"{d.id} {room[d.id]:.1f} MiB" for d in placement.devices)
                + f" (most room: {biggest.id}). This is not split across cards automatically, "
                f"because the tensor_split ratio is a decision about your hardware, not one this "
                f"tool should invent -- splitting also costs about 10% of decode throughput. "
                f"To span cards on purpose, declare it: "
                f"`device: [{placement.devices[0].id}, {placement.devices[-1].id}]` with a "
                f"`tensor_split`. To fit on one card, lower the model's `lanes` or "
                f"`ctx_per_lane` and RE-MEASURE its footprint."
            )
        chosen = fits[0]
        resolved[r.model] = chosen.id
        room[chosen.id] -= need

    return placement.model_copy(
        update={
            "resident": [
                r.model_copy(update={"device": resolved[r.model]}) if r.is_auto else r for r in placement.resident
            ]
        }
    )


def budget_devices(
    placement: Placement,
    specs: Mapping[str, ModelSpec],
    *,
    free_mib: Callable[[int], tuple[float, float]] = sysfs_free_mib,
    assume_empty: bool = True,
) -> list[DeviceBudget]:
    """Charge every resident model to its device(s) and compare against VRAM.

    ``assume_empty`` treats each card as if the models this placement declares are
    not yet loaded — which is what a reconcile means, since YAML wins and the
    router is about to be brought to this state. Set it False to budget against
    the card exactly as it is right now.
    """
    budgets: list[DeviceBudget] = []
    for dev in placement.devices:
        total, free = free_mib(dev.card)
        charges: list[ModelCharge] = []
        for r in placement.resident:
            share = r.share_for(dev.id)
            if share == 0.0:
                continue
            spec = specs.get(r.model)
            if spec is None:
                raise PlacementError(
                    f"placement declares model '{r.model}' resident, but no model config "
                    f"defines it. A placement may not name a model that does not exist."
                )
            fp = spec.require_footprint()
            charges.append(
                ModelCharge(
                    model=r.model,
                    lanes=spec.lanes,
                    share=share,
                    mib=fp.total_mib(spec.lanes) * share,
                )
            )
        budgets.append(
            DeviceBudget(
                device=dev.id,
                card=dev.card,
                total_mib=total,
                free_mib=total if assume_empty else free,
                reserve_mib=float(dev.reserve_mib),
                charges=charges,
            )
        )
    return budgets


def render_preset_ini(placement: Placement, specs: Mapping[str, ModelSpec]) -> str:
    """Render the llama.cpp router preset INI this placement describes.

    The INI is a GENERATED ARTEFACT, never hand-edited: YAML is the source of
    truth, so anything written here is discarded by the next reconcile.
    """
    out: list[str] = [
        "; GENERATED by inferference — do not edit.",
        "; Source of truth is the model YAML + placement YAML. A reconcile overwrites this.",
        "",
    ]
    by_name = {r.model: r for r in placement.resident}
    for name in sorted(by_name):
        r = by_name[name]
        spec = specs[name]
        out.append(f"[{name}]")
        out.append(f"model = {spec.model}")
        out.append(f"device = {','.join(r.devices)}")
        if r.tensor_split:
            out.append(f"tensor-split = {','.join(str(x) for x in r.tensor_split)}")
        # ctx-size is the TOTAL cache across lanes, not the per-lane context.
        out.append(f"ctx-size = {spec.total_ctx}")
        out.append(f"parallel = {spec.lanes}")
        # Always explicit: an unset n-gpu-layers lets -fit silently reduce offload.
        out.append(f"n-gpu-layers = {spec.ngl}")
        if spec.loras:
            # ONE `lora` line, comma-separated -- NOT one line per adapter.
            #
            # The router's preset parser stores options in a
            # `std::map<common_arg, std::string>` (common/preset.h:23) and
            # ASSIGNS on each key it reads (`preset.options[opt] = value`,
            # common/preset.cpp:322, whose own comment says "overwrite existing
            # options"). A repeated `lora =` therefore keeps only the LAST line,
            # and `to_args` emits a single `--lora`. A declared pool of eight
            # adapters silently became one, with no error anywhere -- and a
            # routing-enabled model would then answer every request from
            # whichever adapter happened to be written last.
            #
            # `--lora` takes a comma-separated list for exactly this
            # (common/arg.cpp:2859-2864, `parse_csv_row`), so one line carries
            # the whole pool.
            bad = [str(x) for x in spec.loras if "," in str(x)]
            if bad:
                raise PlacementError(
                    f"model '{name}': adapter path(s) contain a comma and cannot be expressed "
                    f"in the router preset, which separates a LoRA pool with commas: {bad}. "
                    f"Rename the file(s); a silently truncated pool is worse than this refusal."
                )
            out.append("lora = " + ",".join(str(x) for x in spec.loras))
        for key, value in sorted(spec.extra_args.items()):
            out.append(f"{key} = {value}")
        # The declared resident set: loaded at startup, never on demand.
        out.append("load-on-startup = true")
        out.append("")
    return "\n".join(out)


def parse_preset_sections(text: str) -> dict[str, str]:
    """Split a preset INI into ``{section name: normalised body}``.

    Used to tell a CHANGED model from an unchanged one. Comments and blank lines
    are dropped and keys are sorted, so a reordered or re-commented render does
    not read as a change and cause a needless reload.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith((";", "#")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            key, sep, value = line.partition("=")
            sections[current].append(f"{key.strip()}={value.strip()}" if sep else line)
    return {name: "\n".join(sorted(body)) for name, body in sections.items()}


class ReconcilePlan(_Strict):
    """What a reconcile would do. Computed before anything is touched."""

    budgets: list[DeviceBudget]
    models_max: int
    to_unload: list[str]
    to_load: list[str]
    changed: list[str]
    unchanged: list[str]
    needs_router_restart: bool
    restart_reason: str | None = None

    @property
    def is_noop(self) -> bool:
        return not self.to_unload and not self.to_load and not self.needs_router_restart


def plan_reconcile(
    placement: Placement,
    specs: Mapping[str, ModelSpec],
    *,
    loaded_now: Iterable[str],
    router_models_max: int | None,
    free_mib: Callable[[int], tuple[float, float]] = sysfs_free_mib,
    previous_ini: str | None = None,
) -> ReconcilePlan:
    """Decide the reconcile, refusing an over-budget placement before any I/O.

    Reconcile is three steps and no restart: regenerate the INI, request a
    router reload, then ensure each declared model is loaded with its current
    preset. A router may autoload a preset during reload, so callers inspect its
    status and avoid duplicate load requests.

    ``previous_ini`` is the preset file the router is currently running, if it
    exists. It is what makes a CHANGED model visible. Without it a model whose
    preset changed still appears in ``want & have`` and reads as "unchanged", so
    nothing reloads it — while ``?reload=1`` has already taken it down. The model
    would stay down, silently, which is the failure this whole module exists to
    prevent. When it is None every declared model is treated as changed: that
    over-reloads, which costs time, where the other way round loses a model.
    """
    budgets = budget_devices(placement, specs, free_mib=free_mib)
    if any(not b.fits for b in budgets):
        raise BudgetError(budgets)

    want = {r.model for r in placement.resident}
    have = set(loaded_now)
    models_max = placement.models_max

    wanted_ini = parse_preset_sections(render_preset_ini(placement, specs))
    old_ini = parse_preset_sections(previous_ini) if previous_ini is not None else None
    if old_ini is None:
        changed = set(want & have)
    else:
        changed = {n for n in want & have if wanted_ini.get(n) != old_ini.get(n)}

    restart = False
    reason: str | None = None
    declared_max = placement.router.models_max
    if not placement.router.allows(models_max):
        restart = True
        reason = (
            f"the router is declared with --models-max {declared_max} but this placement "
            f"declares {models_max} resident models. It is a router CLI flag, so the router "
            f"must be restarted. Loading past it would return success and SILENTLY evict the "
            f"least recently used model."
        )
    elif router_models_max is not None and router_models_max != declared_max:
        # The running router is not the one placement.yaml describes. Which of
        # the two is right is not knowable here, and guessing the wrong one
        # either evicts a model or refuses a valid placement.
        restart = True
        reason = (
            f"CONFIG DRIFT: the running router was launched with --models-max "
            f"{router_models_max}, but placement.yaml declares {declared_max}. The running "
            f"router is not the one this config describes. Restart it from this config, or "
            f"correct `router.models_max` to match what is actually running."
        )
    elif router_models_max is not None and router_models_max != 0 and router_models_max < models_max:
        restart = True
        reason = (
            f"--models-max is {router_models_max} but this placement declares {models_max} "
            f"resident models. It is a router CLI flag, so the router must be restarted."
        )

    return ReconcilePlan(
        budgets=budgets,
        models_max=models_max,
        to_unload=sorted(have - want),
        # Changed and missing models must be brought up with the declared
        # preset. The router may autoload them during reload.
        to_load=sorted((want - have) | changed),
        changed=sorted(changed),
        unchanged=sorted((want & have) - changed),
        needs_router_restart=restart,
        restart_reason=reason,
    )
