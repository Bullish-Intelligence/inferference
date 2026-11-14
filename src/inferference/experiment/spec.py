"""The declarative experiment spec: schema_version 1.

A spec names every factor held constant (`fixed`), the one factor under test
(`factor`), the workloads to run, the replication discipline, and the
endpoints that matter. YAML, to match the model and placement YAML the
platform already uses (`reconcile.py`).

`fixed.control` carries an optional `pp_table`: a named power-play-table
variant, its sha256, and its field deltas from the stock table. Alongside it
are `fan_profile` and `power_profile`, so `pp_table` is optional and absent
from every spec that does not touch it.

It adds a second optional key, `power_cap_w`: a board power cap in watts that
an operator writes to hwmon `power1_cap`. The pp_table stays stock in that
mode. A spec declares at most one of the two.

It adds a third optional key, `fixed.placement`: a dual-card declaration. It
names two distinct DRM cards. The runner then starts one whole model instance on
each card, never split across cards, and runs both together. `power_cap_w` and
`arm_bound_c` apply to each listed card. A spec without `placement` is a
single-card spec, and its condition digest is unchanged (see `conditions.py`).

It adds a fourth optional key, `fixed.server.extra_args`: server flags that
the arm sets. It maps a flag name without dashes to a string value. An absent
`server` block stays out of the condition digest, like `placement`. A present
block is part of the digest. The runner merges it with the `extra_args` of the
model YAML.

`fixed.host.cpu_epp` is optional. When it is present, every CPU must read that
energy performance preference, or the run does not start. It is part of the
condition digest only when present.

It also carries three identity fields: `ModelFixed.sha256`,
`BuildFixed.build_id`, and `BuildFixed.features`. Each is optional. A preflight
check refuses when the matching `*_must_match` flag is set and the value is
still `None` — that check lives with the runner, not here.

`Replication.order` accepts only `counterbalanced`: ABBA, the start side
alternated per replicate. Any other value fails validation, which `load_spec`
reports as a `SpecError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, StrictInt, model_serializer, model_validator

from .server import refused_flag_keys

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "BatchFixed",
    "CacheFixed",
    "ContextFixed",
    "BuildFixed",
    "ControlFixed",
    "Endpoints",
    "ExperimentSpec",
    "Factor",
    "Fixed",
    "HostFixed",
    "ModelFixed",
    "PPTableArm",
    "PPTableFieldDelta",
    "PlacementFixed",
    "Replication",
    "SafetyFixed",
    "ServerFixed",
    "SpecError",
    "StartGateFixed",
]


class SpecError(ValueError):
    """A spec file is missing, malformed, or internally inconsistent."""


class _Strict(BaseModel):
    # An unknown key is a typo, and a typo silently ignored is a condition
    # nobody actually declared.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelFixed(_Strict):
    ref: str = "resident"
    sha256_must_match: bool = True
    sha256: str | None = None


class BuildFixed(_Strict):
    manifest_must_match: bool = True
    build_id: str | None = None
    features: tuple[str, ...] | None = None


class ContextFixed(_Strict):
    slots: int
    tokens_per_slot: int


class CacheFixed(_Strict):
    k: str
    v: str


class BatchFixed(_Strict):
    size: int
    ubatch: int


class PPTableFieldDelta(_Strict):
    """One field this arm's table changes from the stock blob.

    Matches the shape of :class:`inferference.pptable.FieldDelta`, so a future
    runner can assert a declared delta against the table's own `diff()`.
    """

    field: str
    old: int | tuple[int, ...]
    new: int | tuple[int, ...]


class PPTableArm(_Strict):
    """A named pp_table variant: its identity, and what it changes from stock.

    `sha256` is `None` until the variant has actually been written to disk:
    every uploaded table is written first, with its own sha256, before it is
    ever uploaded.
    A spec reviewed before that point names the variant and its intended
    deltas; it cannot yet name a hash for a blob that does not exist.
    """

    name: str
    sha256: str | None = None
    deltas_from_stock: tuple[PPTableFieldDelta, ...] = ()


class ControlFixed(_Strict):
    fan_profile: str
    power_profile: str
    pp_table: PPTableArm | None = None
    power_cap_w: StrictInt | None = None

    @model_validator(mode="after")
    def _one_power_actuator(self) -> ControlFixed:
        if self.power_cap_w is None:
            return self
        if self.pp_table is not None:
            raise SpecError(
                "fixed.control declares both pp_table and power_cap_w; declare one board power actuator, not both"
            )
        if self.power_cap_w <= 0:
            raise SpecError("fixed.control.power_cap_w must be a positive integer")
        return self


class SafetyFixed(_Strict):
    """Set thermal limits and the allowed memory change on other cards.

    The measured card permits no GTT increase.
    """

    machine_limit_c: int
    arm_bound_c: int
    gtt_limit_mib: int

    @model_validator(mode="after")
    def _arm_bound_below_machine_limit(self) -> SafetyFixed:
        if self.arm_bound_c > self.machine_limit_c:
            raise SpecError(
                f"arm_bound_c ({self.arm_bound_c}) must not exceed machine_limit_c ({self.machine_limit_c})"
            )
        return self


class StartGateFixed(_Strict):
    max_c: int
    tolerance_c: int
    hold_s: int


class HostFixed(_Strict):
    """Host limits. `cpu_epp` is the energy performance preference that every CPU must read.

    The host owns that setting. The runner reads it and refuses the run on a
    mismatch. An absent `cpu_epp` stays out of the dump, so the condition
    digest of a spec without it does not change.
    """

    loadavg_1m_max: float
    cpu_epp: str | None = None

    @model_validator(mode="after")
    def _cpu_epp_is_a_name(self) -> HostFixed:
        if self.cpu_epp is not None and not self.cpu_epp.strip():
            raise SpecError("fixed.host.cpu_epp must not be empty")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_cpu_epp(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if data.get("cpu_epp") is None:
            data.pop("cpu_epp", None)
        return data


class PlacementFixed(_Strict):
    """A dual-card declaration: one whole model instance on each of two cards.

    `cards` holds two distinct DRM card indices. `power_cap_w` and
    `arm_bound_c` apply to each listed card. `on_arm_bound` says what the peer
    run does when one card stops at its arm bound: `continue_peer` lets the
    other run go on. The machine limit stays box-wide and stops both runs.
    """

    mode: Literal["per-card"]
    cards: tuple[StrictInt, StrictInt]
    on_arm_bound: Literal["continue_peer"]

    @model_validator(mode="after")
    def _two_distinct_cards(self) -> PlacementFixed:
        first, second = self.cards
        if first < 0 or second < 0:
            raise SpecError(f"fixed.placement.cards must be zero or greater, got {list(self.cards)}")
        if first == second:
            raise SpecError(f"fixed.placement.cards must name two distinct cards, got {list(self.cards)}")
        return self


class ServerFixed(_Strict):
    """Server flags that this spec sets for every condition.

    `extra_args` maps a flag name without dashes to a string value:
    `{"threads": "2"}` becomes `--threads 2`. The runner refuses a flag that
    it sets itself (or an alias of one), a key with `=`, and a key that the
    model YAML sets to another value.
    """

    extra_args: dict[str, str]

    @model_validator(mode="after")
    def _keys_are_bare_flag_names(self) -> ServerFixed:
        for key in self.extra_args:
            if not key or key.startswith("-") or any(char.isspace() for char in key):
                raise SpecError(f"fixed.server.extra_args key {key!r} must be a flag name without dashes or spaces")
            if "=" in key:
                raise SpecError(f"fixed.server.extra_args key {key!r} must not contain '='; put the value in the map")
        refused = refused_flag_keys(list(self.extra_args))
        if refused:
            raise SpecError(f"fixed.server.extra_args repeats flag(s) that the runner sets itself: {refused}")
        return self


class Fixed(_Strict):
    """Every factor held constant across every condition in this spec."""

    model: ModelFixed
    build: BuildFixed
    context: ContextFixed
    cache: CacheFixed
    batch: BatchFixed
    control: ControlFixed
    safety: SafetyFixed
    start_gate: StartGateFixed
    host: HostFixed
    placement: PlacementFixed | None = None
    server: ServerFixed | None = None

    @model_validator(mode="after")
    def _placement_keeps_the_stock_table(self) -> Fixed:
        arm = self.control.pp_table
        if self.placement is not None and arm is not None and arm.deltas_from_stock:
            raise SpecError(
                "fixed.placement needs the stock pp_table on every card; "
                "declare power_cap_w instead of a pp_table arm that changes the table"
            )
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_options(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # An absent placement or server block is not part of the dump. This
        # keeps the digest and `declaration.json` of every spec that does not
        # declare them as they were.
        data: dict[str, Any] = handler(self)
        for key in ("placement", "server"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class Factor(_Strict):
    """The one factor under test. One condition per value."""

    name: str
    values: tuple[str, ...]


class Replication(_Strict):
    n: int
    order: Literal["counterbalanced"]
    cooldown_between: str


class Endpoints(_Strict):
    primary: str
    band_c: tuple[int, int]
    secondary: tuple[str, ...] = ()


class ExperimentSpec(_Strict):
    """One declarative experiment: `schema_version` is mandatory and checked on read."""

    experiment: str
    schema_version: int
    fixed: Fixed
    factor: Factor
    workloads: tuple[str, ...]
    replication: Replication
    endpoints: Endpoints

    @model_validator(mode="after")
    def _schema_version_is_known(self) -> ExperimentSpec:
        if self.schema_version != SCHEMA_VERSION:
            raise SpecError(
                f"spec {self.experiment!r} declares schema_version={self.schema_version}, "
                f"this module reads only {SCHEMA_VERSION}"
            )
        return self


def load_spec(path: Path) -> ExperimentSpec:
    """Load and validate one experiment spec from a YAML file."""
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise SpecError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise SpecError(f"{path} does not decode to a mapping")
    try:
        return ExperimentSpec.model_validate(raw)
    except SpecError:
        raise
    except Exception as exc:  # pydantic's ValidationError, wrapped to one error type
        raise SpecError(f"{path} failed validation: {exc}") from exc
