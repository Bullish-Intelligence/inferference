#!/usr/bin/env python3
"""Bring the running router to the state the YAML declares. YAML always wins.

    inferference-reconcile.py plan            # what would change, and the budget
    inferference-reconcile.py apply           # do it
    inferference-reconcile.py render          # print the preset INI only
    inferference-reconcile.py status          # what is resident now, and its health
    inferference-reconcile.py evict <device>  # free a card; config untouched
    inferference-reconcile.py config-dir      # print the config dir this tool reads
    inferference-reconcile.py preset-ini      # print the preset file this tool uses

CONFIG (one model per file; the platform names no model itself):

    $INFERFERENCE_CONFIG_DIR/models/<name>.yaml     what a model IS
    $INFERFERENCE_CONFIG_DIR/placement.yaml         which models are resident, where

The config dir is chosen in this order, and the tool prints which one it used:

    1. $INFERFERENCE_CONFIG_DIR, when set.
    2. /etc/inferference, when it exists: the dir that the router service reads.
       The preset file then defaults to /run/inferference/models-preset.ini,
       when that dir exists, as the router unit sets it.
    3. ~/.config/inferference, with a warning: a home copy can be stale.

Other launchers (evict-and-run-*.sh) ask `config-dir` instead of repeating this
order. This CLI is the one source of truth for the default. The order itself
lives in inferference.config_paths, so `inferference.config_dir()` agrees.

The preset file path has one source too. `preset-ini` prints it, and the router
launcher asks for it. The order is: $INFERFERENCE_PRESET_INI; the system router's
/run/inferference/models-preset.ini (system config dir, when /run/inferference
exists); beside a writable config dir that is outside /etc; else
${XDG_STATE_HOME:-~/.local/state}/inferference/models-preset.ini.

PRESET FILE. The router reads one preset file ($INFERFERENCE_PRESET_INI). When
the caller cannot write it (the router runs as its own user), `apply` compares
the declared preset with the router's file first. A difference REFUSES `apply`:
the reload would load the router's preset, not the one that was budgeted. The
declared preset is then rendered to a caller-owned file under $XDG_STATE_HOME.
`plan` reports the same comparison.

WHY THIS EXISTS. llama.cpp will not refuse a placement that does not fit: it
returns success, reports every layer on the GPU, and silently spills into host
memory over PCIe at about 1/14th the speed. So admission control is ours. A
placement that would exceed a card's budget is REFUSED, with a listing of what
is already resident there so the operator decides what comes out — this tool never
evicts to make room.

RECONCILE IS THREE STEPS AND NEEDS NO RESTART:
regenerate the INI, GET /models?reload=1, then ensure every declared model is
loaded with its current preset. The router may start a load-on-startup preset
during reload; apply checks the reported preset and waits for that load instead
of issuing a duplicate request. Missing or stale children are loaded explicitly.
`systemctl reload inferference-router` runs `apply` as the unit user through
ci/runner/inferference-router-reload.sh. `apply` writes the preset file by temp
file + rename, after the budget check, so a refused reload leaves it unchanged.
The router's own /models/load and /models/unload stay enabled; anything they did
is discarded here, by design.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inferference import config_paths  # noqa: E402
from inferference.placement import (  # noqa: E402
    BudgetError,
    ModelSpec,
    Placement,
    PlacementError,
    parse_preset_sections,
    plan_reconcile,
    render_preset_ini,
)
from inferference.reconcile import (  # noqa: E402
    ConfigError,
    child_health,
    crashed_children,
    loaded_names,
    models_with_matching_presets,
    read_config_dir,
    router_models_max,
    running_presets,
)

EX_CONFIG = 78
# Seconds to let amdgpu reclaim freed VRAM before the placement check reads it.
SETTLE_S = float(os.environ.get("INFERFERENCE_SETTLE_S", "20"))
# After a failed apply, the tool unloads the models that it changed. The unloads
# share CLEANUP_S seconds, and one unload takes at most UNLOAD_S. The clean stop
# of one model takes about 14 s at most.
CLEANUP_S = 90.0
UNLOAD_S = 45.0
# The time that `apply` can use after its deadline passes, in the worst case:
# the settle wait, the cleanup, and four router reads of 30 s each. The reload
# script keeps load timeout + this value under the unit timeout.
APPLY_OVERHEAD_S = 240.0


class Deadline:
    """One time limit for the whole of `apply`: every wait and every router call.

    `--timeout` sets it. Two waits in a row must not each take the full limit,
    so both read the time that is left. After the deadline passes, a router call
    still gets one second, so a final unload can run.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def cap(self, seconds: float) -> float:
        """Return ``seconds``, cut to the time that is left (at least one second)."""
        return max(1.0, min(seconds, self.remaining()))


def _default_router() -> str:
    """The router URL to talk to when ``--router`` is not given.

    ``$INFERFERENCE_ROUTER`` wins if set. Otherwise read it from
    ``placement.yaml``'s ``router.host``/``router.port`` — the same file that
    declares ``--models-max``, so one file says how the router was launched.
    Falls back to the router's own default (127.0.0.1:8100) when the config
    cannot be read yet, e.g. on a box with no placement.yaml written.
    """
    env = os.environ.get("INFERFERENCE_ROUTER")
    if env:
        return env
    try:
        placement, _ = read_config_dir(config_dir())
    except (ConfigError, OSError):
        return "http://127.0.0.1:8100"
    return placement.router.url


def die(msg: str) -> NoReturn:
    print(f"!! [reconcile] {msg}", file=sys.stderr)
    raise SystemExit(EX_CONFIG)


# The system-service convention: the config dir of the router service. The
# library owns the order (inferference.config_paths.resolve_config_dir), so that
# the CLI and `inferference.config_dir()` cannot disagree. These names stay here
# so that a test can point the CLI at a fake system dir.
SYSTEM_CONFIG_DIR = config_paths.SYSTEM_CONFIG_DIR
HOME_CONFIG_DIR = config_paths.HOME_CONFIG_DIR
# Where the system router unit keeps its preset file (INFERFERENCE_PRESET_INI).
SYSTEM_PRESET_INI = Path("/run/inferference/models-preset.ini")


def resolve_config_dir() -> tuple[Path, str]:
    """Return the config dir and the reason it was chosen.

    An explicit ``$INFERFERENCE_CONFIG_DIR`` always wins. Without it, the system
    dir wins when it exists, so the CLI budgets from the config that the router
    serves. The home dir is the last resort, and the reason says it is a
    fallback so that the caller can warn.
    """
    return config_paths.resolve_config_dir(SYSTEM_CONFIG_DIR, HOME_CONFIG_DIR)


def config_dir() -> Path:
    return resolve_config_dir()[0]


_ANNOUNCED: set[tuple[Path, str]] = set()


def announce_config_dir() -> None:
    """Say on stderr which config dir this run reads. Never silent."""
    path, how = resolve_config_dir()
    if (path, how) in _ANNOUNCED:
        return
    _ANNOUNCED.add((path, how))
    if how == "home-fallback":
        print(
            f"!! [reconcile] WARNING: reading the home config {path}: $INFERFERENCE_CONFIG_DIR is "
            f"unset and {SYSTEM_CONFIG_DIR} does not exist. A home copy can be stale. "
            f"Set $INFERFERENCE_CONFIG_DIR to the config that the router serves.",
            file=sys.stderr,
        )
    else:
        source = "$INFERFERENCE_CONFIG_DIR" if how == "env" else "system default"
        print(f">> config dir: {path} ({source})", file=sys.stderr)


def load_config() -> tuple[Placement, dict[str, ModelSpec]]:
    """Read the config dir, turning any refusal into the CLI's exit code."""
    announce_config_dir()
    try:
        return read_config_dir(config_dir())
    except ConfigError as exc:
        die(str(exc))


# --------------------------------------------------------------------- router --
def _req(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    timeout: float = 30.0,
    *,
    tolerate_http: bool = False,
) -> Any:
    """One router call.

    An HTTP error and an unreachable router are DIFFERENT faults and are
    reported as such: calling a 400 "unreachable" sends the reader to check a
    router that is running perfectly well.

    ``tolerate_http`` is for a call whose failure is not a problem — unloading a
    model the router has already dropped. A connection failure is still fatal.
    """
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode(errors="replace")[:400]
        except OSError:
            pass
        if tolerate_http:
            return None
        die(f"router refused {method} {url}: HTTP {exc.code} {exc.reason}. {detail}")
    except urllib.error.URLError as exc:
        die(f"router at {url} is unreachable: {exc.reason}. Start it, or set INFERFERENCE_ROUTER.")


def router_models(router: str) -> list[dict[str, Any]]:
    out = _req("GET", f"{router}/models")
    return list(out.get("data", [])) if isinstance(out, dict) else []


# ---------------------------------------------------------------- placement io --
def device_snapshot(placement: Placement) -> dict[str, tuple[float, float]]:
    """(vram_used, gtt_used) MiB per device id."""
    out: dict[str, tuple[float, float]] = {}
    for dev in placement.devices:
        base = Path(f"/sys/class/drm/card{dev.card}/device")
        try:
            vram = int((base / "mem_info_vram_used").read_text()) / 1048576
            gtt = int((base / "mem_info_gtt_used").read_text()) / 1048576
        except (OSError, ValueError) as exc:
            die(f"cannot read VRAM and GTT usage for {dev.id} (card {dev.card}): {exc}")
        if vram < 0 or gtt < 0:
            die(f"invalid VRAM or GTT usage for {dev.id} (card {dev.card})")
        out[dev.id] = (vram, gtt)
    return out


def _under_etc(path: Path) -> bool:
    """True when ``path`` resolves into /etc. The tool never writes a preset there."""
    real = Path(os.path.realpath(path))
    return real == Path("/etc") or Path("/etc") in real.parents


def resolve_ini_path() -> tuple[Path, str]:
    """Return the preset file that the router reads, and the reason it was chosen.

    This is the one source of truth for the default. The router launcher asks it
    through ``preset-ini``, so the launcher and `apply` cannot disagree. The order:

    1. ``$INFERFERENCE_PRESET_INI`` (reason ``env``).
    2. With the system config dir, the system router's runtime path, when its
       dir exists (reason ``system-runtime``): the system dir is read-only, and
       the router unit points at the runtime path.
    3. Beside the config, when the config dir is outside /etc and this process
       can write it (reason ``beside-config``).
    4. ``$XDG_STATE_HOME/inferference/models-preset.ini`` (reason ``state``).
    """
    env = os.environ.get("INFERFERENCE_PRESET_INI")
    if env:
        return Path(env), "env"
    path, how = resolve_config_dir()
    if how == "system" and SYSTEM_PRESET_INI.parent.is_dir():
        return SYSTEM_PRESET_INI, "system-runtime"
    if not _under_etc(path) and path.is_dir() and os.access(path, os.W_OK | os.X_OK):
        return path / "models-preset.ini", "beside-config"
    return local_ini_path(), "state"


def ini_path() -> Path:
    """The preset file that the router reads. See :func:`resolve_ini_path`."""
    return resolve_ini_path()[0]


def _current_ini() -> str | None:
    """The preset file this tool last generated, as a FALLBACK only.

    The router's own reported presets are preferred (`running_presets`): this
    file says what we intended, not what is running. It is used when the router
    has no loaded child to ask.

    None makes the plan treat every declared model as changed, which reloads more
    than it needs to. That is the safe direction: the other one leaves a model
    that ?reload=1 took down and nothing put back.
    """
    path = ini_path()
    try:
        return path.read_text()
    except OSError:
        return None


def _can_write(path: Path) -> bool:
    """True when `_write_atomic` can replace ``path``, or create it where it belongs.

    `_write_atomic` writes a temp file beside ``path`` and renames it over
    ``path``. So the parent dir must be writable even when the file itself is.
    A writable file in a read-only dir is NOT writable here.
    """
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not os.access(parent, os.W_OK | os.X_OK):
        return False
    if path.exists():
        return os.access(path, os.W_OK)
    return True


def local_ini_path() -> Path:
    """A caller-owned path for the rendered preset, when the router's is read-only."""
    state = Path(os.environ.get("XDG_STATE_HOME") or "~/.local/state").expanduser()
    return state / "inferference" / "models-preset.ini"


class PresetCheck:
    """How the router's preset file relates to the preset that would be applied."""

    def __init__(self, path: Path, writable: bool, matches: bool | None, detail: str = "") -> None:
        self.path = path
        self.writable = writable
        # None: the router's file could not be read.
        self.matches = matches
        self.detail = detail

    @property
    def blocks_apply(self) -> bool:
        """The caller cannot write the router's file and it is not the wanted one."""
        return not self.writable and self.matches is not True

    def describe(self) -> str:
        if self.writable:
            return f"router preset {self.path}: writable here; apply rewrites it"
        if self.matches is None:
            return f"router preset {self.path}: read-only here and UNREADABLE ({self.detail})"
        if self.matches:
            return f"router preset {self.path}: read-only here; it MATCHES the declared preset"
        return f"router preset {self.path}: read-only here and DIFFERS from the declared preset"


def check_router_preset(wanted_ini: str) -> PresetCheck:
    path = ini_path()
    writable = _can_write(path)
    try:
        live = path.read_text()
    except OSError as exc:
        return PresetCheck(path, writable, None, str(exc))
    return PresetCheck(path, writable, parse_preset_sections(live) == parse_preset_sections(wanted_ini))


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that a reader sees the old file or the new one, never half.

    The temp file sits in the same dir, so the rename stays on one filesystem. The
    mode is 0644: the router and the other users of the preset file read it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.chmod(0o644)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# -------------------------------------------------------------------- commands --
def cmd_render(args: argparse.Namespace) -> int:
    placement, specs = load_config()
    text = render_preset_ini(placement, specs)
    if args.out:
        # Write the file HERE rather than letting the caller redirect stdout.
        # This repo's devenv prints a banner on shell entry, so
        # `devenv shell -- ... render > file` silently prepends that banner and
        # the router then refuses the file with "failed to parse server config".
        out = Path(args.out).expanduser()
        _write_atomic(out, text)
        print(f">> wrote {out}")
    else:
        print(text, end="")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    placement, specs = load_config()
    models = router_models(args.router) if args.router_reachable else []
    try:
        plan = plan_reconcile(
            placement,
            specs,
            loaded_now=loaded_names(models),
            router_models_max=router_models_max(models),
            previous_ini=running_presets(models) or _current_ini(),
        )
    except BudgetError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except PlacementError as exc:
        die(str(exc))

    print("VRAM budget:")
    for b in plan.budgets:
        mark = "ok " if b.fits else "OVER"
        print(
            f"  [{mark}] {b.device} (card {b.card}): {b.requested_mib:>8.0f} / "
            f"{b.available_mib:>8.0f} MiB usable  ({b.total_mib:.0f} total, "
            f"{b.reserve_mib:.0f} reserved)"
        )
        for c in sorted(b.charges, key=lambda c: -c.mib):
            share = "" if c.share == 1.0 else f"  [{c.share:.0%} of split]"
            print(f"          {c.model:<24} {c.mib:>8.0f} MiB  {c.lanes} lane(s){share}")
    print(f"\n--models-max must be >= {plan.models_max}")
    if plan.needs_router_restart:
        print(f"\n!! ROUTER RESTART REQUIRED: {plan.restart_reason}", file=sys.stderr)
    print(f"\nunload:  {plan.to_unload or '-'}")
    print(f"load:    {plan.to_load or '-'}")
    print(f"changed: {plan.changed or '-'}   (preset differs; reload=1 drops it, so it is reloaded)")
    print(f"keep:    {plan.unchanged or '-'}")
    for child in crashed_children(models):
        print(f"!! CRASHED: {child.name} (exit_code={child.exit_code})", file=sys.stderr)
    check = check_router_preset(render_preset_ini(placement, specs))
    print(f"\n{check.describe()}")
    if check.blocks_apply:
        print("!! apply would REFUSE: the router would reload a preset other than the one budgeted.", file=sys.stderr)
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    placement, specs = load_config()
    models = router_models(args.router)
    try:
        plan = plan_reconcile(
            placement,
            specs,
            loaded_now=loaded_names(models),
            router_models_max=router_models_max(models),
            previous_ini=running_presets(models) or _current_ini(),
        )
    except BudgetError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except PlacementError as exc:
        die(str(exc))

    for child in crashed_children(models):
        # Say it before reconciling: after the reload the evidence is gone, and a
        # child that died on its own is a different problem from a stale config.
        print(
            f"!! [reconcile] '{child.name}' CRASHED (exit_code={child.exit_code}). "
            f"It is being reloaded, but find out why before trusting it.",
            file=sys.stderr,
        )

    if plan.needs_router_restart:
        # Refuse rather than restart: a restart drops every resident model, which
        # is the operator's call to make, not this tool's.
        die(f"{plan.restart_reason}\n   Restart the router with --models-max {plan.models_max}, then re-run apply.")

    wanted_ini = render_preset_ini(placement, specs)
    deadline = Deadline(args.timeout)

    # Decide the preset route BEFORE any unload. When the router's file is
    # read-only here, ?reload=1 loads whatever that file says. If it is not the
    # preset that was budgeted, stop now. "Read-only" includes a writable file in
    # a dir that this process cannot write: the temp file needs the dir.
    check = check_router_preset(wanted_ini)
    if check.blocks_apply:
        local = local_ini_path()
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(wanted_ini)
        die(
            f"{check.describe()}.\n"
            f"   The router would reload that file, not the preset that was budgeted. Nothing was changed.\n"
            f"   The declared preset is rendered at {local}. Install it as the router user, then re-run apply."
        )

    # A reload with nothing to change touches nothing: no preset write, no
    # ?reload=1, no verify, no unload. The router keeps every model that it has.
    if not plan.to_unload and not plan.to_load and check.matches is True:
        print(">> nothing to change: every declared model is resident with the declared preset.")
        print(">> verify and unload skipped; no model was touched.")
        return 0

    before = device_snapshot(placement)

    # Undeclared models come out FIRST, while the router still knows them.
    # Regenerating the INI removes them from it, and ?reload=1 then drops them
    # from the registry — after which unloading them by name is a 400.
    for name in plan.to_unload:
        _req(
            "POST", f"{args.router}/models/unload", {"model": name}, timeout=deadline.cap(UNLOAD_S), tolerate_http=True
        )
        print(f">> unloaded {name}  (not declared in YAML)")

    if check.writable:
        path = check.path
        _write_atomic(path, wanted_ini)
        print(f">> wrote {path}")
    else:
        print(f">> router preset {check.path} is read-only here and matches the declared preset; not rewritten")

    # The models that this run changes: the ones that are missing or whose preset
    # differs. A failed apply unloads these and no other. A model that the run did
    # not change was proven before, and it stays resident.
    changed_here = list(plan.to_load)

    def unload_changed(why: str) -> None:
        if not changed_here:
            print(f"!! {why}. This run changed no model, so none is unloaded.", file=sys.stderr)
            return
        print(f"!! {why}. Unloading the models that this run changed: {changed_here}.", file=sys.stderr)
        cleanup = Deadline(CLEANUP_S)
        for name in changed_here:
            try:
                done = _req(
                    "POST",
                    f"{args.router}/models/unload",
                    {"model": name},
                    timeout=min(UNLOAD_S, cleanup.cap(UNLOAD_S)),
                    tolerate_http=True,
                )
            except (Exception, SystemExit) as exc:
                print(f"!! could not unload {name}: {exc}", file=sys.stderr)
                continue
            if done is None:
                print(f">> {name} was not loaded; nothing to unload", file=sys.stderr)
            else:
                print(f">> unloaded {name} (this run changed it, and the reload then failed)", file=sys.stderr)

    def load_and_verify() -> int:
        # Reload may start a load-on-startup preset.
        _req("GET", f"{args.router}/models?reload=1", timeout=deadline.cap(30.0))
        print(">> reloaded presets")
        current = router_models(args.router)
        loading = [
            name
            for name in plan.to_load
            if any(m.get("id") == name and (m.get("status") or {}).get("value") == "loading" for m in current)
        ]
        if loading:
            _wait_loaded(args.router, loading, deadline=deadline)
            current = router_models(args.router)

        already_current = models_with_matching_presets(current, plan.to_load, wanted_ini)
        still_up = set(loaded_names(current))
        explicit_loads: list[str] = []
        for name in plan.to_load:
            if name in already_current:
                print(f">> {name} already loaded with the declared preset")
                continue
            if name in still_up:
                _req("POST", f"{args.router}/models/unload", {"model": name}, timeout=deadline.cap(UNLOAD_S))
                print(f">> unloaded {name}  (running preset differs from YAML)")
            _req("POST", f"{args.router}/models/load", {"model": name}, timeout=deadline.cap(600.0))
            print(f">> loading {name} ...")
            explicit_loads.append(name)

        if explicit_loads:
            _wait_loaded(args.router, explicit_loads, deadline=deadline)
        observed = router_models(args.router)
        result = _verify(placement, before, plan, observed)
        if result:
            return result
        current = router_models(args.router)
        before_proofs = {item.get("id"): item.get("inferference_residency") for item in observed}
        after_proofs = {item.get("id"): item.get("inferference_residency") for item in current}
        if any(before_proofs.get(r.model) != after_proofs.get(r.model) for r in placement.resident):
            print("!! residency evidence changed during verification", file=sys.stderr)
            return 1
        try:
            _residency_evidence(placement, current)
        except (ValueError, TypeError, KeyError) as exc:
            print(f"!! placement evidence invalid: {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        result = load_and_verify()
    except BaseException:
        unload_changed("the reload failed before the placement was proven")
        raise
    if result:
        unload_changed("the placement was not proven")
    return result


def _wait_loaded(router: str, names: list[str], deadline: Deadline) -> None:
    """Wait until every name in ``names`` has left the `loading` state.

    The wait ends at the shared ``deadline`` of the whole apply. A second wait in
    the same apply gets only the time that the first one left.
    """
    while not deadline.expired():
        models = {m["id"]: m for m in router_models(router)}
        pending = [n for n in names if models.get(n, {}).get("status", {}).get("value") == "loading"]
        if not pending:
            bad = [
                f"{n} ({models.get(n, {}).get('status', {}).get('value')}, "
                f"exit_code={models.get(n, {}).get('status', {}).get('exit_code')})"
                for n in names
                if models.get(n, {}).get("status", {}).get("value") != "loaded"
            ]
            if bad:
                die(f"model(s) failed to load: {', '.join(bad)}")
            return
        time.sleep(min(2.0, max(0.0, deadline.remaining())))
    die(f"the apply deadline of {deadline.seconds:g}s passed while waiting for {names} to load")


def _pci_address(card: int) -> str:
    """The PCI address sysfs gives for a DRM card, as the device's identity."""
    try:
        return Path(f"/sys/class/drm/card{card}/device").resolve().name
    except OSError:
        return "?"


def _residency_evidence(placement: Placement, models: list[dict[str, Any]]) -> dict[str, int]:
    """Validate each child's load-time proof and return GTT limits in bytes."""
    by_name = {model.get("id"): model for model in models}
    devices = {device.id: device for device in placement.devices}
    latest: dict[str, tuple[int, int]] = {}
    for resident in placement.resident:
        model = by_name.get(resident.model)
        status = model.get("status") if isinstance(model, dict) else None
        if not isinstance(status, dict) or status.get("value") != "loaded":
            raise ValueError(f"{resident.model}: model is not loaded")
        assert isinstance(model, dict)
        evidence = model.get("inferference_residency")
        if not isinstance(evidence, dict) or not isinstance(evidence.get("cards"), list):
            raise ValueError(f"{resident.model}: no server residency evidence")
        count = evidence.get("layer_count")
        if type(count) is not int or count <= 0:
            raise ValueError(f"{resident.model}: invalid layer count")
        verified_at = evidence.get("verified_mono_us")
        if type(verified_at) is not int or verified_at <= 0:
            raise ValueError(f"{resident.model}: invalid evidence timestamp")
        seen: set[str] = set()
        layers = 0
        for card in evidence["cards"]:
            if not isinstance(card, dict):
                raise ValueError(f"{resident.model}: invalid card evidence")
            name = card.get("device")
            if not isinstance(name, str) or name in seen or name not in resident.devices or name not in devices:
                raise ValueError(f"{resident.model}: unexpected or duplicate device")
            seen.add(name)
            if card.get("pci") != _pci_address(devices[name].card):
                raise ValueError(f"{resident.model}: PCI address mismatch for {name}")
            values = [card.get(key) for key in ("layers", "gtt_before", "gtt_after", "vram_before", "vram_after")]
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"{resident.model}: invalid residency counter")
            n_layers, gtt_before, gtt_after, vram_before, vram_after = values
            if n_layers == 0 or gtt_after > gtt_before or vram_after <= vram_before:
                raise ValueError(f"{resident.model}: residency failed for {name}")
            layers += n_layers
            previous = latest.get(name)
            if previous is not None and previous[0] == verified_at:
                raise ValueError(f"{resident.model}: ambiguous evidence order for {name}")
            if previous is None or verified_at > previous[0]:
                latest[name] = (verified_at, gtt_after)
        output_device = evidence.get("output_device")
        if (
            not isinstance(output_device, str)
            or seen != set(resident.devices)
            or layers != count
            or output_device not in seen
        ):
            raise ValueError(f"{resident.model}: incomplete layer placement")
    return {name: value for name, (_time, value) in latest.items()}


def _verify(
    placement: Placement, before: dict[str, tuple[float, float]], plan: Any, models: list[dict[str, Any]]
) -> int:
    """Prove the placement landed in VRAM, not in host memory.

    Three independent checks per device, because each one alone can pass on a
    spilled model:

    * GTT did NOT rise. This is the one that catches the silent spill: a model in
      host memory still answers, still reports every layer on the GPU, and still
      returns success.
    * VRAM DID rise, by roughly what the measured footprint said it would. A
      model that never loaded also leaves GTT flat, so flat GTT on its own proves
      nothing.
    * The card is identified by its PCI address, not by a log line. The log
      format has already changed once under this project.

    Do NOT read `CPU_Mapped model buffer size` as a spill signal — it is normal
    for the non-offloaded embedding weights of a perfectly resident model.
    """
    # amdgpu does not return freed VRAM to its own accounting instantly. Measured
    # 2026-09-05: a card read straight after an unload still showed 14083 MiB and
    # settled to 8874 within 20s. Snapshotting too early does not break the
    # floor check, but it prints a figure that is not what the card holds.
    time.sleep(SETTLE_S)
    after = device_snapshot(placement)
    expected = {b.device: b.requested_mib for b in plan.budgets}
    try:
        limits = _residency_evidence(placement, models)
    except (ValueError, TypeError, KeyError) as exc:
        print(f"!! placement evidence invalid: {exc}", file=sys.stderr)
        return 1
    ok = True
    print("\nplacement check:")
    for dev in placement.devices:
        v0, g0 = before[dev.id]
        v1, g1 = after[dev.id]
        if not all(math.isfinite(value) and value >= 0 for value in (v0, g0, v1, g1)):
            print(f"!! {dev.id}: invalid VRAM or GTT counter; placement is not proven", file=sys.stderr)
            ok = False
            continue
        dv, dg = v1 - v0, g1 - g0
        want = expected.get(dev.id, 0.0)
        gtt_increased = dev.id in limits and g1 * 1048576 > limits[dev.id]
        # Compare the ABSOLUTE figure, not the change over this reconcile. A
        # model that was already resident does not move, so its charge never
        # appears in the delta — judging by the delta would call a correct
        # no-op reconcile a failure. The absolute figure also catches the case
        # the delta cannot see at all: a model that was never loaded.
        short = want > 0 and v1 < 0.75 * want
        bad = gtt_increased or short
        ok = ok and not bad
        flag = "GTT RISE" if gtt_increased else ("SHORT" if short else "ok")
        print(
            f"  [{flag:>7}] {dev.id} (card {dev.card}, pci {_pci_address(dev.card)}): "
            f"VRAM {v0:.0f} -> {v1:.0f} MiB (change {dv:+.0f}, declared placement needs {want:.0f}), "
            f"GTT {g0:.6f} -> {g1:.6f} MiB ({dg:+.6f})"
        )
        if short and not gtt_increased:
            print(
                f"!! {dev.id} holds {v1:.0f} MiB but its declared placement needs {want:.0f} MiB. "
                f"A model is not where this placement says it is.",
                file=sys.stderr,
            )
        if gtt_increased:
            print(
                f"!! {dev.id}: GTT is {int(g1 * 1048576) - limits[dev.id]} bytes above "
                "the latest load baseline. VRAM residency is not proven.",
                file=sys.stderr,
            )
    if not ok:
        print("\n!! PLACEMENT NOT PROVEN — see the device line(s) above.", file=sys.stderr)
        return 1
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    placement, _ = load_config()
    models = router_models(args.router)
    declared = {r.model: r for r in placement.resident}
    print(f"{'model':<24} {'status':<10} {'declared on':<22} health")
    for child in sorted(child_health(models), key=lambda c: c.name):
        # A crashed child reports `unloaded` WITH an exit code; a deliberate
        # unload leaves it null. That difference is the only health signal.
        health = f"CRASHED (exit_code={child.exit_code})" if child.crashed else ""
        where = ",".join(declared[child.name].devices) if child.name in declared else "(not declared)"
        print(f"{child.name:<24} {child.status:<10} {where:<22} {health}")
    missing = sorted({r.model for r in placement.resident} - {c.name for c in child_health(models) if c.loaded})
    if missing:
        print(f"\n!! DECLARED BUT NOT RESIDENT: {missing} — run `apply`.")
    snap = device_snapshot(placement)
    print("\ndevices:")
    for dev in placement.devices:
        v, g = snap[dev.id]
        print(f"  {dev.id} (card {dev.card}): VRAM {v:.0f} MiB, GTT {g:.0f} MiB")
    return 0


def cmd_evict(args: argparse.Namespace) -> int:
    """Free a card for other work. The placement config is left untouched, so a
    later `apply` puts everything back."""
    placement, _ = load_config()
    ids = [d.id for d in placement.devices]
    if args.device not in ids:
        die(f"unknown device '{args.device}'. Known: {ids}")
    on_device = {r.model for r in placement.resident if args.device in r.devices}
    models = router_models(args.router)
    live = [m["id"] for m in models if m["id"] in on_device and m.get("status", {}).get("value") == "loaded"]
    if not live:
        print(f">> nothing resident on {args.device}")
        return 0
    for name in live:
        _req("POST", f"{args.router}/models/unload", {"model": name}, timeout=120)
        print(f">> evicted {name} from {args.device}")
    time.sleep(3)
    snap = device_snapshot(placement)
    v, g = snap[args.device]
    print(f">> {args.device} now: VRAM {v:.0f} MiB, GTT {g:.0f} MiB")
    print(">> placement config untouched — `apply` restores it.")
    return 0


def cmd_config_dir(args: argparse.Namespace) -> int:
    """Print the config dir that this tool reads, and nothing else, on stdout.

    A launcher calls this to follow the CLI default. The reason goes to stderr,
    as for every other command. This command reads no file and asks no router.
    """
    announce_config_dir()
    print(config_dir())
    return 0


def cmd_preset_ini(args: argparse.Namespace) -> int:
    """Print the preset file path that this tool uses, and nothing else, on stdout.

    The router launcher calls this to follow the CLI default. A note goes to
    stderr when the path is the caller's state dir. This command reads no file
    and asks no router.
    """
    announce_config_dir()
    path, how = resolve_ini_path()
    if how == "state":
        print(
            f">> the config dir {config_dir()} is not writable here or is inside /etc, "
            f"so the preset goes to the caller's state dir (set $INFERFERENCE_PRESET_INI to choose)",
            file=sys.stderr,
        )
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_router = _default_router()
    ap.add_argument("--router", default=default_router, help=f"router base URL (default {default_router})")
    ap.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="one deadline in seconds for all load waits and router calls of `apply`, not one per wait",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_render = sub.add_parser("render", help="print (or write) the generated preset INI")
    p_render.add_argument(
        "--out",
        help="write to this path instead of stdout; use this rather than a shell redirect",
    )
    p_plan = sub.add_parser("plan", help="show the budget and what would change")
    p_plan.add_argument(
        "--offline",
        dest="router_reachable",
        action="store_false",
        default=True,
        help="budget without asking the router what is loaded",
    )
    sub.add_parser("apply", help="reconcile the router to the YAML")
    sub.add_parser("status", help="what is resident now, and its health")
    p_evict = sub.add_parser("evict", help="unload everything on a device")
    p_evict.add_argument("device", help="device id, e.g. Vulkan1")
    sub.add_parser("config-dir", help="print the config dir that this tool reads")
    sub.add_parser("preset-ini", help="print the preset file path that this tool uses")

    args = ap.parse_args(argv)
    return {
        "render": cmd_render,
        "plan": cmd_plan,
        "apply": cmd_apply,
        "status": cmd_status,
        "evict": cmd_evict,
        "config-dir": cmd_config_dir,
        "preset-ini": cmd_preset_ini,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
