"""Resolve one server role's YAML config into a llama-server argv.

The runtime configuration lives in YAML under ``~/.config/inferference/`` and NOT
in the Nix module. That is deliberate: editing a model path or a pool should be
`$EDITOR` plus `systemctl --user restart`, never a NixOS rebuild. The module
wires the unit; this file decides what the unit runs.

    inferference-config.py <role> [--config-dir DIR] [--print-argv|--print-summary]

Reads ``<config-dir>/<role>.yaml``. Writes the argv, one item per line, to
stdout. ``ci/runner/inferference-server-launch.sh`` execs it.

Schema — every key optional except ``port``:

    port: 8081                  # REQUIRED. No default: two roles on one host
                                # must be given distinct ports by the operator,
                                # not agree by coincidence.
    model: /path/model.gguf     # default: $MODEL_<ROLE> from rig-env.sh
    host: 127.0.0.1
    ctx: 16384
    parallel: 8
    ngl: 999                    # GPU layers
    routing: false              # --lora-seq-routing. Default OFF, see below.
    loras:                      # either form, freely mixed
      - /path/a.gguf
      - {path: /path/b.gguf, scale: 1.0}
    extra_args: ["--verbose"]

**Routing defaults to OFF, and that is a measured decision.** Per-sequence
routing costs traffic that names a single adapter: 0.96x at 8 slots, 0.91x at 16,
0.89x at 32, because each token carries its own mask and the cost
grows with batch width. It wins only when requests in flight name DIFFERENT
adapters — 1.6x-2.0x there. Turn it on for adapter-mixed traffic and leave it off
otherwise.

**YAML is the ONLY source of runtime configuration.** No environment variable
overrides any key above. A second mechanism means a server can run a port one
file names and another file does not, and no one can tell which won by reading
either file. The only environment this file reads are ``$INFERFERENCE_CONFIG_DIR``
(where the YAML lives, not what is in it) and ``$MODEL_<ROLE>`` (the rig's own
inventory from ``rig-env.sh``, which ``model:`` overrides).

**Fail-closed throughout.** A missing config file, an unreadable model, a missing
adapter, a bad type or an unknown key is EX_CONFIG (78) with the reason. systemd
reports 78 distinctly, so a misconfiguration never looks like a crashed server. A
server that starts without an adapter it was told to load would answer requests
with the wrong weights and no error anywhere.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, NoReturn

import yaml

EX_CONFIG = 78

KNOWN_KEYS = {"port", "model", "host", "ctx", "parallel", "ngl", "routing", "loras", "extra_args"}

DEFAULTS: dict[str, Any] = {
    "host": "127.0.0.1",
    "ctx": 16384,
    "parallel": 8,
    "ngl": 999,
    "routing": False,
}


def die(msg: str) -> NoReturn:
    print(f"!! [inferference-config] {msg}", file=sys.stderr)
    raise SystemExit(EX_CONFIG)


def load(role: str, config_dir: Path) -> dict[str, Any]:
    path = config_dir / f"{role}.yaml"
    if not path.is_file():
        die(
            f"no config at {path}. Create it — see ci/runner/server-role.example.yaml. "
            f"The unit reads YAML at start, so this is an edit-and-restart, not a NixOS rebuild."
        )
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        die(f"{path} is not valid YAML: {exc}")
    if raw is None:
        die(f"{path} is empty; it must at least set `port`.")
    if not isinstance(raw, dict):
        die(f"{path} must be a mapping at the top level, got {type(raw).__name__}.")

    # An unknown key is a typo, and a typo silently ignored is a server running a
    # configuration nobody wrote. Refuse instead.
    unknown = {str(k) for k in raw} - KNOWN_KEYS
    if unknown:
        die(f"{path} has unknown key(s) {sorted(unknown)}. Known keys: {sorted(KNOWN_KEYS)}.")

    cfg = {**DEFAULTS, **raw}
    cfg["_path"] = path
    return cfg


def _int(cfg: dict[str, Any], key: str, role: str) -> int:
    val = cfg.get(key)
    if isinstance(val, bool) or not isinstance(val, int):
        die(f"{cfg['_path']}: `{key}` must be an integer, got {val!r}")
    return int(val)


def resolve_loras(cfg: dict[str, Any]) -> list[tuple[Path, float | None]]:
    raw = cfg.get("loras") or []
    if not isinstance(raw, list):
        die(f"{cfg['_path']}: `loras` must be a list, got {type(raw).__name__}")
    out: list[tuple[Path, float | None]] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            path, scale = Path(item).expanduser(), None
        elif isinstance(item, dict):
            if "path" not in item:
                die(f"{cfg['_path']}: loras[{i}] is a mapping with no `path`")
            path = Path(str(item["path"])).expanduser()
            scale = float(item["scale"]) if "scale" in item else None
        else:
            die(f"{cfg['_path']}: loras[{i}] must be a path or a mapping, got {type(item).__name__}")
        if not path.is_file():
            die(
                f"{cfg['_path']}: loras[{i}] does not exist: {path}. Refusing to start a server "
                f"that would answer with weights it was not given."
            )
        out.append((path, scale))
    return out


def build_argv(role: str, cfg: dict[str, Any], binary: Path) -> list[str]:
    # YAML is the ONLY source of runtime configuration. There is no environment
    # override for any key here: two mechanisms mean a server can run a port one
    # file names and another file does not, and the operator cannot tell which
    # won by reading either file.
    port = cfg.get("port")
    if port is None:
        die(f"{cfg['_path']}: `port` is required and has no default — two roles on one host must not collide.")

    # $MODEL_<ROLE> is not a second config mechanism; it is the rig's own
    # inventory (rig-env.sh), and `model:` in the YAML overrides it.
    model = cfg.get("model") or os.environ.get(f"MODEL_{role.upper()}")
    if not model:
        die(
            f"{cfg['_path']}: no `model`, and $MODEL_{role.upper()} is not set "
            f"(rig-env.sh supplies the rig's default for this role)."
        )
    model_path = Path(str(model)).expanduser()
    if not model_path.is_file():
        die(f"model is not readable: {model_path}")

    argv = [
        str(binary),
        "--model",
        str(model_path),
        "--host",
        str(cfg["host"]),
        "--port",
        str(port),
        "--n-gpu-layers",
        str(_int(cfg, "ngl", role)),
        "--parallel",
        str(_int(cfg, "parallel", role)),
        "--ctx-size",
        str(_int(cfg, "ctx", role)),
        "--metrics",
    ]

    loras = resolve_loras(cfg)
    for path, scale in loras:
        if scale is None:
            argv += ["--lora", str(path)]
        else:
            argv += ["--lora-scaled", str(path), str(scale)]

    routing = cfg.get("routing")
    if not isinstance(routing, bool):
        die(f"{cfg['_path']}: `routing` must be true or false, got {routing!r}")
    if routing:
        if not loras:
            die(
                f"{cfg['_path']}: `routing: true` with no `loras` — routing with an empty pool costs and gains nothing."
            )
        argv.append("--lora-seq-routing")
        # R-4: the fork refuses routing with a live prompt cache, because the
        # cache is keyed on prompt tokens alone and would serve one adapter's KV
        # to another adapter's request (R9, live in stock too). Set it here so
        # the operator does not have to know that.
        argv += ["--cache-ram", "0"]

    extra = cfg.get("extra_args") or []
    if not isinstance(extra, list):
        die(f"{cfg['_path']}: `extra_args` must be a list, got {type(extra).__name__}")
    argv += [str(a) for a in extra]
    return argv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role")
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--print-summary", action="store_true", help="human-readable, to stderr")
    args = ap.parse_args()

    config_dir = args.config_dir or Path(
        os.environ.get("INFERFERENCE_CONFIG_DIR", Path.home() / ".config" / "inferference")
    )
    cfg = load(args.role, config_dir)
    argv = build_argv(args.role, cfg, args.binary)

    if args.print_summary:
        loras = resolve_loras(cfg)
        routed = "--lora-seq-routing" in argv
        print(f">>   config          {cfg['_path']}", file=sys.stderr)
        print(f">>   port            {argv[argv.index('--port') + 1]}", file=sys.stderr)
        print(f">>   model           {argv[argv.index('--model') + 1]}", file=sys.stderr)
        print(f">>   ctx / parallel  {cfg['ctx']} / {cfg['parallel']}", file=sys.stderr)
        print(f">>   adapters        {len(loras)}", file=sys.stderr)
        print(
            f">>   routing         {'ON' if routed else 'off'}"
            f"{'' if routed else '  (costs 0.89x-0.96x on single-adapter traffic; on only helps mixed)'}",
            file=sys.stderr,
        )

    print("\n".join(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
