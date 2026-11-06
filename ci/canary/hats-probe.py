#!/usr/bin/env python3
"""Hats (loop-routing) capability probe.

Probes ``inferference.seq_routing`` against a pointed-at lib dir
(``LLAMA_CPP_LIB_PATH``):

  * hats build (``feat/p2-nanbeige-hats``, e.g. ``out-cpu-light-be11e36ac``):
    both ``llama_set_loop_adapters`` / ``llama_set_loop_adapter`` resolve and
    :class:`LoopRoutingBinding` binds (exit 0).
  * integration build (``out-p2fork-2dad55136``): the surface is ABSENT — the
    probe reports ``absent`` and :class:`LoopRoutingUnavailable` raises at BIND
    time, never at import (exit 0 — absence is the expected, fail-closed state).

The bindgen cdef is deliberately NOT extended; this
probe is the hats surface's proof, by direct symbol resolution.

Usage:
    ci/library/.venv/bin/python ci/canary/hats-probe.py [--lib-dir <dir>]
    env: LLAMA_CPP_LIB_PATH (default), LD_LIBRARY_PATH must let llama_cpp load
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from inferference.seq_routing import (
    LOOP_ROUTING_SYMBOLS,
    LoopRoutingBinding,
    LoopRoutingUnavailable,
    library_supports_loop_routing,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib-dir", type=Path, help="lib dir (default: $LLAMA_CPP_LIB_PATH)")
    ap.add_argument("--expect", choices=["present", "absent"], required=True)
    args = ap.parse_args()

    lib_dir = args.lib_dir or Path(os.environ.get("LLAMA_CPP_LIB_PATH", ""))
    if not lib_dir.is_dir():
        raise SystemExit(f"lib dir not found: {lib_dir}")

    # Import happens BEFORE the lib is loaded — the surface must be import-safe.
    import llama_cpp  # noqa: F401

    lib = llama_cpp.llama_cpp._lib
    if lib is None:
        raise SystemExit("llama_cpp loaded no _lib")

    present = library_supports_loop_routing(lib)
    print(f"[hats] lib dir: {lib_dir}")
    print(f"[hats] LOOP_ROUTING_SYMBOLS: {', '.join(LOOP_ROUTING_SYMBOLS)}")
    print(f"[hats] library_supports_loop_routing: {present}")

    if args.expect == "present":
        if not present:
            raise SystemExit(f"[hats] FAIL: expected hats symbols present, got absent ({lib_dir})")
        binding = LoopRoutingBinding(llama_cpp.llama_cpp)
        print(f"[hats] LoopRoutingBinding bound OK: {type(binding).__name__}")
        print("[hats] PASS")
        return 0

    # expect absent: binding must raise LoopRoutingUnavailable, never another error.
    if present:
        raise SystemExit(f"[hats] FAIL: expected hats symbols absent, got present ({lib_dir})")
    try:
        LoopRoutingBinding(llama_cpp.llama_cpp)
    except LoopRoutingUnavailable as exc:
        print(f"[hats] LoopRoutingUnavailable raised at bind (absent-safe): {exc}")
        print("[hats] PASS (absent — expected on integration/stock)")
        return 0
    raise SystemExit("[hats] FAIL: LoopRoutingBinding did not raise LoopRoutingUnavailable")


if __name__ == "__main__":
    sys.exit(main())
