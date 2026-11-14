#!/usr/bin/env python3
"""Synthesize a nanbeige LoRA adapter GGUF for the hats GPU validation (S3/S5).

S3 (P2 seq-routing on a nanbeige base) and S5 (per-pass-hat behavior) need
converted nanbeige LoRA adapters, and none exist on disk (the probe LoRAs are
shaped for another architecture). The new
stock's conversion tooling (convert_lora_to_gguf.py) requires a HF-format
adapter checkpoint; this script instead writes the adapter GGUF directly with
gguf-py's GGUFWriter (the same writer convert_lora_to_gguf.py uses), which is
the honest in-scope equivalent: a synthetic adapter with the exact tensor
naming/shape contract llama_adapter_lora_init enforces
(llama-adapter.cpp:206-230: general.type=adapter, general.architecture must
match the model arch, adapter.type=lora, adapter.lora.alpha f32; :355-366 the
shape check model.ne[0]==lora_a.ne[0], model.ne[1]==lora_b.ne[1],
lora_a.ne[1]==lora_b.ne[0]).

The synthetic deltas are behavioral test inputs, not trained adapters: S5
asserts mechanism properties (reproducible, differs from base / single hats,
near-identity ≈ base), none of which depend on adapter quality. alpha / rank
drive the effective scale (llama-adapter.h:53-57: scale = adapter_scale *
alpha / rank; the hats branch passes adapter_scale=1.0f).

Usage:
    ci/library/.venv/bin/python ci/canary/synth_nanbeige_lora.py \
        --base-model <Nanbeige GGUF> --out <out.gguf> \
        --alpha 64 --seed 7 [--weight-scale 1.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

GGUF_PY = Path(__file__).resolve().parents[2] / "ci/build/.llamacpp-builds/src-llama.cpp/gguf-py"
sys.path.insert(0, str(GGUF_PY))

from gguf import GGUFWriter  # noqa: E402
from gguf.constants import GGUFType, Keys  # noqa: E402

# The physical layers of the nanbeige test model (22) and the weights a LoRA may target
# (the nanbeige GGUF's blk.N.* set, from gguf-py inspection of the model).
# Overridable for other arches (for example 32 layers) via --n-layers/--arch.
N_PHYS = 22
ARCH = "nanbeige"
TARGET_WEIGHTS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
RANK = 8


def base_shapes(reader: object) -> dict[str, list[int]]:
    """Base model GGUF tensor shapes (ne = [out, in] per tensor)."""
    return {t.name: list(t.shape) for t in reader.tensors}  # type: ignore[attr-defined]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        help="multiplier on the synthesized lora_a/lora_b weights (near-identity => ~1e-3)",
    )
    ap.add_argument("--name", default="nanbeige-hat-probe")
    ap.add_argument("--subset", default=None, help="limit to e.g. 'blk.3:attn_k+attn_v+attn_output' (debug)")
    ap.add_argument("--layers", type=int, default=None, help="limit to the first N physical layers (bisection ladder)")
    ap.add_argument("--weights", default=None, help="comma-separated subset of TARGET_WEIGHTS (default: all)")
    ap.add_argument(
        "--up-all-gate-n",
        type=int,
        default=None,
        help="bisection: ffn_up on ALL layers + ffn_gate on the first N layers",
    )
    ap.add_argument(
        "--n-layers", type=int, default=N_PHYS, help=f"physical layer count of the base model (default {N_PHYS})"
    )
    ap.add_argument("--arch", default=ARCH, help=f"adapter general.architecture, must match the model (default {ARCH})")
    args = ap.parse_args()

    from gguf import GGUFReader  # noqa: E402  (after sys.path insert above)

    reader = GGUFReader(str(args.base_model))
    shapes = base_shapes(reader)

    rng = np.random.default_rng(args.seed)
    # GGUFWriter(path, arch): the arch arg becomes general.architecture, which the
    # adapter loader matches against the model's arch (llama-adapter.cpp:220-224) —
    # it must be the model arch string ("nanbeige" / "qwen35" etc.).
    writer = GGUFWriter(str(args.out), args.arch)

    writer.add_type(GGUFType.ADAPTER)
    writer.add_string(Keys.Adapter.TYPE, "lora")
    writer.add_float32(Keys.Adapter.LORA_ALPHA, args.alpha)
    writer.add_architecture()
    writer.add_name(args.name)

    n_pairs = 0
    subset = {}
    if args.subset:
        for item in args.subset.split(","):
            blk, ws = item.split(":")
            subset[blk] = ws.split("+")
    wanted_weights = set(TARGET_WEIGHTS if not args.weights else args.weights.split(","))
    unknown = wanted_weights - set(TARGET_WEIGHTS)
    if unknown:
        print(f"!! unknown weights: {sorted(unknown)} (valid: {TARGET_WEIGHTS})", flush=True)
        return 2
    gate_only_on = None
    if args.up_all_gate_n is not None:
        if args.weights is not None:
            print("!! --up-all-gate-n cannot be combined with --weights", flush=True)
            return 2
        wanted_weights = {"ffn_up", "ffn_gate"}
        gate_only_on = args.up_all_gate_n
    n_phys = args.n_layers
    for i in range(n_phys):
        if args.layers is not None and i >= args.layers:
            continue
        for w in TARGET_WEIGHTS:
            if w not in wanted_weights:
                continue
            if gate_only_on is not None and w == "ffn_gate" and i >= gate_only_on:
                continue
            if subset and (f"blk.{i}" not in subset or w not in subset[f"blk.{i}"]):
                continue
            base = f"blk.{i}.{w}.weight"
            if base not in shapes:
                print(f"!! skip {base}: not in base model", flush=True)
                continue
            ne0, ne1 = int(shapes[base][0]), int(shapes[base][1])
            # GGUFWriter transposes numpy -> stored ne (verified: numpy (2,3) reads
            # back as [3,2]). The adapter loader (llama-adapter.cpp:355-366)
            # requires a.ne[0]==model.ne[0], b.ne[1]==model.ne[1],
            # a.ne[1]==b.ne[0]. With the writer transposing, numpy lora_a
            # (rank, ne0) stores ne [ne0, rank] and numpy lora_b (ne1, rank)
            # stores ne [rank, ne1] — exactly the shapes of the working probe adapters
            # (reader: lora_a [in, rank] / lora_b [rank, out]).
            lora_a = rng.standard_normal((args.rank, ne0)).astype(np.float32) * args.weight_scale
            lora_b = rng.standard_normal((ne1, args.rank)).astype(np.float32) * args.weight_scale
            writer.add_tensor(f"{base}.lora_a", lora_a)
            writer.add_tensor(f"{base}.lora_b", lora_b)
            n_pairs += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = args.out.stat().st_size / 1e6
    print(
        f">> wrote {args.out} ({n_pairs} lora pairs, rank {args.rank}, alpha {args.alpha}, "
        f"weight_scale {args.weight_scale}, {size:.1f} MiB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
