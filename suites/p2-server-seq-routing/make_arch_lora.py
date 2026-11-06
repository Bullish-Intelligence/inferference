"""Synthesize the three adapter sets the routing smoke gate needs, for any model.

This generalises an earlier adapter synthesizer: that script took `--block` and
one target set, and a caller had to know the model's block count. This one
derives both from the GGUF and writes all three sets `routing_smoke.py` looks
for:

    last-ffn-{0,1}.gguf    blk.LAST  ffn_up, ffn_gate, ffn_down
    mid-ffn-{0,1}.gguf     blk.MID   ffn_up, ffn_gate, ffn_down
    last-attn-{0,1}.gguf   blk.LAST  attn_q, attn_k, attn_v, attn_output

The GGUF LoRA format is four keys and a pair of tensors per target. For a base
weight of ggml shape [ne0, ne1], `lora_a` is [ne0, rank] and `lora_b` is
[rank, ne1]; gguf-py writes `reversed(numpy.shape)`, so the numpy arrays are the
transpose of that.

Weights are random. These adapters are shape probes, not quality claims: the
gate asks whether the graph builds and decodes, and no output is read for sense.

`--sigma` is the perturbation size. It has to be large enough that the adapter
reaches the graph at all - a silently dropped adapter also produces "no abort",
and produces it for the wrong reason. 0.02 is the measured default.

Run inside `devenv shell`.
"""

from __future__ import annotations

import argparse
import re
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _gguf():
    for cand in ("src-p2fork", "src-llama.cpp", "src-p2fork-measure"):
        p = REPO / "ci/build/.llamacpp-builds" / cand / "gguf-py"
        if p.is_dir():
            sys.path.insert(0, str(p))
            import gguf  # ty: ignore[unresolved-import]

            return gguf
    raise SystemExit("no gguf-py found under ci/build/.llamacpp-builds/*/gguf-py")


RANK = 16
ALPHA = 64.0
SIGMA = 0.02

FFN_TARGETS = ("ffn_up", "ffn_gate", "ffn_down")
ATTN_TARGETS = ("attn_q", "attn_k", "attn_v", "attn_output")


def model_shape(reader) -> tuple[str, int, int, dict[str, list[int]]]:
    arch = reader.fields["general.architecture"].contents()
    shapes = {t.name: [int(d) for d in t.shape] for t in reader.tensors}
    blocks = {int(m.group(1)) for n in shapes if (m := re.fullmatch(r"blk\.(\d+)\.\w+\.weight", n))}
    if not blocks:
        raise SystemExit("no blk.N.*.weight tensors; not a block-structured model")
    last = max(blocks)
    return arch, last, last // 2, shapes


def pick(shapes: dict[str, list[int]], block: int, targets: tuple[str, ...]) -> dict[str, list[int]]:
    """The base weights this set targets. A missing target is a loud failure.

    A quietly reduced target set would produce an adapter that touches nothing
    on the block under test, and the gate would then pass for the wrong reason.
    """
    want = {f"blk.{block}.{t}.weight" for t in targets}
    found = {n: s for n, s in shapes.items() if n in want}
    missing = want - set(found)
    if missing:
        raise SystemExit(f"model has no {sorted(missing)} — this arch names its weights differently")
    return found


def write_adapter(gguf, out: Path, arch: str, shapes: dict[str, list[int]], seed: int, sigma: float, rank: int) -> None:
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(out), arch)
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", ALPHA)
    w.add_name(out.stem)
    for name, (ne0, ne1) in sorted(shapes.items()):
        a = rng.normal(0.0, sigma, size=(rank, ne0)).astype(np.float32)
        b = rng.normal(0.0, sigma, size=(ne1, rank)).astype(np.float32)
        w.add_tensor(f"{name}.lora_a", a)
        w.add_tensor(f"{name}.lora_b", b)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--count", type=int, default=2, help="adapters per set")
    args = ap.parse_args()

    gguf = _gguf()
    arch, last, mid, shapes = model_shape(gguf.GGUFReader(str(args.model)))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sets = {
        "last-ffn": pick(shapes, last, FFN_TARGETS),
        "mid-ffn": pick(shapes, mid, FFN_TARGETS),
        "last-attn": pick(shapes, last, ATTN_TARGETS),
    }
    print(f"model={args.model.name} arch={arch} last_block={last} mid_block={mid}")
    for prefix, sel in sets.items():
        for i in range(args.count):
            out = args.out_dir / f"{prefix}-{i}.gguf"
            # Seeds differ per set AND per index, so no two adapters in the
            # directory hold the same tensor data under two names. crc32, not
            # hash(): str hashing is salted per process, so hash() would make
            # the adapters unreproducible across runs.
            seed = zlib.crc32(prefix.encode()) % 10_000 + i
            write_adapter(gguf, out, arch, sel, seed=seed, sigma=args.sigma, rank=args.rank)
            print(f"  {out}  targets={sorted(sel)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
