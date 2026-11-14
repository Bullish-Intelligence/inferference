"""The probe-adapter generator: probe-a.gguf and probe-b.gguf for any architecture.

The suites read these two files from `$MODEL_PROBE_DIR`. The tests build a tiny
synthetic base GGUF in a tmp dir, so they need no GPU and no model file.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import gguf
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "suites/p2-server-seq-routing/make_arch_lora.py"
_ARCH = "tinyarch"
_BLOCKS = 4  # last = 3, mid = 1
_N_EMBD, _N_FF, _RANK = 8, 16, 4


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_arch_lora", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base(path: Path, *, drop: str | None = None) -> Path:
    """Write a base GGUF that holds only the tensors the generator reads."""
    w = gguf.GGUFWriter(str(path), _ARCH)
    for blk in range(_BLOCKS):
        # numpy shape is the reverse of the ggml shape: ffn_up is ne [n_embd, n_ff].
        for name, shape in (
            ("ffn_up", (_N_FF, _N_EMBD)),
            ("ffn_gate", (_N_FF, _N_EMBD)),
            ("ffn_down", (_N_EMBD, _N_FF)),
            ("attn_q", (_N_EMBD, _N_EMBD)),
        ):
            if f"blk.{blk}.{name}" == drop:
                continue
            w.add_tensor(f"blk.{blk}.{name}.weight", np.zeros(shape, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _run(base: Path, out_dir: Path) -> None:
    code = _load().main(["--model", str(base), "--out-dir", str(out_dir), "--probe", "--rank", str(_RANK)])
    assert code == 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_probe_writes_exactly_two_adapters(tmp_path: Path) -> None:
    _run(_base(tmp_path / "base.gguf"), tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["probe-a.gguf", "probe-b.gguf"]


def test_probe_adapter_header_and_targets(tmp_path: Path) -> None:
    _run(_base(tmp_path / "base.gguf"), tmp_path / "out")
    reader = gguf.GGUFReader(str(tmp_path / "out" / "probe-a.gguf"))
    assert reader.fields["general.type"].contents() == "adapter"
    assert reader.fields["adapter.type"].contents() == "lora"
    assert reader.fields["general.architecture"].contents() == _ARCH
    assert reader.fields["adapter.lora.alpha"].contents() > 0
    names = {t.name for t in reader.tensors}
    want = {
        f"blk.{blk}.{target}.weight.{half}"
        for blk in (1, 3)
        for target in ("ffn_up", "ffn_gate", "ffn_down")
        for half in ("lora_a", "lora_b")
    }
    assert names == want


def test_probe_shapes_match_the_base(tmp_path: Path) -> None:
    base = gguf.GGUFReader(str(_base(tmp_path / "base.gguf")))
    _run(tmp_path / "base.gguf", tmp_path / "out")
    adapter = gguf.GGUFReader(str(tmp_path / "out" / "probe-b.gguf"))
    shapes = {t.name: [int(d) for d in t.shape] for t in base.tensors}
    for t in adapter.tensors:
        stem, half = t.name.rsplit(".", 1)
        ne0, ne1 = shapes[stem]
        want = [ne0, _RANK] if half == "lora_a" else [_RANK, ne1]
        assert [int(d) for d in t.shape] == want, t.name


def test_probe_adapters_differ_and_reach_the_graph(tmp_path: Path) -> None:
    _run(_base(tmp_path / "base.gguf"), tmp_path / "out")
    a = gguf.GGUFReader(str(tmp_path / "out" / "probe-a.gguf"))
    b = gguf.GGUFReader(str(tmp_path / "out" / "probe-b.gguf"))
    ta = {t.name: np.asarray(t.data) for t in a.tensors}
    tb = {t.name: np.asarray(t.data) for t in b.tensors}
    assert ta.keys() == tb.keys()
    assert all(not np.array_equal(ta[n], tb[n]) for n in ta), "the two adapters share a tensor"
    for stem in {n.rsplit(".", 1)[0] for n in ta}:
        # numpy view of lora_a is (rank, ne0), of lora_b is (ne1, rank): delta = b @ a, nonzero.
        delta = tb[f"{stem}.lora_b"] @ tb[f"{stem}.lora_a"]
        assert float(np.abs(delta).max()) > 0.0, stem


def test_probe_is_reproducible(tmp_path: Path) -> None:
    base = _base(tmp_path / "base.gguf")
    _run(base, tmp_path / "one")
    _run(base, tmp_path / "two")
    for name in ("probe-a.gguf", "probe-b.gguf"):
        assert _sha(tmp_path / "one" / name) == _sha(tmp_path / "two" / name)


def test_probe_fails_loudly_on_a_missing_target(tmp_path: Path) -> None:
    base = _base(tmp_path / "base.gguf", drop="blk.3.ffn_gate")
    with pytest.raises(SystemExit, match="ffn_gate"):
        _run(base, tmp_path / "out")
