"""The ``seq_routed`` backend refuses an adapter that it would apply only in part.

The fork's seq path applies a delta in ``build_lora_mm`` only. The token embedding
(``build_inp_embd``) and the MoE expert matrices (``build_lora_mm_id``) read the
context's own adapter list, and that list is empty under ``seq_routed``. The tests
build tiny synthetic adapter GGUF files with gguf-py. No GPU, no model, no native
library: a fake ``llama_cpp`` module stands in, and its ``Llama`` raises a sentinel
so a test can tell "refused" from "went on to load the model".
"""

from __future__ import annotations

import ctypes
import importlib.util
import struct
import sys
import types
from pathlib import Path

import gguf
import numpy as np
import pytest

from inferference.batching import LlamaContinuousBatchEngine
from inferference.models import EngineConfig
from inferference.router import AdapterSpec, Backend, MultiLoRARouter, RouterConfig
from inferference.seq_adapter_coverage import (
    REASON_EXPERTS,
    REASON_TOKEN_EMBD,
    SeqRoutedAdapterRefused,
    dropped_reason,
    dropped_tensors,
    read_tensor_names,
    refuse_dropped_adapters,
)
from inferference.seq_routing import SeqRoutingUnavailable
from inferference.tied_head_adapter import TiedHeadAdapterRefused, refuse_tied_head_adapters

_RANK = 2


def _adapter(path: Path, bases: list[str], *, extra_kv: bool = True) -> Path:
    """Write a LoRA adapter GGUF that holds a lora_a/lora_b pair for each base tensor name."""
    w = gguf.GGUFWriter(str(path), "tinyarch")
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", 16.0)
    if extra_kv:
        # Value kinds that the header reader must skip: arrays of numbers and of strings, bool, 64-bit.
        w.add_array("test.ints", [1, 2, 3])
        w.add_array("test.strings", ["a", "bc", ""])
        w.add_bool("test.flag", True)
        w.add_uint64("test.big", 1 << 40)
        w.add_float64("test.f64", 0.5)
    for base in bases:
        w.add_tensor(f"{base}.lora_a", np.zeros((_RANK, 8), dtype=np.float32))
        w.add_tensor(f"{base}.lora_b", np.zeros((8, _RANK), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _base(path: Path, *, tied: bool, arch: str = "tinyarch", extra: tuple[str, ...] = ()) -> Path:
    """Write a tiny base-model GGUF. ``tied`` leaves out ``output.weight``: the head reuses token_embd."""
    w = gguf.GGUFWriter(str(path), arch)
    names = ["token_embd.weight", "output_norm.weight", "blk.0.ffn_up.weight", *extra]
    if not tied:
        names.append("output.weight")
    for name in names:
        w.add_tensor(name, np.zeros((8, 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


_NORMAL = [
    "blk.0.ffn_up.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_down.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
    "blk.0.ffn_gate_inp.weight",  # the router matrix of a MoE block goes through build_lora_mm
    "blk.0.ffn_up_shexp.weight",  # shared experts go through build_lora_mm
]


# ---- the name predicate ----


@pytest.mark.parametrize("base", _NORMAL)
def test_normal_targets_are_not_dropped(base: str) -> None:
    assert dropped_reason(f"{base}.lora_a") is None
    assert dropped_reason(f"{base}.lora_b") is None


@pytest.mark.parametrize(
    "base",
    [
        "blk.3.ffn_up_exps.weight",
        "blk.3.ffn_gate_exps.weight",
        "blk.3.ffn_down_exps.weight",
        "blk.3.ffn_gate_up_exps.weight",
        "blk.3.ffn_up_chexps.weight",
        "blk.3.ffn_gate_chexps.weight",
        "blk.3.ffn_down_chexps.weight",
    ],
)
def test_expert_targets_are_dropped(base: str) -> None:
    assert dropped_reason(f"{base}.lora_a") == REASON_EXPERTS
    assert dropped_reason(f"{base}.lora_b") == REASON_EXPERTS


@pytest.mark.parametrize("base", ["token_embd.weight", "per_layer_token_embd.weight"])
def test_token_embedding_is_dropped(base: str) -> None:
    # The fork's loader tests endswith("token_embd.weight"); so does this module.
    assert dropped_reason(f"{base}.lora_a") == REASON_TOKEN_EMBD


def test_the_embedding_norm_is_not_the_embedding() -> None:
    assert dropped_reason("token_embd_norm.weight.lora_a") is None


# ---- the header reader ----


def test_reader_matches_gguf_py(tmp_path: Path) -> None:
    bases = [*_NORMAL, "token_embd.weight", "blk.1.ffn_up_exps.weight"]
    path = _adapter(tmp_path / "a.gguf", bases)
    want = [t.name for t in gguf.GGUFReader(str(path)).tensors]
    assert read_tensor_names(path) == want
    assert len(want) == 2 * len(bases)


def test_reader_handles_a_file_without_extra_kv(tmp_path: Path) -> None:
    path = _adapter(tmp_path / "a.gguf", ["blk.0.ffn_up.weight"], extra_kv=False)
    assert read_tensor_names(path) == ["blk.0.ffn_up.weight.lora_a", "blk.0.ffn_up.weight.lora_b"]


def _gguf_with_array(path: Path, elem_type: int, count: int, payload: bytes = b"") -> Path:
    """Write a GGUF header with one array key whose count the caller sets, with no tensors."""
    key = b"test.array"
    blob = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    blob += struct.pack("<Q", len(key)) + key + struct.pack("<I", 9) + struct.pack("<IQ", elem_type, count)
    path.write_bytes(blob + payload)
    return path


@pytest.mark.parametrize(
    ("elem_type", "count"),
    [
        (6, 1 << 62),  # float32: count * 4 overflows a C ssize_t
        (6, 1 << 40),  # float32: 4 TiB, more than the file and more than memory
        (10, 1 << 30),  # uint64: 8 GiB
        (8, 1 << 62),  # string: one loop pass per element
        (9, 1 << 62),  # array of arrays
    ],
)
def test_a_corrupt_array_count_raises_value_error_not_overflow_or_memory_error(
    tmp_path: Path, elem_type: int, count: int
) -> None:
    path = _gguf_with_array(tmp_path / "x.gguf", elem_type, count, payload=b"\0" * 64)
    with pytest.raises(ValueError):
        read_tensor_names(path)


def test_a_corrupt_array_count_gives_the_refusal_message(tmp_path: Path) -> None:
    path = _gguf_with_array(tmp_path / "x.gguf", 6, 1 << 62)
    with pytest.raises(SeqRoutedAdapterRefused, match="cannot read the tensor names"):
        refuse_dropped_adapters([("x", path)])
    with pytest.raises(TiedHeadAdapterRefused, match="cannot read the tensor names"):
        refuse_tied_head_adapters(tmp_path / "base.gguf", [("x", path)])


def test_a_huge_dimension_count_in_a_tensor_entry_raises_value_error(tmp_path: Path) -> None:
    name = b"t"
    blob = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    blob += struct.pack("<Q", len(name)) + name + struct.pack("<I", 0xFFFFFFFF)
    path = tmp_path / "x.gguf"
    path.write_bytes(blob)
    with pytest.raises(ValueError):
        read_tensor_names(path)


def test_a_valid_array_still_reads_after_the_bound(tmp_path: Path) -> None:
    path = _gguf_with_array(tmp_path / "x.gguf", 6, 3, payload=b"\0" * 12)
    assert read_tensor_names(path) == []


def test_dropped_tensors_names_each_base_once(tmp_path: Path) -> None:
    path = _adapter(tmp_path / "a.gguf", [*_NORMAL, "token_embd.weight", "blk.1.ffn_down_exps.weight"])
    assert dropped_tensors(path) == {
        "token_embd.weight": REASON_TOKEN_EMBD,
        "blk.1.ffn_down_exps.weight": REASON_EXPERTS,
    }


# ---- the refusal ----


def test_normal_adapter_is_accepted(tmp_path: Path) -> None:
    path = _adapter(tmp_path / "ok.gguf", _NORMAL)
    refuse_dropped_adapters([("ok", path)])  # no raise


def test_token_embd_adapter_is_refused_and_named(tmp_path: Path) -> None:
    path = _adapter(tmp_path / "emb.gguf", [*_NORMAL, "token_embd.weight"])
    with pytest.raises(SeqRoutedAdapterRefused) as err:
        refuse_dropped_adapters([("emb-lora", path)])
    msg = str(err.value)
    assert "'emb-lora'" in msg
    assert str(path) in msg
    assert "token_embd.weight" in msg
    assert "build_inp_embd" in msg
    assert "context_pool" in msg
    assert "ffn_up.weight" not in msg  # the applied tensors are not accused


def test_expert_adapter_is_refused_and_named(tmp_path: Path) -> None:
    path = _adapter(tmp_path / "moe.gguf", ["blk.2.ffn_up_exps.weight", "blk.2.ffn_down_exps.weight", *_NORMAL])
    with pytest.raises(SeqRoutedAdapterRefused) as err:
        refuse_dropped_adapters([("moe-lora", path)])
    msg = str(err.value)
    assert "'moe-lora'" in msg
    assert "blk.2.ffn_up_exps.weight" in msg
    assert "blk.2.ffn_down_exps.weight" in msg
    assert "build_lora_mm_id" in msg


def test_refusal_names_every_bad_adapter_and_skips_the_good_one(tmp_path: Path) -> None:
    good = _adapter(tmp_path / "good.gguf", _NORMAL)
    bad1 = _adapter(tmp_path / "b1.gguf", ["token_embd.weight"])
    bad2 = _adapter(tmp_path / "b2.gguf", ["blk.0.ffn_gate_exps.weight"])
    with pytest.raises(SeqRoutedAdapterRefused) as err:
        refuse_dropped_adapters([("good", good), ("bad1", bad1), ("bad2", bad2)])
    msg = str(err.value)
    assert "'bad1'" in msg and "'bad2'" in msg and "'good'" not in msg


def test_long_tensor_list_is_cut_but_counted(tmp_path: Path) -> None:
    path = _adapter(tmp_path / "many.gguf", [f"blk.{i}.ffn_up_exps.weight" for i in range(20)])
    with pytest.raises(SeqRoutedAdapterRefused, match=r"20 tensor\(s\).*and 12 more"):
        refuse_dropped_adapters([("many", path)])


@pytest.mark.parametrize("kind", ["missing", "garbage", "truncated", "bad_version"])
def test_unreadable_adapter_is_refused_not_passed(tmp_path: Path, kind: str) -> None:
    good = _adapter(tmp_path / "good.gguf", _NORMAL)
    path = tmp_path / "x.gguf"
    if kind == "garbage":
        path.write_bytes(b"not a gguf file at all")
    elif kind == "truncated":
        path.write_bytes(good.read_bytes()[:200])
    elif kind == "bad_version":
        data = bytearray(good.read_bytes())
        data[4:8] = (99).to_bytes(4, "little")
        path.write_bytes(bytes(data))
    with pytest.raises(SeqRoutedAdapterRefused, match="cannot read the tensor names"):
        refuse_dropped_adapters([("x", path)])


# ---- where the check runs: the router and the batching engine ----


class _ModelLoadReached(Exception):
    """The fake ``Llama`` raises this: the check let the model load go on."""


class _FakeSymbol:
    def __init__(self) -> None:
        self.argtypes: list[type] = []
        self.restype: type | None = None


class _ForkLib:
    def __init__(self) -> None:
        self.llama_set_seq_adapters = _FakeSymbol()
        self.llama_set_seq_adapter = _FakeSymbol()


class _FakeC:
    llama_context_p_ctypes = ctypes.c_void_p
    llama_adapter_lora_p_ctypes = ctypes.c_void_p

    def __init__(self, lib: object) -> None:
        self._lib = lib


def _install_fake_llama_cpp(monkeypatch: pytest.MonkeyPatch, *, fork: bool) -> None:
    class _Llama:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise _ModelLoadReached

    mod = types.ModuleType("llama_cpp")
    setattr(mod, "Llama", _Llama)  # noqa: B010
    setattr(mod, "llama_cpp", _FakeC(_ForkLib() if fork else object()))  # noqa: B010
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)


def _router_config(path: Path, backend: Backend) -> RouterConfig:
    # An untied base: the tied-head refusal (test_tied_head_adapter.py) stays out of these tests.
    base = _base(path.parent / "base.gguf", tied=False)
    return RouterConfig(
        engine=EngineConfig(model_path=str(base), n_ctx=64, n_batch=16, n_gpu_layers=0, backend="vulkan"),
        adapters=(AdapterSpec(name="a", gguf_path=str(path)),),
        n_seq_max=2,
        backend=backend,
    )


@pytest.fixture
def emb_adapter(tmp_path: Path) -> Path:
    return _adapter(tmp_path / "emb.gguf", [*_NORMAL, "token_embd.weight"])


@pytest.fixture
def moe_adapter(tmp_path: Path) -> Path:
    return _adapter(tmp_path / "moe.gguf", [*_NORMAL, "blk.1.ffn_up_exps.weight"])


@pytest.fixture
def ffn_adapter(tmp_path: Path) -> Path:
    return _adapter(tmp_path / "ffn.gguf", _NORMAL)


@pytest.mark.parametrize("which", ["emb_adapter", "moe_adapter"])
def test_router_seq_routed_refuses_before_the_model_loads(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, which: str
) -> None:
    path = request.getfixturevalue(which)
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(SeqRoutedAdapterRefused, match="seq_routed"):
        MultiLoRARouter(_router_config(path, "seq_routed"))


def test_router_auto_on_a_fork_lib_refuses_too(monkeypatch: pytest.MonkeyPatch, emb_adapter: Path) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(SeqRoutedAdapterRefused):
        MultiLoRARouter(_router_config(emb_adapter, "auto"))


def test_router_seq_routed_accepts_a_normal_adapter(monkeypatch: pytest.MonkeyPatch, ffn_adapter: Path) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(_ModelLoadReached):  # went past the check, to the model load
        MultiLoRARouter(_router_config(ffn_adapter, "seq_routed"))


@pytest.mark.parametrize("which", ["emb_adapter", "moe_adapter"])
def test_router_context_pool_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, which: str
) -> None:
    # context_pool pins the adapter per context (llama_set_adapters_lora), which the
    # normal path applies in full. It must go on to the model load.
    path = request.getfixturevalue(which)
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(_ModelLoadReached):
        MultiLoRARouter(_router_config(path, "context_pool"))


def test_router_auto_on_a_stock_lib_is_context_pool_and_unchanged(
    monkeypatch: pytest.MonkeyPatch, emb_adapter: Path
) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=False)
    with pytest.raises(_ModelLoadReached):
        MultiLoRARouter(_router_config(emb_adapter, "auto"))


def test_router_seq_routed_on_a_stock_lib_still_reports_the_missing_capability(
    monkeypatch: pytest.MonkeyPatch, emb_adapter: Path
) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=False)
    with pytest.raises(SeqRoutingUnavailable):
        MultiLoRARouter(_router_config(emb_adapter, "seq_routed"))


def _engine(path: Path) -> LlamaContinuousBatchEngine:
    base = _base(path.parent / "base.gguf", tied=False)
    return LlamaContinuousBatchEngine(str(base), n_ctx=64, adapters=(("a", str(path), 1.0),))


def test_batching_engine_refuses_before_the_model_loads(monkeypatch: pytest.MonkeyPatch, moe_adapter: Path) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(SeqRoutedAdapterRefused, match="blk.1.ffn_up_exps.weight"):
        _engine(moe_adapter)


def test_batching_engine_accepts_a_normal_adapter(monkeypatch: pytest.MonkeyPatch, ffn_adapter: Path) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(_ModelLoadReached):
        _engine(ffn_adapter)


def test_batching_engine_on_a_stock_lib_has_no_seq_routing_and_no_refusal(
    monkeypatch: pytest.MonkeyPatch, emb_adapter: Path
) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=False)
    with pytest.raises(_ModelLoadReached):
        _engine(emb_adapter)


# ---- the probe adapters of the p2-server-seq-routing suite still load ----

_SCRIPT = Path(__file__).resolve().parents[1] / "suites/p2-server-seq-routing/make_arch_lora.py"


def _make_arch_lora():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("make_arch_lora", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_base(path: Path) -> Path:
    w = gguf.GGUFWriter(str(path), "tinyarch")
    for blk in range(4):
        for name, shape in (
            ("ffn_up", (16, 8)),
            ("ffn_gate", (16, 8)),
            ("ffn_down", (8, 16)),
            ("attn_q", (8, 8)),
            ("attn_k", (8, 8)),
            ("attn_v", (8, 8)),
            ("attn_output", (8, 8)),
        ):
            w.add_tensor(f"blk.{blk}.{name}.weight", np.zeros(shape, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.mark.parametrize("mode", ["probe", "full"])
def test_suite_probe_adapters_pass_the_check(tmp_path: Path, mode: str) -> None:
    base = _tiny_base(tmp_path / "base.gguf")
    out = tmp_path / "out"
    argv = ["--model", str(base), "--out-dir", str(out), "--rank", str(_RANK)]
    if mode == "probe":
        argv.append("--probe")
    assert _make_arch_lora().main(argv) == 0
    written = sorted(out.glob("*.gguf"))
    assert written
    refuse_dropped_adapters((p.stem, p) for p in written)  # no raise
