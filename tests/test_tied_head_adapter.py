"""An adapter on token_embd.weight is refused when the base model ties its output head.

llama.cpp builds a tied output head as a duplicate of ``token_embd.weight`` and the
duplicate keeps that name. The adapter pair for ``token_embd.weight`` then matches the
output head in ``build_lora_mm``, in the wrong layout, and the process aborts at the
first decode. The refusal runs at adapter load time on every backend, before the model
loads. The tests build tiny base and adapter GGUF files with gguf-py. No GPU, no model,
no native library: the fake ``llama_cpp`` module of the seq_routed tests stands in, and
its ``Llama`` raises a sentinel so a test can tell "refused" from "went on to load".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferference.batching import LlamaContinuousBatchEngine
from inferference.models import EngineConfig
from inferference.router import AdapterSpec, Backend, MultiLoRARouter, RouterConfig
from inferference.seq_adapter_coverage import SeqRoutedAdapterRefused
from inferference.tied_head_adapter import (
    TiedHeadAdapterRefused,
    adapter_targets_token_embd,
    refuse_tied_head_adapters,
    tied_head_reason,
)
from tests.test_seq_adapter_coverage import (
    _NORMAL,
    _adapter,
    _base,
    _install_fake_llama_cpp,
    _ModelLoadReached,
)


@pytest.fixture
def tied_base(tmp_path: Path) -> Path:
    return _base(tmp_path / "tied.gguf", tied=True)


@pytest.fixture
def untied_base(tmp_path: Path) -> Path:
    return _base(tmp_path / "untied.gguf", tied=False)


@pytest.fixture
def embd_adapter(tmp_path: Path) -> Path:
    return _adapter(tmp_path / "embd.gguf", [*_NORMAL, "token_embd.weight"])


@pytest.fixture
def ffn_adapter(tmp_path: Path) -> Path:
    return _adapter(tmp_path / "ffn.gguf", _NORMAL)


def _config(base: Path, adapter: Path, backend: Backend) -> RouterConfig:
    return RouterConfig(
        engine=EngineConfig(model_path=str(base), n_ctx=64, n_batch=16, n_gpu_layers=0, backend="vulkan"),
        adapters=(AdapterSpec(name="emb", gguf_path=str(adapter)),),
        n_seq_max=2,
        backend=backend,
    )


# ---- the detection ----


def test_a_base_without_output_weight_is_tied(tied_base: Path) -> None:
    reason = tied_head_reason(tied_base)
    assert reason is not None and "no output.weight" in reason


def test_a_base_with_output_weight_is_not_tied(untied_base: Path) -> None:
    assert tied_head_reason(untied_base) is None


@pytest.mark.parametrize("arch", ["gemma", "gemma2", "cohere2", "command-r", "openelm", "plm", "gemma4-assistant"])
def test_an_always_tied_architecture_is_tied_even_with_output_weight(tmp_path: Path, arch: str) -> None:
    # These architectures never read output.weight: the loader ties the head unconditionally.
    base = _base(tmp_path / "b.gguf", tied=False, arch=arch)
    reason = tied_head_reason(base)
    assert reason is not None and arch in reason


def test_the_exact_tensor_name_is_the_target(tmp_path: Path) -> None:
    emb = _adapter(tmp_path / "emb.gguf", ["token_embd.weight"])
    per_layer = _adapter(tmp_path / "pl.gguf", ["per_layer_token_embd.weight"])
    norm = _adapter(tmp_path / "norm.gguf", ["token_embd_norm.weight"])
    assert adapter_targets_token_embd(emb)
    assert not adapter_targets_token_embd(per_layer)
    assert not adapter_targets_token_embd(norm)


def test_a_split_base_with_the_output_head_in_a_later_shard_is_not_tied(tmp_path: Path) -> None:
    first = _base(tmp_path / "m-00001-of-00002.gguf", tied=True)
    # The second shard holds the output head. Name it as llama.cpp does.
    _base(tmp_path / "m-00002-of-00002.gguf", tied=False)
    assert tied_head_reason(first) is None


# ---- the refusal function ----


def test_refuse_names_the_adapter_the_tensor_and_the_reason(tied_base: Path, embd_adapter: Path) -> None:
    with pytest.raises(TiedHeadAdapterRefused) as err:
        refuse_tied_head_adapters(tied_base, [("my-lora", embd_adapter)])
    msg = str(err.value)
    assert "'my-lora'" in msg
    assert str(embd_adapter) in msg
    assert "token_embd.weight" in msg
    assert "tied output head" in msg
    assert "abort" in msg
    assert str(tied_base) in msg


def test_refuse_accepts_an_embedding_adapter_on_an_untied_base(untied_base: Path, embd_adapter: Path) -> None:
    refuse_tied_head_adapters(untied_base, [("emb", embd_adapter)])  # no raise


def test_refuse_accepts_a_normal_adapter_on_a_tied_base(tied_base: Path, ffn_adapter: Path) -> None:
    refuse_tied_head_adapters(tied_base, [("ffn", ffn_adapter)])  # no raise


def test_refuse_does_not_read_the_base_for_a_normal_adapter(tmp_path: Path, ffn_adapter: Path) -> None:
    refuse_tied_head_adapters(tmp_path / "missing.gguf", [("ffn", ffn_adapter)])  # no raise


def test_refuse_names_every_bad_adapter(tied_base: Path, tmp_path: Path, ffn_adapter: Path) -> None:
    bad1 = _adapter(tmp_path / "b1.gguf", ["token_embd.weight"])
    bad2 = _adapter(tmp_path / "b2.gguf", ["token_embd.weight"])
    with pytest.raises(TiedHeadAdapterRefused) as err:
        refuse_tied_head_adapters(tied_base, [("good", ffn_adapter), ("bad1", bad1), ("bad2", bad2)])
    msg = str(err.value)
    assert "'bad1'" in msg and "'bad2'" in msg and "'good'" not in msg


def test_refuse_fails_closed_on_an_unreadable_base(tmp_path: Path, embd_adapter: Path) -> None:
    garbage = tmp_path / "garbage.gguf"
    garbage.write_bytes(b"not a gguf file at all")
    for base in (garbage, tmp_path / "missing.gguf"):
        with pytest.raises(TiedHeadAdapterRefused, match="cannot be read"):
            refuse_tied_head_adapters(base, [("emb", embd_adapter)])


def test_refuse_fails_closed_on_an_unreadable_adapter(untied_base: Path, tmp_path: Path) -> None:
    garbage = tmp_path / "adapter.gguf"
    garbage.write_bytes(b"GGUF")
    with pytest.raises(TiedHeadAdapterRefused, match="cannot read the tensor names"):
        refuse_tied_head_adapters(untied_base, [("x", garbage)])


# ---- where the check runs: the router and the batching engine ----


@pytest.mark.parametrize("backend", ["context_pool", "seq_routed", "auto"])
def test_router_refuses_a_tied_base_on_every_backend(
    monkeypatch: pytest.MonkeyPatch, tied_base: Path, embd_adapter: Path, backend: Backend
) -> None:
    # A fork lib, so that seq_routed and auto resolve; the model load is never reached.
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(TiedHeadAdapterRefused, match="tied output head"):
        MultiLoRARouter(_config(tied_base, embd_adapter, backend))


def test_router_refuses_a_tied_base_on_a_stock_lib_under_auto(
    monkeypatch: pytest.MonkeyPatch, tied_base: Path, embd_adapter: Path
) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=False)  # auto resolves to context_pool
    with pytest.raises(TiedHeadAdapterRefused):
        MultiLoRARouter(_config(tied_base, embd_adapter, "auto"))


def test_router_loads_an_embedding_adapter_on_an_untied_base_under_context_pool(
    monkeypatch: pytest.MonkeyPatch, untied_base: Path, embd_adapter: Path
) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(_ModelLoadReached):  # went past the check, to the model load
        MultiLoRARouter(_config(untied_base, embd_adapter, "context_pool"))


@pytest.mark.parametrize("backend", ["seq_routed", "auto"])
def test_router_untied_base_still_hits_the_seq_routed_refusal(
    monkeypatch: pytest.MonkeyPatch, untied_base: Path, embd_adapter: Path, backend: Backend
) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(SeqRoutedAdapterRefused):
        MultiLoRARouter(_config(untied_base, embd_adapter, backend))


@pytest.mark.parametrize("backend", ["context_pool", "seq_routed", "auto"])
@pytest.mark.parametrize("tied", [True, False])
def test_router_accepts_a_normal_adapter_on_both_bases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ffn_adapter: Path, backend: Backend, tied: bool
) -> None:
    base = _base(tmp_path / "b.gguf", tied=tied)
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(_ModelLoadReached):
        MultiLoRARouter(_config(base, ffn_adapter, backend))


def test_batching_engine_refuses_a_tied_base(
    monkeypatch: pytest.MonkeyPatch, tied_base: Path, embd_adapter: Path
) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(TiedHeadAdapterRefused, match="token_embd.weight"):
        LlamaContinuousBatchEngine(str(tied_base), n_ctx=64, adapters=(("emb", str(embd_adapter), 1.0),))


def test_batching_engine_accepts_a_normal_adapter_on_a_tied_base(
    monkeypatch: pytest.MonkeyPatch, tied_base: Path, ffn_adapter: Path
) -> None:
    _install_fake_llama_cpp(monkeypatch, fork=True)
    with pytest.raises(_ModelLoadReached):
        LlamaContinuousBatchEngine(str(tied_base), n_ctx=64, adapters=(("ffn", str(ffn_adapter), 1.0),))
