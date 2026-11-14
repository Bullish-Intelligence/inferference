"""CPU-only tests for ``benchkit.gates.load_tokenizer``.

The first group fakes ``transformers`` so the fallback logic runs in the locked core
environment. The second group uses the real libraries and runs where they are installed.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from inferference.benchkit.gates import load_tokenizer

# The tokenizer_config.json that transformers 5 writes: extra_special_tokens is a LIST.
V5_CONFIG = {
    "backend": "tokenizers",
    "tokenizer_class": "GemmaTokenizer",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "extra_special_tokens": ["<|video|>"],
}


class _OldAutoTokenizer:
    """What transformers 4 does with a version 5 config."""

    @staticmethod
    def from_pretrained(path: str, **kwargs: Any) -> Any:
        raise AttributeError("'list' object has no attribute 'keys'")


class _FakeFast:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, auto: type) -> None:
    module = types.ModuleType("transformers")
    setattr(module, "AutoTokenizer", auto)  # noqa: B010
    setattr(module, "PreTrainedTokenizerFast", _FakeFast)  # noqa: B010
    monkeypatch.setitem(sys.modules, "transformers", module)


def _write(directory: Path, config: dict[str, Any] | None, *, tokenizer_json: bool = True) -> None:
    if config is not None:
        (directory / "tokenizer_config.json").write_text(json.dumps(config))
    if tokenizer_json:
        (directory / "tokenizer.json").write_text("{}")


def test_auto_tokenizer_is_the_first_choice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sentinel = object()

    class Auto:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> object:
            assert kwargs == {"local_files_only": True}
            return sentinel

    _install_fake_transformers(monkeypatch, Auto)
    assert load_tokenizer(str(tmp_path)) is sentinel


def test_fallback_reads_the_special_tokens_from_a_newer_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_transformers(monkeypatch, _OldAutoTokenizer)
    _write(tmp_path, V5_CONFIG)
    tokenizer = load_tokenizer(str(tmp_path))
    assert isinstance(tokenizer, _FakeFast)
    assert tokenizer.kwargs == {
        "tokenizer_file": str(tmp_path / "tokenizer.json"),
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "additional_special_tokens": ["<|video|>"],
    }


def test_fallback_accepts_added_token_dicts_and_the_older_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_transformers(monkeypatch, _OldAutoTokenizer)
    config = {
        "eos_token": {"content": "</s>", "lstrip": False},
        "extra_special_tokens": {"image_token": "<image>"},
    }
    _write(tmp_path, config)
    tokenizer = load_tokenizer(str(tmp_path))
    assert tokenizer.kwargs["eos_token"] == "</s>"
    assert tokenizer.kwargs["additional_special_tokens"] == ["<image>"]


def test_fallback_fails_closed_without_tokenizer_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_transformers(monkeypatch, _OldAutoTokenizer)
    _write(tmp_path, V5_CONFIG, tokenizer_json=False)
    with pytest.raises(AttributeError, match="no attribute 'keys'"):
        load_tokenizer(str(tmp_path))


def test_fallback_fails_closed_without_an_eos_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_transformers(monkeypatch, _OldAutoTokenizer)
    _write(tmp_path, {"bos_token": "<bos>"})
    with pytest.raises(RuntimeError, match="no eos_token") as info:
        load_tokenizer(str(tmp_path))
    assert isinstance(info.value.__cause__, AttributeError)


def test_fallback_fails_closed_on_an_unreadable_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_transformers(monkeypatch, _OldAutoTokenizer)
    _write(tmp_path, None)
    with pytest.raises(RuntimeError, match="cannot read tokenizer_config.json"):
        load_tokenizer(str(tmp_path))


# --- the real libraries --------------------------------------------------------------------------


def _tiny_tokenizer_dir(directory: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    vocab = {"<unk>": 0, "<eos>": 1, "<bos>": 2, "<pad>": 3, "hello": 4, "world": 5}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps(V5_CONFIG))


def test_real_transformers_loads_a_newer_config(tmp_path: Path) -> None:
    """Whatever the installed transformers version, the directory loads and the end token is set."""
    _tiny_tokenizer_dir(tmp_path)
    tokenizer = load_tokenizer(str(tmp_path))
    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.eos_token_id == 1
    assert tokenizer.get_vocab()["hello"] == 4
    assert tokenizer.encode("hello world", add_special_tokens=False) == [4, 5]


def test_real_transformers_fallback_path_is_exercised(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Force the fallback with the real fast tokenizer class."""
    _tiny_tokenizer_dir(tmp_path)
    import transformers

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AttributeError("'list' object has no attribute 'keys'")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", refuse)
    tokenizer = load_tokenizer(str(tmp_path))
    assert type(tokenizer).__name__ == "PreTrainedTokenizerFast"
    assert tokenizer.eos_token_id == 1
    assert tokenizer.get_vocab()["world"] == 5
