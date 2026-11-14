"""Refuse a LoRA adapter on ``token_embd.weight`` when the base model ties its output head.

Some models have no separate output head. The fork's model loader then builds
``model.output`` as a duplicate of ``token_embd.weight`` and keeps the name
``token_embd.weight`` on it. The graph computes the logits with
``build_lora_mm(model.output, ...)``. That function finds an adapter weight by tensor
name, so an adapter pair stored for ``token_embd.weight`` matches the output head too.
The adapter loader stores a token-embedding pair in the flipped layout, and
``build_lora_mm`` expects the normal layout. The shapes do not fit,
``GGML_ASSERT(ggml_can_mul_mat(a, b))`` fails in ``ggml_mul_mat``, and the process
aborts (exit 134) at the first decode. This is a crash risk on every backend, so the
check runs at adapter load time, before the model loads and before any VRAM is spent.

There is no override flag. Remove the ``token_embd`` tensors from the adapter, or serve
it on a base model with a separate output head.

Detection reads the base GGUF header with the standard library (the runtime install
has no ``gguf`` package):

* No ``output.weight`` tensor means the model reuses ``token_embd.weight`` as the
  output head. Fork ``ecc0cb0d34e17595ff0546b89c44fa100d076c0d`` (build ``ecc0cb0d3``),
  ``ci/build/.llamacpp-builds/src-p2fork``. There is no common code for this: each
  architecture file under ``src/models/`` loads ``output`` as ``TENSOR_NOT_REQUIRED``
  and, when it is ``NULL``, loads ``LLM_TENSOR_TOKEN_EMBD`` with ``TENSOR_DUPLICATED``
  (for example ``src/models/gemma4.cpp:44-47``). The loader gives the duplicate the
  name of the original (``src/llama-model-loader.cpp:1271-1277``, ``ggml_set_name``).
  ``llama_adapter_lora::get_weight`` finds a weight by that name
  (``src/llama-adapter.cpp:138-146``).
* Seven architectures tie unconditionally: they never read ``output.weight``, even when
  the file holds one. ``general.architecture`` names them in ``_ALWAYS_TIED_ARCHS``.
  Source: the ``output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, ...), TENSOR_DUPLICATED)``
  line of ``gemma.cpp:20``, ``gemma2.cpp:39``, ``cohere2.cpp:31``, ``command-r.cpp:21``,
  ``openelm.cpp:23``, ``plm.cpp:25`` and ``gemma4-assistant.cpp:38``.

The abort needs a graph that calls ``build_lora_mm(model.output, ...)``. 108 of the
architecture files do, and so do the graphs that reuse them. Three tied architectures
(``deepseek2``, ``deepseek32`` and ``glm-dsa``) call plain ``ggml_mul_mat(model.output, ...)``
and would not abort. This module still refuses on the tie in general: a crash risk
is never a silent pass, and the adapter code of those graphs is outside this check.

The tensor test is the exact name ``token_embd.weight``, because ``get_weight`` compares
the full name. ``per_layer_token_embd.weight`` is not the output head and does not
match. The seq_routed check in ``seq_adapter_coverage`` keeps its own, wider test.

Re-read the source lines above when the anchor moves.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from .seq_adapter_coverage import read_gguf_header, read_tensor_names

_TOKEN_EMBD = "token_embd.weight"
_OUTPUT = "output.weight"
_LORA_SUFFIXES: tuple[str, ...] = (".lora_a", ".lora_b")

# Architectures that always take the output head from token_embd.weight.
_ALWAYS_TIED_ARCHS: frozenset[str] = frozenset(
    {"gemma", "gemma2", "cohere2", "command-r", "openelm", "plm", "gemma4-assistant"}
)

# A split GGUF holds its tensors in shards: name-00001-of-00003.gguf.
_SHARD_RE = re.compile(r"^(?P<stem>.*)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


class TiedHeadAdapterRefused(RuntimeError):
    """The adapter targets ``token_embd.weight`` and the base ties its output head, or a file is unreadable."""


def adapter_targets_token_embd(adapter_path: str | Path) -> bool:
    """Return True when the adapter at ``adapter_path`` holds a ``token_embd.weight`` pair."""
    for name in read_tensor_names(adapter_path):
        for suffix in _LORA_SUFFIXES:
            if name.endswith(suffix) and name[: -len(suffix)] == _TOKEN_EMBD:
                return True
    return False


def _shards(path: Path) -> list[Path]:
    m = _SHARD_RE.match(path.name)
    if m is None:
        return [path]
    total = int(m["total"])
    return [path.with_name(f"{m['stem']}-{i:05d}-of-{m['total']}.gguf") for i in range(1, total + 1)]


def tied_head_reason(model_path: str | Path) -> str | None:
    """Return why the base model at ``model_path`` ties its output head, or ``None`` if it does not.

    Raises ``ValueError`` or ``OSError`` when a header cannot be read.
    """
    arch: str | None = None
    names: set[str] = set()
    for shard in _shards(Path(model_path)):
        shard_arch, shard_names = read_gguf_header(shard)
        arch = arch or shard_arch
        names.update(shard_names)
    if arch in _ALWAYS_TIED_ARCHS:
        return f"architecture {arch!r} always takes its output head from {_TOKEN_EMBD}"
    if _OUTPUT not in names:
        return f"the model has no {_OUTPUT} tensor, so it reuses {_TOKEN_EMBD} as the output head"
    return None


def refuse_tied_head_adapters(model_path: str | Path, adapters: Iterable[tuple[str, str | Path]]) -> None:
    """Raise :class:`TiedHeadAdapterRefused` when an adapter would abort the process on a tied base.

    ``adapters`` is a sequence of ``(adapter name, GGUF path)``. The check reads the base
    header only when an adapter targets ``token_embd.weight``. A file that the reader
    cannot parse is refused: the check cannot show that the process is safe.
    """
    problems: list[str] = []
    base_reason: str | None = None
    base_checked = False
    base_error: str | None = None
    for name, path in adapters:
        try:
            targets = adapter_targets_token_embd(path)
        except (OSError, ValueError) as exc:
            problems.append(f"adapter {name!r} ({path}): cannot read the tensor names to check the output head: {exc}")
            continue
        if not targets:
            continue
        if not base_checked:
            base_checked = True
            try:
                base_reason = tied_head_reason(model_path)
            except (OSError, ValueError) as exc:
                base_error = str(exc)
        if base_error is not None:
            problems.append(
                f"adapter {name!r} ({path}) targets {_TOKEN_EMBD}, and the base model ({model_path}) cannot be read "
                f"to check its output head: {base_error}"
            )
        elif base_reason is not None:
            problems.append(f"adapter {name!r} ({path}) targets {_TOKEN_EMBD}; {base_reason} (base model {model_path})")
    if problems:
        raise TiedHeadAdapterRefused(
            "refused before the model loads: an adapter on the token embedding of a model with a tied output head "
            "aborts the process at the first decode (llama.cpp applies the pair to the output head as well, in the "
            "flipped layout, and GGML_ASSERT(ggml_can_mul_mat) fails in build_lora_mm). "
            + " | ".join(problems)
            + ". Remove the token_embd tensors from the adapter, or use a base model with a separate output head."
        )
