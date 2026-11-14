"""Refuse an adapter that the ``seq_routed`` backend would apply only in part.

The P2 fork routes each sequence to its own adapter inside ``build_lora_mm``. That
function is one of three places that apply a LoRA delta. The other two read only
the context's own adapter list (``*loras``), and under ``seq_routed`` that list is
empty, because the pool lives in ``seq_loras``:

* ``build_inp_embd`` applies the delta of the token embedding.
* ``build_lora_mm_id`` applies the delta of each MoE expert matrix.

An adapter that targets one of these tensors loads, runs and returns tokens, and
the router drops that part of the adapter without a message. That is a silent
partial pass (rule 1: fail closed). This module reads the tensor names from the
adapter GGUF header and refuses, at adapter load time, with the adapter, the
tensors and the reason in the message.

The ``context_pool`` backend pins one adapter per context with
``llama_set_adapters_lora``. That call fills ``llama_context::loras``, and all three
functions read it, so ``context_pool`` applies every tensor. It needs no check.

There is no override flag. The way to serve such an adapter is
``backend="context_pool"``.

Source of the patterns: fork ``ecc0cb0d34e17595ff0546b89c44fa100d076c0d`` (build
``ecc0cb0d3``), ``ci/build/.llamacpp-builds/src-p2fork``:

* ``src/llama-graph.cpp:1499-1555`` the ``seq_loras`` path of ``build_lora_mm``.
  ``:1557`` is the ``*loras`` path of the same function.
* ``src/llama-graph.cpp:1578`` ``build_lora_mm_id``. Its only loop (``:1593``)
  reads ``*loras``. It has no ``seq_loras`` path.
  Callers: ``:2153`` ``gate_up_exps``, ``:2172`` ``up_exps``, ``:2185``
  ``gate_exps``, ``:2274`` ``down_exps``, all in ``build_moe_ffn``.
* ``src/llama-graph.cpp:2327`` ``build_inp_embd``. Its only loop (``:2355``) reads
  ``*loras``.
* ``src/llama-adapter.cpp:323`` the loader decides that a tensor is the token
  embedding with ``endswith("token_embd.weight")``. This module uses the same test.
* ``src/llama-arch.cpp:382`` ``token_embd``. ``:401-404`` ``ffn_gate_exps``,
  ``ffn_gate_up_exps``, ``ffn_down_exps``, ``ffn_up_exps``. ``:595-597``
  ``ffn_gate_chexps``, ``ffn_down_chexps``, ``ffn_up_chexps``. The ``chexps``
  tensors reach ``build_lora_mm_id`` through ``build_moe_ffn``
  (``src/models/grovemoe.cpp:154-158``).
* ``src/llama-context.cpp:1264`` ``set_adapters_lora`` fills ``loras``.
  ``:1282`` ``set_seq_adapters`` fills ``seq_loras`` only.

A change of the fork build that moves one of these applications into the seq path
makes this check too strict, not too loose. Re-read the lines above when the anchor
moves.

The reader below uses only the standard library. The runtime install has no
``gguf`` package. An adapter file that the reader cannot parse is refused too.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

# Suffixes of the two halves of a LoRA pair (``llama-adapter.cpp:273,280``).
_LORA_SUFFIXES: tuple[str, ...] = (".lora_a", ".lora_b")

# Loader test for the token embedding (``llama-adapter.cpp:323``).
_TOKEN_EMBD_SUFFIX = "token_embd.weight"

# Expert weights that go through ``build_lora_mm_id``.
# ``ffn_{gate,up,down,gate_up}_exps`` (``llama-arch.cpp:401-404``) and the
# ``chexps`` variants (``llama-arch.cpp:595-597``).
_EXPERT_RE = re.compile(r"(?:^|\.)ffn_(?:gate_up|gate|up|down)_(?:ch)?exps\.weight$")

REASON_TOKEN_EMBD = "token embedding (build_inp_embd applies only the context's own adapters)"
REASON_EXPERTS = "MoE expert matrix (build_lora_mm_id applies only the context's own adapters)"

# Longest tensor list that one message names. The count is always complete.
_MAX_NAMED = 8


class SeqRoutedAdapterRefused(RuntimeError):
    """The ``seq_routed`` backend would drop part of an adapter, or the file is unreadable."""


def _base_name(tensor_name: str) -> str:
    for suffix in _LORA_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def dropped_reason(tensor_name: str) -> str | None:
    """Return why ``seq_routed`` drops the LoRA tensor ``tensor_name``, or ``None``."""
    base = _base_name(tensor_name)
    if base.endswith(_TOKEN_EMBD_SUFFIX):
        return REASON_TOKEN_EMBD
    if _EXPERT_RE.search(base):
        return REASON_EXPERTS
    return None


# ---- a header-only GGUF reader (names only) ----

_GGUF_MAGIC = b"GGUF"
_MAX_STRING = 1 << 24  # a name or key longer than 16 MiB is a corrupt file

# GGUF value type -> byte size, for the fixed-size types.
_FIXED_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_T_STRING = 8
_T_ARRAY = 9
_KEY_ARCH = "general.architecture"


# A read at or below this size needs no check against the file size. A short read
# shows the end of the file.
_SMALL_READ = 1 << 20


def _remaining(f: BinaryIO) -> int:
    """Return the number of bytes between the current position and the end of the file."""
    pos = f.tell()
    end = f.seek(0, 2)
    f.seek(pos)
    return end - pos


def _read(f: BinaryIO, n: int) -> bytes:
    # A corrupt count can ask for more bytes than the file holds, or more than ``read``
    # accepts (OverflowError, MemoryError). Refuse it as a short file, before the read.
    if n > _SMALL_READ and n > _remaining(f):
        raise ValueError(f"{n} bytes requested, but the file ends inside the GGUF header")
    data = f.read(n)
    if len(data) != n:
        raise ValueError("file ends inside the GGUF header")
    return data


def _u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read(f, 4))[0]


def _u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read(f, 8))[0]


def _string(f: BinaryIO) -> str:
    n = _u64(f)
    if n > _MAX_STRING:
        raise ValueError(f"string of {n} bytes in the GGUF header")
    return _read(f, n).decode("utf-8")


def _skip_value(f: BinaryIO, vtype: int) -> None:
    if vtype in _FIXED_SIZE:
        _read(f, _FIXED_SIZE[vtype])
    elif vtype == _T_STRING:
        _string(f)
    elif vtype == _T_ARRAY:
        elem = _u32(f)
        count = _u64(f)
        if elem in _FIXED_SIZE:
            _read(f, _FIXED_SIZE[elem] * count)
        else:
            # Each element takes at least one byte, so a larger count is corrupt.
            if count > _remaining(f):
                raise ValueError(f"array of {count} elements in the GGUF header, but the file is shorter")
            for _ in range(count):
                _skip_value(f, elem)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_header(path: str | Path) -> tuple[str | None, list[str]]:
    """Return ``(general.architecture, tensor names)`` from the GGUF header of ``path``.

    The architecture is ``None`` when the file has no string ``general.architecture``
    key. The names are in file order.

    Raises ``ValueError`` when the file is not a little-endian GGUF v2 or v3 file,
    and ``OSError`` when it cannot be opened.
    """
    with open(path, "rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError("not a GGUF file (bad magic)")
        version = _u32(f)
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors = _u64(f)
        n_kv = _u64(f)
        arch: str | None = None
        for _ in range(n_kv):
            key = _string(f)
            vtype = _u32(f)
            if key == _KEY_ARCH and vtype == _T_STRING:
                arch = _string(f)
            else:
                _skip_value(f, vtype)
        names: list[str] = []
        for _ in range(n_tensors):
            names.append(_string(f))
            n_dims = _u32(f)
            _read(f, 8 * n_dims)  # dims
            _read(f, 4 + 8)  # type, offset
        return arch, names


def read_tensor_names(path: str | Path) -> list[str]:
    """Return the tensor names in the GGUF header of ``path``, in file order.

    Raises ``ValueError`` when the file is not a little-endian GGUF v2 or v3 file,
    and ``OSError`` when it cannot be opened.
    """
    return read_gguf_header(path)[1]


def dropped_tensors(path: str | Path) -> dict[str, str]:
    """Map each base tensor of the adapter at ``path`` that ``seq_routed`` drops to its reason.

    The keys are base names (the lora_a and lora_b suffix removed), in file order.
    """
    out: dict[str, str] = {}
    for name in read_tensor_names(path):
        reason = dropped_reason(name)
        if reason is not None:
            out.setdefault(_base_name(name), reason)
    return out


def refuse_dropped_adapters(adapters: Iterable[tuple[str, str | Path]]) -> None:
    """Raise :class:`SeqRoutedAdapterRefused` when ``seq_routed`` would drop part of an adapter.

    ``adapters`` is a sequence of ``(adapter name, GGUF path)``. The message names
    every offending adapter with its tensors and the reason. An unreadable file is
    refused too: the check cannot show that nothing is dropped.
    """
    problems: list[str] = []
    for name, path in adapters:
        try:
            dropped = dropped_tensors(path)
        except (OSError, ValueError) as exc:
            problems.append(f"adapter {name!r} ({path}): cannot read the tensor names to check coverage: {exc}")
            continue
        if not dropped:
            continue
        items = list(dropped.items())
        shown = "; ".join(f"{t}: {why}" for t, why in items[:_MAX_NAMED])
        more = f"; and {len(items) - _MAX_NAMED} more" if len(items) > _MAX_NAMED else ""
        problems.append(
            f"adapter {name!r} ({path}) targets {len(items)} tensor(s) that seq_routed drops: {shown}{more}"
        )
    if problems:
        raise SeqRoutedAdapterRefused(
            "backend 'seq_routed' would apply these adapters only in part and drop the rest without a message. "
            + " | ".join(problems)
            + ". Serve such an adapter with backend='context_pool', which applies every tensor."
        )
