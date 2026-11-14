"""Which GPU is this process allowed to use, on whichever backend is compiled in.

**The device-visibility variable is per BACKEND, and using the wrong one fails
silently in both directions**:

* ``CUDA_VISIBLE_DEVICES`` is what a CUDA build reads. **The Vulkan backend
  ignores it completely.** A gate that PINS a card with it runs on ``Vulkan0``
  whatever the lease decided, and on a rig whose other card is serving that is
  an overcommit that spills to host RAM and still reports success.
* A gate that GATES READINESS on it skips itself entirely on a Vulkan rig that
  could have run it. A skip reads as "not applicable", which is the silent pass
  the standing rules forbid.

Eleven call sites carried the second bug and one carried the first. They now ask
here instead of naming a vendor variable.

This module deliberately does NOT decide which card to use. That is the GPU
lease's job (``ci/runner/gpu-lease.sh``); this only reports what the environment
was told.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "GPU_BACKEND_LABEL",
    "UNKNOWN_BACKEND_LABEL",
    "BackendUnknownError",
    "VISIBLE_DEVICE_VARS",
    "gpu_env_ready",
    "lib_backend_label",
    "visible_devices",
]

# The ``EngineConfig.backend`` label for a gate that decodes on a GPU.
#
# The label is free text and nothing in ``src/inferference`` reads it: the
# router never branches on it, and a gate that builds an ``EngineConfig`` builds
# no ``LlamaEngineFingerprint`` from it. So a vendor name here would be a claim
# nobody checks, and it was false on a Vulkan build. Use this one neutral label
# at a site that builds no fingerprint. A label that enters a cache key or a
# recorded hash comes from ``lib_backend_label`` instead.
GPU_BACKEND_LABEL = "gpu"

# What ``lib_backend_label`` returns when the library set gives no evidence. It
# is not a backend. Two different backends can both produce it, so a fingerprint
# refuses to build a cache key from it, and a baseline gate refuses to store one.
UNKNOWN_BACKEND_LABEL = "unknown"

# GPU backends that ggml builds as a separate ``libggml-<name>.so``.
_GPU_BACKENDS = ("cann", "cuda", "hip", "metal", "musa", "opencl", "sycl", "vulkan")
_BACKEND_LIB = re.compile(r"^libggml-(" + "|".join(_GPU_BACKENDS) + r")\.(?:so|dylib)")
_BACKEND_FLAG = re.compile(r"-DGGML_(" + "|".join(b.upper() for b in _GPU_BACKENDS) + r")=(?:ON|1|TRUE)\b")

# Ordered by how specific they are, not by preference: whichever is SET wins.
# GGML_VK_VISIBLE_DEVICES is ggml's Vulkan device filter; CUDA_VISIBLE_DEVICES
# is the CUDA one. HIP_VISIBLE_DEVICES is listed for a future ROCm build — it
# costs nothing now and stops the next backend repeating this bug.
VISIBLE_DEVICE_VARS = (
    "GGML_VK_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
)


def visible_devices(env: Mapping[str, str] | None = None) -> str | None:
    """The device-visibility selection in force, or ``None`` if unset.

    An EMPTY value is a deliberate "no GPU" (several refusal gates set
    ``CUDA_VISIBLE_DEVICES=""`` on purpose to force a CPU-only child), so it is
    returned as ``""`` and is distinct from ``None``.
    """
    src: Mapping[str, str] = os.environ if env is None else env
    for var in VISIBLE_DEVICE_VARS:
        if var in src:
            return src[var]
    return None


def gpu_env_ready(env: Mapping[str, str] | None = None) -> bool:
    """Is a lib path pinned AND some device made visible?

    The readiness question every GPU gate in this repo asks. It does NOT probe
    the library — callers still check ``llama_supports_gpu_offload()``, which is
    the only thing that can tell a GPU build from a CPU one.
    """
    src: Mapping[str, str] = os.environ if env is None else env
    visible = visible_devices(src)
    return bool(src.get("LLAMA_CPP_LIB_PATH")) and bool(visible)


class BackendUnknownError(ValueError):
    """The library set gives no evidence of its backend."""


def _loaded_lib_dir() -> Path | None:
    """The directory of the libllama that this process loaded, or ``None``.

    Reads ``sys.modules`` only. It never imports llama-cpp-python, so the call
    costs nothing on a machine with no native stack.
    """
    module = sys.modules.get("llama_cpp.llama_cpp")
    name = getattr(getattr(module, "_lib", None), "_name", None)
    if not isinstance(name, str) or not os.path.isabs(name):
        return None
    return Path(name).parent


def lib_backend_label(
    lib_dir: str | os.PathLike[str] | None = None,
    manifest: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
    *,
    strict: bool = False,
) -> str:
    """The GPU backend of the loaded library set, as a label for a fingerprint.

    This label enters ``LlamaEngineFingerprint.cache_key()``. A CUDA run and a
    Vulkan run must never share a key, so the label comes from the library and
    never from a constant. The evidence, in order:

    1. ``libggml-<backend>.so`` files in the library directory. A shared ggml
       build ships one such file for each compiled backend. Several backends
       give a sorted ``a+b`` label.
    2. ``-DGGML_<BACKEND>=ON`` in the ``cmake_flags`` of the build manifest
       (the manifest next to the library directory, or the one passed in).
       This covers a build with the backend inside ``libggml.so``.
    3. Evidence of a CPU-only build: a ``libggml-cpu.so`` and no GPU library, or
       a manifest with ``cmake_flags`` and no GPU flag: ``cpu``.
    4. Anything else gives ``UNKNOWN_BACKEND_LABEL``: no directory, an unreadable
       or empty directory, or a ``libggml.so`` with no manifest (the backend can
       sit inside it). The label claims no backend, and the fingerprint refuses
       to build a cache key from it. With ``strict=True`` this case raises
       ``BackendUnknownError`` instead. Use ``strict`` for a GPU run.

    The directory is, in order: ``lib_dir``; when ``env`` is ``None``, the
    directory of the libllama that this process loaded; then
    ``LLAMA_CPP_LIB_PATH`` in ``env``. The loaded library is the one that runs,
    so it wins over the variable that was meant to select it.
    """
    label = _lib_backend_label(lib_dir, manifest, env)
    if label == UNKNOWN_BACKEND_LABEL and strict:
        raise BackendUnknownError(
            "the library set gives no evidence of its backend: no libggml-<backend>.so, "
            "no -DGGML_<BACKEND>=ON in a build manifest, and no CPU-only evidence. "
            "Point LLAMA_CPP_LIB_PATH at a built library set that has its build-manifest.json beside it."
        )
    return label


def _lib_backend_label(
    lib_dir: str | os.PathLike[str] | None,
    manifest: Mapping[str, object] | None,
    env: Mapping[str, str] | None,
) -> str:
    src: Mapping[str, str] = os.environ if env is None else env
    directory: Path | None = None
    if lib_dir is not None:
        directory = Path(os.fspath(lib_dir)).expanduser()
    elif env is None and (loaded := _loaded_lib_dir()) is not None:
        directory = loaded
    elif raw := src.get("LLAMA_CPP_LIB_PATH"):
        directory = Path(raw).expanduser()
    if directory is None:
        return UNKNOWN_BACKEND_LABEL
    try:
        names = sorted(entry.name for entry in directory.iterdir())
    except OSError:
        return UNKNOWN_BACKEND_LABEL
    found = {m.group(1) for name in names if (m := _BACKEND_LIB.match(name))}
    flags: str | None = None
    if not found:
        flags = _manifest_flags(directory, manifest)
        found = {m.group(1).lower() for m in _BACKEND_FLAG.finditer(flags or "")}
    if found:
        return "+".join(sorted(found))
    if any(name.startswith("libggml-cpu.") for name in names) or flags:
        return "cpu"
    return UNKNOWN_BACKEND_LABEL


def _manifest_flags(directory: Path, manifest: Mapping[str, object] | None) -> str | None:
    """The ``cmake_flags`` text of the build manifest, or ``None`` when there is none."""
    if manifest is None:
        for candidate in (directory.parent / "build-manifest.json", directory / "build-manifest.json"):
            try:
                data = json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                manifest = data
                break
    return str((manifest or {}).get("cmake_flags") or "") or None
