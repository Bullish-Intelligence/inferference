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

import os
from collections.abc import Mapping

__all__ = ["VISIBLE_DEVICE_VARS", "visible_devices", "gpu_env_ready"]

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
