"""GPU environment facts for the bench harness.

The isolation rule is load-bearing: a job is assigned one card, and a
benchmark must never decode on a card the serving router owns. A benchmark also
records an nvidia-smi snapshot of the cards it can see, so a result carries the
device identity it was measured on.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

_QUERY = "index,name,uuid,driver_version,memory.total,memory.used"


def ci_device() -> str:
    """The CI device index (LLAMA_CI_GPU), defaulting to 0.

    Named-role rule: asserts compare against this variable, never the
    literal index and never a presence check.
    """
    return os.environ.get("LLAMA_CI_GPU", "0")


def require_ci_device() -> None:
    """Fail closed unless CUDA_VISIBLE_DEVICES is the CI device.

    The message names both the expected and the actual device so a mis-routed
    run is refused loudly, not silently.
    """
    expected = ci_device()
    actual = os.environ.get("CUDA_VISIBLE_DEVICES")
    if actual != expected:
        raise RuntimeError(f"CI device mismatch: CUDA_VISIBLE_DEVICES={actual!r}, expected {expected!r} (LLAMA_CI_GPU)")


def gpu_snapshot() -> dict[str, Any]:
    """nvidia-smi snapshot of every GPU (index, name, uuid, driver, memory).

    A missing ``nvidia-smi`` binary reads as returncode 127 with empty stdout, so
    ``require_gpu0_only`` still fails closed on it.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader"], capture_output=True, text=True
        )
    except OSError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def visible_gpu_uuid() -> str | None:
    """The UUID of the GPU selected by CUDA_VISIBLE_DEVICES, else None.

    The index is a function of CUDA_VISIBLE_DEVICES and reads 0 for whichever
    card was exposed; only the UUID names a card. A missing
    ``nvidia-smi`` binary also reads as None.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    try:
        phys = int(visible.split(",")[0])
    except ValueError:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    return lines[phys] if phys < len(lines) else None


def require_gpu0_only(snapshot: dict[str, Any]) -> None:
    """Fail-closed GPU isolation: the VRAM floor check.

    Expects two GPUs. Rejects the run unless the CI device holds the model
    (>= 4000 MiB used) and the server device is idle (<= 200 MiB used) — i.e.
    the model landed on the CI device, not the server's device.
    A busy server device must stop the benchmark before any decode happens.
    """
    lines = snapshot["stdout"].splitlines()
    if len(lines) < 2:
        raise RuntimeError("expected two GPUs in nvidia-smi snapshot")
    values = [line.split(", ") for line in lines]
    ci = int(ci_device())
    srv = int(os.environ.get("LLAMA_SERVER_GPU", "1"))
    used_ci = int(values[ci][-1].split()[0])
    used_srv = int(values[srv][-1].split()[0])
    if used_ci < 4000 or used_srv > 200:
        raise RuntimeError(f"GPU isolation rejected: CI device={used_ci} MiB server device={used_srv} MiB")


def driver_version(snapshot: dict[str, Any]) -> str | None:
    """The CUDA driver version from a snapshot (nvidia-smi column 4)."""
    lines = snapshot["stdout"].splitlines()
    if not lines:
        return None
    fields = lines[0].split(", ")
    return fields[3] if len(fields) > 3 else None
