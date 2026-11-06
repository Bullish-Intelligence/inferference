"""Placement proof for a local eval run.

A speed number from this rig means nothing without it. Measured 2026-09-04 and
re-confirmed 2026-09-05 (007 reports 01 and 04): a model that does not fit its
card does NOT fail to load. The router returns ``{"success":true}``, reports the
child ``loaded``, assigns every layer to the named Vulkan device — and serves
from host memory over PCIe at about **1/14th** the speed. Card1 went to
1512 MiB of GTT with nothing anywhere saying so.

So a local run that cannot prove where the weights are is marked **invalid**,
not "successful with a caveat". A tok/s figure from a spilled model is not a
slow result, it is a measurement of the wrong thing.

The eval starts after the model loads. It requires the router's load-time
evidence and verifies the same model remains loaded after the run.

Do NOT gate on ``CPU_Mapped model buffer size``. It is normal — a perfectly
resident 9B reports 545 MiB of it for non-offloaded embedding weights.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DeviceReading",
    "PlacementProof",
    "ServingConfig",
    "read_device",
    "read_serving_config",
]


class DeviceReading(BaseModel):
    """One card's own accounting at one moment."""

    model_config = ConfigDict(extra="forbid")

    card: int
    pci: str
    vram_used_mib: float
    vram_total_mib: float
    gtt_used_mib: float


def read_device(card: int) -> DeviceReading:
    """Read a DRM card from amdgpu sysfs.

    sysfs is the vendor-truthful source on this rig: ROCm dropped gfx900 after
    5.7, so there is no rocm-smi to ask, and ``ci/runner/gpu-lease.sh`` reads the
    same files.
    """
    base = Path(f"/sys/class/drm/card{card}/device")
    try:
        total = int((base / "mem_info_vram_total").read_text().strip())
        used = int((base / "mem_info_vram_used").read_text().strip())
        gtt = int((base / "mem_info_gtt_used").read_text().strip())
        pci = base.resolve().name
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot read card {card} at {base}: {exc}. A local run cannot be validated "
            f"without the card's own accounting; refusing to call the result valid."
        ) from exc
    return DeviceReading(
        card=card,
        pci=pci,
        vram_used_mib=used / 1048576,
        vram_total_mib=total / 1048576,
        gtt_used_mib=gtt / 1048576,
    )


class ServingConfig(BaseModel):
    """How the router is actually serving this model, read from the router.

    Read rather than assumed. The eval's own ``--concurrency`` says how many
    requests it puts in flight; ``lanes`` says how many the server can decode at
    once. They are different numbers and conflating them is how a "batching"
    result turns out to be a measurement of queueing.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    status: str
    device: str | None = None
    lanes: int | None = Field(default=None, description="the server's --parallel: real decode lanes")
    ctx_size: int | None = Field(default=None, description="TOTAL cache; per-lane context is ctx/lanes")
    preset: str | None = None
    residency: dict[str, Any] | None = None

    @property
    def ctx_per_lane(self) -> int | None:
        if self.ctx_size is None or not self.lanes:
            return None
        return self.ctx_size // self.lanes


def _parse_preset(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith((";", "#", "[")):
            continue
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def read_serving_config(base_url: str, model: str, timeout: float = 15.0) -> ServingConfig:
    """Ask the router how it is serving ``model``.

    ``base_url`` is the OpenAI-compatible URL the eval talks to; the router's
    model registry sits one level up, at ``/models``.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    url = f"{root}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read() or b"null")
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"cannot read the router's model registry at {url}: {exc}") from exc

    entries = payload.get("data", []) if isinstance(payload, dict) else []
    for entry in entries:
        if entry.get("id") != model:
            continue
        status = entry.get("status") or {}
        preset = status.get("preset")
        fields = _parse_preset(preset) if isinstance(preset, str) else {}
        return ServingConfig(
            model=model,
            status=status.get("value", "?"),
            device=fields.get("device"),
            lanes=_as_int(fields.get("parallel")),
            ctx_size=_as_int(fields.get("ctx-size")),
            preset=preset if isinstance(preset, str) else None,
            residency=entry.get("inferference_residency"),
        )
    known = sorted(e.get("id", "?") for e in entries)
    raise RuntimeError(
        f"the router does not serve a model called '{model}'. It serves: {known}. "
        f"There is no demand loading: a model that is not resident is an error, never a load."
    )


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


class PlacementProof(BaseModel):
    """Where the weights were, before and after the run that was timed."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(description="proven | spilled | short | unproven")
    reason: str = ""
    device: str | None = None
    card: int | None = None
    pci: str | None = None
    lanes: int | None = None
    ctx_size: int | None = None
    ctx_per_lane: int | None = None
    expected_mib: float | None = None
    before: DeviceReading | None = None
    after: DeviceReading | None = None

    @property
    def valid(self) -> bool:
        return self.status == "proven"

    @classmethod
    def unproven(cls, reason: str) -> PlacementProof:
        return cls(status="unproven", reason=reason)

    @classmethod
    def evaluate(
        cls,
        *,
        serving: ServingConfig,
        card: int,
        before: DeviceReading,
        after: DeviceReading,
        expected_mib: float | None = None,
    ) -> PlacementProof:
        """Judge one run's placement from its two readings."""
        common: dict[str, Any] = {
            "device": serving.device,
            "card": card,
            "pci": after.pci,
            "lanes": serving.lanes,
            "ctx_size": serving.ctx_size,
            "ctx_per_lane": serving.ctx_per_lane,
            "expected_mib": expected_mib,
            "before": before,
            "after": after,
        }
        gtt_rise = after.gtt_used_mib - before.gtt_used_mib
        if gtt_rise > 0:
            return cls(
                status="unproven",
                reason=(
                    f"GTT rose by {gtt_rise:.6f} MiB on card {card} during the run. "
                    "The readings cannot prove VRAM residency."
                ),
                **common,
            )
        if expected_mib is not None and after.vram_used_mib < 0.75 * expected_mib:
            return cls(
                status="short",
                reason=(
                    f"card {card} holds {after.vram_used_mib:.0f} MiB but the model's measured "
                    f"footprint is {expected_mib:.0f} MiB. The model may not be on the card this "
                    f"run names."
                ),
                **common,
            )
        evidence = serving.residency
        if not isinstance(evidence, dict) or not isinstance(evidence.get("cards"), list) or len(evidence["cards"]) != 1:
            return cls(status="unproven", reason="No single-card load-time residency evidence.", **common)
        entry = evidence["cards"][0]
        if not isinstance(entry, dict):
            return cls(status="unproven", reason="Invalid load-time residency evidence.", **common)
        fields = [entry.get(k) for k in ("layers", "gtt_before", "gtt_after", "vram_before", "vram_after")]
        if any(type(value) is not int or value < 0 for value in fields):
            return cls(status="unproven", reason="Invalid load-time residency counters.", **common)
        layers, gtt_before, gtt_after, vram_before, vram_after = fields
        stamp = evidence.get("verified_mono_us")
        valid = (
            serving.status == "loaded"
            and serving.device == entry.get("device") == evidence.get("output_device")
            and entry.get("pci") == before.pci == after.pci
            and before.card == after.card == card
            and type(stamp) is int
            and stamp > 0
            and layers > 0
            and type(evidence.get("layer_count")) is int
            and layers == evidence.get("layer_count")
            and gtt_after <= gtt_before
            and vram_after > vram_before
            and before.gtt_used_mib * 1048576 <= gtt_after
            and after.gtt_used_mib * 1048576 <= gtt_after
        )
        if not valid:
            return cls(status="unproven", reason="Load-time residency evidence does not match the run.", **common)
        return cls(status="proven", reason="The router proved load-time residency and GTT stayed flat.", **common)
