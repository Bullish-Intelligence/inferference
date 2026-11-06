"""Load a model onto a device and confirm the placement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadReport:
    """What the loader said about one load."""

    success: bool
    status: str
    device: str
    layers_total: int
    layers_on_device: int
    model_buffer_mib: float
    device_vram_used_mib: float


def verify_resident(report: LoadReport) -> bool:
    """Confirm the model sits on the device it was asked for."""
    return (
        report.success
        and report.status == "loaded"
        and report.layers_on_device == report.layers_total
        and report.model_buffer_mib > 0
        and report.device_vram_used_mib > report.model_buffer_mib
    )
