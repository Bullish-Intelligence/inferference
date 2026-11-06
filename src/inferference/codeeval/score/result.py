"""What a grading layer returns."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["LayerResult", "TaskScore", "Status"]

Status = Literal["pass", "fail", "skip"]


class LayerResult(BaseModel):
    """One layer's verdict on one task."""

    model_config = ConfigDict(extra="forbid")

    layer: str
    status: Status
    reason: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True only on an explicit pass. A skip is not a pass."""
        return self.status == "pass"


class TaskScore(BaseModel):
    """Every layer's verdict on one task, plus the request outcome."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    family: str
    difficulty: int
    request_ok: bool
    layers: list[LayerResult] = Field(default_factory=list)

    def layer(self, name: str) -> LayerResult | None:
        for r in self.layers:
            if r.layer == name:
                return r
        return None

    def status_of(self, name: str) -> Status | None:
        found = self.layer(name)
        return found.status if found else None
