"""Render experiment analysis values as Markdown."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from .analysis.compare import Comparison

__all__ = ["ReportError", "render_analysis", "render_comparison"]

_SLOPE_FIELDS = ("slope_c_per_s", "band_c", "method", "n", "se_ar1")


class ReportError(ValueError):
    """The report input does not contain enough data to render a value safely."""


def _items(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return None


def _is_slope(data: Mapping[str, Any]) -> bool:
    return "slope_c_per_s" in data


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render_value(name: str, value: Any, lines: list[str], *, indent: int = 0) -> None:
    prefix = "  " * indent
    data = _items(value)
    if data is not None:
        if _is_slope(data):
            missing = [field for field in _SLOPE_FIELDS if field not in data]
            if missing:
                raise ReportError(f"slope is missing required fields: {', '.join(missing)}")
            band = data["band_c"]
            if not isinstance(band, (list, tuple)) or len(band) != 2:
                raise ReportError("slope band_c must contain its two recorded bounds")
            lines.append(
                f"{prefix}- Band slope: {_format_scalar(data['slope_c_per_s'])} C/s; "
                f"band {band[0]}–{band[1]} C; method={_format_scalar(data['method'])}; "
                f"n={_format_scalar(data['n'])}; AR(1) SE={_format_scalar(data['se_ar1'])} C/s"
            )
            return
        if not data:
            lines.append(f"{prefix}- {name}: {{}}")
            return
        lines.append(f"{prefix}- {name}:")
        for key, child in data.items():
            _render_value(str(key), child, lines, indent=indent + 1)
        return

    if isinstance(value, (list, tuple)):
        if not value:
            lines.append(f"{prefix}- {name}: []")
            return
        lines.append(f"{prefix}- {name}:")
        for index, child in enumerate(value):
            _render_value(str(index), child, lines, indent=indent + 1)
        return

    if "slope" in name.casefold():
        raise ReportError(f"{name} has no band, method, sample count, and AR(1) standard error")
    if name == "offset_c":
        lines.append(f"{prefix}- Offset: {_format_scalar(value)} C")
        return
    lines.append(f"{prefix}- {name}: {_format_scalar(value)}")


def _render_section(name: str, value: Any, lines: list[str]) -> None:
    title = name.capitalize()
    lines.extend((f"## {title}", ""))
    data = _items(value) if value is not None else None
    if data is None:
        lines.append("- No analysis values supplied.")
    elif not data:
        lines.append("- No analysis values supplied.")
    else:
        for key, child in data.items():
            _render_value(str(key), child, lines)
    lines.append("")


def render_analysis(analysis: Mapping[str, Any]) -> str:
    """Render the supplied outcome and decode and prefill analysis values."""
    if not isinstance(analysis, Mapping):
        raise ReportError("analysis must be a mapping")

    lines = ["# Analysis", ""]
    outcome = analysis.get("outcome")
    if outcome is not None:
        outcome_data = _items(outcome)
        if outcome_data is None:
            raise ReportError("outcome must be a mapping")
        state = outcome_data.get("state")
        if state is not None:
            if state == "bounded":
                lines.append("**Outcome:** bounded — not a success.")
            else:
                lines.append(f"**Outcome:** {_format_scalar(state)}")
        for key, value in outcome_data.items():
            if key != "state":
                _render_value(str(key), value, lines)
        lines.append("")

    extra = {key: value for key, value in analysis.items() if key not in {"outcome", "decode", "prefill"}}
    if extra:
        lines.extend(("## Analysis values", ""))
        for key, value in extra.items():
            _render_value(str(key), value, lines)
        lines.append("")

    _render_section("decode", analysis.get("decode"), lines)
    _render_section("prefill", analysis.get("prefill"), lines)
    return "\n".join(lines).rstrip() + "\n"


def render_comparison(comparison: Comparison) -> str:
    """Render one comparison with its recorded scatter and resolved label."""
    if not isinstance(comparison, Comparison):
        raise ReportError("comparison must be a Comparison")

    lines = [
        "# Comparison",
        "",
        f"Endpoint: `{comparison.endpoint}`",
        "",
        f"- A: mean={comparison.a.mean}; SD={comparison.a.sd}; n={comparison.a.n}",
        f"- B: mean={comparison.b.mean}; SD={comparison.b.sd}; n={comparison.b.n}",
        f"- Delta (B − A): {comparison.delta}; scatter={comparison.scatter}; label={comparison.label}",
        f"- Resolved: {_format_scalar(comparison.resolved)}",
        f"- Paired delta: {_format_scalar(comparison.paired_delta)}",
        "- Allowed differences: "
        + (", ".join(comparison.allowed_differences) if comparison.allowed_differences else "none"),
    ]
    return "\n".join(lines) + "\n"
