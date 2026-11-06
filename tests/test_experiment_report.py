"""Tests for Markdown rendering of experiment analysis results."""

from __future__ import annotations

from inferference.experiment.analysis.compare import Comparison
from inferference.experiment.analysis.estimators import SlopeEstimate
from inferference.experiment.analysis.scatter import Scatter
from inferference.experiment.report import ReportError, render_analysis, render_comparison


def _slope() -> SlopeEstimate:
    return SlopeEstimate(
        slope_c_per_s=0.25,
        se_ols=0.02,
        se_ar1=0.03,
        rho1=0.2,
        n=7,
        n_eff=4.7,
        dof=2.7,
        residual_rms=0.1,
        band_c=(52.0, 62.0),
        method="ols+ar1",
        t_start_mono=1.0,
        t_end_mono=8.0,
    )


def test_slope_line_shows_band_method_sample_count_and_ar1_error() -> None:
    report = render_analysis({"decode": {"band_slope": _slope()}})

    assert "Band slope: 0.25 C/s; band 52.0–62.0 C; method=ols+ar1; n=7; AR(1) SE=0.03 C/s" in report


def test_comparison_delta_line_shows_scatter_and_resolved_label() -> None:
    comparison = Comparison(
        endpoint="decode_tok_s",
        a=Scatter(endpoint="decode_tok_s", n=3, mean=10.0, sd=1.0, values=(9.0, 10.0, 11.0)),
        b=Scatter(endpoint="decode_tok_s", n=3, mean=14.0, sd=2.0, values=(12.0, 14.0, 16.0)),
        delta=4.0,
        paired_delta=4.0,
        scatter=2.0,
        resolved=True,
        label="resolved",
        allowed_differences=("control.pp_table",),
    )

    report = render_comparison(comparison)

    assert "Delta (B − A): 4.0; scatter=2.0; label=resolved" in report
    assert "Resolved: true" in report
    assert "Allowed differences: control.pp_table" in report


def test_decode_and_prefill_always_have_separate_sections() -> None:
    report = render_analysis({"decode": {"rate": 12.0}})

    assert "## Decode" in report
    assert "- rate: 12.0" in report
    assert "## Prefill" in report
    assert "- No analysis values supplied." in report
    assert report.index("## Decode") < report.index("## Prefill")


def test_bounded_outcome_is_never_rendered_as_a_success() -> None:
    report = render_analysis({"outcome": {"state": "bounded", "failure_boundary": "arm_bound"}})

    assert "**Outcome:** bounded — not a success." in report
    assert "failure_boundary: arm_bound" in report
    assert "success" not in report.casefold().replace("not a success", "")


def test_offset_and_slope_are_separate_report_values() -> None:
    report = render_analysis({"decode": {"offset_c": 2.5, "band_slope": _slope()}})
    offset_line = next(line for line in report.splitlines() if "Offset:" in line)
    slope_line = next(line for line in report.splitlines() if "Band slope:" in line)

    assert offset_line == "- Offset: 2.5 C"
    assert "0.25 C/s" in slope_line
    assert offset_line != slope_line


def test_a_bare_slope_refuses_incomplete_report_data() -> None:
    try:
        render_analysis({"decode": {"slope_c_per_s": 0.25}})
    except ReportError as exc:
        assert "band, method, sample count, and AR(1) standard error" in str(exc)
    else:
        raise AssertionError("a bare slope must not render without its uncertainty context")
