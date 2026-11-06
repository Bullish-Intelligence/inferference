"""The run artifact: an envelope, and the summary a human reads.

There is no single headline score, by design. A run reports a
vector, sliced by family, so a fast-and-wrong configuration cannot beat a
slow-and-right one on one number.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .drive import DriveResult

__all__ = ["RunReport", "summarize", "write_report"]

LAYERS = ("extract", "parse", "ruff", "pytest", "judge")


class RunReport(BaseModel):
    """Envelope v4: v3 plus batch mode, repeats and a placement proof.

    v4 is what makes two runs comparable across the axes this rig is tuned on:
    build, device, model artifact, context size, lane count, and single versus
    batch. Anything that changes a number belongs in here, because a figure
    whose conditions are not recorded cannot be compared with anything later.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 4
    run_id: str
    kind: str = "codeeval"
    mode: str = "single"
    endpoint: dict[str, Any] = Field(default_factory=dict)
    corpus: dict[str, Any] = Field(default_factory=dict)
    axes: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    placement: dict[str, Any] | None = None
    quality: dict[str, Any] = Field(default_factory=dict)
    speed: dict[str, Any] = Field(default_factory=dict)
    throughput: dict[str, Any] = Field(default_factory=dict)
    thermal: dict[str, Any] = Field(default_factory=dict)
    serving: dict[str, Any] = Field(default_factory=dict)
    validity: dict[str, Any] = Field(default_factory=dict)
    records: str = "records/"
    command: list[str] = Field(default_factory=list)


def _pct(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(100.0 * numerator / denominator, 1)


def _stats(values: list[float | None]) -> dict[str, float] | None:
    """p50/p95/mean over the values that exist. Missing values are dropped, not
    treated as zero: an endpoint that does not report a metric must not drag
    that metric's average down."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    clean.sort()
    p95_index = min(len(clean) - 1, int(round(0.95 * (len(clean) - 1))))
    return {
        "n": len(clean),
        "p50": round(statistics.median(clean), 2),
        "p95": round(clean[p95_index], 2),
        "mean": round(statistics.fmean(clean), 2),
    }


def summarize(result: DriveResult) -> dict[str, Any]:
    """Quality vector and speed distribution, overall and per family."""
    attempted = [o for o in result.outcomes if o.skipped is None]
    skipped = [o for o in result.outcomes if o.skipped is not None]

    def layer_counts(outcomes: list) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for layer in LAYERS:
            statuses = Counter(o.score.status_of(layer) for o in outcomes if o.score.status_of(layer))
            if not statuses:
                continue
            graded = statuses["pass"] + statuses["fail"]
            out[layer] = {
                "pass": statuses["pass"],
                "fail": statuses["fail"],
                "skip": statuses["skip"],
                "pass_rate_pct": _pct(statuses["pass"], graded),
            }
        return out

    ruff_total = 0
    ruff_by_code: Counter[str] = Counter()
    for outcome in attempted:
        found = outcome.score.layer("ruff")
        if found and found.detail:
            ruff_total += int(found.detail.get("count", 0))
            ruff_by_code.update(found.detail.get("by_code", {}))

    by_family: dict[str, Any] = {}
    grouped: dict[str, list] = defaultdict(list)
    for outcome in attempted:
        grouped[outcome.family].append(outcome)
    for family, outcomes in sorted(grouped.items()):
        by_family[family] = {
            "n": len(outcomes),
            "layers": layer_counts(outcomes),
        }

    request_failures = [o for o in attempted if not o.score.request_ok]
    truncated = [o for o in attempted if o.response.finish_reason == "length"]

    speed = {
        "ttft_ms": _stats([_t(o, "ttft_ms") for o in attempted]),
        "e2e_ms": _stats([_t(o, "e2e_ms") for o in attempted]),
        "client_decode_tok_s": _stats([_t(o, "client_decode_tok_s") for o in attempted]),
        "server_decode_tok_s": _stats([_t(o, "decode_tok_s") for o in attempted]),
        "server_prefill_tok_s": _stats([_t(o, "prefill_tok_s") for o in attempted]),
        "n_prompt": _stats([_t(o, "n_prompt") for o in attempted]),
        "n_predicted": _stats([_t(o, "n_predicted") for o in attempted]),
        "n_reasoning_tokens": _stats([_t(o, "n_reasoning_tokens") for o in attempted]),
        "ttft_any_ms": _stats([_t(o, "ttft_any_ms") for o in attempted]),
        "wall_s": round(result.wall_s, 1),
    }

    # Aggregate versus per-request. Raising concurrency is SUPPOSED to improve
    # the first and worsen the second; reporting only one of them hides the
    # trade the run was made to measure.
    throughput = {
        "mode": result.mode,
        "client_concurrency": result.concurrency,
        "wall_s": round(result.wall_s, 2),
        "active_s": round(result.active_s, 2),
        "idle_s": round(result.idle_s, 2),
        "completion_tokens": result.completion_tokens,
        "aggregate_tok_s": (None if result.aggregate_tok_s is None else round(result.aggregate_tok_s, 2)),
        "wall_tok_s": (None if result.wall_tok_s is None else round(result.wall_tok_s, 2)),
        "thermal": result.thermal,
        "aborted": result.aborted,
        "requests_per_s": (None if result.requests_per_s is None else round(result.requests_per_s, 3)),
        "per_request_decode_tok_s": speed["server_decode_tok_s"],
        "per_request_e2e_ms": speed["e2e_ms"],
        "per_request_ttft_ms": speed["ttft_ms"],
    }

    return {
        "tasks_total": len(result.outcomes),
        "tasks_attempted": len(attempted),
        "tasks_skipped": len(skipped),
        "skipped_detail": [{"task_id": o.task_id, "reason": o.skipped} for o in skipped],
        "truncated": [o.task_id for o in truncated],
        "request_failures": [{"task_id": o.task_id, "error": o.response.error} for o in request_failures],
        "layers": layer_counts(attempted),
        "ruff_violations_total": ruff_total,
        "ruff_by_code": dict(ruff_by_code.most_common()),
        "by_family": by_family,
        "speed": speed,
        "throughput": throughput,
    }


def _t(outcome: Any, key: str) -> float | None:
    value = (outcome.response.timings or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def write_report(report: RunReport, summary: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "envelope.json").write_text(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    (out_dir / "summary.md").write_text(render_summary(report, summary))


def render_summary(report: RunReport, s: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# codeeval run `{report.run_id}`\n")
    add(
        f"- endpoint: `{report.endpoint.get('name')}` "
        f"model `{report.endpoint.get('model')}` "
        f"(hosted: {report.endpoint.get('hosted')})"
    )
    add(f"- mode: `{report.mode}`  axes: `{report.axes}`")
    if report.serving:
        add(
            f"- serving: device `{report.serving.get('device')}` "
            f"lanes `{report.serving.get('lanes')}` "
            f"ctx `{report.serving.get('ctx_size')}` "
            f"(per lane `{report.serving.get('ctx_per_lane')}`)"
        )
    if report.placement:
        pl = report.placement
        add(f"- placement: **{pl.get('status')}** — {pl.get('reason')}")
    add(f"- corpus: {report.corpus.get('n_tasks')} tasks, hash `{str(report.corpus.get('corpus_hash'))[:12]}`")
    add(f"- tasks attempted {s['tasks_attempted']} / skipped {s['tasks_skipped']}")
    if report.validity.get("status") != "valid":
        add(f"- **VALIDITY: {report.validity.get('status')}** — {report.validity.get('reason')}")
    add("")

    add("## Quality\n")
    add("| layer | pass | fail | skip | pass rate |")
    add("|---|---|---|---|---|")
    for layer, c in s["layers"].items():
        rate = "—" if c["pass_rate_pct"] is None else f"{c['pass_rate_pct']}%"
        add(f"| {layer} | {c['pass']} | {c['fail']} | {c['skip']} | {rate} |")
    add("")
    if s["truncated"]:
        add(f"**Truncated at the token budget: {len(s['truncated'])} task(s)** — {', '.join(s['truncated'])}\n")
    add(f"ruff violations total: **{s['ruff_violations_total']}** — {s['ruff_by_code'] or 'none'}\n")

    add("## By family\n")
    add("| family | n | extract | parse | ruff | pytest |")
    add("|---|---|---|---|---|---|")
    for family, f in s["by_family"].items():
        cells = []
        for layer in ("extract", "parse", "ruff", "pytest"):
            c = f["layers"].get(layer)
            cells.append("—" if not c else f"{c['pass']}/{c['pass'] + c['fail']}")
        add(f"| {family} | {f['n']} | " + " | ".join(cells) + " |")
    add("")

    t = s.get("throughput") or {}
    if t:
        add("## Throughput\n")
        lanes = (report.serving or {}).get("lanes")
        add(f"- mode: **{t['mode']}**, client concurrency **{t['client_concurrency']}**, server lanes **{lanes}**")
        if lanes and t["client_concurrency"] > lanes:
            # Say it rather than let a reader infer batching from the mode name.
            add(
                f"- **{t['client_concurrency']} in flight against {lanes} decode lane(s): the surplus "
                f"QUEUED.** Queueing lands in time-to-first-token, not in decode rate."
            )
        add(f"- wall clock **{t['wall_s']} s**, {t['completion_tokens']} completion tokens")
        th = t.get("thermal") or {}
        if th.get("enabled") and th.get("pacing"):
            # Say it loudly: a paced run's wall clock is not a build measurement.
            add(
                f"- **thermally paced** ({th['pause_c']}/{th['resume_c']} C): {th['pauses']} pause(s), "
                f"{th['paused_s']} s idle, peak {th['peak_c']} C. Active time **{t['active_s']} s** "
                f"of {t['wall_s']} s wall."
            )
            active_rate = f"- **aggregate {t['aggregate_tok_s']} tok/s of ACTIVE time**"
            add(f"{active_rate} ({t['wall_tok_s']} tok/s wall, pauses included)")
        else:
            add(f"- **aggregate {t['aggregate_tok_s']} tok/s**, {t['requests_per_s']} requests/s")
            if th.get("enabled"):
                # Guarded but not paced: no pauses were taken, so wall == active.
                add(f"- thermal guard armed on card {th['card']} (abort {th['abort_c']} C), peak **{th['peak_c']} C**")
        pr = t.get("per_request_decode_tok_s")
        if pr:
            add(f"- per-request decode: p50 {pr['p50']}, p95 {pr['p95']} tok/s")
        pe = t.get("per_request_e2e_ms")
        if pe:
            add(f"- per-request latency: p50 {pe['p50']} ms, p95 {pe['p95']} ms")
        add("")

    add("## Speed\n")
    add(
        "_A hosted endpoint's client-side rates include the network. "
        "Server-side rates are blank unless the endpoint reports them._\n"
    )
    add("| metric | p50 | p95 | mean | n |")
    add("|---|---|---|---|---|")
    for key, st in s["speed"].items():
        if isinstance(st, dict):
            add(f"| {key} | {st['p50']} | {st['p95']} | {st['mean']} | {st['n']} |")
    add(f"\nwall clock: {s['speed']['wall_s']} s\n")

    if s["request_failures"]:
        add("## Request failures\n")
        for f in s["request_failures"]:
            add(f"- `{f['task_id']}`: {f['error']}")
        add("")
    if s["skipped_detail"]:
        add("## Skipped\n")
        for f in s["skipped_detail"]:
            add(f"- `{f['task_id']}`: {f['reason']}")
        add("")
    return "\n".join(lines)
