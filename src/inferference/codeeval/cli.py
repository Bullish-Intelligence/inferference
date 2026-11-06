"""``python -m inferference.codeeval`` — run the harness.

Two endpoint shapes, one code path:

* ``--endpoint local`` — the router on this machine, OpenAI-compatible.
* ``--endpoint deepseek`` — a hosted baseline. The key comes from the gitignored
  ``.env``; it is never written into an artifact.

No model string is defaulted. ``--model`` is required, because this repo does
not name models in its own files.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import sys
import time
from pathlib import Path
from uuid import uuid4

from .client import Endpoint, load_dotenv
from .drive import DriveResult, run_batched, run_single
from .placement import PlacementProof, ServingConfig, read_device, read_serving_config
from .report import RunReport, summarize, write_report
from .task import CorpusError, load_task_set
from .thermal import ThermalPacer

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = REPO / "tasks" / "codeeval"

ENDPOINTS: dict[str, dict[str, str | bool | None]] = {
    "local": {
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key_env": None,
        "hosted": False,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "hosted": True,
    },
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codeeval", description=__doc__)
    p.add_argument("--endpoint", choices=sorted(ENDPOINTS), required=True)
    p.add_argument("--model", required=True, help="model string, verbatim")
    p.add_argument("--base-url", help="override the endpoint's base URL")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--out", type=Path, default=REPO / "runs" / "codeeval")
    p.add_argument("--family", action="append", dest="families")
    p.add_argument("--task", action="append", dest="ids")
    p.add_argument("--limit", type=int)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--structured", action="store_true")
    p.add_argument(
        "--token-budget-scale",
        type=float,
        default=1.0,
        help=(
            "multiply every task's max_output_tokens. Use this to give an "
            "answer more room, not to pay for a reasoning model's thinking."
        ),
    )
    p.add_argument(
        "--reasoning-headroom",
        type=int,
        default=0,
        help=(
            "tokens ADDED to every task's budget to pay for a reasoning "
            "model's thinking. It is added, not multiplied, because the task "
            "declares how big the ANSWER is and the endpoint decides how much "
            "it thinks first. Without it a reasoning model spends the whole "
            "budget before the answer starts and returns nothing."
        ),
    )
    p.add_argument(
        "--max-prompt-tokens",
        type=int,
        help="skip (never truncate) any task whose prompt is larger",
    )
    p.add_argument("--label", default="", help="a note stored in the envelope")
    p.add_argument(
        "--mode",
        choices=("single", "batch"),
        default="single",
        help="single = one request at a time (best-case latency); batch = many in flight (throughput)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "requests IN FLIGHT in batch mode. This is not the server's batch "
            "size: the server decodes at most its --parallel lanes together, and "
            "the surplus queues. Both numbers go in the envelope."
        ),
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="run the whole workload this many times, to expose unstable results",
    )
    p.add_argument(
        "--card",
        type=int,
        help=("DRM card index to read during a local run. The router must provide load-time residency evidence."),
    )
    p.add_argument(
        "--pace-card",
        type=int,
        help=(
            "hold this DRM card inside a temperature band by pausing BETWEEN "
            "requests. These cards have no fan, so this is the only throttle "
            "available. Rates are then reported over ACTIVE time, with the "
            "pauses recorded separately."
        ),
    )
    p.add_argument("--pause-c", type=int, default=55, help="pause new requests at this temperature")
    p.add_argument(
        "--resume-c",
        type=int,
        default=50,
        help=(
            "resume at this temperature. It must leave room for a WHOLE request: "
            "a pause can only be taken between requests, and one request here "
            "raises HBM by about 30 C. Resuming at 66 C drove a card to 95 C."
        ),
    )
    p.add_argument(
        "--max-loadavg",
        type=float,
        default=3.0,
        help=(
            "flag the run as noisy if the 1-minute host load exceeds this at "
            "start. Measured 2026-09-05: the same workload gave 30.2 tok/s at "
            "load 5.5 and 35.2 at load 1.6, a 16%% swing, because host load "
            "starves the GPU feed. 0 disables the check."
        ),
    )
    p.add_argument(
        "--abort-c",
        type=int,
        default=85,
        help=(
            "hard ceiling. A watchdog samples the card DURING requests and voids "
            "the run if it is reached, because the pacer cannot interrupt a "
            "request that is already in flight."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(REPO / ".env")

    try:
        task_set = load_task_set(args.corpus)
    except CorpusError as exc:
        print(f"corpus error: {exc}", file=sys.stderr)
        return 2
    task_set = task_set.slice(families=args.families, ids=args.ids, limit=args.limit)
    if not len(task_set):
        print("no tasks selected", file=sys.stderr)
        return 2

    spec = ENDPOINTS[args.endpoint]
    base_url = args.base_url or spec["base_url"]
    api_key_env = spec["api_key_env"]
    endpoint = Endpoint(
        name=args.endpoint,
        base_url=str(base_url),
        model=args.model,
        api_key_env=None if api_key_env is None else str(api_key_env),
        hosted=bool(spec["hosted"]),
    )
    if endpoint.api_key_env and not os.environ.get(endpoint.api_key_env):
        print(
            f"{endpoint.api_key_env} is not set. Put it in {REPO / '.env'} (gitignored) or export it.",
            file=sys.stderr,
        )
        return 2

    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {run_id}: {len(task_set)} task(s) -> {out_dir}", file=sys.stderr)

    def progress(i: int, total: int, outcome) -> None:
        marks = " ".join(f"{r.layer}={r.status}" for r in outcome.score.layers)
        print(f"  [{i}/{total}] {outcome.task_id}: {marks}", file=sys.stderr)

    # Read how the router is REALLY serving this model before timing anything.
    # The eval's concurrency and the server's lane count are different numbers.
    serving: ServingConfig | None = None
    if not endpoint.hosted:
        try:
            serving = read_serving_config(endpoint.base_url, endpoint.model)
        except RuntimeError as exc:
            print(f"cannot read the router's model registry: {exc}", file=sys.stderr)
            return 2
        if serving.status != "loaded":
            print(
                f"model '{endpoint.model}' is '{serving.status}', not loaded. There is no demand "
                f"loading: reconcile the router first.",
                file=sys.stderr,
            )
            return 2

    before = None
    if args.card is not None:
        try:
            before = read_device(args.card)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    # The abort watchdog arms whenever we know which card to watch. Pacing is
    # the separate, opt-in part: an unpaced run still must not cook the card.
    pacer_card = args.pace_card if args.pace_card is not None else args.card
    do_pace = args.pace_card is not None
    try:
        template = ThermalPacer(
            card=pacer_card, pace=do_pace, pause_c=args.pause_c, resume_c=args.resume_c, abort_c=args.abort_c
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if template.enabled and template.observe() is None:
        print(f"cannot read temperatures for card {pacer_card}; refusing to pace blindly", file=sys.stderr)
        return 2

    async def drive_all() -> list[DriveResult]:
        out: list[DriveResult] = []
        for rep in range(1, args.repeats + 1):
            tag = "" if args.repeats == 1 else f" (repeat {rep}/{args.repeats})"
            print(f"  -- {args.mode} pass{tag}", file=sys.stderr)
            repeat = 0 if args.repeats == 1 else rep
            if args.mode == "batch":
                out.append(
                    await run_batched(
                        task_set,
                        endpoint,
                        out_dir=out_dir,
                        concurrency=args.concurrency,
                        temperature=args.temperature,
                        max_prompt_tokens=args.max_prompt_tokens,
                        structured=args.structured,
                        token_budget_scale=args.token_budget_scale,
                        reasoning_headroom=args.reasoning_headroom,
                        repeat=repeat,
                        pacer=ThermalPacer(
                            card=pacer_card,
                            pace=do_pace,
                            pause_c=args.pause_c,
                            resume_c=args.resume_c,
                            abort_c=args.abort_c,
                        ),
                        progress=progress,
                    )
                )
            else:
                out.append(
                    await run_single(
                        task_set,
                        endpoint,
                        out_dir=out_dir,
                        temperature=args.temperature,
                        max_prompt_tokens=args.max_prompt_tokens,
                        structured=args.structured,
                        token_budget_scale=args.token_budget_scale,
                        reasoning_headroom=args.reasoning_headroom,
                        repeat=repeat,
                        pacer=ThermalPacer(
                            card=pacer_card,
                            pace=do_pace,
                            pause_c=args.pause_c,
                            resume_c=args.resume_c,
                            abort_c=args.abort_c,
                        ),
                        progress=progress,
                    )
                )
        return out

    passes = asyncio.run(drive_all())

    # Repeats are reported as one workload: the outcomes are pooled so the
    # quality vector covers every attempt, and each pass keeps its own
    # throughput figure, because a rate averaged across passes hides an
    # unstable one.
    result = passes[0]
    if len(passes) > 1:
        merged = DriveResult(
            outcomes=[o for p in passes for o in p.outcomes],
            started_at=passes[0].started_at,
            ended_at=passes[-1].ended_at,
            mode=passes[0].mode,
            concurrency=passes[0].concurrency,
        )
        result = merged

    after = None
    if args.card is not None:
        try:
            after = read_device(args.card)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    load_start = os.getloadavg()[0]
    noisy = bool(args.max_loadavg) and load_start > args.max_loadavg
    if noisy:
        print(
            f"!! HOST LOAD {load_start:.2f} exceeds {args.max_loadavg} at start. Throughput on a "
            f"loaded box reads LOW — the GPU is starved of work, not slow. This run is "
            f"recorded as noisy and must not be compared with a quiet-box run.",
            file=sys.stderr,
        )

    aborted = next((p.aborted for p in passes if p.aborted), None)
    if aborted:
        print(f"\n!! RUN VOID: {aborted}", file=sys.stderr)

    if endpoint.hosted:
        proof = None
    elif before is None or after is None or serving is None:
        proof = PlacementProof.unproven(
            "no --card given, so the cards' own accounting was never read. A local speed "
            "number without a placement proof is not a valid result: an over-budget model "
            "loads successfully into host memory and runs about 14x slower."
        )
    else:
        try:
            serving_after = read_serving_config(endpoint.base_url, endpoint.model)
        except RuntimeError as exc:
            proof = PlacementProof.unproven(f"cannot confirm the model after the run: {exc}")
        else:
            if serving_after.status != "loaded" or serving_after.residency != serving.residency:
                proof = PlacementProof.unproven("the model or its residency evidence changed during the run")
            else:
                proof = PlacementProof.evaluate(serving=serving, card=args.card, before=before, after=after)

    summary = summarize(result)
    per_pass = [
        {
            "repeat": i + 1,
            "wall_s": round(p.wall_s, 2),
            "completion_tokens": p.completion_tokens,
            "aggregate_tok_s": (None if p.aggregate_tok_s is None else round(p.aggregate_tok_s, 2)),
            "requests_per_s": (None if p.requests_per_s is None else round(p.requests_per_s, 3)),
        }
        for i, p in enumerate(passes)
    ]
    throughput = dict(summary["throughput"])
    throughput["per_pass"] = per_pass
    throughput["server_lanes"] = None if serving is None else serving.lanes

    report = RunReport(
        run_id=run_id,
        mode=args.mode,
        endpoint=endpoint.redacted(),
        corpus={
            "root": str(args.corpus),
            "n_tasks": len(task_set),
            "corpus_hash": task_set.corpus_hash,
            "task_ids": [t.id for t in task_set.tasks],
        },
        axes={
            "mode": args.mode,
            "client_concurrency": args.concurrency,
            "server_lanes": None if serving is None else serving.lanes,
            "ctx_size": None if serving is None else serving.ctx_size,
            "ctx_per_lane": None if serving is None else serving.ctx_per_lane,
            "device": None if serving is None else serving.device,
            "structured": bool(args.structured),
            "speculative": "off",
            "temperature": args.temperature,
            "token_budget_scale": args.token_budget_scale,
            "reasoning_headroom": args.reasoning_headroom,
            "repeats": args.repeats,
            "label": args.label,
        },
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "loadavg": list(os.getloadavg()),
            "build": _build_fingerprint(),
        },
        placement=None if proof is None else proof.model_dump(mode="json"),
        serving=({} if serving is None else serving.model_dump(mode="json")),
        quality=summary,
        speed=summary["speed"],
        throughput=throughput,
        validity=_validity(endpoint, proof, aborted, noisy=noisy, loadavg=load_start),
        command=sys.argv,
    )
    write_report(report, summary, out_dir)
    print((out_dir / "summary.md").read_text())
    return 0


def _build_fingerprint() -> dict[str, str]:
    """Which llama.cpp build served this run, as far as it can be known here.

    The harness talks HTTP and never links llama_cpp, so it cannot read the
    build directly. What it can record is the router's own /props if offered,
    and the environment variable the launcher sets. Absent is recorded as
    absent, never guessed.
    """
    return {
        "llama_server": os.environ.get("INFERFERENCE_LLAMA_SERVER", "unrecorded"),
        "anchor": os.environ.get("INFERFERENCE_ABI_ANCHOR", "unrecorded"),
    }


def _validity(
    endpoint: Endpoint,
    proof: PlacementProof | None,
    aborted: str | None = None,
    *,
    noisy: bool = False,
    loadavg: float | None = None,
) -> dict[str, str]:
    if aborted:
        # A thermal abort voids the run whatever else was true of it: the
        # timings describe a card that was throttling.
        return {"status": "invalid", "reason": f"run aborted: {aborted}"}
    if endpoint.hosted:
        return {
            "status": "quality-only",
            "reason": (
                "hosted endpoint: latency includes the network and the "
                "provider's own load, so speed figures are NOT comparable "
                "with a local run"
            ),
        }
    if proof is None or not proof.valid:
        # INVALID, not "successful with a note". A spilled model answers every
        # request correctly and reports success; its timings simply describe
        # host memory rather than the card.
        return {
            "status": "invalid",
            "reason": (
                f"local run whose placement was not proven ({'' if proof is None else proof.status}): "
                f"{'no proof attempted' if proof is None else proof.reason}"
            ),
        }
    if noisy:
        # Not invalid — the measurement is real, it just describes a contended
        # box. Comparing it with a quiet-box run is what would be wrong.
        return {
            "status": "noisy",
            "reason": (
                f"host loadavg was {loadavg:.2f} at start. A loaded host starves the GPU feed and "
                f"depresses throughput by up to 16% (measured). Valid on its own terms; NOT "
                f"comparable with a quiet-box run. {proof.reason}"
            ),
        }
    return {"status": "valid", "reason": proof.reason}


if __name__ == "__main__":
    raise SystemExit(main())
