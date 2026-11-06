"""The per-model routing smoke gate: does routing serve on THIS architecture?

A survey of 140 architectures found **103** on the ungated row-reduction pattern
that produced R12, and only `llama` was ever run. Auditing the other 102 is the
wrong answer: it is a code reading, and it goes stale on the next anchor bump. A
gate any (model, adapter) pair can run is the right one, and this is an earlier
`run_llama_arch.py` generalised into it.

**What changed from run_llama_arch.py.** The model, the adapter directory and
the two target blocks were module constants pinned to TinyLlama. They are
parameters now, and the block indices derive from the model's own GGUF instead
of being written down (`arch_blocks`). Everything else is deliberately
unchanged.

**The control arms stay.** They are what located all three failed builds: the
first bad build turned `mid-ffn-routed` and `last-attn-routed` from serving to
aborting, and an arm-under-test alone would have read as "still
aborts" and sent the session back into logic that was already correct.

| arm | adapter target | routing | what it proves |
|---|---|---|---|
| `last-ffn-routed`  | blk.LAST ffn_*  | ON  | the arm under test |
| `last-ffn-classic` | blk.LAST ffn_*  | OFF | model and adapters are sound; a failure is routing-only |
| `mid-ffn-routed`   | blk.MID ffn_*   | ON  | an FFN LoRA under routing is fine away from the reduction |
| `last-attn-routed` | blk.LAST attn_* | ON  | the last BLOCK is not the trigger; being AFTER the reduction is |
| `last-ffn-routed-1tok` | blk.LAST ffn_* | ON | prompt length is not a mitigation |

**Pass/fail comes from the server's own decode timings, never from HTTP status.**
A synthetic adapter makes the model emit gibberish and `common_chat_peg_parse`
answers 500 after a completed decode. A 500 there is a formatting
refusal, not a graph failure.

**Fail-closed.** A missing model or adapter set is a loud skip carrying the
command that fixes it. An arm that neither serves nor aborts is a failure.

GPU discipline: this is a CI job that happens to launch a server, so it takes
the CI role's device (`$LLAMA_CI_GPU`, or `$ROUTING_SMOKE_GPU` to override). It
asserts the variable, never a literal index.

Run inside `devenv shell`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RIG_ENV = REPO / "ci" / "runner" / "rig-env.sh"

# Long enough that prefill is unambiguously many tokens against n_outputs = 1,
# which is the shape the defect needs.
PROMPT = " ".join(["Explain in detail why the sky appears blue during the day."] * 12)
HEALTH_TIMEOUT_S = 180.0
ASSERT_RE = re.compile(r"GGML_ASSERT\(.*\) failed|ggml_can_repeat|Aborted|terminate called")
DECODE_RE = re.compile(r"prompt eval time =.*?/\s*(\d+) tokens")
EVAL_RE = re.compile(r"\beval time =.*?/\s*(\d+) tokens")

ADAPTER_PREFIXES = ("last-ffn", "mid-ffn", "last-attn")


# --------------------------------------------------------------------------- #
# the model's own shape
# --------------------------------------------------------------------------- #


def arch_blocks(model: Path) -> dict[str, Any]:
    """Architecture, block count, and the two blocks the arms target.

    Derived from the GGUF, not written down: the whole point of this gate is
    that a new architecture needs no new constant here.
    """
    import sys

    gguf_py = REPO / "ci/build/.llamacpp-builds/src-p2fork/gguf-py"
    if not gguf_py.is_dir():
        gguf_py = REPO / "ci/build/.llamacpp-builds/src-llama.cpp/gguf-py"
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # ty: ignore[unresolved-import]

    reader = GGUFReader(str(model))
    arch = reader.fields["general.architecture"].contents()
    blocks = set()
    for t in reader.tensors:
        m = re.fullmatch(r"blk\.(\d+)\.\w+\.weight", t.name)
        if m:
            blocks.add(int(m.group(1)))
    if not blocks:
        raise SystemExit(f"{model.name}: no blk.N.*.weight tensors; not a block-structured model")
    last = max(blocks)
    # A middle block, far from the last one. The control's value is that it sits
    # before any end-of-loop row reduction; the exact index does not matter.
    mid = last // 2
    return {"arch": arch, "n_blocks": len(blocks), "last_block": last, "mid_block": mid}


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #


def arm_specs() -> list[dict[str, Any]]:
    return [
        {"name": "last-ffn-routed", "prefix": "last-ffn", "routing": True},
        {"name": "last-ffn-classic", "prefix": "last-ffn", "routing": False},
        {"name": "mid-ffn-routed", "prefix": "mid-ffn", "routing": True},
        {"name": "last-attn-routed", "prefix": "last-attn", "routing": True},
        {"name": "last-ffn-routed-1tok", "prefix": "last-ffn", "routing": True, "prompt": "A"},
    ]


def server_argv(binary: Path, model: Path, loras: list[Path], port: int, routing: bool) -> list[str]:
    # THE CARD IS NAMED TWO DIFFERENT WAYS AND ONLY ONE WORKS PER BACKEND.
    # launch() exports CUDA_VISIBLE_DEVICES, which is how a CUDA build is pinned.
    # THE VULKAN BACKEND IGNORES IT ENTIRELY: a Vulkan server with
    # CUDA_VISIBLE_DEVICES=1 still places the model on Vulkan0, so every arm
    # would silently run on the wrong card — and on a rig whose other card is
    # serving, that is an overcommit that spills to host RAM and still passes.
    # $ROUTING_SMOKE_DEVICE names the device explicitly (e.g. "Vulkan1") and is
    # passed through as --device. Unset keeps the CUDA behaviour unchanged.
    argv = [
        str(binary),
        "-m", str(model),
        "-ngl", "99",
        "--port", str(port),
        "--host", "127.0.0.1",
        "--parallel", "4",
        "-c", "8192",
        "--kv-unified",
        "--metrics",
        "--cache-ram", "0",
    ]  # fmt: skip
    device = os.environ.get("ROUTING_SMOKE_DEVICE")
    if device:
        argv += ["--device", device]
    for lora in loras:
        argv += ["--lora", str(lora)]
    if routing:
        argv.append("--lora-seq-routing")
    return argv


def launch(argv: list[str], log: Path) -> subprocess.Popen[bytes]:
    # This gate launches a server, but it is a CI JOB, so it takes the CI role's
    # device. An earlier version pinned $LLAMA_SERVER_GPU: the cell then held the
    # CI lock while computing on a model server's card, which is harmless only
    # while no server is resident there.
    gpu = os.environ.get("ROUTING_SMOKE_GPU") or os.environ.get("LLAMA_CI_GPU")
    if not gpu:
        raise SystemExit("LLAMA_CI_GPU is unset; source ci/runner/rig-env.sh")
    quoted = " ".join(f"'{a}'" for a in argv)
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("wb")
    return subprocess.Popen(  # noqa: S603
        ["bash", "-c", f"source '{RIG_ENV}'; export CUDA_VISIBLE_DEVICES='{gpu}'; exec {quoted}"],
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_startup(proc: subprocess.Popen[bytes], port: int) -> str:
    """'healthy' | 'exited' | 'timeout' — never a default, never an unbounded wait."""
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return "exited"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:  # noqa: S310
                if r.status == 200:
                    return "healthy"
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(0.5)
    return "timeout"


def completion(port: int, routed: bool, prompt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": 8,
        "temperature": 0.0,
        "seed": 1234,
        "cache_prompt": False,
        "stream": False,
        "return_tokens": True,
    }
    if routed:
        payload["lora"] = [{"id": 0, "scale": 1.0}]
    req = urllib.request.Request(  # noqa: S310
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
        return json.loads(r.read().decode())


def run_arm(
    binary: Path,
    model: Path,
    lora_dir: Path,
    spec: dict[str, Any],
    port: int,
    tag: str,
    log_dir: Path,
) -> dict[str, Any]:
    loras = sorted(lora_dir.glob(f"{spec['prefix']}-*.gguf"))
    if len(loras) != 2:
        raise SystemExit(f"expected 2 adapters for {spec['prefix']} in {lora_dir}, found {len(loras)}")
    # The log name carries the build id AND the model: one binary runs several
    # models here, and a shared log name would attribute one model's stack to
    # another. A provenance defect already voided one gate run this way.
    log = log_dir / f"{tag}.{model.stem}.{spec['name']}.log"
    argv = server_argv(binary, model, loras, port, spec["routing"])
    prompt = spec.get("prompt", PROMPT)
    out: dict[str, Any] = {
        "arm": spec["name"],
        "routing": spec["routing"],
        "adapters": [p.name for p in loras],
        "prompt_words": len(prompt.split()),
    }
    proc = launch(argv, log)
    try:
        out["startup"] = wait_startup(proc, port)
        if out["startup"] == "healthy":
            try:
                body = completion(port, spec["routing"], prompt)
                out["request"] = "ok"
                out["n_prompt_tokens"] = body.get("tokens_evaluated")
                out["tokens"] = body.get("tokens", [])[:8]
            except Exception as exc:  # noqa: BLE001
                # Recorded, never fatal: a chat-parser 500 lands here and the
                # decode timings below are what decides the arm.
                out["request"] = f"failed: {type(exc).__name__}: {exc}"
            time.sleep(1.0)  # a CUDA abort can land while the response is in flight
            out["alive_after_request"] = proc.poll() is None
        else:
            out["request"] = "not attempted"
            out["exit_code"] = proc.poll()
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    text = log.read_text(errors="replace")
    out["assert_lines"] = [ln.strip() for ln in text.splitlines() if ASSERT_RE.search(ln)][:6]
    out["aborted"] = bool(out["assert_lines"])
    prompt_hit = DECODE_RE.search(text)
    eval_hit = EVAL_RE.search(text)
    out["prompt_eval_tokens"] = int(prompt_hit.group(1)) if prompt_hit else 0
    out["decode_tokens"] = int(eval_hit.group(1)) if eval_hit else 0
    out["decode_completed"] = out["prompt_eval_tokens"] > 0 and out["decode_tokens"] > 0
    out["serves"] = out["decode_completed"] and not out["aborted"]
    out["log"] = str(log)
    return out


def run_gate(
    binary: Path,
    model: Path,
    lora_dir: Path,
    log_dir: Path,
    base_port: int = 8501,
) -> dict[str, Any]:
    """Every arm, one server each. Returns the record; raises only on a setup fault."""
    tag = binary.parents[1].name.rsplit("-", 1)[-1]
    shape = arch_blocks(model)
    result: dict[str, Any] = {
        "binary": str(binary),
        "build": tag,
        "model": model.name,
        "lora_dir": str(lora_dir),
        **shape,
        "arms": [],
    }
    for i, spec in enumerate(arm_specs()):
        print(f"== {model.stem} / {spec['name']}", flush=True)
        result["arms"].append(run_arm(binary, model, lora_dir, spec, base_port + i, tag, log_dir))

    by = {a["arm"]: a for a in result["arms"]}
    result["verdict"] = {
        # The arm under test. On a pre-R12 library this is where llama aborted.
        "routed_serves_after_the_reduction": by["last-ffn-routed"]["serves"],
        # Controls. A control that stops serving means the harness or the build
        # is at fault, not the arm under test.
        "classic_path_serves": by["last-ffn-classic"]["serves"],
        "mid_layer_ffn_serves": by["mid-ffn-routed"]["serves"],
        "last_layer_attn_serves": by["last-attn-routed"]["serves"],
        "single_token_prompt_serves": by["last-ffn-routed-1tok"]["serves"],
    }
    result["pass"] = all(result["verdict"].values())
    return result


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, type=Path, help="a routed llama-server")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--lora-dir", required=True, type=Path, help="holds last-ffn-*, mid-ffn-*, last-attn-*")
    ap.add_argument("--log-dir", type=Path, default=Path("/tmp/routing-smoke"))
    ap.add_argument("--base-port", type=int, default=8501)
    ap.add_argument("--out", type=Path, default=Path("routing-smoke.json"))
    args = ap.parse_args()

    for p in (args.binary, args.model):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    if not args.lora_dir.is_dir():
        raise SystemExit(f"missing adapter dir {args.lora_dir} — see make_arch_lora.py")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    result = run_gate(args.binary, args.model, args.lora_dir, args.log_dir, args.base_port)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}\n")
    print(f"arch={result['arch']} blocks={result['n_blocks']} last={result['last_block']} mid={result['mid_block']}")
    for k, v in result["verdict"].items():
        print(f"{'YES' if v else 'NO '}  {k}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
