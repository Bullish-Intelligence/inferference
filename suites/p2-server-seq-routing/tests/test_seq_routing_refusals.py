"""The stage-4 startup contract for ``llama-server --lora-seq-routing``.

*Every* refusal R-1..R-7,
warning W-1..W-4 and per-request refusal Q-1..Q-2 has a test that asserts the
status code and that the message names the missing thing. The conditions and the
message text are ``spikes/S11/interactions.md`` section 8 - observed by a probe,
not derived from a reading.

Two properties of this file are load-bearing:

* **Nothing is read out of the source.** Each case launches the real binary and
  reads its exit code and its stderr. R-2's pre-fix failure mode is ``abort()``,
  so a code reading proves nothing about it (kickoff section 7 gate 8).
* **No GPU.** ``CUDA_VISIBLE_DEVICES`` is emptied for every child, so the server
  is unambiguously CPU-only and needs no place in the rig's GPU arbitration.
  A refusal fires after the model loads and before the port binds; the device is
  not part of any condition under test.

Three conditions cannot be produced by this server and one cannot be produced by
this rig, so the binary carries two named test hooks for them. See
``LLAMA_SEQ_ROUTING_TEST_FORCE`` and ``LLAMA_SEQ_ROUTING_TEST_SCAN`` in
``tools/server/server-context.cpp``.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from inferference.gpu_env import VISIBLE_DEVICE_VARS

MODELS = Path(os.environ.get("INFERFERENCE_MODELS_DIR", "~/.cache/inferference/models")).expanduser()
MODEL = Path(os.environ.get("MODEL_NANBEIGE", str(MODELS / "Nanbeige4.2-3B-UD-Q4_K_XL.gguf")))
PROBE_DIR = Path(os.environ.get("MODEL_NANBEIGE_LORA_PROBE", str(MODELS / "nanbeige-lora-probe")))
LORA_A = PROBE_DIR / "suite-probe-a.gguf"
LORA_B = PROBE_DIR / "suite-probe-b.gguf"

# Nanbeige is dense and 201 tensors, so at --batch-size 128 an unfixed library
# admits floor((max(128*40, 32*201) - 1371) / 2156) = 2 pool adapters and aborts
# at 3 (S11-P6, re-measured in 08 section 4).
UNFIXED_CEILING_BATCH = 128
UNFIXED_CEILING_POOL = 3

_LIB_DIR = os.environ.get("LLAMA_CPP_LIB_PATH")
_OUT_DIR = Path(_LIB_DIR).parent if _LIB_DIR else None
_SERVER = _OUT_DIR / "bin" / "llama-server" if _OUT_DIR else None


def _ready() -> bool:
    return bool(_SERVER and _SERVER.is_file() and MODEL.is_file() and LORA_A.is_file() and LORA_B.is_file())


pytestmark = pytest.mark.skipif(
    not _ready(),
    reason=(
        "requires a stage-4 build (LLAMA_CPP_LIB_PATH -> <out>/lib with <out>/bin/llama-server), "
        f"the Nanbeige model at {MODEL} and the probe adapters in {PROBE_DIR}"
    ),
)


# --------------------------------------------------------------------------- #
# launching
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _argv(port: int, loras: list[Path], extra: list[str]) -> list[str]:
    argv = [
        str(_SERVER),
        "-m",
        str(MODEL),
        "-ngl",
        "0",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--parallel",
        "4",
        "-c",
        "2048",
        "--kv-unified",
        "--cache-ram",
        "0",
    ]
    for lora in loras:
        argv += ["--lora", str(lora)]
    return argv + extra


def _child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    # No device at all: these cases are device-independent, and a server that
    # takes no device needs no place in the rig's GPU arbitration.
    # FORCE CPU-ONLY ON EVERY BACKEND. Emptying CUDA_VISIBLE_DEVICES alone does
    # NOT do this on a Vulkan build -- ggml ignores it, the child still finds
    # both cards, and these refusal cases then run WITH offload, silently
    # testing something other than what they claim.
    for _var in VISIBLE_DEVICE_VARS:
        env[_var] = ""
    env.update(extra or {})
    return env


def start_refused(
    loras: list[Path],
    extra: list[str],
    env: dict[str, str] | None = None,
    timeout: float = 240.0,
) -> tuple[int, str]:
    """Launch the server, expect it to exit on its own, return (rc, output)."""
    port = _free_port()
    proc = subprocess.Popen(  # noqa: S603
        _argv(port, loras, extra),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_child_env(env),
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        raise AssertionError(f"server did not exit within {timeout}s; it should have refused.\n{out[-4000:]}") from None
    return proc.returncode, out


class RunningServer:
    def __init__(self, loras: list[Path], extra: list[str], env: dict[str, str] | None = None) -> None:
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._argv = _argv(self.port, loras, extra)
        self._env = _child_env(env)
        self.proc: subprocess.Popen[str] | None = None
        self.output = ""

    def __enter__(self) -> RunningServer:
        self.proc = subprocess.Popen(  # noqa: S603
            self._argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self._env,
            text=True,
            start_new_session=True,
        )
        deadline = time.time() + 300.0
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(f"server exited with {self.proc.returncode} before becoming healthy")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=3.0) as r:  # noqa: S310
                    if "ok" in r.read().decode():
                        return self
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(1.0)
        raise AssertionError("server never became healthy")

    def __exit__(self, *exc: object) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def completion(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        req = urllib.request.Request(  # noqa: S310
            f"{self.base_url}/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode())


def assert_refused(rc: int, out: str, ident: str, *fragments: str) -> None:
    assert rc != 0, f"expected a non-zero exit for {ident}; got {rc}\n{out[-3000:]}"
    assert ident in out, f"{ident} is not named in the server output\n{out[-3000:]}"
    for frag in fragments:
        assert frag in out, f"{ident}: message does not name {frag!r}\n{out[-3000:]}"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def moe_gguf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """S11-P5's positive control: metadata naming an expert weight.

    Written rather than found, because no adapter on this rig targets an expert
    tensor and no MoE model is present. It is never loaded as an adapter - the
    shipped scanner reads tensor names, and that is the whole of R-5.

    S11 wrote a metadata-only file for this. ggml's own reader rejects that one
    (``gguf_init_from_reader: failed to seek to beginning of data section``), so
    the control here carries a real, aligned data section: four 1x1 f32 tensors
    at the default 32-byte alignment.
    """
    names = [
        "blk.0.attn_q.weight.lora_a",
        "blk.0.attn_q.weight.lora_b",
        "blk.0.ffn_down_exps.weight.lora_a",
        "blk.0.ffn_down_exps.weight.lora_b",
    ]
    align = 32
    body = bytearray(b"GGUF")
    body += struct.pack("<I", 3)
    body += struct.pack("<QQ", len(names), 0)
    for i, name in enumerate(names):
        raw = name.encode()
        body += struct.pack("<Q", len(raw)) + raw
        body += struct.pack("<I", 1)  # n_dims
        body += struct.pack("<q", 1)  # ne[0]
        body += struct.pack("<I", 0)  # ggml type f32
        body += struct.pack("<Q", i * align)  # offset, aligned
    body += b"\0" * ((align - len(body) % align) % align)
    body += b"\0" * (len(names) * align)
    path = tmp_path_factory.mktemp("seq-routing") / "moe-positive-control.gguf"
    path.write_bytes(bytes(body))
    return path


@pytest.fixture(scope="module")
def routed_server() -> Iterator[RunningServer]:
    with RunningServer([LORA_A, LORA_B], ["--lora-seq-routing"]) as srv:
        yield srv


def _lib_without_scaled() -> Path | None:
    """A sibling build whose libllama predates option B - R-1's library."""
    if _OUT_DIR is None:
        return None
    for cand in sorted(p.resolve() for p in _OUT_DIR.parent.glob("out-*/lib")):
        if _LIB_DIR and cand == Path(_LIB_DIR).resolve():
            continue
        libs = list(cand.glob("libllama.so.*"))
        if not libs:
            continue
        syms = subprocess.run(  # noqa: S603
            ["nm", "-D", str(libs[0])], capture_output=True, text=True, check=False
        ).stdout
        if "llama_set_seq_adapters" in syms and "llama_set_seq_adapter_scaled" not in syms:
            return cand
    return None


# --------------------------------------------------------------------------- #
# R-1 .. R-7
# --------------------------------------------------------------------------- #


def test_r1_library_without_the_routing_symbols() -> None:
    """R-1 [S9] - name the symbol the loaded libllama does not export."""
    older = _lib_without_scaled()
    if older is None:
        pytest.skip(
            "no sibling build carries the P2 pair without llama_set_seq_adapter_scaled; "
            "R-1 needs a pre-option-B lib to load against"
        )
    # The binary's RUNPATH is $ORIGIN/../lib, and LD_LIBRARY_PATH is searched
    # first, so this swaps the library the server resolves at load time.
    env = {"LD_LIBRARY_PATH": f"{older}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    rc, out = start_refused([LORA_A, LORA_B], ["--lora-seq-routing"], env=env)
    assert_refused(rc, out, "R-1", "llama_set_seq_adapter_scaled", "does not export")


def test_r2_pool_above_the_node_ceiling_on_an_unfixed_library() -> None:
    """R-2 [S11-P6] - refuse, rather than abort inside llama_decode.

    Gate 9, half one. The pre-fix failure mode is ``abort()``: reading the code
    proves nothing, so this drives an actual unfixed library.

    This SKIPS in CI, and the skip is deliberate. Every workflow cell now builds
    ``integration``, which carries the fix, so CI holds no unfixed
    routed lib to swap in. Close it by hand against a build of a branch that
    predates the fold - measured 2026-08-26 on
    ``out-p2fork-server-68ff00d60/lib`` (feat/p2-seq-lora-narrow), which exports
    llama_set_seq_adapter_scaled and lacks the node-budget fix::

        SEQ_ROUTING_UNFIXED_LIB=.../out-p2fork-server-68ff00d60/lib \\
          ci/runner/run-suite.sh --suite p2-server-seq-routing --out <integration out>

    Rebuild that lib from the ref on origin if the out dir has been pruned.
    """
    unfixed = os.environ.get("SEQ_ROUTING_UNFIXED_LIB")
    if not unfixed or not Path(unfixed).is_dir():
        pytest.skip(
            "set SEQ_ROUTING_UNFIXED_LIB to a lib dir that exports "
            "llama_set_seq_adapter_scaled and lacks the stage 0.5 node-budget fix"
        )
    env = {"LD_LIBRARY_PATH": f"{unfixed}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    rc, out = start_refused(
        [LORA_A] * UNFIXED_CEILING_POOL,
        ["--lora-seq-routing", "-b", str(UNFIXED_CEILING_BATCH)],
        env=env,
    )
    assert_refused(rc, out, "R-2", "graph node ceiling", "node-budget fix")
    assert "ceiling of 2" in out, f"R-2 must name the measured ceiling of 2 at n_batch 128\n{out[-3000:]}"


def test_r2_does_not_fire_on_a_fixed_library() -> None:
    """Gate 9, half two - the fixed library has no ceiling, so R-2 stays quiet.

    Same pool and same --batch-size that refuse above. Stage 0.5 measured this
    tree to pool 32 at n_batch 128, where its own parent aborts at pool 3.
    """
    with RunningServer(
        [LORA_A] * UNFIXED_CEILING_POOL,
        ["--lora-seq-routing", "-b", str(UNFIXED_CEILING_BATCH)],
    ) as srv:
        with urllib.request.urlopen(f"{srv.base_url}/health", timeout=10) as r:  # noqa: S310
            assert "ok" in r.read().decode()


def test_r3_routing_with_hats() -> None:
    """R-3 [S11-P2] - registering hats disables routing entirely, silently."""
    rc, out = start_refused([LORA_A, LORA_B], ["--lora-seq-routing"], env={"LLAMA_SEQ_ROUTING_TEST_FORCE": "hats"})
    assert_refused(rc, out, "R-3", "loop-adapter (hats) routing", "bit-identical to the base model")


def test_r4_routing_with_the_prompt_cache() -> None:
    """R-4 [S11-P3] - the cache entry carries no adapter to compare against."""
    rc, out = start_refused([LORA_A, LORA_B], ["--lora-seq-routing", "--cache-ram", "8192"])
    assert_refused(rc, out, "R-4", "--cache-ram 0", "no adapter identity")


def test_r4_slot_save_path() -> None:
    """R-4's second half - --slot-save-path is independent of --cache-ram."""
    rc, out = start_refused([LORA_A, LORA_B], ["--lora-seq-routing", "--slot-save-path", "/tmp"])
    assert_refused(rc, out, "R-4", "--slot-save-path")


def test_r4_does_not_fire_with_the_cache_off() -> None:
    """The negative control: --cache-ram 0 is enough, despite cache_idle_slots.

    ``cache_idle_slots`` defaults to true and the server turns it off itself
    when the cache is off (server-context.cpp:1575-1578). R-4 must not refuse a
    server whose cache is already disabled, or the flag would be unusable.
    """
    with RunningServer([LORA_A, LORA_B], ["--lora-seq-routing"]) as srv:
        with urllib.request.urlopen(f"{srv.base_url}/health", timeout=10) as r:  # noqa: S310
            assert "ok" in r.read().decode()


def test_r5_adapter_targeting_an_expert_weight(moe_gguf: Path) -> None:
    """R-5 [S11-P5] - build_lora_mm_id has no seq branch, so the delta vanishes."""
    rc, out = start_refused(
        [LORA_A, LORA_B],
        ["--lora-seq-routing"],
        env={"LLAMA_SEQ_ROUTING_TEST_SCAN": str(moe_gguf)},
    )
    assert_refused(rc, out, "R-5", "ffn_down_exps", "silently dropped", str(moe_gguf))


def test_r5_stays_quiet_on_the_real_adapters() -> None:
    """The negative control S11-P5 also ran: neither probe adapter is MoE."""
    with RunningServer([LORA_A, LORA_B], ["--lora-seq-routing"]) as srv:
        with urllib.request.urlopen(f"{srv.base_url}/health", timeout=10) as r:  # noqa: S310
            assert "ok" in r.read().decode()


def test_r6_parallel_above_n_seq_max() -> None:
    """R-6 [S11-P1] - defence in depth; unreachable without the test hook."""
    rc, out = start_refused([LORA_A, LORA_B], ["--lora-seq-routing"], env={"LLAMA_SEQ_ROUTING_TEST_FORCE": "n_seq_max"})
    assert_refused(rc, out, "R-6", "exceeds the context's n_seq_max", "out of bounds")


def test_r7_alora_adapter() -> None:
    """R-7 [S2] - alora activation is per-slot state the routing path drops."""
    rc, out = start_refused([LORA_A, LORA_B], ["--lora-seq-routing"], env={"LLAMA_SEQ_ROUTING_TEST_FORCE": "alora"})
    assert_refused(rc, out, "R-7", "activated LoRA (alora)", str(LORA_A))


def test_refusals_are_silent_when_the_flag_is_off() -> None:
    """The whole battery is opt-in: without the flag nothing is checked at all."""
    with RunningServer([LORA_A, LORA_B], ["--cache-ram", "8192"]) as srv:
        with urllib.request.urlopen(f"{srv.base_url}/health", timeout=10) as r:  # noqa: S310
            assert "ok" in r.read().decode()


# --------------------------------------------------------------------------- #
# W-1 .. W-4 - the server starts, loudly
# --------------------------------------------------------------------------- #


def _warnings_of(loras: list[Path], extra: list[str], env: dict[str, str] | None = None) -> str:
    """Start the server, wait for it to be healthy, stop it, return its output.

    A warning path does not exit by itself, so the child's output goes to a file
    and the file is read after the process is gone. Reading a pipe while the
    server runs would deadlock on a full buffer.
    """
    port = _free_port()
    with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False) as fh:
        log = Path(fh.name)
    try:
        with log.open("wb") as sink:
            proc = subprocess.Popen(  # noqa: S603
                _argv(port, loras, extra),
                stdout=sink,
                stderr=subprocess.STDOUT,
                env=_child_env(env),
                text=False,
                start_new_session=True,
            )
        deadline = time.time() + 300.0
        healthy = False
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3.0) as r:  # noqa: S310
                    if "ok" in r.read().decode():
                        healthy = True
                        break
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(1.0)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        out = log.read_text(errors="replace")
        assert healthy, f"a warning must not stop the server from starting\n{out[-3000:]}"
        return out
    finally:
        log.unlink(missing_ok=True)


def test_w1_parallel_at_or_below_eight() -> None:
    """W-1 - the gain is smallest at low concurrency, and it is not monotone.

    The first text said "routing pays above 8". Measurement showed
    the whole grid and refuted that shape: the win peaks near c=16 and falls
    again at c=32, because it tracks how badly stock serializes rather than
    concurrency itself.
    """
    out = _warnings_of([LORA_A, LORA_B], ["--lora-seq-routing"])
    assert "W-1" in out, out[-3000:]
    assert "--parallel 4" in out, out[-3000:]
    assert "peaked near c=16, not monotone" in out, out[-3000:]
    assert "Routing pays above 8" not in out, out[-3000:]


def test_w5_names_the_cost_to_single_adapter_traffic() -> None:
    """W-5 - every request pays for every REGISTERED adapter, in flight or not.

    Measured 0.71x-0.99x on homogeneous traffic across the envelope grid. No
    other warning covers it: W-1 is about concurrency, W-2 and W-4 about pool
    size.
    """
    out = _warnings_of([LORA_A, LORA_B], ["--lora-seq-routing"])
    assert "W-5" in out, out[-3000:]
    assert "largest here: 3 pairs" in out, out[-3000:]
    # Stage 6A moved the cost from the pool to the mask; the text must say so.
    assert "The cost is the mask, not the pool" in out, out[-3000:]


def test_w5_stays_quiet_at_pool_one() -> None:
    """A pool of one charges nothing extra, so there is nothing to warn about."""
    out = _warnings_of([LORA_A], ["--lora-seq-routing"])
    assert "W-5" not in out, out[-3000:]


def test_w2_names_the_lora_pair_count_not_only_the_pool_size() -> None:
    """W-2 [S8, S10] - pair count is what predicts the cost. Both probes are 3."""
    out = _warnings_of([LORA_A, LORA_B], ["--lora-seq-routing"])
    assert "W-2" in out, out[-3000:]
    assert "pool of 2 adapters carrying 3, 3 LoRA pairs" in out, out[-3000:]
    # The cost figures are this project's own, measured at full batch width.
    assert "costs per adapter IN FLIGHT" in out, out[-3000:]


def test_w4_falls_back_and_says_so_without_a_device() -> None:
    """W-4 [19] - D3's cap is computed from VRAM, and this suite has no GPU.

    Every case here runs with CUDA_VISIBLE_DEVICES emptied, so no offload device
    reports free memory and the bound cannot be computed. The fallback to D3's
    constant of 3 must be LOUD: a missing capability is never a silent pass.
    """
    out = _warnings_of([LORA_A, LORA_B, LORA_A, LORA_B], ["--lora-seq-routing"])
    assert "W-4" in out, out[-3000:]
    assert "could not be computed" in out, out[-3000:]
    assert "Falling back to D3's constant of 3" in out, out[-3000:]


def test_w4_fallback_stays_quiet_at_the_constant() -> None:
    """The negative control: at pool 3 the fallback threshold is not crossed."""
    out = _warnings_of([LORA_A, LORA_B, LORA_A], ["--lora-seq-routing"])
    assert "W-4" not in out, out[-3000:]


def test_w4_refuses_to_report_on_a_possibly_split_model() -> None:
    """W-4's third state - several GPUs, so no bound.

    ``seq_routing_device_memory`` used to take the FIRST offload device
    unconditionally. With two GPUs the model may be split over them
    (``--split-mode layer`` is the default) and the model's own device list is
    not on ``llama.h``, so that figure is about a card holding part of the
    weights - a number that reads as a valid bound and is not one. It refuses
    now: both sizes stay 0 and the fallback names the split.

    A real split needs a server started across both cards, which this rig cannot
    do while a card is leased to CI, so the device COUNT is the forced input.
    Everything downstream of it is shipped code.
    """
    out = _warnings_of(
        [LORA_A, LORA_B, LORA_A, LORA_B],
        ["--lora-seq-routing"],
        env={"LLAMA_SEQ_ROUTING_TEST_FORCE": "vram_split"},
    )
    assert "W-4" in out, out[-3000:]
    assert "could not be computed" in out, out[-3000:]
    assert "split over several GPUs" in out, out[-3000:]
    assert "Falling back to D3's constant of 3" in out, out[-3000:]
    # It must name the split rather than the CPU-only case, or the two
    # indistinguishable failures are back.
    assert "no offload device reported free memory" not in out, out[-3000:]


def test_w4_names_the_measured_numbers_when_the_device_is_tight() -> None:
    """W-4's computed half. The hook supplies the one input this rig cannot.

    Filling a card to within two adapters of full is not reproducible here, so
    ``vram_tight`` sets free memory to one adapter's worth - the same reason
    hats, n_seq_max and alora are forced. Everything downstream of that input is
    the shipped code: the adapter bytes come from the real GGUFs and the
    per-adapter figure carries the measured 1.25x margin.
    """
    out = _warnings_of(
        [LORA_A, LORA_B],
        ["--lora-seq-routing"],
        env={"LLAMA_SEQ_ROUTING_TEST_FORCE": "vram_tight"},
    )
    assert "W-4" in out, out[-3000:]
    assert "room for about 0 more" in out, out[-3000:]
    # The message must carry real quantities, not a bare threshold - that is the
    # whole difference between the computed bound and the constant it replaced.
    assert "of adapter weights" in out, out[-3000:]
    assert "free" in out, out[-3000:]
    # And it must not claim to refuse: the loader is what fails past the device.
    assert "the loader fails outright, not this warning" in out, out[-3000:]


def test_w3_draft_model() -> None:
    """W-3 [S11-P4] - the draft context carries no adapters. 82 drafted, 0 taken."""
    # --spec-type is what actually builds the draft context. Without it the
    # server loads the draft model and never speculates, so there is no hazard
    # and W-3 must stay quiet - the second case below is that control.
    out = _warnings_of(
        [LORA_A, LORA_B],
        ["--lora-seq-routing", "-md", str(MODEL), "-ngld", "0", "--spec-type", "draft-simple"],
    )
    assert "W-3" in out and "draft context carries no adapters" in out, out[-3000:]


def test_w3_stays_quiet_without_a_draft_context() -> None:
    """No draft context means no drafts, so nothing to warn about."""
    out = _warnings_of([LORA_A, LORA_B], ["--lora-seq-routing"])
    assert "W-3" not in out, out[-3000:]


def test_w4_pool_above_the_documented_budget() -> None:
    """W-4 - pool > 3, which after stage 6A is a memory statement, not a speed one.

    Four pool entries, from the two probe adapters loaded twice each. Repeating
    a path gives four handles and so four pool slots, which is what W-4 counts.
    """
    out = _warnings_of([LORA_A, LORA_B, LORA_A, LORA_B], ["--lora-seq-routing"])
    assert "W-4" in out and "pool of 4" in out, out[-3000:]
    # D3's cap was cost-driven and stage 6A removed the cost, so W-4 now warns
    # about memory. Asserting the old throughput claim would pin a false one.
    assert "Registered adapters no longer cost throughput" in out, out[-3000:]


def test_w4_stays_quiet_at_pool_two() -> None:
    out = _warnings_of([LORA_A, LORA_B], ["--lora-seq-routing"])
    assert "W-4" not in out, out[-3000:]


# --------------------------------------------------------------------------- #
# Q-1, Q-2 - per-request, HTTP 400
# --------------------------------------------------------------------------- #


def test_q1_adapter_id_outside_the_pool(routed_server: RunningServer) -> None:
    status, body = routed_server.completion({"prompt": "hi", "n_predict": 1, "lora": [{"id": 9, "scale": 1.0}]})
    assert status == 400, body
    msg = body["error"]["message"]
    assert "lora id 9 is not in the routing pool" in msg, msg
    assert "pool size 2" in msg, msg


def test_q2_more_than_one_enabled_adapter(routed_server: RunningServer) -> None:
    status, body = routed_server.completion(
        {"prompt": "hi", "n_predict": 1, "lora": [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 1.0}]}
    )
    assert status == 400, body
    msg = body["error"]["message"]
    assert "routes one sequence to one pool slot" in msg, msg
    assert "this request enables 2 adapters" in msg, msg


def test_a_single_enabled_adapter_is_accepted(routed_server: RunningServer) -> None:
    """The negative control for Q-1 and Q-2 - one adapter at one scale is fine."""
    status, body = routed_server.completion(
        {
            "prompt": "hi",
            "n_predict": 2,
            "temperature": 0.0,
            "seed": 1234,
            "cache_prompt": False,
            "ignore_eos": True,
            "lora": [{"id": 1, "scale": 0.5}],
        }
    )
    assert status == 200, body
    assert body["tokens_predicted"] == 2, body


def test_a_zero_scale_request_is_the_base_model(routed_server: RunningServer) -> None:
    """A base seat costs one mask column of zeros and no matmul (S6 section 2.3)."""
    status, body = routed_server.completion(
        {
            "prompt": "hi",
            "n_predict": 2,
            "temperature": 0.0,
            "seed": 1234,
            "cache_prompt": False,
            "ignore_eos": True,
            "lora": [{"id": 0, "scale": 0.0}],
        }
    )
    assert status == 200, body
    assert body["tokens_predicted"] == 2, body
