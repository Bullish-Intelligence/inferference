"""Start, watch, and stop one `llama-server` process for one GPU card.

This module builds the server's argv and environment and runs it. It never
starts a real server itself: every caller supplies `server_bin`, and every
test in this repo points `server_bin` at a fake stand-in
(`tests/fixtures/experiment/fake_llama_server.py`).

Two safety rules are load-bearing, both verbatim from
`ci/runner/gpu-thermal-guard.sh`:

* `build_command` never sets `GPU_THERMAL_MAX_C`, and refuses an `extra_args`
  key that collides with a key the command already sets. The machine's
  thermal ceiling has exactly one source of truth, and it is not this module
  (the guard owns the ceiling; a caller must not override it through argv).
* `ServerProcess.stop` sends exactly ONE `SIGINT` to the server's process
  group, then escalates to `SIGKILL` after a grace period. Two `SIGINT`s a
  few milliseconds apart made `llama-server` call `exit()` from inside its own
  signal handler and deadlock — see the `_forward_stop` comment in
  `ci/runner/gpu-thermal-guard.sh`. Never send a second `SIGINT` under any
  code path in this module.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from inferference.gpu_env import VISIBLE_DEVICE_VARS


class ServerError(RuntimeError):
    """The server did not start, did not become healthy, or could not be read."""


def _pin_card(env: dict[str, str], card: int) -> None:
    """Pin the child to one card on every backend, as `ci/runner/gpu-lease.sh` does.

    Each backend reads its own variable: Vulkan reads `GGML_VK_VISIBLE_DEVICES`,
    CUDA reads `CUDA_VISIBLE_DEVICES`, and ROCm reads `HIP_VISIBLE_DEVICES`. Set
    all of them, so an inherited value cannot move the child to another card.
    """
    for name in VISIBLE_DEVICE_VARS:
        env[name] = str(card)


# The fixed part of the argv, in the plan's exact order. `extra_args` is
# refused if it repeats any of these keys, or an alias of one.
_FIXED_FLAGS = (
    "--model",
    "--host",
    "--port",
    "--device",
    "--split-mode",
    "--n-gpu-layers",
    "--fit",
    "--ctx-size",
    "--parallel",
    "--batch-size",
    "--ubatch-size",
    "--flash-attn",
    "--cache-type-k",
    "--cache-type-v",
    "--alias",
    "--jinja",
    "--log-verbosity",
)

# Other names for a fixed flag, and flags that move weights off the card.
# `--n-gpu-layers 999` and `--split-mode none` keep the whole model in VRAM
# (AGENTS.md rule 3). An alias or a placement flag could undo that, so the
# refusal list names each one. Names have no dashes and use `-` for `_`.
_REFUSED_ALIASES = frozenset(
    {
        "gpu-layers",
        "ngl",
        "main-gpu",
        "mg",
        "tensor-split",
        "ts",
        "n-cpu-moe",
        "ncmoe",
        "cpu-moe",
        "cmoe",
        "override-tensor",
        "ot",
    }
)


def normalize_flag_name(key: str) -> str:
    """Return the flag name of an `extra_args` key, as the refusal list spells it.

    The name has no leading dashes, uses `-` for `_`, and is lower case.
    """
    return key.strip().lstrip("-").replace("_", "-").lower()


def refused_flag_keys(keys: Sequence[str]) -> list[str]:
    """Return the sorted `extra_args` keys that the command must not repeat.

    A key is refused when its normalised name is a fixed flag or an alias.
    """
    fixed = {flag[2:] for flag in _FIXED_FLAGS}
    return sorted(key for key in keys if normalize_flag_name(key) in fixed | _REFUSED_ALIASES)


@dataclass(frozen=True, slots=True)
class ServerCommand:
    """A fully resolved command: what to run, with what environment, on what port."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    port: int
    host: str = "127.0.0.1"


def merge_extra_args(model_args: Mapping[str, str], spec_args: Mapping[str, str]) -> dict[str, str]:
    """Merge the `extra_args` of the model YAML with `fixed.server.extra_args` of the spec.

    A key in both with equal values is kept once. A key in both with different
    values is refused with `ServerError`: the spec and the config root would
    then name two arms. `build_command` still refuses a flag that the runner
    sets itself.
    """
    merged = {str(key): str(value) for key, value in model_args.items()}
    conflicts = sorted(key for key, value in spec_args.items() if key in merged and merged[key] != str(value))
    if conflicts:
        raise ServerError(f"fixed.server.extra_args conflicts with the model YAML extra_args for: {conflicts}")
    merged.update({str(key): str(value) for key, value in spec_args.items()})
    return merged


def build_command(
    *,
    server_bin: Path,
    model: Path,
    alias: str,
    card: int,
    port: int,
    slots: int,
    tokens_per_slot: int,
    batch: int,
    ubatch: int,
    cache_k: str,
    cache_v: str,
    extra_args: Mapping[str, str],
    wrapper: Sequence[str],
    base_env: Mapping[str, str],
    fan_status_file: Path,
) -> ServerCommand:
    """Build the server's argv and environment. Never executes anything.

    `extra_args` keys must not repeat a flag the command already sets
    (`_FIXED_FLAGS`, plus `--device` taking its value from `card`), or an
    alias of one. A key with `=` is refused. A collision is refused with `ServerError` naming the key, because a silent
    second flag would let a caller override a safety-relevant value (for
    example `--device`) by accident.
    """
    embedded = sorted(key for key in extra_args if "=" in key)
    if embedded:
        raise ServerError(f"extra_args keys must not contain '=': {embedded}")
    collisions = refused_flag_keys(list(extra_args))
    if collisions:
        raise ServerError(f"extra_args repeats flag(s) the command already sets: {collisions}")

    # Scoped visibility maps the selected physical card to logical Vulkan0.
    device = "Vulkan0"
    argv: list[str] = [
        *wrapper,
        str(server_bin),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        device,
        "--split-mode",
        "none",
        "--n-gpu-layers",
        "999",
        "--fit",
        "off",
        "--ctx-size",
        str(slots * tokens_per_slot),
        "--parallel",
        str(slots),
        "--batch-size",
        str(batch),
        "--ubatch-size",
        str(ubatch),
        "--flash-attn",
        "auto",
        "--cache-type-k",
        cache_k,
        "--cache-type-v",
        cache_v,
        "--alias",
        alias,
        "--jinja",
        "--log-verbosity",
        "5",
    ]
    for key, value in extra_args.items():
        argv.extend([f"--{key}", value])

    env = dict(base_env)
    _pin_card(env, card)
    env["INFERFERENCE_GTT_BASELINE_CARD"] = str(card)
    env["GPU_LEASE_CARDS"] = str(card)
    env["GPU_THERMAL_FAN_STATUS_FILE"] = str(fan_status_file)
    return ServerCommand(argv=tuple(argv), env=env, port=port, host="127.0.0.1")


def list_devices(
    server_bin: Path,
    *,
    card: int,
    base_env: Mapping[str, str],
    timeout_s: float = 60,
    stop: threading.Event | None = None,
    poll_s: float = 0.2,
) -> str:
    """Run `<server_bin> --list-devices`, scoped to one card, and return stdout+stderr.

    No wrapper: the device list does not touch the GPU lease or the thermal
    guard, so it runs direct, as the plan specifies. When `stop` is set, the
    call kills the child and raises `ServerError` within `poll_s` seconds.
    """
    env = dict(base_env)
    _pin_card(env, card)
    process = subprocess.Popen(
        [str(server_bin), "--list-devices"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            if stop is not None and stop.is_set():
                raise ServerError("--list-devices stopped by a stop request")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ServerError(f"--list-devices did not exit within {timeout_s}s")
            try:
                stdout, stderr = process.communicate(timeout=min(poll_s, remaining))
            except subprocess.TimeoutExpired:
                continue
            return stdout + stderr
    finally:
        if process.poll() is None:
            # Kill the whole group: a child of the server would keep the pipes open.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()


def wait_for_bindable_port(
    port: int,
    *,
    host: str = "127.0.0.1",
    timeout_s: float = 240,
    poll_s: float = 0.5,
    stop: threading.Event | None = None,
) -> None:
    """Block until `(host, port)` is actually bindable, never merely "not listening".

    A closed connection to a just-stopped server holds its port in
    `TIME_WAIT` for a while: a process-liveness check sees no process and
    returns at once, while the socket still refuses a real `bind()`. Probe
    the real condition instead: bind without `SO_REUSEADDR`, which is what
    the server does, so a probe that succeeds means the server could bind
    too. When `stop` is set, the wait ends at once with `ServerError`.
    """
    deadline = time.monotonic() + timeout_s
    last: OSError | None = None
    while time.monotonic() < deadline:
        if stop is not None and stop.is_set():
            raise ServerError(f"wait for port {port} stopped by a stop request")
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((host, port))
        except OSError as exc:
            last = exc
            probe.close()
            if stop is not None:
                stop.wait(poll_s)
            else:
                time.sleep(poll_s)
            continue
        probe.close()
        return
    raise ServerError(f"port {port} on {host} was not bindable within {timeout_s}s: {last!r}")


class ServerProcess:
    """One `llama-server` child, run in its own process group.

    `start` uses `start_new_session=True` (the Python equivalent of `setsid`),
    so `stop` can signal the whole group with exactly one `SIGINT` and never
    risk a bare `kill` hitting only a wrapper script while the real server
    lives on underneath it.
    """

    def __init__(self, command: ServerCommand, log_path: Path) -> None:
        self._command = command
        self._log_path = log_path
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file: BinaryIO | None = None

    def start(self) -> None:
        """Launch the process group. Does not wait for health."""
        if self._process is not None:
            raise ServerError("start() called twice on the same ServerProcess")
        log_file = self._log_path.open("wb")
        self._log_file = log_file
        self._process = subprocess.Popen(
            list(self._command.argv),
            env=dict(self._command.env),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def alive(self) -> bool:
        """Is the process still running? Never raises."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def wait_healthy(self, timeout_s: float = 600) -> None:
        """Poll `GET /health` until it answers 200, or the process exits first.

        Fail-closed: a process that exits before health is observed raises
        `ServerError` at once. This never waits out the full timeout for a
        process that has already died, and it never assumes success.
        """
        if self._process is None:
            raise ServerError("wait_healthy() called before start()")
        url = f"http://{self._command.host}:{self._command.port}/health"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.alive():
                raise ServerError(
                    f"server exited with code {self._process.returncode} before becoming healthy; see {self._log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - localhost only
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(0.25)
        raise ServerError(f"server did not become healthy within {timeout_s}s; see {self._log_path}")

    def stop(self, *, grace_s: float = 30) -> int:
        """Send exactly one `SIGINT` to the process group, then `SIGKILL` after `grace_s`.

        Never sends a second `SIGINT` under any code path: two `SIGINT`s a
        few milliseconds apart deadlock `llama-server`. If the process already
        exited, this is a no-op that returns its exit code.
        """
        if self._process is None:
            raise ServerError("stop() called before start()")
        if not self.alive():
            return self._process.returncode  # type: ignore[return-value]

        pgid = os.getpgid(self._process.pid)
        os.killpg(pgid, signal.SIGINT)

        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if not self.alive():
                break
            time.sleep(0.1)
        else:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        returncode = self._process.wait()
        if self._log_file is not None:
            self._log_file.close()
        return returncode

    def log_text(self) -> str:
        """The server's combined stdout+stderr, verbatim, read fresh from disk."""
        return self._log_path.read_text(errors="replace")
