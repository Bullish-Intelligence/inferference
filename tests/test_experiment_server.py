"""Tests for `inferference.experiment.server`.

Every test here runs the FAKE server fixture
(`tests/fixtures/experiment/fake_llama_server.py`), never a real
`llama-server` binary, and never touches a real GPU. See WP10 in
the runner plan for the pp_table actuator.
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import pytest

from inferference.experiment.server import (
    ServerError,
    ServerProcess,
    build_command,
    list_devices,
    wait_for_bindable_port,
)

FAKE_SERVER = Path(__file__).parent / "fixtures" / "experiment" / "fake_llama_server.py"


def _free_port() -> int:
    """An ephemeral port that is bindable right now, picked by the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _fake_server_command(tmp_path: Path, *, port: int, extra_env: dict[str, str] | None = None):
    base_env = dict(os.environ)
    base_env.update(extra_env or {})
    return build_command(
        server_bin=FAKE_SERVER,
        model=Path("/models/model.gguf"),
        alias="test-alias",
        card=0,
        port=port,
        slots=1,
        tokens_per_slot=1024,
        batch=256,
        ubatch=256,
        cache_k="q8_0",
        cache_v="q8_0",
        extra_args={},
        wrapper=(),
        base_env=base_env,
        fan_status_file=tmp_path / "fan-status",
    )


# -- build_command: argv shape --------------------------------------------------


def test_build_command_argv_shape() -> None:
    command = build_command(
        server_bin=Path("/opt/llama-server"),
        model=Path("/models/model.gguf"),
        alias="test-alias",
        card=1,
        port=8080,
        slots=4,
        tokens_per_slot=2048,
        batch=512,
        ubatch=256,
        cache_k="q8_0",
        cache_v="f16",
        extra_args={"spec-type": "ngram-mod"},
        wrapper=("/usr/bin/wrapper", "--flag"),
        base_env={},
        fan_status_file=Path("/run/arctic-fan/status"),
    )

    assert command.argv == (
        "/usr/bin/wrapper",
        "--flag",
        "/opt/llama-server",
        "--model",
        "/models/model.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--device",
        "Vulkan0",
        "--split-mode",
        "none",
        "--n-gpu-layers",
        "999",
        "--fit",
        "off",
        "--ctx-size",
        "8192",
        "--parallel",
        "4",
        "--batch-size",
        "512",
        "--ubatch-size",
        "256",
        "--flash-attn",
        "auto",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "f16",
        "--alias",
        "test-alias",
        "--jinja",
        "--log-verbosity",
        "5",
        "--spec-type",
        "ngram-mod",
    )
    assert command.port == 8080
    assert command.host == "127.0.0.1"


def test_build_command_refuses_extra_args_key_collision() -> None:
    with pytest.raises(ServerError):
        build_command(
            server_bin=Path("/opt/llama-server"),
            model=Path("/models/model.gguf"),
            alias="test-alias",
            card=0,
            port=8080,
            slots=1,
            tokens_per_slot=1024,
            batch=256,
            ubatch=256,
            cache_k="q8_0",
            cache_v="q8_0",
            extra_args={"device": "Vulkan5"},
            wrapper=(),
            base_env={},
            fan_status_file=Path("/run/arctic-fan/status"),
        )


# -- build_command: preserve the inherited machine limit ----------------------


def test_build_command_preserves_inherited_thermal_max_c() -> None:
    command = build_command(
        server_bin=Path("/opt/llama-server"),
        model=Path("/models/model.gguf"),
        alias="test-alias",
        card=2,
        port=8080,
        slots=1,
        tokens_per_slot=1024,
        batch=256,
        ubatch=256,
        cache_k="q8_0",
        cache_v="q8_0",
        extra_args={},
        wrapper=(),
        base_env={"GPU_THERMAL_MAX_C": "95", "PATH": "/usr/bin"},
        fan_status_file=Path("/run/arctic-fan/status"),
    )

    assert command.env["GPU_THERMAL_MAX_C"] == "95"
    assert command.env["GGML_VK_VISIBLE_DEVICES"] == "2"
    assert command.env["GPU_LEASE_CARDS"] == "2"
    assert command.env["GPU_THERMAL_FAN_STATUS_FILE"] == str(Path("/run/arctic-fan/status"))


def test_build_command_env_holds_no_thermal_max_c_even_when_base_env_lacks_it() -> None:
    command = build_command(
        server_bin=Path("/opt/llama-server"),
        model=Path("/models/model.gguf"),
        alias="test-alias",
        card=0,
        port=8080,
        slots=1,
        tokens_per_slot=1024,
        batch=256,
        ubatch=256,
        cache_k="q8_0",
        cache_v="q8_0",
        extra_args={},
        wrapper=(),
        base_env={},
        fan_status_file=Path("/run/arctic-fan/status"),
    )

    assert "GPU_THERMAL_MAX_C" not in command.env


def test_list_devices_preserves_inherited_thermal_max_c(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class _Completed:
        returncode = 0
        stdout = "Available devices:\n  Vulkan0: test device\n"
        stderr = ""

    def fake_run(*args, **kwargs):
        captured.update(kwargs["env"])
        return _Completed()

    monkeypatch.setattr("inferference.experiment.server.subprocess.run", fake_run)

    list_devices(Path("/fake/server"), card=1, base_env={"GPU_THERMAL_MAX_C": "95"})

    assert captured["GPU_THERMAL_MAX_C"] == "95"
    assert captured["GGML_VK_VISIBLE_DEVICES"] == "1"


# -- ServerProcess: healthy start -----------------------------------------------


def test_healthy_start_against_fake_server_succeeds(tmp_path: Path) -> None:
    port = _free_port()
    sigint_file = tmp_path / "sigint-count"
    command = _fake_server_command(tmp_path, port=port, extra_env={"FAKE_SERVER_SIGINT_COUNT_FILE": str(sigint_file)})
    process = ServerProcess(command, log_path=tmp_path / "server.log")

    process.start()
    try:
        process.wait_healthy(timeout_s=15)
        assert process.alive()
        assert "fake server ready" in process.log_text()
    finally:
        process.stop(grace_s=5)


# -- ServerProcess: exit before healthy raises ----------------------------------


def test_exit_before_healthy_raises_server_error(tmp_path: Path) -> None:
    port = _free_port()
    command = _fake_server_command(tmp_path, port=port, extra_env={"FAKE_SERVER_EXIT_BEFORE_BIND": "1"})
    process = ServerProcess(command, log_path=tmp_path / "server.log")

    process.start()
    with pytest.raises(ServerError):
        process.wait_healthy(timeout_s=10)
    assert not process.alive()


# -- ServerProcess: stop sends exactly one SIGINT -------------------------------


def test_stop_delivers_exactly_one_sigint(tmp_path: Path) -> None:
    port = _free_port()
    sigint_file = tmp_path / "sigint-count"
    command = _fake_server_command(tmp_path, port=port, extra_env={"FAKE_SERVER_SIGINT_COUNT_FILE": str(sigint_file)})
    process = ServerProcess(command, log_path=tmp_path / "server.log")

    process.start()
    process.wait_healthy(timeout_s=15)
    returncode = process.stop(grace_s=10)

    assert returncode == 0
    assert sigint_file.read_text().strip() == "1"
    assert not process.alive()


# -- wait_for_bindable_port: TIME_WAIT blocks until timeout ---------------------


def test_wait_for_bindable_port_blocks_during_time_wait() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.listen(1)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    accepted, _ = listener.accept()

    # The side that closes first does the active close and lands in
    # TIME_WAIT, holding the port. Close it before the listener itself, so
    # the eventual failure is attributable to TIME_WAIT, not to the listener
    # still being bound.
    accepted.close()
    client.close()
    listener.close()
    time.sleep(0.05)

    started = time.monotonic()
    with pytest.raises(ServerError):
        wait_for_bindable_port(port, timeout_s=1.0, poll_s=0.1)
    elapsed = time.monotonic() - started
    assert elapsed >= 0.9
