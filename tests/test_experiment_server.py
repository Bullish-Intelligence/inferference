"""Tests for `inferference.experiment.server`.

Every test here runs the FAKE server fixture
(`tests/fixtures/experiment/fake_llama_server.py`), never a real
`llama-server` binary, and never touches a real GPU. See WP10 in
the runner plan for the pp_table actuator.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import pytest

from inferference.experiment.server import (
    ServerError,
    ServerProcess,
    build_command,
    list_devices,
    merge_extra_args,
    wait_for_bindable_port,
)

_PATH = {"PATH": os.environ["PATH"]}
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


def _command_with_extra_args(extra_args: dict[str, str]):
    return build_command(
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
        extra_args=extra_args,
        wrapper=(),
        base_env={},
        fan_status_file=Path("/run/arctic-fan/status"),
    )


@pytest.mark.parametrize("key", ["port=1", "threads=2", "--port=1"])
def test_build_command_refuses_an_extra_args_key_with_equals(key: str) -> None:
    with pytest.raises(ServerError, match="="):
        _command_with_extra_args({key: "1"})


@pytest.mark.parametrize(
    "key",
    [
        "gpu-layers",
        "ngl",
        "main-gpu",
        "tensor-split",
        "n-cpu-moe",
        "cpu-moe",
        "override-tensor",
        "n_gpu_layers",
        "--ngl",
        "GPU-Layers",
    ],
)
def test_build_command_refuses_an_alias_of_a_fixed_flag(key: str) -> None:
    with pytest.raises(ServerError, match="already sets"):
        _command_with_extra_args({key: "0"})


def test_build_command_keeps_n_gpu_layers_999_and_passes_a_plain_flag() -> None:
    command = _command_with_extra_args({"threads": "2"})

    assert command.argv[command.argv.index("--n-gpu-layers") + 1] == "999"
    assert command.argv[-2:] == ("--threads", "2")


def test_merge_extra_args_joins_model_and_spec_flags() -> None:
    merged = merge_extra_args({"threads": "2"}, {"spec-type": "ngram-mod"})

    assert merged == {"threads": "2", "spec-type": "ngram-mod"}


def test_merge_extra_args_keeps_a_key_that_both_sides_set_to_one_value() -> None:
    assert merge_extra_args({"threads": "2"}, {"threads": "2"}) == {"threads": "2"}


def test_merge_extra_args_refuses_a_key_with_two_values() -> None:
    with pytest.raises(ServerError, match="threads"):
        merge_extra_args({"threads": "2"}, {"threads": "4"})


def test_merged_extra_args_still_refuse_a_flag_the_command_sets() -> None:
    merged = merge_extra_args({}, {"ctx-size": "1"})

    with pytest.raises(ServerError, match="ctx-size"):
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
            extra_args=merged,
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


def _device_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake-list-devices.sh"
    script.write_text(f"#!/usr/bin/env bash\n{body}\n")
    script.chmod(0o755)
    return script


def test_list_devices_preserves_inherited_thermal_max_c(tmp_path: Path) -> None:
    script = _device_script(
        tmp_path, 'echo "Available devices:"; echo "max=$GPU_THERMAL_MAX_C vk=$GGML_VK_VISIBLE_DEVICES" >&2'
    )

    text = list_devices(script, card=1, base_env={**_PATH, "GPU_THERMAL_MAX_C": "95"})

    assert "Available devices:" in text
    assert "max=95 vk=1" in text


def test_list_devices_stops_at_once_when_a_stop_is_requested(tmp_path: Path) -> None:
    script = _device_script(tmp_path, "sleep 30")
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()

    started = time.monotonic()
    with pytest.raises(ServerError, match="stop request"):
        list_devices(script, card=0, base_env=_PATH, timeout_s=30, stop=stop, poll_s=0.05)

    assert time.monotonic() - started < 5


def test_list_devices_still_times_out(tmp_path: Path) -> None:
    script = _device_script(tmp_path, "sleep 30")

    with pytest.raises(ServerError, match="did not exit"):
        list_devices(script, card=0, base_env=_PATH, timeout_s=0.3, poll_s=0.05)


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


def test_wait_for_bindable_port_ends_when_a_stop_is_requested() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    try:
        started = time.monotonic()
        with pytest.raises(ServerError, match="stop request"):
            wait_for_bindable_port(port, timeout_s=60, poll_s=0.1, stop=stop)
        assert time.monotonic() - started < 5
    finally:
        listener.close()


@pytest.mark.parametrize("name", ["GGML_VK_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"])
def test_build_command_pins_the_card_on_every_backend_variable(name: str) -> None:
    command = build_command(
        server_bin=Path("/opt/llama-server"),
        model=Path("/models/model.gguf"),
        alias="test-alias",
        card=1,
        port=8080,
        slots=1,
        tokens_per_slot=1024,
        batch=256,
        ubatch=256,
        cache_k="q8_0",
        cache_v="q8_0",
        extra_args={},
        wrapper=(),
        base_env={"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0", "PATH": "/usr/bin"},
        fan_status_file=Path("/run/arctic-fan/status"),
    )

    assert command.env[name] == "1"


def test_list_devices_pins_the_card_on_every_backend_variable(tmp_path: Path) -> None:
    script = _device_script(
        tmp_path,
        'echo "Available devices:"; '
        'echo "vk=$GGML_VK_VISIBLE_DEVICES cuda=$CUDA_VISIBLE_DEVICES hip=$HIP_VISIBLE_DEVICES" >&2',
    )

    text = list_devices(script, card=1, base_env={**_PATH, "CUDA_VISIBLE_DEVICES": "0"})

    assert "vk=1 cuda=1 hip=1" in text
