#!/usr/bin/env python3
"""A fake `llama-server`, for `tests/test_experiment_server.py` only.

This script stands in for a real `llama-server` binary so that
`inferference.experiment.server` can be tested with no GPU and no real
inference binary, ever. It is executable directly (its shebang resolves
`python3` on `PATH`, which the devenv shell provides), so a test can pass its
own path as `server_bin`.

Modes, selected by argv and environment:

* `--list-devices` on argv: print the fact-1-shaped device list to stdout and
  exit 0. No server starts.
* Otherwise: print fact-2/3-shaped log lines to stderr (the PCI address,
  device name, and layer count come from `FAKE_SERVER_PCI`,
  `FAKE_SERVER_DEVICE_NAME`, and `FAKE_SERVER_LAYERS`, so a test controls what
  "residency" text appears), then bind `--port` and serve:
    - `GET /health` -> 200
    - `POST /tokenize` -> `{"tokens": [...]}`, one token per whitespace-split
      word in `content`, so a test can predict the count.
    - `POST /v1/chat/completions` -> an SSE stream, ending with a chunk
      carrying `usage` and `timings`, then `data: [DONE]`.

`FAKE_SERVER_EXIT_BEFORE_BIND=1` makes it exit 1 before binding, to test the
exit-before-healthy path.

It traps `SIGINT` and exits 0 after exactly one handler run, writing the
count of `SIGINT`s it received to `FAKE_SERVER_SIGINT_COUNT_FILE` (atomically:
a temp file in the same directory, then `os.replace`), so a test can assert
that `ServerProcess.stop` sent exactly one.
"""

from __future__ import annotations

import http.server
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

_LIST_DEVICES_TEXT = "Available devices:\n  Vulkan0: AMD Radeon Pro V620 (RADV NAVI21) (30720 MiB, 30700 MiB free)\n"


def _arg_value(argv: list[str], flag: str, default: str) -> str:
    if flag in argv:
        index = argv.index(flag)
        if index + 1 < len(argv):
            return argv[index + 1]
    return default


def _log_lines() -> list[str]:
    pci = os.environ.get("FAKE_SERVER_PCI", "0000:03:00.0")
    device_name = os.environ.get("FAKE_SERVER_DEVICE_NAME", "AMD Radeon Pro V620 (RADV NAVI21)")
    logical = os.environ.get("FAKE_SERVER_LOGICAL_DEVICE", "Vulkan0")
    layers = int(os.environ.get("FAKE_SERVER_LAYERS", "66"))
    lines = [
        f"0.00.244.998 I llama_prepare_model_devices: using device {logical} ({device_name}) ({pci}) - 30700 MiB free"
    ]
    for layer in range(layers):
        lines.append(f"0.00.607.440 D load_tensors: layer {layer:4d} assigned to device {logical}, is_swa = 0")
    lines.append(f"0.01.272.102 I load_tensors: offloaded {layers}/{layers} layers to GPU")
    lines.append("0.01.272.106 I load_tensors:   CPU_Mapped model buffer size =   682.03 MiB")
    lines.append("0.08.466.268 I llama_init_from_model: enabling flash_attn since it is required for quantized V cache")
    lines.append("0.08.466.292 I llama_context: flash_attn    = enabled")
    return lines


def _write_sigint_count(count_file: str, count: int) -> None:
    target = Path(count_file)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(str(count))
    os.replace(temp, target)


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    # HTTP/1.1 so chunked streaming truncation raises under the client, like
    # llama-server (a cut HTTP/1.0 body would just look like a short body).
    protocol_version = "HTTP/1.1"
    shutdown_event: threading.Event | None = None

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - keep stderr free of per-request noise
        return

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming convention
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - http.server's naming convention
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw)
        if self.path == "/tokenize":
            content = str(payload.get("content", ""))
            tokens = list(range(len(content.split())))
            self._send_json(200, {"tokens": tokens})
        elif self.path == "/v1/chat/completions":
            self._stream_chat_completion(payload)
        else:
            self._send_json(404, {"error": "not found"})

    def _stream_chat_completion(self, payload: dict[str, object]) -> None:
        messages = payload.get("messages", [])
        parts: list[str] = []
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    parts.append(str(message.get("content", "")))
        prompt_tokens = len(" ".join(parts).split())

        def _chunk(data: bytes) -> None:
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        delta_chunk = {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]}
        _chunk(f"data: {json.dumps(delta_chunk)}\n\n".encode())
        stream_delay_s = float(os.environ.get("FAKE_SERVER_STREAM_DELAY_S", "0") or "0")
        if stream_delay_s > 0:
            # Hold the stream open mid-body so a runner-initiated shutdown
            # closes it under the client, like a real SIGINT mid-generation.
            time.sleep(stream_delay_s)
            if self.shutdown_event is not None and self.shutdown_event.is_set():
                # SIGINT landed mid-stream: llama-server cancels the streaming
                # task and the chunked body ends without the terminating
                # zero chunk (live failure 2026-10-08: "peer closed connection
                # without sending complete message body (incomplete chunked
                # read)"), so cut the connection rather than finish.
                self.close_connection = True
                return
        final_chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1},
            "timings": {
                "prompt_n": prompt_tokens,
                "prompt_ms": 1.0,
                "predicted_n": 1,
                "predicted_ms": 1.0,
                "cache_n": 0,
            },
        }
        _chunk(f"data: {json.dumps(final_chunk)}\n\n".encode())
        _chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")


def main(argv: list[str]) -> int:
    if "--list-devices" in argv:
        sys.stdout.write(_LIST_DEVICES_TEXT)
        sys.stdout.flush()
        return 0

    if "INFERFERENCE_GTT_BASELINE_CARD" in os.environ and "FAKE_SERVER_DRM_ROOT" in os.environ:
        card = os.environ["INFERFERENCE_GTT_BASELINE_CARD"]
        device = Path(os.environ["FAKE_SERVER_DRM_ROOT"]) / f"card{card}" / "device"
        pci = device.resolve().name
        gtt = (device / "mem_info_gtt_used").read_text().strip()
        vram = (device / "mem_info_vram_used").read_text().strip()
        print(f"INFERFERENCE_BASELINE card={card} pci={pci} gtt={gtt} vram={vram}", file=sys.stderr)
    for line in _log_lines():
        print(line, file=sys.stderr)
    sys.stderr.flush()

    if os.environ.get("FAKE_SERVER_EXIT_BEFORE_BIND") == "1":
        print("fake server exiting before bind, by request", file=sys.stderr)
        return 1

    host = _arg_value(argv, "--host", "127.0.0.1")
    port = int(_arg_value(argv, "--port", "0"))

    server = http.server.ThreadingHTTPServer((host, port), _FakeHandler)
    stop_event = threading.Event()
    _FakeHandler.shutdown_event = stop_event
    state = {"count": 0}
    count_file = os.environ.get("FAKE_SERVER_SIGINT_COUNT_FILE")

    def handle_sigint(_signum: int, _frame: object) -> None:
        state["count"] += 1
        if count_file:
            _write_sigint_count(count_file, state["count"])
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print("fake server ready", file=sys.stderr)
    sys.stderr.flush()

    stop_event.wait()
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
