"""Serve Atuin AI's protocol in front of an OpenAI-compatible endpoint.

`atuin ai` does NOT speak OpenAI. It POSTs to `{endpoint}/api/cli/chat`
with `Accept: text/event-stream` and its own schema, and expects a named-event
SSE stream back. This adapter translates both directions so the inferference
routed llama-server can serve `atuin ai`.

    atuin ai  --(Atuin protocol)-->  THIS  --(OpenAI)-->  llama-server

Every claim below is read from atuin v18.18.1's source, not inferred:

REQUEST, `crates/atuin-ai/src/stream.rs:134-170`
    POST {endpoint}/api/cli/chat        Accept: text/event-stream, Bearer <token>
    { "messages":[...], "context":{...}, "invocation_id":"...",
      "config": {"capabilities":[...], "user_contexts":?, "skills":?, "model":?},
      "session_id": ? }

RESPONSE, `stream.rs:216-290` — SSE with NAMED events. Unnamed `data:` frames are
dropped on the floor by that match, which is the trap here: an OpenAI-shaped body
gets a 200 and renders nothing.
    event: text         data: {"content":"..."}
    event: tool_call    data: {"id":"...","name":"...","input":{...}}
    event: tool_result  data: {"tool_use_id":"...","content":"...","is_error":bool,
                               "remote":bool,"content_length":int}
    event: status       data: {"state":"..."}
    event: error        data: {"message":"..."}
    event: done         data: {"session_id":"...","credits":...}   <- terminates
    header x-atuin-ai-session-id  (optional, yields SessionIdentity)

TOOLS. Atuin advertises `client_v1_*` capabilities; the wire names the model must
emit are the canonical ones (`tools/descriptor.rs:24-85`) — `read_file`,
`edit_file`, `write_file`, `execute_shell_command`, `atuin_history`,
`atuin_output`, `load_skill`. The client executes them locally and returns the
results in the NEXT request's messages. Input fields are read from each executor
in `tools/mod.rs` (lines 325, 450, 648, 763, 1075, 1208, 1351) rather than
guessed.

    atuin-ai-adapter.py [--listen 127.0.0.1:8100] [--upstream http://127.0.0.1:8081/v1]
                        [--token SECRET] [--model NAME]

Point atuin at it:

    [ai]
    enabled  = true
    endpoint = "http://127.0.0.1:8100"
    api_token = "..."          # must match --token, or omit both

SPIKE STATUS: this is a working spike, not a shipped service. It has no TLS, no
concurrency limit beyond the threading server, and it forwards whatever the model
emits. Run it on loopback only.
"""

from __future__ import annotations

import argparse
import json
import socketserver
import sys
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# capability advertised by atuin -> (wire tool name, JSON schema)
# Field names come from the executors in crates/atuin-ai/src/tools/mod.rs.
TOOLS: dict[str, tuple[str, dict[str, Any]]] = {
    "client_v1_read_file": (
        "read_file",
        {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to read."},
                "offset": {"type": "integer", "description": "First line (1-based)."},
                "limit": {"type": "integer", "description": "How many lines."},
            },
            "required": ["file_path"],
        },
    ),
    "client_v1_edit_file": (
        "edit_file",
        {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    ),
    "client_v1_write_file": (
        "write_file",
        {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            "required": ["file_path", "content"],
        },
    ),
    "client_v1_execute_shell_command": (
        "execute_shell_command",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "description": {"type": "string", "description": "One line on what it does."},
                "dir": {"type": "string", "description": "Working directory."},
                "shell": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds."},
            },
            "required": ["command"],
        },
    ),
    "client_v1_atuin_history": (
        "atuin_history",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search the user's shell history."},
                "limit": {"type": "integer"},
                "only_failed": {"type": "boolean"},
                "authors": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
    ),
    "client_v1_atuin_output": (
        "atuin_output",
        {
            "type": "object",
            "properties": {
                "history_id": {"type": "string", "description": "History entry to fetch output for."},
                "ranges": {"type": "string"},
            },
            "required": ["history_id"],
        },
    ),
    "client_v1_load_skill": (
        "load_skill",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    ),
}

WIRE_TO_CAP = {wire: cap for cap, (wire, _) in TOOLS.items()}

SYSTEM_PROMPT = (
    "You are the Atuin shell assistant, running inside the user's terminal. "
    "Answer briefly and concretely. When a tool would answer the question, call it "
    "rather than describing what you would do. Prefer one tool call at a time. "
    "The user's shell context is supplied below."
)


def build_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Atuin messages + its context block -> OpenAI messages."""
    ctx = body.get("context") or {}
    parts = [f"{k}: {v}" for k, v in ctx.items() if v not in (None, "", "unknown")]
    system = SYSTEM_PROMPT + ("\n\nShell context:\n" + "\n".join(parts) if parts else "")

    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "user"
        content = m.get("content")
        # Atuin replays tool results as message content; keep them as plain text
        # rather than inventing OpenAI tool-role plumbing the client never sent.
        if isinstance(content, list):
            content = "\n".join(
                str(c.get("text") or c.get("content") or c) if isinstance(c, dict) else str(c) for c in content
            )
        out.append({"role": role if role in ("user", "assistant", "system") else "user", "content": content or ""})
    return out


def build_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    caps = ((body.get("config") or {}).get("capabilities")) or []
    tools = []
    for cap in caps:
        if cap in TOOLS:
            wire, schema = TOOLS[cap]
            tools.append({"type": "function", "function": {"name": wire, "parameters": schema}})
    return tools


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = "http://127.0.0.1:8081/v1"
    token = ""
    model = "inferference"
    verbose = False

    def _sse(self, event: str, payload: dict[str, Any]) -> None:
        chunk = f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()
        self.wfile.write(chunk)
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        if self.verbose:
            sys.stderr.write("[adapter] " + (format % args) + "\n")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/api/cli/chat":
            self.send_error(404, "only /api/cli/chat is served")
            return

        # Atuin distinguishes 401 from other failures and tells the user which
        # setting to fix (stream.rs:193-207), so answer 401 rather than 403.
        if self.token:
            sent = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
            if sent != self.token:
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError as exc:
            self.send_error(400, f"bad JSON: {exc}")
            return

        session_id = body.get("session_id") or str(uuid.uuid4())

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Atuin-AI-Session-Id", session_id)
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            self._relay(body)
        except Exception as exc:  # noqa: BLE001 - the stream must always terminate
            self._sse("error", {"message": f"adapter: {exc}"})
        # `done` is what breaks atuin's read loop (stream.rs:267-281). Without it
        # the TUI waits forever, so it is emitted on every path.
        self._sse("done", {"session_id": session_id})

    def _relay(self, body: dict[str, Any]) -> None:
        payload = {
            "model": (body.get("config") or {}).get("model") or self.model,
            "messages": build_messages(body),
            "stream": True,
            "temperature": 0.3,
        }
        tools = build_tools(body)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        req = urllib.request.Request(  # noqa: S310 - loopback, operator-supplied
            f"{self.upstream.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # id -> {name, args} accumulated across deltas; OpenAI streams tool call
        # arguments in fragments, and atuin needs one whole object per call.
        calls: dict[int, dict[str, Any]] = {}
        sent_status = False

        try:
            resp = urllib.request.urlopen(req, timeout=600)  # noqa: S310
        except urllib.error.URLError as exc:
            self._sse("error", {"message": f"upstream unreachable at {self.upstream}: {exc}"})
            return

        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}

                text = delta.get("content")
                if text:
                    self._sse("text", {"content": text})

                # A reasoning model emits its thinking here. Surfacing it as a
                # status keeps the TUI alive during long think phases without
                # pasting the whole chain of thought into the answer.
                if delta.get("reasoning_content") and not sent_status:
                    self._sse("status", {"state": "thinking"})
                    sent_status = True

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = calls.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]

        for slot in calls.values():
            name = slot["name"]
            if name not in WIRE_TO_CAP:
                self._sse("error", {"message": f"model asked for unknown tool {name!r}"})
                continue
            try:
                parsed = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                self._sse("error", {"message": f"tool {name}: arguments were not JSON: {slot['args'][:200]}"})
                continue
            self._sse("tool_call", {"id": slot["id"] or str(uuid.uuid4()), "name": name, "input": parsed})


class ThreadingHTTP(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="127.0.0.1:8100")
    ap.add_argument("--upstream", default="http://127.0.0.1:8081/v1")
    ap.add_argument("--token", default="", help="require this bearer token; empty = no auth")
    ap.add_argument("--model", default="inferference")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    host, _, port = args.listen.rpartition(":")
    Handler.upstream = args.upstream
    Handler.token = args.token
    Handler.model = args.model
    Handler.verbose = args.verbose

    srv = ThreadingHTTP((host or "127.0.0.1", int(port)), Handler)
    print(f">> atuin-ai adapter on http://{host or '127.0.0.1'}:{port}/api/cli/chat", file=sys.stderr)
    print(f">> upstream {args.upstream}  auth {'on' if args.token else 'off'}", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
