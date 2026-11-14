"""A thin replica balancer in front of the llama.cpp router.

The router maps one model name to one child process. To serve one model from
two cards, the placement declares the model resident twice under two names. A
client that uses one name then leaves one card idle. This balancer gives a
GROUP of such children one public name::

    placement.yaml
      balancer:
        host: 127.0.0.1
        port: 8100
        groups:
          <public-name>: [<member>, <member>]

It listens on ``balancer.host:port`` and proxies every request to the router
(``router.host:port``). It changes a request only in two cases:

* The request names a group in its JSON body (``model``) or its query string
  (``model``). The balancer picks one member and rewrites ``model`` to the
  member's name. The member is the loaded one with the fewest requests in
  flight that the balancer counts. Equal counts take turns.
* The request carries ``autoload`` in its query string. The balancer removes it.
  A request for a model that is not loaded is an error, never a load.

Every other name and path passes through unchanged. A request for a group
member's own name is not special: a group may reuse the name of one of its
members, and then that name means the group.

The balancer answers a few paths itself:

* ``GET /health``: 200 when the router answered a poll recently and every group
  has at least one loaded member. Otherwise 503 with the reasons.
* ``GET /balancer/status``: counters, per-member in-flight and state.
* ``GET /models`` and ``GET /v1/models``: the router's list plus one entry per
  group. The group entry carries ``members`` and the TOTAL ``--parallel`` and
  ``--ctx-size`` of its loaded members, so a client sizes its concurrency to the
  group. It has no ``preset`` and no ``inferference_residency``: both describe
  one child. A tool that needs them reads the router.
* ``GET /slots?model=<group>``: the slot lists of the loaded members, merged.

Failover. If the connection to the router fails, or the router answers 5xx, or
answers 400 "model is not loaded", BEFORE the balancer sends any byte to the
client, the balancer marks that member down until the next poll and tries the
next member once. After the first byte reaches the client it never retries.

Streaming passes through unbuffered. The balancer reads whatever bytes the
router has sent and writes them to the client at once. It closes the upstream
connection when the client disconnects.

Each connection carries one request (``Connection: close`` on both sides). The
balancer is small on purpose: it reads HTTP/1.1 with the standard library only.

    python -m inferference.balancer --config-dir DIR [--check]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import http
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

from .config_paths import ConfigError, read_placement
from .placement import PlacementError

__all__ = ["Balancer", "Group", "Member", "main"]

log = logging.getLogger("inferference.balancer")

EX_CONFIG = 78

POLL_INTERVAL_S = 2.0
POLL_TIMEOUT_S = 5.0
CONNECT_TIMEOUT_S = 10.0
#: Same as the router's own default (--timeout 3600). It is the idle gap on a socket.
READ_TIMEOUT_S = 3600.0
#: Time a client may take to send its request head.
HEAD_TIMEOUT_S = 60.0
#: A response that the balancer buffers (errors, lists) stays under this size.
BUFFER_LIMIT = 4 * 1024 * 1024
MAX_REQUEST_BODY = 1024 * 1024 * 1024
#: Health needs a poll no older than this many poll cycles (interval + timeout each).
HEALTH_WINDOW_CYCLES = 2

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-connection",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
    "transfer-encoding",
    "content-length",
    "expect",
}
_BODY_METHODS = {"POST", "PUT", "PATCH"}
_STATE_RANK = {"loaded": 0, "loading": 1, "sleeping": 2, "unloaded": 3, "unknown": 4, "missing": 5}


# --------------------------------------------------------------------------- state


@dataclass
class Member:
    """One resident model of a group, as the balancer sees it."""

    name: str
    state: str = "unknown"
    args: list[str] = field(default_factory=list)
    #: Set by a failed request. The next successful poll clears it.
    down: bool = False
    in_flight: int = 0
    requests: int = 0
    failures: int = 0
    last_poll_unix: float | None = None

    @property
    def usable(self) -> bool:
        return self.state == "loaded" and not self.down

    def describe(self) -> str:
        return f"{self.name}={self.state}{' (down)' if self.down else ''}"


@dataclass
class Group:
    name: str
    members: list[Member]
    cursor: int = 0

    def choose(self, exclude: Iterable[str] = ()) -> Member | None:
        """Least in-flight among the usable members. Ties take turns.

        The scan starts at ``cursor`` and ``min`` keeps the first of equal
        counts, so equal members rotate in declared order.
        """
        skip = set(exclude)
        n = len(self.members)
        order = [self.members[(self.cursor + i) % n] for i in range(n)]
        candidates = [m for m in order if m.usable and m.name not in skip]
        if not candidates:
            return None
        chosen = min(candidates, key=lambda m: m.in_flight)
        self.cursor = (self.members.index(chosen) + 1) % n
        return chosen

    def states(self) -> str:
        return ", ".join(m.describe() for m in self.members)


# ---------------------------------------------------------------- HTTP primitives


class BadMessage(Exception):
    """The peer sent something that is not HTTP the balancer can read."""


@dataclass
class Head:
    """A start line and its headers, in order."""

    first: str
    headers: list[tuple[str, str]]

    def get(self, name: str) -> str | None:
        low = name.lower()
        for key, value in self.headers:
            if key.lower() == low:
                return value
        return None

    @property
    def status(self) -> int:
        parts = self.first.split(None, 2)
        try:
            return int(parts[1])
        except (IndexError, ValueError) as exc:
            raise BadMessage(f"bad status line {self.first!r}") from exc

    @property
    def chunked(self) -> bool:
        return "chunked" in (self.get("transfer-encoding") or "").lower()

    @property
    def content_length(self) -> int | None:
        raw = self.get("content-length")
        if raw is None:
            return None
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise BadMessage(f"bad Content-Length {raw!r}") from exc
        if value < 0:
            raise BadMessage(f"bad Content-Length {raw!r}")
        return value


def _parse_head(raw: bytes) -> Head:
    lines = raw.decode("latin-1").split("\r\n")
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        if not line:
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise BadMessage(f"bad header line {line!r}")
        headers.append((key.strip(), value.strip()))
    return Head(lines[0], headers)


async def _read_head(reader: asyncio.StreamReader, timeout: float) -> Head:
    try:
        raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout)
    except asyncio.IncompleteReadError as exc:
        raise BadMessage("connection closed inside the message head") from exc
    except asyncio.LimitOverrunError as exc:
        raise BadMessage("message head is too large") from exc
    return _parse_head(raw)


async def _read_chunked(reader: asyncio.StreamReader, limit: int, timeout: float) -> bytes:
    out = bytearray()
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout)
        if not line:
            raise BadMessage("connection closed inside a chunked body")
        size_text = line.split(b";", 1)[0].strip()
        try:
            size = int(size_text, 16)
        except ValueError as exc:
            raise BadMessage(f"bad chunk size {size_text!r}") from exc
        if size == 0:
            while (await asyncio.wait_for(reader.readline(), timeout)).strip():
                pass  # trailers
            return bytes(out)
        if len(out) + size > limit:
            raise BadMessage("body is too large")
        out += await asyncio.wait_for(reader.readexactly(size), timeout)
        await asyncio.wait_for(reader.readexactly(2), timeout)  # the CRLF after the data


async def _read_body(reader: asyncio.StreamReader, head: Head, limit: int, timeout: float, *, eof_ends: bool) -> bytes:
    """Read a message body: chunked, Content-Length, or (responses) until EOF."""
    try:
        if head.chunked:
            return await _read_chunked(reader, limit, timeout)
        length = head.content_length
        if length is not None:
            if length > limit:
                raise BadMessage("body is too large")
            return await asyncio.wait_for(reader.readexactly(length), timeout)
        if not eof_ends:
            return b""
        data = await asyncio.wait_for(reader.read(limit + 1), timeout)
        if len(data) > limit:
            raise BadMessage("body is too large")
        return data
    except asyncio.IncompleteReadError as exc:
        raise BadMessage("connection closed inside the body") from exc


def _no_body(method: str, status: int) -> bool:
    return method == "HEAD" or status < 200 or status in (204, 304)


def _serialize(first: str, headers: Iterable[tuple[str, str]], body: bytes | None = None) -> bytes:
    lines = [first, *(f"{k}: {v}" for k, v in headers), "", ""]
    out = "\r\n".join(lines).encode("latin-1")
    return out + body if body else out


def _response_bytes(
    status: int, body: bytes, content_type: str = "application/json", extra: Sequence[tuple[str, str]] = ()
) -> bytes:
    phrase = http.HTTPStatus(status).phrase
    headers = [("Content-Type", content_type), ("Content-Length", str(len(body))), ("Connection", "close"), *extra]
    return _serialize(f"HTTP/1.1 {status} {phrase}", headers, body)


def _json_response(status: int, payload: Any) -> bytes:
    return _response_bytes(status, json.dumps(payload).encode())


def _error_response(status: int, message: str, **extra: Any) -> bytes:
    kind = "unavailable_error" if status in (502, 503) else "invalid_request_error"
    return _json_response(status, {"error": {"code": status, "message": message, "type": kind, **extra}})


def _normalized(head: Head, body: bytes) -> bytes:
    """A buffered upstream response, re-framed with Content-Length."""
    skip = _HOP_BY_HOP
    headers = [(k, v) for k, v in head.headers if k.lower() not in skip]
    headers += [("Content-Length", str(len(body))), ("Connection", "close")]
    return _serialize(head.first, headers, body)


def _streamed_head(head: Head) -> bytes:
    """A response head for a streamed body. Only the connection headers change."""
    headers = [(k, v) for k, v in head.headers if k.lower() not in ("connection", "keep-alive")]
    headers.append(("Connection", "close"))
    return _serialize(head.first, headers)


@dataclass
class Request:
    method: str
    target: str
    headers: list[tuple[str, str]]
    body: bytes

    @property
    def path(self) -> str:
        return urlsplit(self.target).path

    @property
    def query(self) -> list[tuple[str, str]]:
        return parse_qsl(urlsplit(self.target).query, keep_blank_values=True)

    def header(self, name: str) -> str | None:
        low = name.lower()
        return next((v for k, v in self.headers if k.lower() == low), None)


def _with_query(path: str, pairs: list[tuple[str, str]]) -> str:
    return f"{path}?{urlencode(pairs)}" if pairs else path


def _arg_int(args: Sequence[Any], flag: str) -> int | None:
    for i, item in enumerate(args[:-1]):
        if item == flag:
            try:
                return int(args[i + 1])
            except (TypeError, ValueError):
                return None
    return None


@dataclass
class Failure:
    """An attempt that failed before any byte reached the client."""

    reason: str
    response: bytes | None = None  # the router's own answer, when it gave one


# -------------------------------------------------------------------- the balancer


class Balancer:
    def __init__(
        self,
        groups: dict[str, list[str]],
        router_host: str,
        router_port: int,
        *,
        poll_interval: float = POLL_INTERVAL_S,
        poll_timeout: float = POLL_TIMEOUT_S,
        connect_timeout: float = CONNECT_TIMEOUT_S,
        read_timeout: float = READ_TIMEOUT_S,
    ) -> None:
        self.router_host = router_host
        self.router_port = router_port
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.groups = {name: Group(name, [Member(m) for m in members]) for name, members in groups.items()}
        self.last_poll_ok: float | None = None  # time.monotonic()
        self.last_poll_ok_unix: float | None = None
        self.last_poll_error: str | None = None
        self.polls_ok = 0
        self.polls_failed = 0
        self._fail_streak = 0
        self.requests = 0
        self.rejected = 0
        self.failovers = 0
        self.started_unix = time.time()
        self._tasks: set[asyncio.Task[Any]] = set()

    # ---- the router connection (one connection per request) --------------------

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await asyncio.wait_for(
            asyncio.open_connection(self.router_host, self.router_port, limit=2**20), self.connect_timeout
        )

    def _upstream_request(self, method: str, target: str, headers: list[tuple[str, str]], body: bytes) -> bytes:
        out = [(k, v) for k, v in headers if k.lower() not in _HOP_BY_HOP and k.lower() != "host"]
        out.append(("Host", f"{self.router_host}:{self.router_port}"))
        if body or method in _BODY_METHODS:
            out.append(("Content-Length", str(len(body))))
        out.append(("Connection", "close"))
        return _serialize(f"{method} {target} HTTP/1.1", out, body)

    async def _fetch(
        self, method: str, target: str, headers: list[tuple[str, str]] | None = None
    ) -> tuple[Head, bytes]:
        """One buffered request to the router. Raises OSError or BadMessage."""
        sent = [(k, v) for k, v in (headers or []) if k.lower() != "accept-encoding"] + [
            ("Accept-Encoding", "identity")
        ]
        reader, writer = await self._connect()
        try:
            writer.write(self._upstream_request(method, target, sent, b""))
            await writer.drain()
            head = await _read_head(reader, self.read_timeout)
            body = b""
            if not _no_body(method, head.status):
                body = await _read_body(reader, head, BUFFER_LIMIT, self.read_timeout, eof_ends=True)
            return head, body
        finally:
            writer.close()

    # ---- polling ---------------------------------------------------------------

    async def poll_once(self) -> bool:
        """Read GET /models and update every member. On failure keep the old state."""
        try:
            head, body = await asyncio.wait_for(self._fetch("GET", "/models"), self.poll_timeout)
            if head.status != 200:
                raise BadMessage(f"GET /models answered {head.first}")
            entries = json.loads(body)["data"]
            by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
        except (OSError, BadMessage, TimeoutError, ValueError, KeyError, TypeError) as exc:
            self.polls_failed += 1
            self._fail_streak += 1
            self.last_poll_error = f"{type(exc).__name__}: {exc}"
            if self._fail_streak == 1 or self._fail_streak % 15 == 0:
                log.warning(
                    "poll failed (%d in a row), keeping the last known state: %s",
                    self._fail_streak,
                    self.last_poll_error,
                )
            return False
        if self._fail_streak:
            log.info("poll recovered after %d failure(s)", self._fail_streak)
        self._fail_streak = 0
        self.last_poll_error = None
        self.polls_ok += 1
        self.last_poll_ok = time.monotonic()
        self.last_poll_ok_unix = time.time()
        for group in self.groups.values():
            for m in group.members:
                status = (by_id.get(m.name) or {}).get("status") or {}
                state = status.get("value", "unknown") if m.name in by_id else "missing"
                if state != m.state:
                    log.info("member %s: %s -> %s", m.name, m.state, state)
                m.state = state
                args = status.get("args")
                m.args = [str(a) for a in args] if isinstance(args, list) else []
                m.down = False
                m.last_poll_unix = self.last_poll_ok_unix
        return True

    async def poll_loop(self) -> None:
        while True:
            started = time.monotonic()
            await self.poll_once()
            await asyncio.sleep(max(0.0, self.poll_interval - (time.monotonic() - started)))

    # ---- the server ------------------------------------------------------------

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            try:
                request = await self._read_request(reader, writer)
            except (BadMessage, TimeoutError, ConnectionError) as exc:
                with contextlib.suppress(Exception):
                    writer.write(_error_response(400, f"bad request: {exc}"))
                    await writer.drain()
                return
            work = asyncio.ensure_future(self._dispatch(request, writer))
            watch = asyncio.ensure_future(self._watch_disconnect(reader))
            await asyncio.wait({work, watch}, return_when=asyncio.FIRST_COMPLETED)
            if not work.done():
                log.info("client left: cancelled %s %s", request.method, request.path)
                work.cancel()
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work
            with contextlib.suppress(asyncio.CancelledError):
                await watch
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("request failed")
        finally:
            if task is not None:
                self._tasks.discard(task)
            writer.close()

    @staticmethod
    async def _watch_disconnect(reader: asyncio.StreamReader) -> None:
        with contextlib.suppress(ConnectionError, OSError):
            while await reader.read(65536):
                pass  # one request per connection: later bytes are ignored

    async def _read_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> Request:
        head = await _read_head(reader, HEAD_TIMEOUT_S)
        parts = head.first.split()
        if len(parts) != 3 or not parts[2].startswith("HTTP/"):
            raise BadMessage(f"bad request line {head.first!r}")
        if (head.get("expect") or "").lower() == "100-continue":
            writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            await writer.drain()
        body = await _read_body(reader, head, MAX_REQUEST_BODY, self.read_timeout, eof_ends=False)
        return Request(parts[0].upper(), parts[1], head.headers, body)

    async def serve(self, host: str, port: int) -> asyncio.Server:
        return await asyncio.start_server(self.handle, host, port, limit=2**20)

    async def close_connections(self, grace: float = 5.0) -> None:
        tasks = list(self._tasks)
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=grace)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    # ---- routing ---------------------------------------------------------------

    async def _send(self, writer: asyncio.StreamWriter, data: bytes) -> None:
        writer.write(data)
        await writer.drain()

    async def _dispatch(self, req: Request, writer: asyncio.StreamWriter) -> None:
        self.requests += 1
        path = req.path
        if req.method in ("GET", "HEAD"):
            if path == "/health":
                return await self._send(writer, self._health_response())
            if path == "/balancer/status":
                return await self._send(writer, _json_response(200, self.status()))

        pairs = req.query
        if any(k == "autoload" for k, _ in pairs):
            log.warning("stripped ?autoload from %s %s: a request never loads a model", req.method, path)
            pairs = [(k, v) for k, v in pairs if k != "autoload"]
            req = Request(req.method, _with_query(path, pairs), req.headers, req.body)

        if req.method == "GET" and path in ("/models", "/v1/models"):
            return await self._list_models(req, writer)

        match = self._match_group(req, pairs)
        if match is None:
            failure = await self._attempt(req.method, req.target, req.headers, req.body, writer)
            if failure is not None:
                await self._send(
                    writer, failure.response or _error_response(502, f"router unreachable: {failure.reason}")
                )
            return
        group, build = match
        if req.method == "GET" and path == "/slots":
            return await self._merged_slots(group, req, build, writer)
        await self._route(group, req, build, writer)

    def _match_group(
        self, req: Request, pairs: list[tuple[str, str]]
    ) -> tuple[Group, Callable[[str], tuple[str, bytes]]] | None:
        """Find the group a request names. Return it with a builder: member -> (target, body)."""
        path = req.path
        if path in ("/models", "/v1/models") or path.startswith(("/models/", "/v1/models/")):
            return None  # router control paths name one model: never rewrite them
        parsed: Any = None
        if req.body and req.method in _BODY_METHODS:
            with contextlib.suppress(ValueError):
                parsed = json.loads(req.body)
        if isinstance(parsed, dict) and isinstance(parsed.get("model"), str) and parsed["model"] in self.groups:
            doc = parsed

            def build_body(member: str) -> tuple[str, bytes]:
                return req.target, json.dumps({**doc, "model": member}, ensure_ascii=False).encode()

            return self.groups[doc["model"]], build_body
        named = next((v for k, v in pairs if k == "model"), None)
        if named in self.groups:

            def build_query(member: str) -> tuple[str, bytes]:
                return _with_query(path, [(k, member if k == "model" else v) for k, v in pairs]), req.body

            return self.groups[named], build_query
        return None

    def _no_member_response(self, group: Group) -> bytes:
        return _error_response(
            503,
            f"no member of group '{group.name}' is loaded: {group.states()}",
            group=group.name,
            members={m.name: m.state for m in group.members},
        )

    async def _route(
        self, group: Group, req: Request, build: Callable[[str], tuple[str, bytes]], writer: asyncio.StreamWriter
    ) -> None:
        tried: list[str] = []
        last: Failure | None = None
        for _attempt in range(2):
            member = group.choose(exclude=tried)
            if member is None:
                break
            tried.append(member.name)
            target, body = build(member.name)
            member.in_flight += 1
            member.requests += 1
            log.info(
                "route group=%s member=%s in_flight=%d %s %s",
                group.name,
                member.name,
                member.in_flight,
                req.method,
                req.path,
            )
            try:
                last = await self._attempt(req.method, target, req.headers, body, writer)
            finally:
                member.in_flight -= 1
            if last is None:
                return
            member.down = True
            member.failures += 1
            self.failovers += 1
            log.warning("member %s of group %s failed before any byte: %s", member.name, group.name, last.reason)
        if last is None:
            self.rejected += 1
            log.warning("group %s has no loaded member: %s", group.name, group.states())
            await self._send(writer, self._no_member_response(group))
        else:
            await self._send(
                writer, last.response or _error_response(502, f"router unreachable: {last.reason}", group=group.name)
            )

    async def _attempt(
        self, method: str, target: str, headers: list[tuple[str, str]], body: bytes, writer: asyncio.StreamWriter
    ) -> Failure | None:
        """Forward one request and relay the answer.

        Returns None when the client has the answer (good or not), or a Failure
        when nothing reached the client and the caller may try another member.
        """
        try:
            up_reader, up_writer = await self._connect()
        except (OSError, TimeoutError) as exc:
            return Failure(f"connect to router failed: {exc!r}")
        sent_to_client = False
        try:
            up_writer.write(self._upstream_request(method, target, headers, body))
            await up_writer.drain()
            head = await _read_head(up_reader, self.read_timeout)
            status = head.status
            if status >= 500 or status == 400:
                reply = await _read_body(up_reader, head, BUFFER_LIMIT, 30.0, eof_ends=True)
                response = _normalized(head, reply)
                if status >= 500 or b"not loaded" in reply.lower():
                    return Failure(f"router answered {status}", response)
                sent_to_client = True
                await self._send(writer, response)
                return None
            sent_to_client = True
            await self._send(writer, _streamed_head(head))
            remaining = 0 if _no_body(method, status) else (None if head.chunked else head.content_length)
            while remaining != 0:
                chunk = await asyncio.wait_for(
                    up_reader.read(65536 if remaining is None else min(65536, remaining)), self.read_timeout
                )
                if not chunk:
                    break
                await self._send(writer, chunk)
                if remaining is not None:
                    remaining -= len(chunk)
            return None
        except (OSError, BadMessage, TimeoutError, asyncio.IncompleteReadError) as exc:
            if sent_to_client:
                log.warning("stream broke after bytes went to the client (%s %s): %r", method, target, exc)
                writer.close()  # the client sees a truncated answer; never retry
                return None
            return Failure(f"router connection failed: {exc!r}")
        finally:
            up_writer.close()

    # ---- the paths the balancer answers or merges --------------------------------

    def _health_response(self) -> bytes:
        reasons: list[str] = []
        window = HEALTH_WINDOW_CYCLES * (self.poll_interval + self.poll_timeout)
        if self.last_poll_ok is None:
            reasons.append("the router has not answered a poll yet")
        elif time.monotonic() - self.last_poll_ok > window:
            reasons.append(f"the router has not answered a poll for more than {window:.0f} s: {self.last_poll_error}")
        for group in self.groups.values():
            if not any(m.usable for m in group.members):
                reasons.append(f"group '{group.name}' has no loaded member: {group.states()}")
        if reasons:
            return _json_response(503, {"status": "unavailable", "reasons": reasons})
        return _json_response(200, {"status": "ok"})

    def status(self) -> dict[str, Any]:
        now_mono = time.monotonic()
        return {
            "router": {
                "url": f"http://{self.router_host}:{self.router_port}",
                "last_poll_ok_unix": self.last_poll_ok_unix,
                "last_poll_ok_age_s": None if self.last_poll_ok is None else round(now_mono - self.last_poll_ok, 3),
                "last_poll_error": self.last_poll_error,
                "polls_ok": self.polls_ok,
                "polls_failed": self.polls_failed,
            },
            "groups": {
                g.name: {
                    "members": {
                        m.name: {
                            "state": m.state,
                            "down": m.down,
                            "in_flight": m.in_flight,
                            "requests": m.requests,
                            "failures": m.failures,
                            "last_poll_unix": m.last_poll_unix,
                        }
                        for m in g.members
                    }
                }
                for g in self.groups.values()
            },
            "totals": {
                "requests": self.requests,
                "routed": sum(m.requests for g in self.groups.values() for m in g.members),
                "in_flight": sum(m.in_flight for g in self.groups.values() for m in g.members),
                "failovers": self.failovers,
                "rejected": self.rejected,
                "started_unix": self.started_unix,
            },
        }

    async def _list_models(self, req: Request, writer: asyncio.StreamWriter) -> None:
        try:
            head, body = await self._fetch("GET", req.target, req.headers)
        except (OSError, BadMessage, TimeoutError) as exc:
            return await self._send(writer, _error_response(502, f"router unreachable: {exc!r}"))
        try:
            doc = json.loads(body)
            entries = doc["data"]
            assert head.status == 200 and isinstance(entries, list)
        except (ValueError, KeyError, TypeError, AssertionError):
            return await self._send(writer, _normalized(head, body))  # not a list: pass it through
        doc["data"] = self._merge_groups(entries)
        await self._send(writer, _normalized(head, json.dumps(doc).encode()))

    def _merge_groups(self, entries: list[Any]) -> list[Any]:
        by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
        merged = list(entries)
        for group in self.groups.values():
            entry = self._group_entry(group, by_id)
            index = next((i for i, e in enumerate(merged) if isinstance(e, dict) and e.get("id") == group.name), None)
            if index is None:
                merged.append(entry)
            else:
                merged[index] = entry
        return merged

    def _group_entry(self, group: Group, by_id: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for m in group.members:
            raw = by_id.get(m.name)
            status = (raw or {}).get("status") or {}
            raw_args = status.get("args")
            args: list[Any] = list(raw_args) if isinstance(raw_args, list) else []
            rows.append((m, raw, status.get("value", "unknown") if raw else "missing", args))
        loaded = [r for r in rows if r[2] == "loaded"]
        best = min((r[2] for r in rows), key=lambda s: _STATE_RANK.get(s, 9))
        source = (loaded or rows)[0]
        entry = copy.deepcopy(source[1]) if isinstance(source[1], dict) else {}
        entry.pop("inferference_residency", None)
        args = list(source[3])
        # Total lanes and total cache, so a client that divides one by the other still
        # gets the context of one lane.
        for flag in ("--parallel", "--ctx-size"):
            values = [_arg_int(r[3], flag) for r in loaded]
            if loaded and all(v is not None for v in values) and flag in args:
                args[args.index(flag) + 1] = str(sum(v for v in values if v is not None))
        entry.update(
            {"id": group.name, "object": "model", "owned_by": "inferference-balancer", "aliases": [], "tags": []}
        )
        entry["status"] = {"value": "loaded" if loaded else best, "args": args}  # no `preset`: it describes one child
        entry["members"] = [
            {"id": m.name, "state": state, "in_flight": m.in_flight, "args": member_args}
            for m, _raw, state, member_args in rows
        ]
        return entry

    async def _merged_slots(
        self,
        group: Group,
        req: Request,
        build: Callable[[str], tuple[str, bytes]],
        writer: asyncio.StreamWriter,
    ) -> None:
        """GET /slots?model=<group>: every loaded member's slots in one list.

        Each slot gets ``member`` (the member's name) and ``member_slot_id`` (the
        id the member reported). ``id`` is renumbered 0..n-1 across the list so it stays unique.
        """
        members = [m for m in group.members if m.usable]
        if not members:
            self.rejected += 1
            return await self._send(writer, self._no_member_response(group))

        async def one(m: Member) -> list[Any] | None:
            try:
                head, body = await self._fetch("GET", build(m.name)[0], req.headers)
                data = json.loads(body)
                if head.status == 200 and isinstance(data, list):
                    return data
                log.warning("slots of %s: router answered %s", m.name, head.first)
            except (OSError, BadMessage, TimeoutError, ValueError) as exc:
                log.warning("slots of %s: %r", m.name, exc)
            return None

        results = await asyncio.gather(*(one(m) for m in members))
        if all(r is None for r in results):
            return await self._send(
                writer, _error_response(502, f"no member of group '{group.name}' returned its slots")
            )
        slots: list[Any] = []
        for m, result in zip(members, results, strict=True):
            for slot in result or []:
                if isinstance(slot, dict):
                    slot = {**slot, "member": m.name, "member_slot_id": slot.get("id")}
                    slot["id"] = len(slots)
                slots.append(slot)
        await self._send(writer, _json_response(200, slots))


# ----------------------------------------------------------------------- the CLI


def _fail(message: str) -> int:
    print(f"!! [balancer] {message}", file=sys.stderr)
    return EX_CONFIG


def build(config_dir: Path) -> tuple[Balancer, str, int]:
    """Read placement.yaml. Return the balancer and its listen address."""
    placement = read_placement(config_dir)
    if placement.balancer is None:
        raise ConfigError(f"{config_dir / 'placement.yaml'} has no `balancer:` key. Declare one to run a balancer.")
    cfg = placement.balancer
    return Balancer(cfg.groups, placement.router.host, placement.router.port), cfg.host, cfg.port


async def _run(balancer: Balancer, host: str, port: int) -> None:
    server = await balancer.serve(host, port)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    poller = asyncio.ensure_future(balancer.poll_loop())
    log.info(
        "listening on %s:%d, router at %s:%d, groups: %s",
        host,
        port,
        balancer.router_host,
        balancer.router_port,
        {g.name: [m.name for m in g.members] for g in balancer.groups.values()},
    )
    await stop.wait()
    log.info("stop requested")
    server.close()
    poller.cancel()
    await balancer.close_connections()
    await asyncio.gather(poller, return_exceptions=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m inferference.balancer", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config-dir", help="where placement.yaml lives (default: $INFERFERENCE_CONFIG_DIR)")
    ap.add_argument("--check", action="store_true", help="validate the config, print the plan, and exit")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s", stream=sys.stderr)

    raw = args.config_dir or os.environ.get("INFERFERENCE_CONFIG_DIR")
    if not raw:
        return _fail("no config dir: pass --config-dir or set $INFERFERENCE_CONFIG_DIR.")
    config_dir = Path(raw).expanduser()
    try:
        balancer, host, port = build(config_dir)
    except (ConfigError, PlacementError, OSError) as exc:
        return _fail(str(exc))
    if args.check:
        groups = {g.name: [m.name for m in g.members] for g in balancer.groups.values()}
        print(
            f">> balancer listen {host}:{port}, router {balancer.router_host}:{balancer.router_port}, groups {groups}"
        )
        return 0
    try:
        asyncio.run(_run(balancer, host, port))
    except OSError as exc:
        return _fail(f"cannot listen on {host}:{port}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
