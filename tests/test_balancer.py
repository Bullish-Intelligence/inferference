"""The replica balancer: config rules, member choice, rewriting, failover, streaming.

No GPU and no router. An in-process stub stands in for the router. It serves
GET /models, GET /slots, and chat requests, and records what it receives.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from inferference import config_paths
from inferference.balancer import Balancer, Group, Member, build, main
from inferference.config_paths import ConfigError
from inferference.placement import Placement

# ------------------------------------------------------------------ config rules


def _placement(balancer: dict[str, Any] | None, router_port: int = 8101) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "devices": [{"id": "D0", "card": 0}, {"id": "D1", "card": 1}],
        "resident": [{"model": "a", "device": "D0"}, {"model": "b", "device": "D1"}, {"model": "c", "device": "D1"}],
        "router": {"host": "127.0.0.1", "port": router_port},
    }
    if balancer is not None:
        doc["balancer"] = balancer
    return doc


def _bal(groups: dict[str, list[str]], port: int = 8100, host: str = "127.0.0.1") -> dict[str, Any]:
    return {"host": host, "port": port, "groups": groups}


def test_no_balancer_key_is_valid() -> None:
    assert Placement.model_validate(_placement(None)).balancer is None


def test_a_valid_balancer_is_read() -> None:
    p = Placement.model_validate(_placement(_bal({"pub": ["a", "b"], "c": ["c"]})))
    assert p.balancer is not None
    assert p.balancer.url == "http://127.0.0.1:8100"
    assert p.balancer.groups["pub"] == ["a", "b"]


@pytest.mark.parametrize(
    ("balancer", "message"),
    [
        (_bal({}), "groups is empty"),
        (_bal({"pub": []}), "has no members"),
        (_bal({"pub": ["a", "zzz"]}), "not a resident model"),
        (_bal({"p1": ["a", "b"], "p2": ["b"]}), "member of two groups"),
        (_bal({"pub": ["a", "a"]}), "lists a member twice"),
        (_bal({"b": ["a"]}), "not a member"),
        (_bal({"bad name": ["a"]}), "alphanumeric"),
        (_bal({"pub": ["a"]}, port=8101), "equals router.port"),
        (_bal({"pub": ["a"]}, host="0.0.0.0", port=8101), "equals router.port"),
        ({"host": "127.0.0.1", "groups": {"pub": ["a"]}}, "port"),
    ],
)
def test_bad_balancer_configs_are_refused(balancer: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Placement.model_validate(_placement(balancer))


def test_a_group_may_reuse_the_name_of_its_own_member() -> None:
    p = Placement.model_validate(_placement(_bal({"a": ["a", "b"]})))
    assert p.balancer is not None


def test_the_same_port_on_another_host_is_allowed() -> None:
    p = Placement.model_validate(_placement(_bal({"pub": ["a"]}, host="10.0.0.5", port=8101)))
    assert p.balancer is not None


def test_an_unknown_balancer_key_is_refused() -> None:
    with pytest.raises(ValueError, match="extra"):
        Placement.model_validate(_placement({**_bal({"pub": ["a"]}), "typo": 1}))


def _config_dir(tmp_path: Path, balancer: dict[str, Any] | None) -> Path:
    (tmp_path / "placement.yaml").write_text(yaml.safe_dump(_placement(balancer)))
    return tmp_path


def test_resolve_router_url_returns_the_balancer_for_clients(tmp_path: Path) -> None:
    root = _config_dir(tmp_path, _bal({"pub": ["a", "b"]}))
    assert config_paths.resolve_router_url(root) == "http://127.0.0.1:8100/v1"


def test_resolve_router_url_returns_the_router_without_a_balancer(tmp_path: Path) -> None:
    root = _config_dir(tmp_path, None)
    assert config_paths.resolve_router_url(root) == "http://127.0.0.1:8101/v1"


def test_management_reads_the_router_even_with_a_balancer(tmp_path: Path) -> None:
    root = _config_dir(tmp_path, _bal({"pub": ["a", "b"]}))
    assert config_paths.read_placement(root).router.url == "http://127.0.0.1:8101"


def test_build_needs_a_balancer_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no `balancer:` key"):
        build(_config_dir(tmp_path, None))


def test_check_prints_the_plan_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _config_dir(tmp_path, _bal({"pub": ["a", "b"]}))
    assert main(["--config-dir", str(root), "--check"]) == 0
    out = capsys.readouterr().out
    assert "127.0.0.1:8100" in out and "127.0.0.1:8101" in out and "pub" in out


def test_check_refuses_a_bad_config_with_exit_78(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _config_dir(tmp_path, _bal({"pub": ["a", "nope"]}))
    assert main(["--config-dir", str(root), "--check"]) == 78
    assert "not a resident model" in capsys.readouterr().err


def test_no_config_dir_is_refused(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("INFERFERENCE_CONFIG_DIR", raising=False)
    assert main(["--check"]) == 78
    assert "no config dir" in capsys.readouterr().err


def test_the_config_dir_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INFERFERENCE_CONFIG_DIR", str(_config_dir(tmp_path, _bal({"pub": ["a"]}))))
    assert main(["--check"]) == 0


# ------------------------------------------------------------------ member choice


def _group(*states: str) -> Group:
    return Group("g", [Member(f"m{i}", state=s) for i, s in enumerate(states)])


def test_choose_takes_the_least_in_flight() -> None:
    g = _group("loaded", "loaded", "loaded")
    g.members[0].in_flight = 3
    g.members[1].in_flight = 1
    g.members[2].in_flight = 2
    chosen = g.choose()
    assert chosen is not None and chosen.name == "m1"


def test_ties_take_turns() -> None:
    g = _group("loaded", "loaded", "loaded")
    names = [m.name for m in (g.choose() for _ in range(6)) if m]
    assert names == ["m0", "m1", "m2", "m0", "m1", "m2"]


def test_choose_skips_members_that_are_not_loaded_or_are_down() -> None:
    g = _group("unloaded", "loaded", "loading")
    assert [m.name for m in (g.choose(), g.choose()) if m] == ["m1", "m1"]
    g.members[1].down = True
    assert g.choose() is None


def test_choose_honours_the_exclude_list() -> None:
    g = _group("loaded", "loaded")
    chosen = g.choose(exclude=["m0"])
    assert chosen is not None and chosen.name == "m1"
    assert g.choose(exclude=["m0", "m1"]) is None


# ------------------------------------------------------------------ the stub router


def _entry(name: str, state: str = "loaded", parallel: int = 4, ctx: int = 4096 * 4) -> dict[str, Any]:
    return {
        "id": name,
        "object": "model",
        "aliases": [],
        "tags": [],
        "status": {
            "value": state,
            "args": ["srv", "--ctx-size", str(ctx), "--parallel", str(parallel), "--device", f"dev-{name}"],
            "preset": f"[{name}]\nparallel = {parallel}\n",
        },
        "inferference_residency": {"output_device": f"dev-{name}"},
    }


class Stub:
    """A fake router. ``chat`` is replaced per test to script an answer."""

    def __init__(self) -> None:
        self.models: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.models_hits = 0
        self.chat: Callable[[dict[str, Any], asyncio.StreamWriter], Awaitable[None]] = self._ok
        self.server: asyncio.Server | None = None
        self.port = 0
        self.held: list[asyncio.Event] = []

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        assert self.server
        self.server.close()
        await self.server.wait_closed()

    @staticmethod
    def reply(status: int, body: bytes, ctype: str = "application/json") -> bytes:
        return (
            f"HTTP/1.1 {status} X\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        ).encode() + body

    async def _ok(self, req: dict[str, Any], w: asyncio.StreamWriter) -> None:
        w.write(self.reply(200, json.dumps({"served_by": req["json"].get("model")}).encode()))

    async def _handle(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            raw = await r.readuntil(b"\r\n\r\n")
            lines = raw.decode().split("\r\n")
            method, target, _ = lines[0].split()
            headers = {k.lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:] if ln)}
            body = await r.readexactly(int(headers.get("content-length", 0)))
            path, _, query = target.partition("?")
            try:
                parsed = json.loads(body) if body else {}
            except ValueError:
                parsed = {}
            req = {
                "method": method,
                "target": target,
                "path": path,
                "query": query,
                "headers": headers,
                "body": body,
                "json": parsed,
            }
            self.requests.append(req)
            if path in ("/models", "/v1/models"):
                self.models_hits += 1
                w.write(self.reply(200, json.dumps({"object": "list", "data": self.models}).encode()))
            elif path == "/slots":
                name = dict(p.split("=") for p in query.split("&")).get("model", "?")
                w.write(self.reply(200, json.dumps([{"id": 0, "owner": name}, {"id": 1, "owner": name}]).encode()))
            else:
                await self.chat(req, w)
            await w.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            w.close()

    def chat_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["path"].startswith(("/v1/chat", "/props", "/tokenize"))]


class Rig:
    """A stub router, a balancer in front of it, and an httpx client."""

    def __init__(self, groups: dict[str, list[str]], states: dict[str, str] | None = None) -> None:
        self.stub = Stub()
        self.groups = groups
        names = [m for ms in groups.values() for m in ms]
        self.stub.models = [_entry(n, (states or {}).get(n, "loaded")) for n in names]
        self.bal: Balancer
        self.server: asyncio.Server
        self.client: httpx.AsyncClient
        self.base = ""

    async def __aenter__(self) -> Rig:
        await self.stub.start()
        self.bal = Balancer(self.groups, "127.0.0.1", self.stub.port, poll_interval=0.05, poll_timeout=2.0)
        self.server = await self.bal.serve("127.0.0.1", 0)
        self.base = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        self.client = httpx.AsyncClient(timeout=10.0)
        assert await self.bal.poll_once()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.client.aclose()
        self.server.close()
        await self.bal.close_connections(grace=1.0)
        await self.stub.stop()


def run(coro: Awaitable[Any]) -> Any:
    async def _wrap() -> Any:
        return await asyncio.wait_for(coro, 30)  # type: ignore[arg-type]

    return asyncio.run(_wrap())


CHAT = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}


def test_a_group_name_in_the_body_is_rewritten_to_a_member() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
            assert r.status_code == 200
            served = r.json()["served_by"]
            assert served in ("a", "b")
            seen = rig.stub.chat_requests()[0]["json"]
            assert seen["model"] == served and seen["messages"] == CHAT["messages"]

    run(scenario())


def test_requests_alternate_between_idle_members() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            names = []
            for _ in range(4):
                r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
                names.append(r.json()["served_by"])
            assert names == ["a", "b", "a", "b"]

    run(scenario())


def test_the_query_model_of_a_get_is_rewritten() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            r = await rig.client.get(f"{rig.base}/props", params={"model": "pub", "x": "1"})
            assert r.status_code == 200
            req = rig.stub.chat_requests()[0]
            assert req["query"] in ("model=a&x=1", "model=b&x=1")

    run(scenario())


def test_other_names_and_paths_pass_through_unchanged() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            body = {"model": "other", **CHAT}
            r = await rig.client.post(f"{rig.base}/v1/chat/completions?x=1", json=body)
            assert r.json() == {"served_by": "other"}
            req = rig.stub.chat_requests()[0]
            assert req["target"] == "/v1/chat/completions?x=1" and req["json"] == body
            # a request for the model path of a member is a plain pass-through when no group has that name
            r = await rig.client.post(f"{rig.base}/tokenize", content=b"not json")
            assert rig.stub.chat_requests()[1]["body"] == b"not json"

    run(scenario())


def test_control_paths_are_never_rewritten() -> None:
    async def scenario() -> None:
        async with Rig({"a": ["a", "b"]}) as rig:
            await rig.client.post(f"{rig.base}/models/unload", json={"model": "a"})
            sent = [r for r in rig.stub.requests if r["path"] == "/models/unload"]
            assert sent[0]["json"] == {"model": "a"}

    run(scenario())


def test_a_group_may_reuse_a_member_name() -> None:
    async def scenario() -> None:
        async with Rig({"a": ["a", "b"]}) as rig:
            names = {
                (await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "a", **CHAT})).json()[
                    "served_by"
                ]
                for _ in range(2)
            }
            assert names == {"a", "b"}

    run(scenario())


def test_not_loaded_members_are_skipped_and_none_loaded_is_503() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}, states={"a": "unloaded"}) as rig:
            for _ in range(3):
                r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
                assert r.json()["served_by"] == "b"
            rig.stub.models = [_entry("a", "unloaded"), _entry("b", "unloaded")]
            await rig.bal.poll_once()
            r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
            assert r.status_code == 503
            msg = r.json()["error"]["message"]
            assert "pub" in msg and "a=unloaded" in msg and "b=unloaded" in msg

    run(scenario())


def test_least_in_flight_wins_under_load() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            release = asyncio.Event()

            async def slow(req: dict[str, Any], w: asyncio.StreamWriter) -> None:
                if req["json"]["model"] == "a":
                    await release.wait()
                w.write(Stub.reply(200, json.dumps({"served_by": req["json"]["model"]}).encode()))

            rig.stub.chat = slow
            first = asyncio.ensure_future(
                rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
            )
            await asyncio.sleep(0.2)  # the first request holds member a
            assert rig.bal.groups["pub"].members[0].in_flight == 1
            others = [
                (await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})).json()[
                    "served_by"
                ]
                for _ in range(3)
            ]
            assert others == ["b", "b", "b"]
            release.set()
            assert (await first).json()["served_by"] == "a"

    run(scenario())


def test_autoload_is_stripped_from_every_request() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            await rig.client.post(f"{rig.base}/v1/chat/completions?autoload=true&k=v", json={"model": "pub", **CHAT})
            await rig.client.post(f"{rig.base}/v1/chat/completions?autoload=true", json={"model": "other", **CHAT})
            await rig.client.get(f"{rig.base}/props", params={"model": "pub", "autoload": "true"})
            for req in rig.stub.chat_requests():
                assert "autoload" not in req["target"]
            assert rig.stub.chat_requests()[0]["query"] == "k=v"

    run(scenario())


# ------------------------------------------------------------------ failover


def test_a_5xx_before_any_byte_fails_over_to_the_next_member() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:

            async def chat(req: dict[str, Any], w: asyncio.StreamWriter) -> None:
                if req["json"]["model"] == "a":
                    w.write(Stub.reply(500, b'{"error":"boom"}'))
                else:
                    w.write(Stub.reply(200, b'{"served_by":"b"}'))

            rig.stub.chat = chat
            r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
            assert r.status_code == 200 and r.json()["served_by"] == "b"
            members = rig.bal.groups["pub"].members
            assert members[0].down and members[0].failures == 1
            # a is skipped until the next poll
            r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
            assert [x["json"]["model"] for x in rig.stub.chat_requests()] == ["a", "b", "b"]
            await rig.bal.poll_once()
            assert not members[0].down

    run(scenario())


def test_a_400_model_not_loaded_fails_over() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:

            async def chat(req: dict[str, Any], w: asyncio.StreamWriter) -> None:
                if req["json"]["model"] == "a":
                    w.write(Stub.reply(400, b'{"error":{"message":"model is not loaded"}}'))
                else:
                    w.write(Stub.reply(200, b'{"served_by":"b"}'))

            rig.stub.chat = chat
            r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
            assert r.status_code == 200 and r.json()["served_by"] == "b"

    run(scenario())


def test_an_ordinary_400_goes_to_the_client_without_a_retry() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:

            async def chat(req: dict[str, Any], w: asyncio.StreamWriter) -> None:
                w.write(Stub.reply(400, b'{"error":{"message":"prompt too long"}}'))

            rig.stub.chat = chat
            r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
            assert r.status_code == 400 and "prompt too long" in r.text
            assert len(rig.stub.chat_requests()) == 1

    run(scenario())


def test_a_failed_connection_fails_over_once(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            real = rig.bal._connect
            calls = {"n": 0}

            async def flaky() -> Any:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise ConnectionRefusedError("refused")
                return await real()

            monkeypatch.setattr(rig.bal, "_connect", flaky)
            r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
            assert r.status_code == 200 and r.json()["served_by"] == "b"
            assert rig.bal.groups["pub"].members[0].down

    run(scenario())


def test_two_failures_give_the_last_answer_and_never_a_third_try() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b", "c"]}) as rig:

            async def chat(req: dict[str, Any], w: asyncio.StreamWriter) -> None:
                w.write(Stub.reply(503, b'{"error":"busy"}'))

            rig.stub.chat = chat
            r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
            assert r.status_code == 503 and "busy" in r.text
            assert len(rig.stub.chat_requests()) == 2

    run(scenario())


def test_the_router_down_gives_502() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            await rig.stub.stop()
            rig.stub.server = None
            r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
            assert r.status_code == 502
            r = await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "other", **CHAT})
            assert r.status_code == 502
            rig.stub.server = await asyncio.start_server(lambda *_: None, "127.0.0.1", 0)  # for __aexit__

    run(scenario())


def test_no_retry_after_bytes_reached_the_client() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:

            async def chat(req: dict[str, Any], w: asyncio.StreamWriter) -> None:
                w.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                w.write(b"7\r\ndata: 1\r\n")  # a chunk cut short: the stub then closes
                await w.drain()
                w.close()

            rig.stub.chat = chat
            with pytest.raises(httpx.HTTPError):
                async with rig.client.stream(
                    "POST", f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT}
                ) as r:
                    assert r.status_code == 200
                    async for _ in r.aiter_raw():
                        pass
            assert len(rig.stub.chat_requests()) == 1

    run(scenario())


# ------------------------------------------------------------------ streaming


def test_sse_passes_through_unbuffered() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            first_seen = asyncio.Event()

            async def chat(req: dict[str, Any], w: asyncio.StreamWriter) -> None:
                w.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                for i in range(3):
                    data = f"data: {i}\n\n".encode()
                    w.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                    await w.drain()
                    # The stub sends the next chunk only after the client has read this one.
                    await asyncio.wait_for(first_seen.wait(), 5)
                    first_seen.clear()
                w.write(b"0\r\n\r\n")

            rig.stub.chat = chat
            got: list[str] = []
            async with rig.client.stream("POST", f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT}) as r:
                assert r.status_code == 200
                async for text in r.aiter_text():
                    got.append(text)
                    first_seen.set()
            assert "".join(got) == "data: 0\n\ndata: 1\n\ndata: 2\n\n"
            assert len(got) == 3  # each chunk arrived on its own, before the next was sent

    run(scenario())


def test_a_large_body_passes() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            big = "x" * (3 * 1024 * 1024)
            r = await rig.client.post(
                f"{rig.base}/v1/chat/completions", json={"model": "pub", "messages": [{"role": "user", "content": big}]}
            )
            assert r.status_code == 200
            assert len(rig.stub.chat_requests()[0]["json"]["messages"][0]["content"]) == len(big)

    run(scenario())


def test_client_disconnect_closes_the_upstream_request() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            upstream_closed = asyncio.Event()

            async def chat(req: dict[str, Any], w: asyncio.StreamWriter) -> None:
                w.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                await w.drain()
                try:
                    for _ in range(200):
                        w.write(b"2\r\n.\n\r\n")
                        await w.drain()
                        await asyncio.sleep(0.05)
                except (ConnectionError, OSError):
                    upstream_closed.set()

            rig.stub.chat = chat
            host, port = rig.base.removeprefix("http://").split(":")
            reader, writer = await asyncio.open_connection(host, int(port))
            body = json.dumps({"model": "pub", **CHAT}).encode()
            writer.write(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            await reader.read(64)
            writer.close()
            await asyncio.wait_for(upstream_closed.wait(), 5)
            for _ in range(50):
                if all(m.in_flight == 0 for m in rig.bal.groups["pub"].members):
                    break
                await asyncio.sleep(0.05)
            assert all(m.in_flight == 0 for m in rig.bal.groups["pub"].members)

    run(scenario())


# ------------------------------------------------------------------ health, status, lists


def test_health_follows_the_members() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            assert (await rig.client.get(f"{rig.base}/health")).status_code == 200
            rig.stub.models = [_entry("a", "unloaded"), _entry("b", "loaded")]
            await rig.bal.poll_once()
            assert (await rig.client.get(f"{rig.base}/health")).status_code == 200
            rig.stub.models = [_entry("a", "unloaded"), _entry("b", "unloaded")]
            await rig.bal.poll_once()
            r = await rig.client.get(f"{rig.base}/health")
            assert r.status_code == 503 and "no loaded member" in r.text

    run(scenario())


def test_health_is_503_before_the_first_poll_and_when_the_router_is_silent() -> None:
    bal = Balancer({"pub": ["a"]}, "127.0.0.1", 1, poll_interval=0.01, poll_timeout=0.01)
    assert b"503" in bal._health_response().split(b"\r\n")[0]
    bal.groups["pub"].members[0].state = "loaded"
    bal.last_poll_ok = 0.0  # a very old poll
    assert b"503" in bal._health_response().split(b"\r\n")[0]


def test_a_failed_poll_keeps_the_last_state() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            await rig.stub.stop()
            rig.stub.server = None
            assert not await rig.bal.poll_once()
            assert [m.state for m in rig.bal.groups["pub"].members] == ["loaded", "loaded"]
            assert rig.bal.polls_failed == 1 and rig.bal.last_poll_error
            rig.stub.server = await asyncio.start_server(lambda *_: None, "127.0.0.1", 0)

    run(scenario())


def test_status_reports_members_and_totals() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            for _ in range(4):
                await rig.client.post(f"{rig.base}/v1/chat/completions", json={"model": "pub", **CHAT})
            s = (await rig.client.get(f"{rig.base}/balancer/status")).json()
            members = s["groups"]["pub"]["members"]
            assert members["a"]["requests"] == 2 and members["b"]["requests"] == 2
            assert members["a"]["in_flight"] == 0 and members["a"]["state"] == "loaded"
            assert members["a"]["last_poll_unix"] is not None
            assert s["totals"]["routed"] == 4 and s["router"]["polls_ok"] >= 1

    run(scenario())


def test_models_lists_the_router_entries_and_one_entry_per_group() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            for path in ("/models", "/v1/models"):
                data = (await rig.client.get(f"{rig.base}{path}")).json()["data"]
                ids = [e["id"] for e in data]
                assert ids == ["a", "b", "pub"]
                group = data[2]
                assert group["status"]["value"] == "loaded"
                assert [m["id"] for m in group["members"]] == ["a", "b"]
                assert [m["state"] for m in group["members"]] == ["loaded", "loaded"]
                assert "inferference_residency" not in group
                assert "preset" not in group["status"]
                # total lanes and total cache of the loaded members
                args = group["status"]["args"]
                assert args[args.index("--parallel") + 1] == "8"
                assert args[args.index("--ctx-size") + 1] == str(2 * 4096 * 4)
                # the member entries stay as the router sent them
                assert data[0]["inferference_residency"]

    run(scenario())


def test_the_group_entry_replaces_a_member_of_the_same_name() -> None:
    async def scenario() -> None:
        async with Rig({"a": ["a", "b"]}, states={"b": "unloaded"}) as rig:
            data = (await rig.client.get(f"{rig.base}/v1/models")).json()["data"]
            assert [e["id"] for e in data] == ["a", "b"]
            assert "members" in data[0] and data[0]["status"]["value"] == "loaded"
            args = data[0]["status"]["args"]
            assert args[args.index("--parallel") + 1] == "4"  # only the loaded member counts
            assert [m["state"] for m in data[0]["members"]] == ["loaded", "unloaded"]

    run(scenario())


def test_a_group_with_nothing_loaded_reports_the_best_state() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}, states={"a": "unloaded", "b": "loading"}) as rig:
            data = (await rig.client.get(f"{rig.base}/models")).json()["data"]
            assert data[-1]["status"]["value"] == "loading"

    run(scenario())


def test_models_reload_query_passes_through() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            await rig.client.get(f"{rig.base}/models", params={"reload": "1", "autoload": "true"})
            last = [r for r in rig.stub.requests if r["path"] == "/models"][-1]
            assert last["query"] == "reload=1"

    run(scenario())


def test_slots_of_a_group_are_merged() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}) as rig:
            slots = (await rig.client.get(f"{rig.base}/slots", params={"model": "pub"})).json()
            assert [s["id"] for s in slots] == [0, 1, 2, 3]
            assert [s["member"] for s in slots] == ["a", "a", "b", "b"]
            assert [s["member_slot_id"] for s in slots] == [0, 1, 0, 1]
            assert [s["owner"] for s in slots] == ["a", "a", "b", "b"]
            # a slot request for a name that is not a group passes through
            other = (await rig.client.get(f"{rig.base}/slots", params={"model": "a-other"})).json()
            assert other[0] == {"id": 0, "owner": "a-other"}

    run(scenario())


def test_props_tokenize_detokenize_and_template_go_to_a_loaded_member() -> None:
    async def scenario() -> None:
        async with Rig({"pub": ["a", "b"]}, states={"a": "unloaded"}) as rig:
            await rig.client.get(f"{rig.base}/props", params={"model": "pub"})
            for path in ("/tokenize", "/detokenize", "/apply-template"):
                await rig.client.post(f"{rig.base}{path}", json={"model": "pub", "content": "x"})
            sent = [
                r for r in rig.stub.requests if r["path"] in ("/props", "/tokenize", "/detokenize", "/apply-template")
            ]
            assert len(sent) == 4
            assert all(("model=b" in r["query"]) or r["json"].get("model") == "b" for r in sent)

    run(scenario())
