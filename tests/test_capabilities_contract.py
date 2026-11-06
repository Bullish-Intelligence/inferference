"""The upward contract: what a running server can do, answered honestly."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from inferference.capabilities import (
    AdapterInfo,
    Capabilities,
    CapabilityError,
    ConstraintKind,
    ModelInfo,
)
from inferference.client import InferferenceClient

BASE_URL = "http://rig.invalid:8081/v1"


def _capabilities() -> Capabilities:
    return Capabilities(
        server="llama-server-router",
        models=(ModelInfo(name="base", n_ctx=16384), ModelInfo(name="ner-json")),
        adapters=(AdapterInfo(name="acrouter", lora_id=0),),
        constraints=frozenset({ConstraintKind.SCHEMA, ConstraintKind.CHOICE}),
    )


# ---- the model itself --------------------------------------------------------


def test_capabilities_answers_support_model_and_adapter_lookups() -> None:
    caps = _capabilities()

    assert caps.supports(ConstraintKind.SCHEMA)
    assert caps.supports("choice")
    assert not caps.supports(ConstraintKind.GRAMMAR_GBNF)
    assert caps.model("base") == ModelInfo(name="base", n_ctx=16384)
    assert caps.model("absent") is None
    assert caps.adapter("acrouter") == AdapterInfo(name="acrouter", lora_id=0)
    assert caps.adapter("absent") is None


def test_capabilities_is_frozen_and_refuses_unknown_keys() -> None:
    caps = _capabilities()
    with pytest.raises(ValidationError):
        caps.server = "other"  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError):
        Capabilities(server="s", routing=True)  # ty: ignore[unknown-argument]


def test_unknown_constraint_kind_is_unsupported_not_an_error() -> None:
    assert not _capabilities().supports("regex")


# ---- resolving a name onto the wire -----------------------------------------


def test_resident_model_resolves_to_itself_on_the_model_field() -> None:
    assert _capabilities().resolve_model_field("ner-json") == "ner-json"


def test_loaded_adapter_that_is_not_a_model_refuses_with_the_reason() -> None:
    """Fail closed. The adapter exists, but nothing proves it is addressable."""
    with pytest.raises(CapabilityError, match="not addressable"):
        _capabilities().resolve_model_field("acrouter")


def test_unknown_name_names_what_is_resident() -> None:
    with pytest.raises(CapabilityError, match="ner-json"):
        _capabilities().resolve_model_field("nonesuch")


# ---- probing a live server ---------------------------------------------------


Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, **kwargs: Any) -> InferferenceClient:
    return InferferenceClient(
        BASE_URL,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def _json(payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=json.dumps(payload), headers={"content-type": "application/json"})


def test_probe_reads_resident_models_props_and_adapters() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/v1/models":
            return _json({"data": [{"id": "base"}]})
        if request.url.path == "/props":
            return _json({"default_generation_settings": {"n_ctx": 16384}})
        if request.url.path == "/lora-adapters":
            return _json([{"id": 0, "path": "/m/acrouter.gguf", "scale": 1.0}])
        return _json({}, 404)

    caps = asyncio.run(_client(handler).capabilities())

    base = caps.model("base")
    assert [model.name for model in caps.models] == ["base"]
    assert base is not None and base.n_ctx == 16384
    assert caps.adapter("acrouter") == AdapterInfo(name="acrouter", lora_id=0, scale=1.0)
    assert "/v1/models" in seen and "/props" in seen and "/lora-adapters" in seen


def test_context_length_is_dropped_when_several_models_are_resident() -> None:
    """`/props` describes one set of settings. With two children it names neither."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return _json({"data": [{"id": "base"}, {"id": "ner-json"}]})
        if request.url.path == "/props":
            return _json({"default_generation_settings": {"n_ctx": 16384}})
        return _json({}, 404)

    caps = asyncio.run(_client(handler).capabilities())

    assert [model.n_ctx for model in caps.models] == [None, None]


def test_adapter_entry_prefers_a_reported_name_over_its_path_stem() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return _json({"data": [{"id": "base"}]})
        if request.url.path == "/lora-adapters":
            return _json([{"id": 1, "name": "acrouter-v2", "path": "/m/adapter-1.gguf"}])
        return _json({}, 404)

    caps = asyncio.run(_client(handler).capabilities())

    assert caps.adapter("acrouter-v2") == AdapterInfo(name="acrouter-v2", lora_id=1, scale=1.0)


def test_absent_optional_endpoint_removes_a_capability_and_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return _json({"data": [{"id": "base"}]})
        return _json({"error": "not found"}, 404)

    caps = asyncio.run(_client(handler).capabilities())

    base = caps.model("base")
    assert caps.adapters == ()
    assert base is not None and base.n_ctx is None


def test_a_server_that_cannot_list_models_is_a_loud_failure() -> None:
    """The resident set is declared. A server that will not name it is unusable."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _json({"error": "boom"}, 500)

    with pytest.raises(CapabilityError, match="models"):
        asyncio.run(_client(handler).capabilities())


def test_declared_constraints_are_carried_and_overridable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return _json({"data": [{"id": "base"}]})
        return _json({}, 404)

    default = asyncio.run(_client(handler).capabilities())
    assert default.supports(ConstraintKind.SCHEMA)
    assert default.supports(ConstraintKind.GRAMMAR_GBNF)

    narrowed = asyncio.run(_client(handler, constraints=frozenset({ConstraintKind.SCHEMA})).capabilities())
    assert not narrowed.supports(ConstraintKind.GRAMMAR_GBNF)


def test_probe_is_cached_until_refresh_is_asked_for() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v1/models":
            return _json({"data": [{"id": "base"}]})
        return _json({}, 404)

    async def probe_twice() -> None:
        client = _client(handler)
        await client.capabilities()
        await client.capabilities()
        assert calls.count("/v1/models") == 1
        await client.capabilities(refresh=True)
        assert calls.count("/v1/models") == 2

    asyncio.run(probe_twice())


# ---- the dependency boundary -------------------------------------------------


def test_contract_modules_import_without_the_native_stack() -> None:
    """The consumer installs this contract without the ABI-anchor pair."""
    probe = (
        "import sys, inferference.capabilities, inferference.client;"
        "banned = [m for m in ('numpy', 'llama_cpp', 'xgrammar') if m in sys.modules];"
        "print(banned)"
    )
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "[]"
