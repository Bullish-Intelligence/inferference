"""An async client that negotiates the serving plane's contract.

Scope is deliberately narrow: this answers *what can this server do*, not
*generate this text*. Generation keeps going over the OpenAI-compatible
transport ``llama-server`` already serves, which a consumer's agent framework
already speaks. Owning a generation path here would mean owning chat
templating, tool-call parsing, streaming and usage accounting — surface the
server does for free today.

Depends on pydantic and httpx only. See ``capabilities.py`` for why.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

import httpx

from .capabilities import AdapterInfo, Capabilities, CapabilityError, ConstraintKind, ModelInfo

# The constraint dialects `llama-server` enforces. This is a DECLARED fact, not
# a probe: no endpoint reports it, and this library builds the server's argv
# (`ci/runner/inferference-config.py`), so this library is the honest place to
# state it. Pass `constraints=` to narrow it for a server built differently.
LLAMA_SERVER_CONSTRAINTS = frozenset({ConstraintKind.SCHEMA, ConstraintKind.CHOICE, ConstraintKind.GRAMMAR_GBNF})

_DEFAULT_TIMEOUT = 10.0


def _server_root(base_url: str) -> str:
    """Strip an OpenAI ``/v1`` suffix; ``/props`` and friends sit at the root."""
    trimmed = base_url.rstrip("/")
    return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed


def _arg_value(args: list[Any], flag: str) -> str | None:
    """Return the value of ``--flag value`` or ``--flag=value`` from an argv."""
    prefix = f"{flag}="
    for index, token in enumerate(args):
        if token == flag and index + 1 < len(args):
            return str(args[index + 1])
        if isinstance(token, str) and token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _positive_int(text: str | None) -> int | None:
    """Parse a positive integer, or ``None``. Absent or bad removes a capability."""
    if text is None:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _model_info(entry: dict[str, Any]) -> ModelInfo:
    """Read one ``/v1/models`` entry, preferring its reported launch argv."""
    status = entry.get("status")
    status = status if isinstance(status, dict) else {}
    args = status.get("args")
    args = args if isinstance(args, list) else []
    return ModelInfo(
        name=entry["id"],
        n_ctx=_positive_int(_arg_value(args, "--ctx-size")),
        n_parallel=_positive_int(_arg_value(args, "--parallel")),
        loaded=status.get("value") == "loaded",
    )


class InferferenceClient:
    """Ask one server what it can do, and cache the answer."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        constraints: frozenset[ConstraintKind] = LLAMA_SERVER_CONSTRAINTS,
        server: str = "llama-server-router",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.root_url = _server_root(base_url)
        self.constraints = constraints
        self.server = server
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._capabilities: Capabilities | None = None

    # ---- probes --------------------------------------------------------------

    async def _get(self, url: str) -> Any | None:
        """Return decoded JSON, or ``None`` when the server does not answer.

        A missing optional endpoint removes a capability. It never adds one and
        it never raises, so an older server degrades to a smaller contract
        instead of failing the caller.
        """
        try:
            response = await self._client.get(url, headers=self._headers)
        except httpx.HTTPError:
            return None
        if response.status_code != httpx.codes.OK:
            return None
        try:
            return response.json()
        except ValueError:
            return None

    async def _probe_model_entries(self) -> list[dict[str, Any]]:
        """Read the resident entries. A server that will not name them is unusable.

        The entry's ``status.args`` is the argv the router launched the child
        with. It is a report of what is running, not a guess; the contract
        reads its per-model facts from there.
        """
        payload = await self._get(f"{self.base_url}/models")
        if not isinstance(payload, dict):
            raise CapabilityError(
                f"server at {self.base_url} did not answer GET /models; "
                "the resident set is declared, so a server that cannot name it cannot be routed to"
            )
        entries = payload.get("data")
        if not isinstance(entries, list):
            raise CapabilityError(f"server at {self.base_url} returned no `data` list from GET /models")
        return [entry for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), str)]

    async def _probe_n_ctx(self, model_count: int) -> int | None:
        """Read the context length from ``/props``, and only for one model.

        The fallback for a bare ``llama-server`` whose argv the caller cannot
        see. ``/props`` reports one set of generation settings; with several
        children resident it cannot say which one it describes, so the value is
        dropped rather than spread across models it may not describe.
        """
        if model_count != 1:
            return None
        payload = await self._get(f"{self.root_url}/props")
        if not isinstance(payload, dict):
            return None
        settings = payload.get("default_generation_settings")
        if not isinstance(settings, dict):
            return None
        n_ctx = settings.get("n_ctx")
        return n_ctx if isinstance(n_ctx, int) and n_ctx > 0 else None

    async def _probe_adapters(self) -> tuple[AdapterInfo, ...]:
        """Read the loaded LoRA pool. Absent endpoint means no adapters."""
        payload = await self._get(f"{self.root_url}/lora-adapters")
        if not isinstance(payload, list):
            return ()
        adapters: list[AdapterInfo] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str):
                path = entry.get("path")
                if not isinstance(path, str):
                    continue
                name = PurePosixPath(path).stem
            lora_id = entry.get("id")
            scale = entry.get("scale")
            adapters.append(
                AdapterInfo(
                    name=name,
                    lora_id=lora_id if isinstance(lora_id, int) and lora_id >= 0 else None,
                    scale=float(scale) if isinstance(scale, int | float) and scale > 0 else 1.0,
                )
            )
        return tuple(adapters)

    # ---- the contract --------------------------------------------------------

    async def capabilities(self, *, refresh: bool = False) -> Capabilities:
        """Negotiate and cache what this server offers."""
        if self._capabilities is not None and not refresh:
            return self._capabilities

        entries = await self._probe_model_entries()
        models = tuple(_model_info(entry) for entry in entries)
        if len(models) == 1 and models[0].n_ctx is None:
            n_ctx = await self._probe_n_ctx(1)
            if n_ctx is not None:
                models = (models[0].model_copy(update={"n_ctx": n_ctx}),)
        adapters = await self._probe_adapters()

        self._capabilities = Capabilities(
            server=self.server,
            models=models,
            adapters=adapters,
            constraints=self.constraints,
        )
        return self._capabilities

    async def aclose(self) -> None:
        """Close the transport, unless the caller supplied and owns it."""
        if self._owns_client:
            await self._client.aclose()
