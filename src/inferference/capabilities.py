"""The upward contract: what one running server can actually do.

A consumer must not guess at the serving plane's abilities. It asks. Everything here is plain pydantic over
plain data, so the contract imports without numpy, ``llama_cpp`` or xgrammar:
a consumer installs it without the ABI-anchor pair.

The rule throughout is the repo's standing one. **Fail closed.** An ability
this module cannot prove is absent, and asking for an absent ability is a loud
refusal, never a silent fallback.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CapabilityError(RuntimeError):
    """A server cannot do what the caller asked of it."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConstraintKind(StrEnum):
    """A constrained-decoding dialect a server enforces during generation.

    ``grammar_gbnf`` is named for its dialect on purpose. GBNF is close to, but
    not the same as, the EBNF xgrammar accepts, and a constraint that cannot
    name its own dialect cannot be validated against the engine that runs it.
    """

    SCHEMA = "schema"
    CHOICE = "choice"
    GRAMMAR_GBNF = "grammar_gbnf"


class AdapterInfo(_FrozenModel):
    """One LoRA adapter the server reports as loaded."""

    name: str
    lora_id: int | None = Field(default=None, ge=0)
    scale: float = Field(default=1.0, gt=0)


class ModelInfo(_FrozenModel):
    """One resident model, named as the OpenAI ``model`` field must name it.

    ``n_ctx`` and ``n_parallel`` come from the child's launch argv, which the
    router reports per model. ``loaded`` is the router's own warm/cold state; an
    unloaded model is still addressable, because the router swaps it in.
    """

    name: str
    n_ctx: int | None = Field(default=None, gt=0)
    n_parallel: int | None = Field(default=None, gt=0)
    loaded: bool = False


class Capabilities(_FrozenModel):
    """What a server offers, at one moment, as far as it could be established.

    ``models`` are the names valid in the OpenAI ``model`` field. ``adapters``
    are the LoRA adapters the server reports; they are a separate axis, and
    being loaded does not make one addressable — see
    :meth:`resolve_model_field`.
    """

    server: str
    models: tuple[ModelInfo, ...] = ()
    adapters: tuple[AdapterInfo, ...] = ()
    constraints: frozenset[ConstraintKind] = frozenset()
    seq_routing: bool = False

    def supports(self, kind: ConstraintKind | str) -> bool:
        """Report whether the server enforces a constraint dialect.

        An unrecognised name is unsupported, not an error: a consumer may hold
        constraint kinds this serving plane has never heard of.
        """
        try:
            resolved = ConstraintKind(kind)
        except ValueError:
            return False
        return resolved in self.constraints

    def model(self, name: str) -> ModelInfo | None:
        """Return the named resident model, or ``None``."""
        return next((model for model in self.models if model.name == name), None)

    def adapter(self, name: str) -> AdapterInfo | None:
        """Return the named loaded adapter, or ``None``."""
        return next((adapter for adapter in self.adapters if adapter.name == name), None)

    def resolve_model_field(self, name: str) -> str:
        """Return the value to send as the OpenAI ``model`` field for ``name``.

        A resident model resolves to itself. An adapter that the server loaded
        but does not also publish as a model is REFUSED: nothing establishes
        that the model field selects it, and a request that silently answers
        from the base weights is worse than one that fails.
        """
        if self.model(name) is not None:
            return name
        if self.adapter(name) is not None:
            raise CapabilityError(
                f"adapter {name!r} is loaded but not addressable through the OpenAI `model` field; "
                "this server publishes it only as a LoRA adapter, so it needs the native `lora` "
                "request parameter, which this contract does not yet carry"
            )
        resident = ", ".join(model.name for model in self.models) or "<none>"
        raise CapabilityError(f"unknown model or adapter {name!r}; resident models: {resident}")
