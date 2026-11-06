"""Cross-chain pull-in: splice another node's context into a base chain.

``node_delta_live.py`` reconstructs a single ancestor chain with zero prefill.
This module handles the harder case the demo needs: while answering at node S,
pull in a *different* chain -- the definition a symbol references, a type, a
find-refs target -- selected by the router/model2vec layer.

Why this cannot be a plain ``restore_seq_state`` of the pull-in's blob:

  * **RoPE positions are baked in.**  The pull-in's K vectors were rotated for
    the positions it occupied on its own chain.  Loaded after a different-length
    base chain they land at the wrong positions and every attention score is
    wrong.  ``NodeDelta.base_position`` records where each span was captured, so
    the required position shift is exactly ``new_base - delta.base_position``.
  * **No cross-attention to the base.**  The pull-in never attended to S's
    ancestors, so even re-anchored its hidden states are stale w.r.t. the base.

Two strategies, both keyed off ``base_position``:

  * :func:`blend_by_redecode` (implemented, robust) -- reuse only the pull-in's
    *token IDs* and re-decode them at their assigned positions on top of the
    restored base seq.  This is an honest *partial* prefill (|pull-in| tokens,
    not the whole context) and is automatically correct: the re-decoded tokens
    attend to the base, RoPE is applied at the true positions, no cache-layout
    surgery.  ``token_budget`` caps how many pull-in tokens are admitted.

  * :func:`blend_by_reanchor` (stub, fast) -- the CacheBlend-style path: reuse
    the pull-in's cached K/V, rotate K by the position delta in place, and
    recompute attention for only a ``heal_fraction`` of tokens.  This needs
    direct KV-buffer access llama.cpp does not expose through the stable API, so
    it raises ``NotImplementedError`` with the exact hook documented.

Both leave the base seq positioned so the caller decodes the user's prompt next
via the normal decode loop.  Failure is data where the tree is at fault.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .node_delta import (
    ChainDecision,
    ChainRejectionReason,
    NodeDelta,
    NodeDeltaIndex,
    resolve_chain,
)
from .prefix_cache_live import (
    LiveRestoreResult,
    LlamaSeqStateBridge,
    PrefixStateCache,
    restore_and_continue,
)


@dataclass(frozen=True, slots=True)
class PullInSpec:
    """One cross-chain reference to admit into the base context.

    ``node_id`` is the pull-in's tree node.  Its effective tokens are its own
    ``span_token_ids`` by default (the referenced symbol/def body); pass
    ``use_cumulative_chain=True`` to admit the pull-in's whole ancestor chain
    (rarely needed -- usually the base already shares those ancestors).
    """

    node_id: str
    use_cumulative_chain: bool = False


@dataclass(frozen=True, slots=True)
class BlendResult:
    """Outcome of assembling a base chain plus zero or more pull-ins.

    ``base_restore`` is the live restore of the base chain; ``admitted`` lists
    the pull-in node ids actually decoded (in order); ``dropped`` maps a pull-in
    node id to why it was skipped (tree miss, budget exhausted); ``end_position``
    is where the caller should decode the prompt from.
    """

    base_restore: LiveRestoreResult
    admitted: tuple[str, ...]
    dropped: tuple[tuple[str, str], ...]
    end_position: int
    prompt_next_token: int | None = None

    @property
    def ok(self) -> bool:
        return self.base_restore.restored


def _pull_in_tokens(index: NodeDeltaIndex, spec: PullInSpec) -> tuple[NodeDelta, tuple[int, ...]] | None:
    delta = index.get(spec.node_id)
    if delta is None:
        return None
    if spec.use_cumulative_chain:
        return delta, delta.key.prefix_token_ids
    return delta, delta.span_token_ids


def blend_by_redecode(
    bridge: LlamaSeqStateBridge,
    ctx: Any,
    *,
    cache: PrefixStateCache,
    index: NodeDeltaIndex,
    base_node_id: str,
    pull_ins: Sequence[PullInSpec],
    prompt_token_ids: Sequence[int],
    n_seq_max: int,
    token_budget: int | None = None,
    seq_id: int = 0,
) -> BlendResult | ChainDecision:
    """Restore the base chain, re-decode admitted pull-ins after it, then prompt.

    Correct-by-construction reduced prefill: only the base chain is free (its
    whole-seq blob is restored); each admitted pull-in and the prompt are
    decoded live, at contiguous positions right after the base.  Because the
    pull-ins are decoded *with the base resident*, they attend to it and are
    RoPE-rotated at their true positions -- no re-anchoring required.

    ``token_budget`` (if set) caps the total pull-in tokens admitted; pull-ins
    are considered in the given order and one that would overflow the budget is
    dropped whole (never truncated mid-symbol).  Tree-level failures on the base
    return a ``ChainDecision``; a missing pull-in is recorded in ``dropped`` and
    skipped, never fatal.
    """
    prompt = tuple(prompt_token_ids)
    if not prompt:
        raise ValueError("prompt_token_ids must be non-empty")

    base_chain = resolve_chain(index, base_node_id)
    if not base_chain.can_restore:
        return base_chain
    assert base_chain.plan is not None
    base = base_chain.plan.target

    # 1) Restore the base chain and decode the FIRST pull-in token (or, if no
    #    pull-in survives budgeting, the prompt) as the mandatory suffix that
    #    refreshes logits.  We compute the admitted set first so we know what to
    #    hand restore_and_continue as the suffix.
    admitted: list[tuple[NodeDelta, tuple[int, ...]]] = []
    dropped: list[tuple[str, str]] = []
    spent = 0
    for spec in pull_ins:
        found = _pull_in_tokens(index, spec)
        if found is None:
            dropped.append((spec.node_id, str(ChainRejectionReason.NODE_NOT_FOUND)))
            continue
        delta, tokens = found
        if not tokens:
            dropped.append((spec.node_id, "empty_span"))
            continue
        if token_budget is not None and spent + len(tokens) > token_budget:
            dropped.append((spec.node_id, "token_budget_exhausted"))
            continue
        spent += len(tokens)
        admitted.append((delta, tokens))

    # Everything decoded after the base, in order: pull-in spans then the prompt.
    tail: list[int] = []
    for _, tokens in admitted:
        tail.extend(tokens)
    tail.extend(prompt)

    base_pos = base.cumulative_token_count
    request = base.key.prefix_token_ids + tuple(tail)

    result = restore_and_continue(
        bridge,
        ctx,
        cache=cache,
        key=base.key,
        request_token_ids=request,
        n_seq_max=n_seq_max,
        seq_id=seq_id,
    )

    end_position = base_pos + len(tail)
    return BlendResult(
        base_restore=result,
        admitted=tuple(d.node_id for d, _ in admitted),
        dropped=tuple(dropped),
        end_position=end_position if result.restored else base_pos,
        prompt_next_token=result.next_token if result.restored else None,
    )


def blend_by_reanchor(
    bridge: LlamaSeqStateBridge,
    ctx: Any,
    *,
    pull_in: NodeDelta,
    new_base_position: int,
    heal_fraction: float,
    seq_id: int = 0,
) -> LiveRestoreResult:
    """CacheBlend-style fast path: reuse cached KV, rotate RoPE, heal a fraction.

    Reuses the pull-in's cached K/V instead of re-decoding: apply the position
    delta ``new_base_position - pull_in.base_position`` as a RoPE rotation to the
    cached K (V is position-free), then recompute attention for only a
    ``heal_fraction`` of the pull-in's tokens to restore cross-attention to the
    base.  This is the only path that beats a partial prefill on wall-clock, but
    it requires writing into llama.cpp's per-cell KV buffers, which the stable
    C API (``llama_state_seq_*``, ``llama_memory_*``) does not expose -- the
    blob is an opaque, layout-versioned dump.

    Landing this needs one of: (a) a custom ``ggml`` op over the cache tensors
    behind a native shim, or (b) an upstream ``llama_kv_self_rope_shift``-style
    entry point.  Until then the robust :func:`blend_by_redecode` is the
    supported primitive.  The position delta and heal budget are validated here
    so a caller can wire the dial before the kernel exists.
    """
    if not 0.0 <= heal_fraction <= 1.0:
        raise ValueError("heal_fraction must be in [0, 1]")
    position_delta = new_base_position - pull_in.base_position
    raise NotImplementedError(
        "blend_by_reanchor needs direct KV-buffer RoPE shift + selective recompute "
        f"(position_delta={position_delta}, heal_fraction={heal_fraction}); "
        "use blend_by_redecode until the native kernel lands"
    )
