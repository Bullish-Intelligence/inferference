"""Live driver: build and replay a codebase context tree over llama.cpp KV state.

``node_delta.py`` is the pure tree policy/persistence layer.  This module wires
it to the proven per-seq capture/restore primitives in ``prefix_cache_live.py``:

  * :func:`capture_tree` walks the codebase tree depth-first in ONE sequence,
    decoding only each node's incremental ``span_token_ids`` (its ancestors are
    already resident in the seq), snapshotting the whole-seq KV blob at every
    node, and recording a :class:`NodeDelta`.  Each node is decoded exactly once
    across the whole tree -- that is the "precompute on startup, store deltas"
    step.  Sibling recursion rewinds the KV to the parent boundary with
    ``llama_memory_seq_rm`` so a sibling's span lands at the same base position.

  * :func:`reconstruct_chain` answers "contextual response at node S": look up
    S's whole-seq blob, restore it, and decode the user's prompt as the suffix
    -- zero prefill of the codebase context, only the prompt is decoded.  It is
    a thin composition of :func:`plan_chain` and the exact-prefix
    ``restore_and_continue`` lifecycle, so it inherits every fail-closed rule.

The cross-chain "pull-in" (go-to-definition) blend is intentionally NOT here:
that needs RoPE re-anchoring + selective recompute and belongs in a later
``node_blend_live.py``; ``base_position`` on each delta is the hook it will use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .fingerprint import LlamaEngineFingerprint
from .node_delta import (
    ChainDecision,
    NodeDelta,
    NodeDeltaIndex,
    NodeInclusion,
    plan_chain,
)
from .prefix_cache import PrefixCacheKey, RestorePlanDecision
from .prefix_cache_live import (
    LiveRestoreResult,
    LlamaSeqStateBridge,
    PrefixStateCache,
    restore_and_continue,
)


@dataclass(frozen=True, slots=True)
class TreeNodeSpec:
    """Input node for a capture pass: identity, parent, and the tokens it adds.

    ``span_token_ids`` is only this node's own raw body (already tokenized by the
    caller/LSP layer); the driver concatenates the *effective* span of each
    ancestor to form the cumulative chain key.  ``inclusion`` mirrors
    :class:`NodeInclusion`:

      * ``RAW``       -- the effective span is ``span_token_ids``.
      * ``SUMMARIZE`` -- the effective span is ``summary_token_ids`` (a shorter,
        caller-produced digest) decoded in place of the raw body, so children
        inherit the summary instead of the full text.  ``summary_token_ids``
        must be non-empty when this policy is used.
      * ``EXCLUDE``   -- the effective span is empty; the node contributes no
        tokens and gets no blob, but may still parent children.
    """

    node_id: str
    parent_node_id: str | None
    span_token_ids: tuple[int, ...]
    inclusion: NodeInclusion = NodeInclusion.RAW
    summary_token_ids: tuple[int, ...] = ()
    source_path: str | None = None

    def __post_init__(self) -> None:
        if self.inclusion is NodeInclusion.SUMMARIZE and not self.summary_token_ids:
            raise ValueError(f"node {self.node_id!r} is SUMMARIZE but has no summary_token_ids")

    @property
    def effective_span(self) -> tuple[int, ...]:
        """The tokens actually decoded into the chain for this node's policy."""
        if self.inclusion is NodeInclusion.EXCLUDE:
            return ()
        if self.inclusion is NodeInclusion.SUMMARIZE:
            return tuple(self.summary_token_ids)
        return tuple(self.span_token_ids)


def _seq_rm(bridge: LlamaSeqStateBridge, ctx: Any, seq_id: int, p0: int, p1: int) -> None:
    """Drop KV cells for ``seq_id`` in position range [p0, p1) to rewind a branch.

    Uses the same memory API the continuous batcher relies on for slot reuse.
    ``p1 == -1`` clears to the end, matching llama.cpp's convention.
    """
    native = bridge._native  # same native handle the bridge decodes through
    memory = native.llama_get_memory(ctx)
    native.llama_memory_seq_rm(memory, seq_id, p0, p1)


def capture_tree(
    bridge: LlamaSeqStateBridge,
    ctx: Any,
    *,
    cache: PrefixStateCache,
    index: NodeDeltaIndex,
    namespace: str,
    fingerprint: LlamaEngineFingerprint,
    nodes: Sequence[TreeNodeSpec],
    n_seq_max: int,
    seq_id: int = 0,
) -> tuple[NodeDelta, ...]:
    """Precompute the whole tree into per-node whole-seq blobs, once each.

    Decodes the tree depth-first in a single sequence so every node's blob is a
    valid cumulative-chain snapshot, then rewinds to each parent boundary before
    descending into the next sibling.  Publishes one blob per node under its
    cumulative-chain key (reusing the exact-prefix cache) and records one
    :class:`NodeDelta` row per node.  Returns the recorded deltas in DFS order.

    Raises ``ValueError`` on a malformed tree (missing parent, duplicate id,
    forest with no single traversal root is fine -- every ``parent is None`` node
    is treated as a top-level root).
    """
    by_parent: dict[str | None, list[TreeNodeSpec]] = {}
    ids: set[str] = set()
    for spec in nodes:
        if spec.node_id in ids:
            raise ValueError(f"duplicate node_id {spec.node_id!r}")
        ids.add(spec.node_id)
        by_parent.setdefault(spec.parent_node_id, []).append(spec)
    for spec in nodes:
        if spec.parent_node_id is not None and spec.parent_node_id not in ids:
            raise ValueError(f"node {spec.node_id!r} references unknown parent {spec.parent_node_id!r}")

    recorded: list[NodeDelta] = []

    def descend(spec: TreeNodeSpec, ancestor_tokens: tuple[int, ...], base_position: int) -> None:
        # The effective span reflects the node's policy: raw body, a shorter
        # summary, or nothing (EXCLUDE).  Children inherit exactly what is
        # decoded here, so a SUMMARIZE parent gives its subtree the digest.
        span = spec.effective_span
        if span:
            bridge.decode_tokens(ctx, span, seq_id, base_position)

        cumulative = ancestor_tokens + span
        child_base = base_position + len(span)

        if spec.inclusion is not NodeInclusion.EXCLUDE:
            if not cumulative:
                raise ValueError(f"node {spec.node_id!r} has an empty cumulative chain; a root must add tokens")
            key = PrefixCacheKey.from_fingerprint(
                namespace=namespace, fingerprint=fingerprint, prefix_token_ids=cumulative
            )
            blob = bridge.capture_seq_state(ctx, seq_id)
            cache.publish(
                key,
                blob,
                llama_state_version="node_delta_seq_v1",
                runtime_facts={"n_seq_max": str(n_seq_max)},
            )
            delta = NodeDelta(
                node_id=spec.node_id,
                parent_node_id=spec.parent_node_id,
                key=key,
                span_token_ids=span,
                base_position=base_position,
                inclusion=spec.inclusion,
                source_path=spec.source_path,
            )
            index.put(delta)
            recorded.append(delta)

        for child in by_parent.get(spec.node_id, ()):
            descend(child, cumulative, child_base)

        # Rewind this node's span so the next sibling reuses the parent boundary.
        if span:
            _seq_rm(bridge, ctx, seq_id, base_position, -1)

    for root in by_parent.get(None, ()):
        descend(root, (), 0)

    return tuple(recorded)


def reconstruct_chain(
    bridge: LlamaSeqStateBridge,
    ctx: Any,
    *,
    cache: PrefixStateCache,
    index: NodeDeltaIndex,
    node_id: str,
    prompt_token_ids: Sequence[int],
    n_seq_max: int,
    seq_id: int = 0,
) -> LiveRestoreResult | ChainDecision:
    """Restore node ``node_id``'s whole-chain KV and decode the prompt as suffix.

    This is the query path: no codebase prefill, only ``prompt_token_ids`` are
    decoded.  Resolves the tree to the target's cumulative-chain key, looks up
    its blob, and delegates the restore-then-suffix lifecycle to the exact-prefix
    ``restore_and_continue`` so every fail-closed rule (n_seq_max portability,
    set_data==0 reject, mandatory suffix) is inherited unchanged.

    Returns a ``ChainDecision`` rejection when the tree walk fails (fall back to
    a cold prefill of the whole context), otherwise the ``LiveRestoreResult``.
    """
    prompt = tuple(prompt_token_ids)
    if not prompt:
        raise ValueError("prompt_token_ids must be non-empty")

    target = index.get(node_id)
    if target is None:
        return _node_miss(node_id)

    # Look up the target's entry only to hand plan_chain the exact key it will
    # re-verify; the heavy blob read happens inside restore_and_continue.
    lookup = cache.lookup(target.key)
    if not lookup.hit or lookup.entry is None:
        return _node_miss(node_id, detail=lookup.reason)

    decision = plan_chain(index, lookup.entry, node_id, prompt)
    if isinstance(decision, ChainDecision):
        return decision  # tree-level rejection
    assert isinstance(decision, RestorePlanDecision)
    if decision.plan is None:
        # Exact-prefix policy rejected the reconstructed request; surface it as
        # a live result so the caller falls back to cold prefill.
        assert decision.rejection is not None
        return LiveRestoreResult(False, str(decision.rejection.reason))

    request = decision.plan.cached_prefix_token_ids + decision.plan.uncached_suffix_token_ids
    return restore_and_continue(
        bridge,
        ctx,
        cache=cache,
        key=target.key,
        request_token_ids=request,
        n_seq_max=n_seq_max,
        seq_id=seq_id,
    )


def _node_miss(node_id: str, *, detail: str | None = None) -> ChainDecision:
    from .node_delta import ChainRejectionReason

    return ChainDecision.reject(ChainRejectionReason.NODE_NOT_FOUND, detail or node_id)
