"""Codebase-tree node deltas layered over the exact-prefix state cache.

This is the pure policy/persistence layer for the "context tree" demo: a
codebase is a tree (repo root -> dir -> file -> symbol) where every node's
LLM context is the concatenation of its ancestors' tokens.  Because that chain
is a single causal, monotonically-positioned token sequence, a node captured
*after its ancestors were decoded into the same sequence* has a whole-seq KV
blob (``llama_state_seq_get_data``) that already contains the entire chain.

So this module adds three things on top of ``prefix_cache.py`` and reuses its
identity/compatibility/integrity machinery verbatim:

  * :class:`NodeDelta` -- tree metadata for one node (parent link, the token
    span this node *adds*, the absolute base position that span starts at, and
    an inclusion policy) paired with the :class:`PrefixCacheKey` of the node's
    cumulative chain;
  * a small SQLite index (:class:`NodeDeltaIndex`) that stores the tree
    alongside the existing ``prefix_cache_entries`` blobs, so navigation is a
    parent-pointer walk, not a re-tokenization;
  * :func:`plan_chain` -- resolve "answer at node S" to the exact cumulative
    token IDs to restore, reusing :func:`plan_restore` for the restore lifecycle.

Like ``prefix_cache.py`` this module has no llama.cpp dependency and performs no
inference; the DFS capture driver and live reconstruct live in
``node_delta_live.py``.  Failure is always data, never an inference exception.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .prefix_cache import (
    PrefixCacheEntry,
    PrefixCacheKey,
    RestorePlanDecision,
    plan_restore,
)

NODE_TREE_FORMAT_VERSION = 1


class NodeInclusion(StrEnum):
    """Per-node policy deciding how a node contributes to a child's context."""

    RAW = "raw"  # decode the node's full token span into the chain
    SUMMARIZE = "summarize"  # decode a summary token span in place of the raw body
    EXCLUDE = "exclude"  # skip entirely; children attach to this node's parent


class ChainRejectionReason(StrEnum):
    """Why a chain cannot be reconstructed from stored deltas."""

    NODE_NOT_FOUND = "node_not_found"
    BROKEN_PARENT_LINK = "broken_parent_link"
    MISSING_NODE_ENTRY = "missing_node_entry"
    POSITION_DISCONTINUITY = "position_discontinuity"
    ENGINE_FINGERPRINT_MISMATCH = "engine_fingerprint_mismatch"


@dataclass(frozen=True, slots=True)
class NodeDelta:
    """One codebase-tree node: its chain key plus incremental tree metadata.

    ``key`` identifies the node's *cumulative* chain (all ancestor tokens plus
    this node's own span), so a whole-seq blob published under ``key`` restores
    the entire chain at once.  ``span_token_ids`` is only the increment this node
    contributes, and ``base_position`` is the absolute position that increment
    starts at (== the cumulative token count of the parent chain).  Those two
    fields are what a future cross-chain splice needs to re-anchor RoPE; within
    a single chain they are bookkeeping that lets the DFS driver decode each
    node exactly once.
    """

    node_id: str
    parent_node_id: str | None
    key: PrefixCacheKey
    span_token_ids: tuple[int, ...]
    base_position: int
    inclusion: NodeInclusion = NodeInclusion.RAW
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if self.base_position < 0:
            raise ValueError("base_position must be non-negative")
        if self.parent_node_id == self.node_id:
            raise ValueError("a node cannot be its own parent")
        # The cumulative chain must be exactly parent-prefix followed by this
        # node's span, ending at the recorded checkpoint.  We can verify the
        # tail locally; the parent-prefix half is verified during a chain walk.
        chain = self.key.prefix_token_ids
        span = self.span_token_ids
        if span and chain[len(chain) - len(span) :] != span:
            raise ValueError("key.prefix_token_ids must end with span_token_ids")
        if self.base_position + len(span) != self.key.checkpoint_token_count:
            raise ValueError("base_position + len(span) must equal cumulative checkpoint count")

    @property
    def cumulative_token_count(self) -> int:
        return self.key.checkpoint_token_count

    @property
    def is_root(self) -> bool:
        return self.parent_node_id is None


@dataclass(frozen=True, slots=True)
class ChainPlan:
    """Resolved reconstruction for a target node: the node whose blob to restore.

    Because each node's blob is a whole-seq snapshot of its cumulative chain,
    reconstructing "context at ``target``" is a single restore of ``target``'s
    own entry followed by the mandatory suffix decode -- the ancestor walk only
    validates continuity and supplies the token IDs the restore plan checks.
    """

    target: NodeDelta
    chain: tuple[NodeDelta, ...]  # root..target, EXCLUDE nodes dropped
    cumulative_token_ids: tuple[int, ...]  # exact IDs the target blob covers


@dataclass(frozen=True, slots=True)
class ChainDecision:
    """A chain plan or an explicit reason to fall back to a cold prefill."""

    plan: ChainPlan | None = None
    reason: ChainRejectionReason | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if (self.plan is None) == (self.reason is None):
            raise ValueError("chain decision must contain exactly one of plan or reason")

    @property
    def can_restore(self) -> bool:
        return self.plan is not None

    @classmethod
    def reject(cls, reason: ChainRejectionReason, detail: str) -> ChainDecision:
        return cls(plan=None, reason=reason, detail=detail)


class NodeDeltaIndex:
    """SQLite tree index that lives beside the prefix-cache blob store.

    Deliberately mirrors ``PersistentPrefixCache``'s connection PRAGMAs and its
    "one row per checkpoint" simplicity: no eviction daemon, manual retention.
    A ``NodeDelta`` row references its cumulative chain by ``storage_key`` -- the
    same key ``PersistentPrefixCache`` files the whole-seq blob under -- so the
    blob store and the tree share identity without duplicating bytes.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.db_path = self.root / "node_tree.sqlite3"
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS node_deltas (
                    node_id TEXT PRIMARY KEY,
                    parent_node_id TEXT,
                    storage_key TEXT NOT NULL,
                    key_json TEXT NOT NULL,
                    span_token_ids_json TEXT NOT NULL,
                    base_position INTEGER NOT NULL,
                    inclusion TEXT NOT NULL,
                    source_path TEXT,
                    tree_format_version INTEGER NOT NULL
                )"""
            )
            db.execute("CREATE INDEX IF NOT EXISTS node_deltas_parent ON node_deltas(parent_node_id)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @staticmethod
    def _key_json(key: PrefixCacheKey) -> str:
        return json.dumps(
            {
                "namespace": key.namespace,
                "engine_fingerprint_key": key.engine_fingerprint_key,
                "prefix_token_ids": key.prefix_token_ids,
                "checkpoint_token_count": key.checkpoint_token_count,
                "format_version": key.format_version,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _key_from_json(raw_json: str) -> PrefixCacheKey:
        raw = json.loads(raw_json)
        return PrefixCacheKey(
            namespace=raw["namespace"],
            engine_fingerprint_key=raw["engine_fingerprint_key"],
            prefix_token_ids=tuple(raw["prefix_token_ids"]),
            checkpoint_token_count=raw["checkpoint_token_count"],
            format_version=raw["format_version"],
        )

    @classmethod
    def _row_to_delta(cls, row: tuple[Any, ...]) -> NodeDelta:
        return NodeDelta(
            node_id=str(row[0]),
            parent_node_id=row[1],
            key=cls._key_from_json(row[3]),
            span_token_ids=tuple(json.loads(row[4])),
            base_position=int(row[5]),
            inclusion=NodeInclusion(row[6]),
            source_path=row[7],
        )

    def put(self, delta: NodeDelta) -> None:
        """Record one node.  Idempotent replace, matching the blob store's upsert."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT OR REPLACE INTO node_deltas VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    delta.node_id,
                    delta.parent_node_id,
                    delta.key.storage_key,
                    self._key_json(delta.key),
                    json.dumps(delta.span_token_ids, separators=(",", ":")),
                    delta.base_position,
                    str(delta.inclusion),
                    delta.source_path,
                    NODE_TREE_FORMAT_VERSION,
                ),
            )

    def get(self, node_id: str) -> NodeDelta | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM node_deltas WHERE node_id = ?", (node_id,)).fetchone()
            return None if row is None else self._row_to_delta(row)

    def children(self, node_id: str | None) -> tuple[NodeDelta, ...]:
        with self._connect() as db:
            if node_id is None:
                rows = db.execute("SELECT * FROM node_deltas WHERE parent_node_id IS NULL").fetchall()
            else:
                rows = db.execute("SELECT * FROM node_deltas WHERE parent_node_id = ?", (node_id,)).fetchall()
            return tuple(self._row_to_delta(r) for r in rows)


def resolve_chain(index: NodeDeltaIndex, node_id: str) -> ChainDecision:
    """Walk parent links to the root, honoring per-node inclusion policy.

    Returns the ancestor list (root..target) with ``EXCLUDE`` nodes dropped and
    the exact cumulative token IDs the target blob is expected to cover.  All
    failure modes are data so a caller falls back to a cold prefill.
    """
    target = index.get(node_id)
    if target is None:
        return ChainDecision.reject(ChainRejectionReason.NODE_NOT_FOUND, node_id)

    reversed_chain: list[NodeDelta] = []
    seen: set[str] = set()
    cursor: NodeDelta | None = target
    engine = target.key.engine_fingerprint_key
    while cursor is not None:
        if cursor.node_id in seen:
            return ChainDecision.reject(ChainRejectionReason.BROKEN_PARENT_LINK, f"cycle at {cursor.node_id}")
        seen.add(cursor.node_id)
        if cursor.key.engine_fingerprint_key != engine:
            return ChainDecision.reject(ChainRejectionReason.ENGINE_FINGERPRINT_MISMATCH, cursor.node_id)
        if cursor.inclusion is not NodeInclusion.EXCLUDE:
            reversed_chain.append(cursor)
        if cursor.parent_node_id is None:
            break
        parent = index.get(cursor.parent_node_id)
        if parent is None:
            return ChainDecision.reject(
                ChainRejectionReason.BROKEN_PARENT_LINK,
                f"{cursor.node_id} -> {cursor.parent_node_id}",
            )
        cursor = parent

    chain = tuple(reversed(reversed_chain))
    return ChainDecision(
        plan=ChainPlan(
            target=target,
            chain=chain,
            cumulative_token_ids=target.key.prefix_token_ids,
        )
    )


def plan_chain(
    index: NodeDeltaIndex,
    entry: PrefixCacheEntry,
    node_id: str,
    prompt_token_ids: tuple[int, ...],
) -> RestorePlanDecision | ChainDecision:
    """Resolve node ``node_id`` to a concrete restore plan for its cached blob.

    Composes the tree walk with the existing ``plan_restore`` lifecycle.  The
    target node's whole-seq blob is published under the key whose
    ``prefix_token_ids`` are the node's cumulative chain, so a valid restore
    request is ``cumulative_chain + prompt_token_ids`` and the prompt is exactly
    the mandatory >= 1 suffix token ``plan_restore`` requires for fresh logits.

    ``entry`` must be the cache entry published for this node (its ``key`` equal
    to the node's cumulative-chain key); ``plan_restore`` re-checks that exact
    identity, so a stale or mismatched entry is rejected as data, not trusted.

    Returns the ``ChainDecision`` rejection when the tree itself is unusable,
    otherwise the ``RestorePlanDecision`` from the exact-prefix policy.
    """
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must be non-empty (the suffix that refreshes logits)")
    decision = resolve_chain(index, node_id)
    if not decision.can_restore:
        return decision
    assert decision.plan is not None
    request = decision.plan.cumulative_token_ids + tuple(prompt_token_ids)
    return plan_restore(entry, entry.key, request)
