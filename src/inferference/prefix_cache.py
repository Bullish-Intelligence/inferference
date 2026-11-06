"""Contracts for a persistent, exact-prefix llama.cpp state snapshot cache.

This module deliberately has no llama.cpp dependency and performs no I/O.  It
defines the compatibility boundary that a later filesystem implementation and
state-capture codec must obey.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import time_ns
from typing import Any, Protocol

from .fingerprint import LlamaEngineFingerprint

CACHE_FORMAT_VERSION = 1


class CacheRejectionReason(StrEnum):
    """Why an entry must not be used for a restore."""

    NAMESPACE_MISMATCH = "namespace_mismatch"
    FORMAT_VERSION_MISMATCH = "format_version_mismatch"
    ENGINE_FINGERPRINT_MISMATCH = "engine_fingerprint_mismatch"
    PREFIX_TOKEN_IDS_MISMATCH = "prefix_token_ids_mismatch"
    CHECKPOINT_TOKEN_COUNT_MISMATCH = "checkpoint_token_count_mismatch"
    REQUEST_DOES_NOT_EXTEND_PREFIX = "request_does_not_extend_prefix"
    SUFFIX_DECODE_REQUIRED = "suffix_decode_required"
    N_SEQ_MAX_MISMATCH = "n_seq_max_mismatch"
    STATE_SET_DATA_REJECTED = "state_set_data_rejected"
    STATE_SIZE_MISMATCH = "state_size_mismatch"
    STATE_CHECKSUM_MISMATCH = "state_checksum_mismatch"


@dataclass(frozen=True, slots=True)
class PrefixCacheKey:
    """Identity of one exact checkpoint, including its compatible engine."""

    namespace: str
    engine_fingerprint_key: str
    prefix_token_ids: tuple[int, ...]
    checkpoint_token_count: int
    format_version: int = CACHE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("namespace must be non-empty")
        if not self.engine_fingerprint_key:
            raise ValueError("engine_fingerprint_key must be non-empty")
        if self.format_version <= 0:
            raise ValueError("format_version must be positive")
        if self.checkpoint_token_count != len(self.prefix_token_ids):
            raise ValueError("checkpoint_token_count must equal len(prefix_token_ids)")
        if any(token < 0 for token in self.prefix_token_ids):
            raise ValueError("prefix_token_ids must contain non-negative token IDs")

    @classmethod
    def from_fingerprint(
        cls,
        *,
        namespace: str,
        fingerprint: LlamaEngineFingerprint,
        prefix_token_ids: tuple[int, ...],
        format_version: int = CACHE_FORMAT_VERSION,
    ) -> PrefixCacheKey:
        """Construct a key from the complete, frozen engine fingerprint."""
        return cls(
            namespace=namespace,
            engine_fingerprint_key=fingerprint.cache_key(),
            prefix_token_ids=prefix_token_ids,
            checkpoint_token_count=len(prefix_token_ids),
            format_version=format_version,
        )

    @property
    def storage_key(self) -> str:
        """Injective, deterministic serialization for an index or blob name.

        The complete serialized fields remain in the key instead of relying on
        a digest alone, so distinct fingerprints or token sequences cannot map
        to the same value through this contract.
        """
        fields = (
            str(self.format_version),
            _encode_text(self.namespace),
            _encode_text(self.engine_fingerprint_key),
            str(self.checkpoint_token_count),
            ".".join(str(token) for token in self.prefix_token_ids),
        )
        return "prefix-cache:" + "|".join(fields)


@dataclass(frozen=True, slots=True)
class PrefixCacheEntry:
    """Indexed metadata for one persisted whole-state snapshot."""

    key: PrefixCacheKey
    state_size_bytes: int
    state_checksum_sha256: str
    state_blob_path: str | None = None
    created_at_ns: int | None = None
    accessed_at_ns: int | None = None
    llama_state_version: str | None = None
    runtime_facts: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.state_size_bytes < 0:
            raise ValueError("state_size_bytes must be non-negative")
        if len(self.state_checksum_sha256) != 64:
            raise ValueError("state_checksum_sha256 must be a SHA-256 hex digest")
        try:
            int(self.state_checksum_sha256, 16)
        except ValueError as exc:
            raise ValueError("state_checksum_sha256 must be a SHA-256 hex digest") from exc

    @property
    def format_version(self) -> int:
        return self.key.format_version


@dataclass(frozen=True, slots=True)
class CacheCompatibility:
    """A non-throwing compatibility result suitable for a cache fallback."""

    accepted: bool
    reason: CacheRejectionReason | None = None
    detail: str | None = None

    @classmethod
    def reject(cls, reason: CacheRejectionReason, detail: str) -> CacheCompatibility:
        return cls(accepted=False, reason=reason, detail=detail)


@dataclass(frozen=True, slots=True)
class RestorePlan:
    """The only safe restore data flow: restore then decode fresh suffix tokens."""

    entry: PrefixCacheEntry
    cached_prefix_token_ids: tuple[int, ...]
    uncached_suffix_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.uncached_suffix_token_ids:
            raise ValueError("restore plan requires at least one suffix token decode for fresh logits")


@dataclass(frozen=True, slots=True)
class RestorePlanDecision:
    """A restore plan or an explicit reason to fall back to normal prefill."""

    plan: RestorePlan | None = None
    rejection: CacheCompatibility | None = None

    def __post_init__(self) -> None:
        if (self.plan is None) == (self.rejection is None):
            raise ValueError("restore decision must contain exactly one of plan or rejection")

    @property
    def can_restore(self) -> bool:
        return self.plan is not None


class PrefixCacheBlobStore(Protocol):
    """Blob boundary for a later atomic filesystem implementation."""

    def read_blob(self, entry: PrefixCacheEntry) -> bytes | None: ...

    def write_blob(self, entry: PrefixCacheEntry, state: bytes) -> None: ...


class PrefixCacheIndex(Protocol):
    """Checkpoint-boundary metadata lookup; deliberately not a radix tree."""

    def get(self, key: PrefixCacheKey) -> PrefixCacheEntry | None: ...

    def put(self, entry: PrefixCacheEntry) -> None: ...


@dataclass(frozen=True, slots=True)
class CacheLookup:
    """Safe lookup result; failure is data, never an inference exception."""

    entry: PrefixCacheEntry | None
    state: bytes | None
    hit: bool
    reason: str


class PersistentPrefixCache:
    """Small SQLite-indexed, atomic whole-state cache with manual retention.

    This deliberately stores one blob per checkpoint boundary.  It has no
    eviction daemon or radix tree: deletion/retention is manual for the MVP.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.blobs = self.root / "blobs"
        self.db_path = self.root / "index.sqlite3"
        self.blobs.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS prefix_cache_entries (
                storage_key TEXT PRIMARY KEY, key_json TEXT NOT NULL,
                size_bytes INTEGER NOT NULL, checksum TEXT NOT NULL,
                blob_path TEXT NOT NULL, created_at_ns INTEGER NOT NULL,
                accessed_at_ns INTEGER NOT NULL, llama_state_version TEXT,
                runtime_facts_json TEXT NOT NULL)"""
            )

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
    def _entry(row: tuple[Any, ...]) -> PrefixCacheEntry:
        raw = json.loads(row[1])
        key = PrefixCacheKey(
            namespace=raw["namespace"],
            engine_fingerprint_key=raw["engine_fingerprint_key"],
            prefix_token_ids=tuple(raw["prefix_token_ids"]),
            checkpoint_token_count=raw["checkpoint_token_count"],
            format_version=raw["format_version"],
        )
        return PrefixCacheEntry(
            key,
            int(row[2]),
            str(row[3]),
            str(row[4]),
            int(row[5]),
            int(row[6]),
            row[7],
            tuple(tuple(item) for item in json.loads(row[8])),
        )

    def publish(
        self,
        key: PrefixCacheKey,
        state: bytes,
        *,
        llama_state_version: str | None = None,
        runtime_facts: dict[str, str] | None = None,
    ) -> PrefixCacheEntry:
        """Durably publish a verified blob before its SQLite metadata row."""
        checksum = hashlib.sha256(state).hexdigest()
        digest = hashlib.sha256(key.storage_key.encode()).hexdigest()
        target = self.blobs / f"{digest}.bin"
        temporary = self.blobs / f".{digest}.{os.getpid()}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(state)
                handle.flush()
                os.fsync(handle.fileno())
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
                raise OSError("state blob checksum changed before publication")
            os.replace(temporary, target)
            now = time_ns()
            entry = PrefixCacheEntry(
                key,
                len(state),
                checksum,
                str(target.relative_to(self.root)),
                now,
                now,
                llama_state_version,
                tuple(sorted((runtime_facts or {}).items())),
            )
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """INSERT OR REPLACE INTO prefix_cache_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        key.storage_key,
                        self._key_json(key),
                        entry.state_size_bytes,
                        entry.state_checksum_sha256,
                        entry.state_blob_path,
                        now,
                        now,
                        llama_state_version,
                        json.dumps(entry.runtime_facts, separators=(",", ":")),
                    ),
                )
            return entry
        finally:
            temporary.unlink(missing_ok=True)

    def lookup(self, key: PrefixCacheKey) -> CacheLookup:
        entry, reason = self.lookup_entry(key)
        if entry is None:
            return CacheLookup(None, None, False, reason)
        return self.read_entry(entry)

    def lookup_entry(self, key: PrefixCacheKey) -> tuple[PrefixCacheEntry | None, str]:
        """Return compatible metadata only; caller may time disk read separately."""
        try:
            with self._connect() as db:
                row = db.execute(
                    "SELECT * FROM prefix_cache_entries WHERE storage_key = ?", (key.storage_key,)
                ).fetchone()
                if row is None:
                    return None, "miss_not_found"
                entry = self._entry(row)
                compatible = check_compatibility(entry, key)
                if not compatible.accepted:
                    return None, str(compatible.reason)
                now = time_ns()
                db.execute(
                    "UPDATE prefix_cache_entries SET accessed_at_ns = ? WHERE storage_key = ?", (now, key.storage_key)
                )
                return entry, "hit"
        except (OSError, sqlite3.Error, ValueError, json.JSONDecodeError) as exc:
            return None, f"miss_storage_error:{type(exc).__name__}"

    def read_entry(self, entry: PrefixCacheEntry) -> CacheLookup:
        """Read and checksum a previously accepted entry without raising."""
        try:
            state = (self.root / str(entry.state_blob_path)).read_bytes()
            integrity = check_state_integrity(entry, state)
            if not integrity.accepted:
                return CacheLookup(entry, None, False, str(integrity.reason))
            return CacheLookup(entry, state, True, "hit")
        except OSError as exc:
            return CacheLookup(entry, None, False, f"miss_storage_error:{type(exc).__name__}")


def restore_then_decode_suffix(
    plan: RestorePlan,
    state: bytes,
    *,
    load_state: Callable[[bytes], None],
    decode_suffix: Callable[[tuple[int, ...]], None],
) -> None:
    """Enforce the only valid lifecycle: load state, then refresh logits."""
    load_state(state)
    decode_suffix(plan.uncached_suffix_token_ids)


def check_compatibility(entry: PrefixCacheEntry, requested_key: PrefixCacheKey) -> CacheCompatibility:
    """Require an exact namespace, format, engine, and prefix-token match."""
    cached = entry.key
    if cached.namespace != requested_key.namespace:
        return CacheCompatibility.reject(CacheRejectionReason.NAMESPACE_MISMATCH, "cache namespace differs")
    if cached.format_version != requested_key.format_version:
        return CacheCompatibility.reject(CacheRejectionReason.FORMAT_VERSION_MISMATCH, "cache format version differs")
    if cached.engine_fingerprint_key != requested_key.engine_fingerprint_key:
        return CacheCompatibility.reject(
            CacheRejectionReason.ENGINE_FINGERPRINT_MISMATCH, "frozen engine fingerprint differs"
        )
    if cached.checkpoint_token_count != requested_key.checkpoint_token_count:
        return CacheCompatibility.reject(
            CacheRejectionReason.CHECKPOINT_TOKEN_COUNT_MISMATCH, "checkpoint token count differs"
        )
    if cached.prefix_token_ids != requested_key.prefix_token_ids:
        return CacheCompatibility.reject(CacheRejectionReason.PREFIX_TOKEN_IDS_MISMATCH, "exact token IDs differ")
    return CacheCompatibility(accepted=True)


def check_state_integrity(entry: PrefixCacheEntry, state: bytes) -> CacheCompatibility:
    """Verify blob size and checksum before passing bytes to a restore codec."""
    if len(state) != entry.state_size_bytes:
        return CacheCompatibility.reject(CacheRejectionReason.STATE_SIZE_MISMATCH, "persisted state size differs")
    actual = hashlib.sha256(state).hexdigest()
    if actual != entry.state_checksum_sha256.lower():
        return CacheCompatibility.reject(
            CacheRejectionReason.STATE_CHECKSUM_MISMATCH, "persisted state checksum differs"
        )
    return CacheCompatibility(accepted=True)


def plan_restore(
    entry: PrefixCacheEntry, requested_key: PrefixCacheKey, request_token_ids: tuple[int, ...]
) -> RestorePlanDecision:
    """Make a safe restore plan, requiring a post-restore suffix decode.

    Saved llama.cpp state excludes the output logits buffer.  A zero-token
    suffix would therefore invite callers to consume stale logits.
    """
    compatible = check_compatibility(entry, requested_key)
    if not compatible.accepted:
        return RestorePlanDecision(rejection=compatible)
    prefix = entry.key.prefix_token_ids
    if request_token_ids[: len(prefix)] != prefix:
        return RestorePlanDecision(
            rejection=CacheCompatibility.reject(
                CacheRejectionReason.REQUEST_DOES_NOT_EXTEND_PREFIX, "request does not begin with cached token IDs"
            )
        )
    suffix = request_token_ids[len(prefix) :]
    if not suffix:
        return RestorePlanDecision(
            rejection=CacheCompatibility.reject(
                CacheRejectionReason.SUFFIX_DECODE_REQUIRED, "restore requires one or more uncached suffix tokens"
            )
        )
    return RestorePlanDecision(plan=RestorePlan(entry, prefix, suffix))


def _encode_text(value: str) -> str:
    """Length-prefix UTF-8 text so delimiter characters remain unambiguous."""
    encoded = value.encode("utf-8").hex()
    return f"{len(encoded)}:{encoded}"
