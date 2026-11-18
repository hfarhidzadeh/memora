"""SQLite-backed typed memory, hybrid retrieval, and archive pruning."""

from __future__ import annotations

import math
import re
import sqlite3
import uuid
from array import array
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Union

from .embeddings import Embedder

SCHEMA_VERSION = 4


class MemoryType(str, Enum):
    USER_PREFERENCE = "user_preference"
    STRATEGIC_INSIGHT = "strategic_insight"
    TECHNICAL_CONSTRAINT = "technical_constraint"
    CORRECTION = "correction"
    TASK = "task"
    FACT = "fact"
    EPISODE = "episode"


class Relationship(str, Enum):
    RELATED_TO = "related_to"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    DEPENDS_ON = "depends_on"


class LinkCarryover(str, Enum):
    """What happens to a memory's links when a newer version replaces it.

    ``keep`` leaves them on the old, archived version. ``copy`` copies them to
    the new version when the replacement happens. ``topic`` leaves them where
    they are, and search treats a link to any version of a fact as a link to
    its current version.

    ``copy`` acts when an update is written and ``topic`` when searching, so
    they give the same search results only if ``copy`` was in effect for every
    update. A database updated under ``keep`` and searched under ``copy``
    behaves like ``keep``, and links copied under ``copy`` stay whichever
    option later searches use.
    """

    KEEP = "keep"
    COPY = "copy"
    TOPIC = "topic"


@dataclass(frozen=True)
class Memory:
    id: str
    kind: MemoryType
    content: str
    memory_key: Optional[str]
    source: str
    confidence: float
    importance: float
    created_at: str
    updated_at: str
    last_accessed_at: Optional[str]
    access_count: int
    archived_at: Optional[str]
    superseded_by: Optional[str]
    channel: Optional[str] = None
    speaker: Optional[str] = None
    asserted_as_admin: bool = False
    fact_subject: Optional[str] = None
    asserted_as_delegate: bool = False


@dataclass(frozen=True)
class Link:
    """A directed relationship that reads ``source relation target``."""

    source_id: str
    target_id: str
    relation: Relationship


@dataclass(frozen=True)
class SearchResult:
    """One ranked memory.

    ``score`` only orders the results of a single search (the best direct match
    scores 1). The raw signals behind it are comparable across searches, so a
    caller can tell a weak best match from a strong one and stay silent:
    ``lexical_score`` is the BM25-style score, ``vector_score`` the cosine
    similarity (0 below the embedder's ``min_similarity``), and ``coverage``
    the share of the query's inverse-document-frequency mass that the memory
    contains. Each is 0 when the memory did not match that way.

    ``claim_status`` is ``current``, ``disputed``, ``agrees`` or ``overruled``
    for active keyed channel claims. It uses exact text after trimming and
    case folding, not semantic disagreement detection.
    """

    memory: Memory
    score: float
    matched_by: str
    links: tuple[Link, ...] = ()
    lexical_score: float = 0.0
    vector_score: float = 0.0
    coverage: float = 0.0
    claim_status: str = "current"


@dataclass(frozen=True)
class PruneCandidate:
    memory: Memory
    score: float


class AudienceReader:
    """Read-only view of a store for a fixed, nonempty set of viewers.

    The caller must obtain the viewer identities from a trusted source. Each
    read still checks current channel membership, so leaving takes effect on
    the next call.
    """

    def __init__(
        self, store: MemoryStore, viewers: Union[str, Iterable[str]]
    ) -> None:
        self._store = store
        viewer_ids = store._viewer_ids(viewers)
        if viewer_ids is None:
            raise ValueError("viewers must name at least one member")
        self._viewers = viewer_ids

    def search(
        self,
        query: str,
        *,
        kind: Optional[Union[MemoryType, str]] = None,
        limit: int = 10,
        record_access: bool = True,
    ) -> list[SearchResult]:
        return self._store.search(
            query,
            kind=kind,
            limit=limit,
            record_access=record_access,
            viewers=self._viewers,
        )

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        return self._store.get_memory(memory_id, viewers=self._viewers)

    def history(self, memory_id: str) -> list[Memory]:
        return self._store.history(memory_id, viewers=self._viewers)

    def list_memories(
        self, *, include_archived: bool = False, limit: int = 100
    ) -> list[Memory]:
        return self._store.list_memories(
            include_archived=include_archived,
            limit=limit,
            viewers=self._viewers,
        )


class ChannelWriter:
    """Write channel claims as one identity supplied by trusted caller code."""

    def __init__(self, store: MemoryStore, member_id: str) -> None:
        self._store = store
        self._member_id = store._name(member_id, "member_id")

    def add_memory(
        self, content: str, kind: Union[MemoryType, str], *,
        channel: str, memory_key: Optional[str] = None,
        fact_subject: Optional[str] = None, source: str = "conversation",
        confidence: float = 0.8, importance: float = 0.5,
    ) -> Memory:
        return self._store._add_memory(
            content, kind, channel=channel, speaker=self._member_id,
            memory_key=memory_key, fact_subject=fact_subject, source=source,
            confidence=confidence, importance=importance, check_membership=True,
        )


def retention_score(
    importance: float,
    confidence: float,
    idle_days: float,
    access_count: int,
    decay_days: float,
) -> float:
    """Score used by pruning; memories below the threshold are archived."""
    return (
        importance
        * confidence
        * math.exp(-idle_days / decay_days)
        * (1 + math.log1p(access_count) * 0.25)
    )


class MemoryStore:
    """Persist memories in a local SQLite database.

    Without an embedder, search ranks memories lexically. With one, it fuses
    the lexical and vector rankings with reciprocal rank fusion. Either way,
    results are then expanded by one relationship hop.

    A memory may carry the ``channel`` it was said in and its ``speaker``.
    Reads that pass ``viewers`` use only the memories in channels that every
    viewer currently belongs to. By default, reads without ``viewers`` use
    everything for single-user or administrative use. ``require_viewers=True``
    rejects those reads, and ``for_viewers`` binds a reader to one audience.
    Use ``for_writer`` for member writes. ``require_writer=True`` prevents
    direct writes through this administrative store instance.
    """

    _TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
    _RELATION_WEIGHTS = {
        Relationship.RELATED_TO.value: 0.45,
        Relationship.SUPPORTS.value: 0.5,
        Relationship.CONTRADICTS.value: 0.6,
        Relationship.SUPERSEDES.value: 0.55,
        Relationship.DEPENDS_ON.value: 0.45,
    }
    _RRF_K = 60
    _EMBED_BATCH = 256
    _ID_CHUNK = 400

    def __init__(
        self,
        db_path: Union[str, Path] = ".memora.sqlite3",
        *,
        embedder: Optional[Embedder] = None,
        link_carryover: Union[LinkCarryover, str] = LinkCarryover.KEEP,
        require_viewers: bool = False,
        require_writer: bool = False,
    ) -> None:
        self.embedder = embedder
        self.require_viewers = require_viewers
        self.require_writer = require_writer
        try:
            self.link_carryover = LinkCarryover(link_carryover)
        except ValueError as error:
            raise ValueError(
                "unsupported link carryover: {}".format(link_carryover)
            ) from error
        self._use_uri = str(db_path) == ":memory:"
        self.db_path = (
            "file:memora-{}?mode=memory&cache=shared".format(uuid.uuid4().hex)
            if self._use_uri
            else str(Path(db_path).expanduser())
        )
        self._keeper = (
            sqlite3.connect(self.db_path, uri=True) if self._use_uri else None
        )
        if not self._use_uri:
            Path(self.db_path).resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._initialize()

    def close(self) -> None:
        """Release an in-memory database; file databases need no closing."""
        if self._keeper is not None:
            self._keeper.close()
            self._keeper = None

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self, *, write: bool = True) -> Iterator[sqlite3.Connection]:
        """Run statements in one transaction.

        A write transaction takes the write lock before its first read, so
        two writers cannot both read the same state and then both act on it.
        """
        with closing(
            sqlite3.connect(
                self.db_path, timeout=5.0, uri=self._use_uri, isolation_level=None
            )
        ) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")

    def _initialize(self) -> None:
        tables = (
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN (
                    'user_preference', 'strategic_insight',
                    'technical_constraint', 'correction', 'task',
                    'fact', 'episode'
                )),
                content TEXT NOT NULL CHECK (length(trim(content)) > 0),
                memory_key TEXT,
                source TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                importance REAL NOT NULL CHECK (importance BETWEEN 0 AND 1),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT,
                access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),
                archived_at TEXT,
                superseded_by TEXT REFERENCES memories(id),
                channel TEXT,
                speaker TEXT,
                asserted_as_admin INTEGER NOT NULL DEFAULT 0
                    CHECK (asserted_as_admin IN (0, 1)),
                fact_subject TEXT,
                asserted_as_delegate INTEGER NOT NULL DEFAULT 0
                    CHECK (asserted_as_delegate IN (0, 1))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS relationships (
                source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation TEXT NOT NULL CHECK (relation IN (
                    'related_to', 'supports', 'contradicts',
                    'supersedes', 'depends_on'
                )),
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, relation),
                CHECK (source_id <> target_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (memory_id, model)
            )
            """,
            # One row per stay: leaving closes the row, joining again adds one.
            """
            CREATE TABLE IF NOT EXISTS channel_members (
                channel TEXT NOT NULL CHECK (length(trim(channel)) > 0),
                member_id TEXT NOT NULL CHECK (length(trim(member_id)) > 0),
                joined_at TEXT NOT NULL,
                left_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS admins (
                member_id TEXT PRIMARY KEY CHECK (length(trim(member_id)) > 0)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS delegations (
                channel TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                member_id TEXT NOT NULL,
                granted_by TEXT NOT NULL,
                granted_at TEXT NOT NULL,
                revoked_by TEXT,
                revoked_at TEXT,
                PRIMARY KEY (channel, memory_key, member_id, granted_at)
            )
            """,
        )
        indexes = (
            # Channel keys have one active claim per speaker; memories without
            # a channel share one namespace, as before channels existed.
            """
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_memory_per_key
                ON memories(memory_key)
                WHERE memory_key IS NOT NULL AND archived_at IS NULL
                  AND channel IS NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_memory_per_channel_key
                ON memories(channel, memory_key, speaker)
                WHERE memory_key IS NOT NULL AND archived_at IS NULL
                  AND channel IS NOT NULL AND speaker IS NOT NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_unattributed_channel_key
                ON memories(channel, memory_key)
                WHERE memory_key IS NOT NULL AND archived_at IS NULL
                  AND channel IS NOT NULL AND speaker IS NULL
            """,
            """
            CREATE INDEX IF NOT EXISTS memories_active_kind
                ON memories(archived_at, kind)
            """,
            """
            CREATE INDEX IF NOT EXISTS memories_superseded_by
                ON memories(superseded_by)
            """,
            """
            CREATE INDEX IF NOT EXISTS memories_channel
                ON memories(channel)
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS one_current_membership
                ON channel_members(channel, member_id)
                WHERE left_at IS NULL
            """,
            """
            CREATE INDEX IF NOT EXISTS current_memberships_by_member
                ON channel_members(member_id)
                WHERE left_at IS NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_delegation
                ON delegations(channel, memory_key, member_id)
                WHERE revoked_at IS NULL
            """,
        )
        with self._transaction() as connection:
            for statement in tables:
                connection.execute(statement)
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(memories)")
            }
            for column in ("channel", "speaker"):
                if column not in columns:
                    connection.execute(
                        "ALTER TABLE memories ADD COLUMN {} TEXT".format(column)
                    )
            if "asserted_as_admin" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN asserted_as_admin INTEGER NOT NULL DEFAULT 0"
                )
            if "fact_subject" not in columns:
                connection.execute("ALTER TABLE memories ADD COLUMN fact_subject TEXT")
            if "asserted_as_delegate" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN asserted_as_delegate INTEGER NOT NULL DEFAULT 0"
                )
            if version < 2:
                # Keys were unique across the whole store; the index is
                # rebuilt below to cover only memories without a channel.
                connection.execute("DROP INDEX IF EXISTS one_active_memory_per_key")
            if version < 3:
                connection.execute("DROP INDEX IF EXISTS one_active_memory_per_channel_key")
            for statement in indexes:
                connection.execute(statement)
            if version < 1:
                # Key rewrites used to store supersedes edges as old -> new.
                # Every relation now reads source -> target: new supersedes old.
                # Rows whose newer memory no longer exists are left as they are.
                legacy_edge = """
                    relation = 'supersedes'
                    AND EXISTS (
                        SELECT 1 FROM memories
                        WHERE memories.id = relationships.source_id
                          AND memories.superseded_by = relationships.target_id
                    )
                    AND EXISTS (
                        SELECT 1 FROM memories
                        WHERE memories.id = relationships.target_id
                    )
                """
                connection.execute(
                    "UPDATE OR IGNORE relationships "
                    "SET source_id = target_id, target_id = source_id "
                    "WHERE " + legacy_edge
                )
                connection.execute("DELETE FROM relationships WHERE " + legacy_edge)
            if version < SCHEMA_VERSION:
                connection.execute("PRAGMA user_version = {}".format(SCHEMA_VERSION))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")

    @staticmethod
    def _memory_from_row(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            kind=MemoryType(row["kind"]),
            content=row["content"],
            memory_key=row["memory_key"],
            source=row["source"],
            confidence=row["confidence"],
            importance=row["importance"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"],
            archived_at=row["archived_at"],
            superseded_by=row["superseded_by"],
            channel=row["channel"],
            speaker=row["speaker"],
            asserted_as_admin=bool(row["asserted_as_admin"]),
            fact_subject=row["fact_subject"],
            asserted_as_delegate=bool(row["asserted_as_delegate"]),
        )

    @staticmethod
    def _validate_unit_interval(value: float, name: str) -> None:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("{} must be between 0 and 1".format(name))

    @staticmethod
    def _name(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("{} must not be empty".format(name))
        return value.strip()

    @classmethod
    def _viewer_ids(
        cls, viewers: Optional[Union[str, Iterable[str]]]
    ) -> Optional[tuple[str, ...]]:
        """The distinct viewers of a read, or ``None`` if it has no audience."""
        if viewers is None:
            return None
        if isinstance(viewers, str):
            viewers = (viewers,)
        names = {cls._name(viewer, "a viewer") for viewer in viewers}
        if not names:
            # An empty audience would otherwise mean "no one to hide from".
            raise ValueError("viewers must name at least one member")
        return tuple(sorted(names))

    def for_viewers(self, viewers: Union[str, Iterable[str]]) -> AudienceReader:
        """Bind all user-facing reads to these viewers.

        Give this reader, rather than the administrative store, to code that
        serves people. Authentication and viewer identity are the caller's job.
        """
        viewer_ids = self._viewer_ids(viewers)
        if viewer_ids is None:
            raise ValueError("viewers must name at least one member")
        return AudienceReader(self, viewer_ids)

    def for_writer(self, member_id: str) -> ChannelWriter:
        """Bind writes to an authenticated member supplied by the caller."""
        return ChannelWriter(self, member_id)

    def _read_viewers(
        self, viewers: Optional[Union[str, Iterable[str]]]
    ) -> Optional[tuple[str, ...]]:
        viewer_ids = self._viewer_ids(viewers)
        if viewer_ids is None and self.require_viewers:
            raise ValueError("this store requires viewers for user-facing reads")
        return viewer_ids

    @staticmethod
    def _visible(
        viewers: Optional[tuple[str, ...]], column: str = "memories.channel"
    ) -> tuple[str, list]:
        """An SQL condition, and its parameters, that hides what viewers may not see.

        It keeps rows whose channel every viewer currently belongs to, so a
        row without a channel is kept for no viewer. Membership is read in the
        same transaction as the rows. Without viewers it keeps everything.
        """
        if viewers is None:
            return "", []
        return (
            " AND {} IN ("
            "SELECT channel_members.channel FROM channel_members "
            "WHERE channel_members.left_at IS NULL "
            "AND channel_members.member_id IN ({}) "
            "GROUP BY channel_members.channel HAVING COUNT(*) = ?)".format(
                column, ",".join("?" for _ in viewers)
            ),
            [*viewers, len(viewers)],
        )

    def join_channel(self, channel: str, member_id: str) -> None:
        """Add a member to a channel; joining twice changes nothing.

        Access is decided by membership at the time of a read, so a new member
        can read everything already in the channel.
        """
        channel = self._name(channel, "channel")
        member_id = self._name(member_id, "member_id")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO channel_members (channel, member_id, joined_at)
                VALUES (?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (channel, member_id, self._now()),
            )

    def leave_channel(self, channel: str, member_id: str) -> None:
        """End a membership.

        From the next read the member no longer sees the channel's memories,
        including the ones they said themselves. Leaving a channel the member
        is not in is an error, so a misspelt name cannot pass for a removal.
        """
        channel = self._name(channel, "channel")
        member_id = self._name(member_id, "member_id")
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE channel_members SET left_at = ?
                WHERE channel = ? AND member_id = ? AND left_at IS NULL
                """,
                (self._now(), channel, member_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(
                    "{} is not a member of {}".format(member_id, channel)
                )
            connection.execute(
                """UPDATE delegations SET revoked_at = ?, revoked_by = ?
                WHERE channel = ? AND member_id = ? AND revoked_at IS NULL""",
                (self._now(), "channel_departure", channel, member_id),
            )

    def members_of(self, channel: str) -> list[str]:
        """The current members of a channel."""
        with self._transaction(write=False) as connection:
            return [
                row["member_id"]
                for row in connection.execute(
                    """
                    SELECT member_id FROM channel_members
                    WHERE channel = ? AND left_at IS NULL ORDER BY member_id
                    """,
                    (self._name(channel, "channel"),),
                )
            ]

    def channels_of(self, member_id: str) -> list[str]:
        """The channels a member is currently in."""
        with self._transaction(write=False) as connection:
            return [
                row["channel"]
                for row in connection.execute(
                    """
                    SELECT channel FROM channel_members
                    WHERE member_id = ? AND left_at IS NULL ORDER BY channel
                    """,
                    (self._name(member_id, "member_id"),),
                )
            ]

    def add_admin(self, member_id: str) -> None:
        """Register a workspace-wide admin through the trusted store interface."""
        member_id = self._name(member_id, "member_id")
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO admins (member_id) VALUES (?)", (member_id,)
            )

    def is_admin(self, member_id: str) -> bool:
        """Whether a member can resolve team-fact disagreements."""
        member_id = self._name(member_id, "member_id")
        with self._transaction(write=False) as connection:
            return connection.execute(
                "SELECT 1 FROM admins WHERE member_id = ?", (member_id,)
            ).fetchone() is not None

    def grant_delegation(
        self, channel: str, memory_key: str, member_id: str, *, granted_by: str
    ) -> None:
        """Grant one team claim to a channel member; caller must be an admin."""
        channel = self._name(channel, "channel")
        memory_key = self._name(memory_key, "memory_key")
        member_id = self._name(member_id, "member_id")
        granted_by = self._name(granted_by, "granted_by")
        with self._transaction() as connection:
            if not connection.execute(
                "SELECT 1 FROM admins WHERE member_id = ?", (granted_by,)
            ).fetchone():
                raise ValueError("only an admin may grant delegation")
            if not connection.execute(
                """SELECT 1 FROM channel_members WHERE channel = ?
                AND member_id = ? AND left_at IS NULL""", (channel, member_id)
            ).fetchone():
                raise ValueError("delegate must be a current channel member")
            if connection.execute(
                """SELECT 1 FROM memories WHERE channel = ? AND memory_key = ?
                AND fact_subject IS NOT NULL""", (channel, memory_key)
            ).fetchone():
                raise ValueError("personal facts cannot be delegated")
            if not connection.execute(
                """SELECT 1 FROM delegations WHERE channel = ? AND memory_key = ?
                AND member_id = ? AND revoked_at IS NULL""",
                (channel, memory_key, member_id),
            ).fetchone():
                connection.execute(
                    """INSERT INTO delegations
                    (channel, memory_key, member_id, granted_by, granted_at)
                    VALUES (?, ?, ?, ?, ?)""",
                    (channel, memory_key, member_id, granted_by, self._now()),
                )

    def revoke_delegation(
        self, channel: str, memory_key: str, member_id: str, *, revoked_by: str
    ) -> None:
        """Revoke a team-fact grant; old claims keep their write-time role."""
        channel = self._name(channel, "channel")
        memory_key = self._name(memory_key, "memory_key")
        member_id = self._name(member_id, "member_id")
        revoked_by = self._name(revoked_by, "revoked_by")
        with self._transaction() as connection:
            if not connection.execute(
                "SELECT 1 FROM admins WHERE member_id = ?", (revoked_by,)
            ).fetchone():
                raise ValueError("only an admin may revoke delegation")
            cursor = connection.execute(
                """UPDATE delegations SET revoked_at = ?, revoked_by = ?
                WHERE channel = ? AND memory_key = ? AND member_id = ?
                AND revoked_at IS NULL""",
                (self._now(), revoked_by, channel, memory_key, member_id),
            )
            if cursor.rowcount == 0:
                raise ValueError("no active delegation for this claim")

    @staticmethod
    def _claim_rank(row: Union[sqlite3.Row, Mapping[str, object]]) -> int:
        if row["fact_subject"] is not None:
            return 2 if row["speaker"] == row["fact_subject"] else 0
        if row["asserted_as_admin"]:
            return 2
        return 1 if row["asserted_as_delegate"] else 0

    def add_memory(
        self,
        content: str,
        kind: Union[MemoryType, str],
        *,
        memory_key: Optional[str] = None,
        source: str = "conversation",
        confidence: float = 0.8,
        importance: float = 0.5,
        channel: Optional[str] = None,
        speaker: Optional[str] = None,
        fact_subject: Optional[str] = None,
    ) -> Memory:
        if self.require_writer:
            raise ValueError("this store requires a bound writer for user-facing writes")
        return self._add_memory(
            content, kind, memory_key=memory_key, source=source,
            confidence=confidence, importance=importance, channel=channel,
            speaker=speaker, fact_subject=fact_subject, check_membership=False,
        )

    def _add_memory(
        self, content: str, kind: Union[MemoryType, str], *,
        memory_key: Optional[str], source: str, confidence: float,
        importance: float, channel: Optional[str], speaker: Optional[str],
        fact_subject: Optional[str], check_membership: bool,
    ) -> Memory:
        """Store a memory; a keyed channel claim belongs to its speaker.

        ``channel`` is where the memory was said and ``speaker`` who said it.
        In a channel, the same speaker corrects their own claim. A different
        speaker's differing claim stays active and is linked as a contradiction.
        A workspace admin resolves non-admin claims; two admins' first claims
        can disagree until one restates. Unchannelled keys keep their original
        single-user replacement behavior.
        """
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        if memory_key is not None and not memory_key.strip():
            raise ValueError("memory_key must not be empty")
        if not source or not source.strip():
            raise ValueError("source must not be empty")
        if channel is not None:
            channel = self._name(channel, "channel")
        if speaker is not None:
            speaker = self._name(speaker, "speaker")
        if channel is not None and memory_key is not None and speaker is None:
            raise ValueError("keyed channel memories require a speaker")
        if fact_subject is not None:
            fact_subject = self._name(fact_subject, "fact_subject")
            if channel is None or memory_key is None:
                raise ValueError("personal facts require a channel and memory_key")
        if check_membership and channel is None:
            raise ValueError("bound writers require a channel")
        self._validate_unit_interval(confidence, "confidence")
        self._validate_unit_interval(importance, "importance")
        try:
            memory_type = MemoryType(kind)
        except ValueError as error:
            raise ValueError("unsupported memory type: {}".format(kind)) from error

        vector = None
        if self.embedder is not None:
            vector = self._embed([content.strip()])[0]

        now = self._now()
        memory_id = uuid.uuid4().hex
        with self._transaction() as connection:
            if check_membership and not connection.execute(
                """SELECT 1 FROM channel_members WHERE channel = ?
                AND member_id = ? AND left_at IS NULL""", (channel, speaker)
            ).fetchone():
                raise ValueError("writer is not a current channel member")
            predecessors = []
            conflicts = []
            asserted_as_admin = bool(
                channel is not None and speaker is not None
                and connection.execute(
                    "SELECT 1 FROM admins WHERE member_id = ?", (speaker,)
                ).fetchone() is not None
            )
            asserted_as_delegate = bool(
                channel is not None and memory_key is not None
                and fact_subject is None and speaker is not None
                and connection.execute(
                    """SELECT 1 FROM delegations WHERE channel = ? AND memory_key = ?
                    AND member_id = ? AND revoked_at IS NULL""",
                    (channel, memory_key, speaker),
                ).fetchone() is not None
            )
            if memory_key is not None:
                heads = connection.execute(
                    """
                    SELECT id, speaker, content, archived_at, asserted_as_admin,
                           asserted_as_delegate, fact_subject
                    FROM memories
                    WHERE memory_key = ? AND superseded_by IS NULL
                      AND channel IS ?
                    """,
                    (memory_key, channel),
                ).fetchall()
                if channel is not None and connection.execute(
                    """SELECT 1 FROM memories WHERE memory_key = ? AND channel = ?
                    AND fact_subject IS NOT ? LIMIT 1""",
                    (memory_key, channel, fact_subject),
                ).fetchone():
                    raise ValueError("claim scope or personal subject differs from key history")
                if channel is None:
                    predecessors = heads
                else:
                    incoming = {
                        "speaker": speaker, "fact_subject": fact_subject,
                        "asserted_as_admin": asserted_as_admin,
                        "asserted_as_delegate": asserted_as_delegate,
                    }
                    rank = self._claim_rank(incoming)
                    if any(
                        head["speaker"] == speaker
                        and head["archived_at"] is None
                        and self._claim_rank(head) > rank
                        for head in heads
                    ):
                        raise ValueError(
                            "a lapsed authority cannot revise its earlier claim"
                        )
                    other_active = [head for head in heads
                                    if head["speaker"] != speaker
                                    and head["archived_at"] is None]
                    other_max = max((self._claim_rank(head) for head in other_active), default=-1)
                    restatement = any(
                        head["speaker"] == speaker and head["archived_at"] is None
                        and self._claim_rank(head) == rank for head in heads
                    )
                    resolve_all = rank > 0 and (
                        rank > other_max or (rank == other_max and restatement)
                    )
                    predecessors = [
                        head for head in heads
                        if head["speaker"] == speaker or
                        (resolve_all and self._claim_rank(head) <= rank)
                    ]
                    predecessor_ids = {head["id"] for head in predecessors}
                    conflicts = [
                        head for head in heads
                        if head["id"] not in predecessor_ids
                        and head["archived_at"] is None
                        and head["content"].strip().casefold()
                        != content.strip().casefold()
                    ]
                for previous in predecessors:
                    if previous["archived_at"] is None:
                        connection.execute(
                            "UPDATE memories SET archived_at = ?, updated_at = ? WHERE id = ?",
                            (now, now, previous["id"]),
                        )

            connection.execute(
                """
                INSERT INTO memories (
                    id, kind, content, memory_key, source, confidence,
                    importance, created_at, updated_at, channel, speaker,
                    asserted_as_admin, fact_subject, asserted_as_delegate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    memory_type.value,
                    content.strip(),
                    memory_key,
                    source.strip(),
                    confidence,
                    importance,
                    now,
                    now,
                    channel,
                    speaker,
                    int(asserted_as_admin),
                    fact_subject,
                    int(asserted_as_delegate),
                ),
            )
            if self.embedder is not None:
                self._store_vector(connection, memory_id, vector)

            for previous in predecessors:
                self._record_supersession(connection, memory_id, previous["id"], now)
            for other in conflicts:
                connection.execute(
                    """
                    INSERT INTO relationships (source_id, target_id, relation, created_at)
                    VALUES (?, ?, 'contradicts', ?)
                    ON CONFLICT DO NOTHING
                    """,
                    (memory_id, other["id"], now),
                )

            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            return self._memory_from_row(row)

    def _record_supersession(
        self, connection: sqlite3.Connection, new_id: str, old_id: str, now: str
    ) -> None:
        connection.execute(
            "UPDATE memories SET superseded_by = ?, updated_at = ? WHERE id = ?",
            (new_id, now, old_id),
        )
        if self.link_carryover is LinkCarryover.COPY:
            connection.execute(
                """
                INSERT INTO relationships (source_id, target_id, relation, created_at)
                SELECT
                    CASE WHEN source_id = :old THEN :new ELSE source_id END,
                    CASE WHEN target_id = :old THEN :new ELSE target_id END,
                    relation,
                    :now
                FROM relationships
                WHERE (source_id = :old OR target_id = :old)
                  AND relation != 'supersedes'
                  AND source_id != :new AND target_id != :new
                ON CONFLICT(source_id, target_id, relation) DO NOTHING
                """,
                {"old": old_id, "new": new_id, "now": now},
            )
        connection.execute(
            """
            INSERT INTO relationships (source_id, target_id, relation, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relation) DO NOTHING
            """,
            (new_id, old_id, Relationship.SUPERSEDES.value, now),
        )

    def relate(
        self,
        source_id: str,
        target_id: str,
        relation: Union[Relationship, str] = Relationship.RELATED_TO,
    ) -> None:
        """Link two active memories; the link reads ``source relation target``.

        ``supersedes`` archives the target and records the source as its
        replacement. A source without a key takes over the target's key.
        Both must be in the same channel. Within a channel, the source must
        have the same speaker or higher write-time authority for that claim.
        """
        try:
            relation_type = Relationship(relation)
        except ValueError as error:
            raise ValueError("unsupported relationship: {}".format(relation)) from error
        if source_id == target_id:
            raise ValueError("a memory cannot be related to itself")

        now = self._now()
        with self._transaction() as connection:
            rows = {
                row["id"]: row
                for row in connection.execute(
                    """
                    SELECT id, memory_key, created_at, rowid, channel, speaker,
                           asserted_as_admin, asserted_as_delegate, fact_subject
                    FROM memories
                    WHERE id IN (?, ?) AND archived_at IS NULL
                    """,
                    (source_id, target_id),
                ).fetchall()
            }
            if set(rows) != {source_id, target_id}:
                raise ValueError("both memories must exist and be active")

            if relation_type is Relationship.SUPERSEDES:
                newer, older = rows[source_id], rows[target_id]
                if newer["channel"] != older["channel"]:
                    raise ValueError(
                        "a memory can only supersede one in the same channel"
                    )
                if newer["channel"] is not None and (
                    newer["fact_subject"] != older["fact_subject"]
                    or (newer["memory_key"] is not None
                        and older["memory_key"] is not None
                        and newer["memory_key"] != older["memory_key"])
                ):
                    raise ValueError("supersession requires the same claim scope and key")
                if (
                    newer["channel"] is not None
                    and (
                        self._claim_rank(newer) < self._claim_rank(older)
                        or (newer["speaker"] != older["speaker"]
                            and self._claim_rank(newer) == self._claim_rank(older))
                    )
                ):
                    raise ValueError(
                        "only a same-speaker claim with retained authority or a higher-authority holder can supersede a channel claim"
                    )
                # Older databases store whole seconds; insertion order breaks ties.
                if (newer["created_at"], newer["rowid"]) < (
                    older["created_at"],
                    older["rowid"],
                ):
                    raise ValueError(
                        "a memory can only supersede an older one; "
                        "add the content again as a new memory instead"
                    )
                connection.execute(
                    "UPDATE memories SET archived_at = ?, updated_at = ? WHERE id = ?",
                    (now, now, target_id),
                )
                if newer["memory_key"] is None and older["memory_key"] is not None:
                    connection.execute(
                        "UPDATE memories SET memory_key = ?, updated_at = ? WHERE id = ?",
                        (older["memory_key"], now, source_id),
                    )
                self._record_supersession(connection, source_id, target_id, now)
                return

            connection.execute(
                """
                INSERT INTO relationships (source_id, target_id, relation, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, relation) DO NOTHING
                """,
                (source_id, target_id, relation_type.value, now),
            )

    def get_memory(
        self,
        memory_id: str,
        *,
        viewers: Optional[Union[str, Iterable[str]]] = None,
    ) -> Optional[Memory]:
        """Return a memory, or ``None`` if it is missing or hidden from ``viewers``."""
        visible, visible_parameters = self._visible(self._read_viewers(viewers))
        with self._transaction(write=False) as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?" + visible,
                [memory_id] + visible_parameters,
            ).fetchone()
            return self._memory_from_row(row) if row is not None else None

    def history(
        self,
        memory_id: str,
        *,
        viewers: Optional[Union[str, Iterable[str]]] = None,
    ) -> list[Memory]:
        """Return every version of a memory, newest first.

        Works from any version: it follows ``superseded_by`` forward to the
        current version, then collects everything that version replaced,
        directly or indirectly. With ``viewers``, a memory hidden from them
        has no history and hidden versions are left out.
        """
        visible, visible_parameters = self._visible(self._read_viewers(viewers))
        with self._transaction(write=False) as connection:
            row = connection.execute(
                "SELECT id, superseded_by FROM memories WHERE id = ?" + visible,
                [memory_id] + visible_parameters,
            ).fetchone()
            if row is None:
                return []
            seen = {row["id"]}
            while row["superseded_by"] is not None and row["superseded_by"] not in seen:
                newer = connection.execute(
                    "SELECT id, superseded_by FROM memories WHERE id = ?" + visible,
                    [row["superseded_by"]] + visible_parameters,
                ).fetchone()
                if newer is None:
                    break
                seen.add(newer["id"])
                row = newer

            rows = connection.execute(
                """
                WITH RECURSIVE versions(id) AS (
                    SELECT ?
                    UNION
                    SELECT memories.id FROM memories
                    JOIN versions ON memories.superseded_by = versions.id{0}
                )
                SELECT memories.* FROM memories
                JOIN versions ON memories.id = versions.id{0}
                ORDER BY memories.created_at DESC, memories.rowid DESC
                """.format(visible),
                [row["id"]] + visible_parameters * 2,
            ).fetchall()
            return [self._memory_from_row(version) for version in rows]

    def restore(self, memory_id: str) -> Memory:
        """Reactivate a memory that pruning archived.

        Restoring resets the last-access time, so the memory gets a fresh
        ``min_age_days`` window before pruning considers it again.
        """
        now = self._now()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise ValueError("memory not found: {}".format(memory_id))
            if row["archived_at"] is None:
                raise ValueError("memory is already active: {}".format(memory_id))
            if row["superseded_by"] is not None:
                raise ValueError(
                    "memory was superseded by {} and cannot be restored".format(
                        row["superseded_by"]
                    )
                )
            if row["memory_key"] is not None:
                holder = connection.execute(
                    """
                    SELECT id FROM memories
                    WHERE memory_key = ? AND archived_at IS NULL
                      AND channel IS ?
                      AND (channel IS NULL OR speaker IS ?)
                    """,
                    (row["memory_key"], row["channel"], row["speaker"]),
                ).fetchone()
                if holder is not None:
                    raise ValueError(
                        "key {} is held by active memory {}".format(
                            row["memory_key"], holder["id"]
                        )
                    )
            connection.execute(
                """
                UPDATE memories
                SET archived_at = NULL, updated_at = ?, last_accessed_at = ?
                WHERE id = ?
                """,
                (now, now, memory_id),
            )
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            return self._memory_from_row(row)

    def list_memories(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        viewers: Optional[Union[str, Iterable[str]]] = None,
    ) -> list[Memory]:
        """The most recently updated memories, only visible ones with ``viewers``."""
        if limit < 1:
            raise ValueError("limit must be positive")
        visible, visible_parameters = self._visible(self._read_viewers(viewers))
        clause = "1" if include_archived else "archived_at IS NULL"
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE {}{}
                ORDER BY updated_at DESC, id
                LIMIT ?
                """.format(clause, visible),
                visible_parameters + [limit],
            ).fetchall()
            return [self._memory_from_row(row) for row in rows]

    def search(
        self,
        query: str,
        *,
        kind: Optional[Union[MemoryType, str]] = None,
        limit: int = 10,
        record_access: bool = True,
        viewers: Optional[Union[str, Iterable[str]]] = None,
    ) -> list[SearchResult]:
        """Rank active memories for a query.

        Each result's ``links`` lists its relationships to the other results,
        the links behind its graph score, and every ``contradicts`` link to an
        active memory, even one cut off by ``limit`` or filtered out by
        ``kind``. A disputed memory therefore always shows that it is disputed.

        ``viewers`` names the people who will see the answer (one member id or
        several). The search then runs as if the store held only the memories
        in channels that every viewer currently belongs to. Hidden memories are
        left out before ranking, not after it, so they change no score, add no
        graph neighbour and appear in no link. A memory without a channel is
        hidden from every viewer.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be positive")
        kind_value = None
        if kind is not None:
            try:
                kind_value = MemoryType(kind).value
            except ValueError as error:
                raise ValueError("unsupported memory type: {}".format(kind)) from error
        viewer_ids = self._read_viewers(viewers)
        visible, visible_parameters = self._visible(viewer_ids)

        query_tokens = self._TOKEN_PATTERN.findall(query.lower())
        query_vector = None
        if self.embedder is not None:
            self._embed_missing(viewer_ids)
            query_vector = self._embed([query.strip()])[0]
        if not query_tokens and query_vector is None:
            return []

        with self._transaction(write=record_access) as connection:
            sql = "SELECT * FROM memories WHERE archived_at IS NULL" + visible
            parameters = list(visible_parameters)
            if kind_value is not None:
                sql += " AND kind = ?"
                parameters.append(kind_value)
            active_memories = {
                row["id"]: self._memory_from_row(row)
                for row in connection.execute(sql, parameters).fetchall()
            }
            if not active_memories:
                return []

            lexical_scores, coverage = self._lexical_scores(active_memories, query_tokens)
            vector_scores = {}
            if query_vector is not None:
                vector_scores = self._vector_scores(
                    connection, active_memories, query_vector, viewer_ids
                )
            if self.embedder is None:
                direct_scores = lexical_scores
            else:
                direct_scores = self._fuse_ranks(
                    [
                        self._ranked(lexical_scores, active_memories),
                        self._ranked(vector_scores, active_memories),
                    ]
                )
            if not direct_scores:
                return []

            maximum_direct_score = max(direct_scores.values())
            seed_scores = {
                memory_id: score / maximum_direct_score
                for memory_id, score in direct_scores.items()
            }
            seed_links = self._search_links(connection, seed_scores, viewer_ids)
            graph_scores = {}
            for link in seed_links:
                for seed_id, neighbor_id in (
                    (link.source_id, link.target_id),
                    (link.target_id, link.source_id),
                ):
                    if seed_id not in seed_scores or neighbor_id not in active_memories:
                        continue
                    graph_score = (
                        seed_scores[seed_id] * self._RELATION_WEIGHTS[link.relation.value]
                    )
                    graph_scores[neighbor_id] = max(
                        graph_scores.get(neighbor_id, 0.0), graph_score
                    )

            result_scores = {}
            for memory_id in set(seed_scores) | set(graph_scores):
                seed_score = seed_scores.get(memory_id, 0.0)
                graph_score = graph_scores.get(memory_id, 0.0)
                if seed_score:
                    result_scores[memory_id] = min(
                        1.35, seed_score + graph_score * 0.35
                    )
                else:
                    result_scores[memory_id] = graph_score * 0.65

            ranked_ids = self._ranked(result_scores, active_memories)[:limit]
            now = self._now()
            if record_access:
                for memory_id in ranked_ids:
                    connection.execute(
                        """
                        UPDATE memories
                        SET access_count = access_count + 1, last_accessed_at = ?
                        WHERE id = ?
                        """,
                        (now, memory_id),
                    )

            links_by_memory = {memory_id: set() for memory_id in ranked_ids}
            result_links = self._search_links(connection, ranked_ids, viewer_ids)
            active_disputants = self._active_ids(
                connection,
                {
                    end
                    for link in result_links
                    if link.relation is Relationship.CONTRADICTS
                    for end in (link.source_id, link.target_id)
                    if end not in links_by_memory
                },
                viewer_ids,
            )
            for link in result_links:
                for memory_id, other_id in (
                    (link.source_id, link.target_id),
                    (link.target_id, link.source_id),
                ):
                    if memory_id in links_by_memory and (
                        other_id in links_by_memory
                        or (
                            link.relation is Relationship.CONTRADICTS
                            and other_id in active_disputants
                        )
                    ):
                        links_by_memory[memory_id].add(link)
            for link in seed_links:
                for memory_id, other_id in (
                    (link.source_id, link.target_id),
                    (link.target_id, link.source_id),
                ):
                    if (
                        memory_id in links_by_memory
                        and memory_id in graph_scores
                        and other_id in seed_scores
                    ):
                        links_by_memory[memory_id].add(link)

            claim_statuses = self._claim_statuses(connection, viewer_ids)
            results = []
            for memory_id in ranked_ids:
                memory = active_memories[memory_id]
                matched_by = "+".join(
                    name
                    for name, scores in (
                        ("lexical", lexical_scores),
                        ("vector", vector_scores),
                        ("graph", graph_scores),
                    )
                    if memory_id in scores
                )
                results.append(
                    SearchResult(
                        memory=(
                            replace(
                                memory,
                                access_count=memory.access_count + 1,
                                last_accessed_at=now,
                            )
                            if record_access
                            else memory
                        ),
                        score=result_scores[memory_id],
                        matched_by=matched_by,
                        links=tuple(
                            sorted(
                                links_by_memory[memory_id],
                                key=lambda link: (
                                    link.relation.value,
                                    link.source_id,
                                    link.target_id,
                                ),
                            )
                        ),
                        lexical_score=lexical_scores.get(memory_id, 0.0),
                        vector_score=vector_scores.get(memory_id, 0.0),
                        coverage=coverage.get(memory_id, 0.0),
                        claim_status=claim_statuses.get(memory_id, "current"),
                    )
                )
            return results

    def _claim_statuses(
        self, connection: sqlite3.Connection,
        viewers: Optional[tuple[str, ...]],
    ) -> dict[str, str]:
        """Classify active keyed claims using all visible kinds in each channel."""
        visible, parameters = self._visible(viewers)
        groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for row in connection.execute(
            """
            SELECT id, channel, memory_key, speaker, content,
                   asserted_as_admin, asserted_as_delegate, fact_subject FROM memories
            WHERE archived_at IS NULL AND channel IS NOT NULL
              AND memory_key IS NOT NULL{}
            """.format(visible),
            parameters,
        ):
            groups.setdefault((row["channel"], row["memory_key"]), []).append(row)
        statuses = {}
        for claims in groups.values():
            values = {claim["content"].strip().casefold() for claim in claims}
            max_rank = max(self._claim_rank(claim) for claim in claims)
            authority_values = {
                claim["content"].strip().casefold() for claim in claims
                if self._claim_rank(claim) == max_rank
            }
            for claim in claims:
                value = claim["content"].strip().casefold()
                if max_rank > 0 and len(authority_values) == 1:
                    if self._claim_rank(claim) == max_rank:
                        status = "current"
                    elif value in authority_values:
                        status = "agrees"
                    else:
                        status = "overruled"
                elif len(values) == 1:
                    status = "current"
                else:
                    status = "disputed"
                statuses[claim["id"]] = status
        return statuses

    def _lexical_scores(
        self, memories: Mapping[str, Memory], query_tokens: Sequence[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """BM25-style scores and query coverage of memories sharing a query token.

        Coverage is the inverse-document-frequency mass of the distinct query
        tokens a memory contains, as a share of the mass of all of them. A rare
        query word that no memory contains still counts in the total, so a
        question about something never mentioned cannot be fully covered.
        """
        if not query_tokens:
            return {}, {}
        tokenized = {
            memory.id: self._TOKEN_PATTERN.findall(
                "{} {}".format(memory.content, memory.memory_key or "").lower()
            )
            for memory in memories.values()
        }
        document_frequency = {}
        for tokens in tokenized.values():
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        average_length = sum(map(len, tokenized.values())) / float(len(memories))
        inverse_document_frequency = {
            token: math.log(
                1
                + (len(memories) - document_frequency.get(token, 0) + 0.5)
                / (document_frequency.get(token, 0) + 0.5)
            )
            # Sorted, so the sums below do not depend on the hash seed.
            for token in sorted(set(query_tokens))
        }
        total_mass = sum(inverse_document_frequency.values())
        scores = {}
        coverage = {}
        for memory_id, terms in tokenized.items():
            term_frequency = {}
            for term in terms:
                term_frequency[term] = term_frequency.get(term, 0) + 1

            score = 0.0
            for token in query_tokens:
                frequency = term_frequency.get(token, 0)
                if not frequency:
                    continue
                length_factor = 1.2 * (
                    0.25 + 0.75 * len(terms) / max(average_length, 1.0)
                )
                score += inverse_document_frequency[token] * (
                    frequency * 2.2 / (frequency + length_factor)
                )
            if score > 0:
                scores[memory_id] = score
                coverage[memory_id] = (
                    sum(
                        mass
                        for token, mass in inverse_document_frequency.items()
                        if token in term_frequency
                    )
                    / total_mass
                )
        return scores, coverage

    def _vector_scores(
        self,
        connection: sqlite3.Connection,
        memories: Mapping[str, Memory],
        query_vector: array,
        viewers: Optional[tuple[str, ...]] = None,
    ) -> dict[str, float]:
        """Cosine similarities that reach the embedder's ``min_similarity``."""
        visible, visible_parameters = self._visible(viewers)
        scores = {}
        for row in connection.execute(
            """
            SELECT embeddings.memory_id, embeddings.vector FROM embeddings
            JOIN memories ON memories.id = embeddings.memory_id
            WHERE embeddings.model = ? AND memories.archived_at IS NULL{}
            """.format(visible),
            [self.embedder.name] + visible_parameters,
        ):
            if row["memory_id"] not in memories or not row["vector"]:
                continue
            vector = array("f")
            vector.frombytes(row["vector"])
            if len(vector) != len(query_vector):
                continue
            similarity = sum(left * right for left, right in zip(query_vector, vector))
            if similarity >= self.embedder.min_similarity:
                scores[row["memory_id"]] = similarity
        return scores

    @staticmethod
    def _ranked(scores: Mapping[str, float], memories: Mapping[str, Memory]) -> list[str]:
        """Order by score; equal scores put the newer memory first."""
        return sorted(
            scores,
            key=lambda memory_id: (
                scores[memory_id],
                memories[memory_id].created_at,
                memory_id,
            ),
            reverse=True,
        )

    def _fuse_ranks(self, rankings: Sequence[Sequence[str]]) -> dict[str, float]:
        fused = {}
        for ranking in rankings:
            for rank, memory_id in enumerate(ranking, start=1):
                fused[memory_id] = fused.get(memory_id, 0.0) + 1.0 / (self._RRF_K + rank)
        return fused

    def _active_ids(
        self,
        connection: sqlite3.Connection,
        memory_ids: Iterable[str],
        viewers: Optional[tuple[str, ...]] = None,
    ) -> set[str]:
        memory_ids = list(memory_ids)
        visible, visible_parameters = self._visible(viewers)
        active = set()
        for start in range(0, len(memory_ids), self._ID_CHUNK):
            chunk = memory_ids[start:start + self._ID_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            active.update(
                row["id"]
                for row in connection.execute(
                    """
                    SELECT id FROM memories
                    WHERE archived_at IS NULL AND id IN ({}){}
                    """.format(placeholders, visible),
                    chunk + visible_parameters,
                )
            )
        return active

    def _search_links(
        self,
        connection: sqlite3.Connection,
        memory_ids: Iterable[str],
        viewers: Optional[tuple[str, ...]] = None,
    ) -> list[Link]:
        """Links touching the given memories, as search should see them.

        With ``topic`` carryover, a link to an earlier version of a fact counts
        as a link to its current version. With ``viewers``, a link counts only
        if both of its ends are visible, and so does every version it is
        followed through.
        """
        memory_ids = list(memory_ids)
        if self.link_carryover is not LinkCarryover.TOPIC:
            return self._without_resolved_contradictions(
                connection, self._links_touching(connection, memory_ids, viewers)
            )
        links = self._links_touching(
            connection,
            memory_ids
            + sorted(self._earlier_versions(connection, memory_ids, viewers)),
            viewers,
        )
        current = self._current_versions(
            connection,
            {end for link in links for end in (link.source_id, link.target_id)},
            viewers,
        )
        resolved = set()
        for link in links:
            # A version's own supersedes link resolves to a self-link and drops out.
            source_id = current.get(link.source_id, link.source_id)
            target_id = current.get(link.target_id, link.target_id)
            if source_id != target_id:
                resolved.add(Link(source_id, target_id, link.relation))
        return self._without_resolved_contradictions(
            connection,
            sorted(
                resolved,
                key=lambda link: (link.source_id, link.target_id, link.relation.value),
            ),
        )

    def _without_resolved_contradictions(
        self, connection: sqlite3.Connection, links: Sequence[Link]
    ) -> list[Link]:
        """A carried contradiction no longer applies when current claims agree."""
        ends = {
            end
            for link in links
            if link.relation is Relationship.CONTRADICTS
            for end in (link.source_id, link.target_id)
        }
        if not ends:
            return list(links)
        active = {}
        end_ids = sorted(ends)
        for start in range(0, len(end_ids), self._ID_CHUNK):
            chunk = end_ids[start:start + self._ID_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            active.update(
                (row["id"], row)
                for row in connection.execute(
                    """
                    SELECT id, channel, memory_key, content FROM memories
                    WHERE archived_at IS NULL AND id IN ({})
                    """.format(placeholders),
                    chunk,
                )
            )
        return [
            link
            for link in links
            if not (
                link.relation is Relationship.CONTRADICTS
                and link.source_id in active
                and link.target_id in active
                and active[link.source_id]["channel"] is not None
                and active[link.source_id]["channel"] == active[link.target_id]["channel"]
                and active[link.source_id]["memory_key"] is not None
                and active[link.source_id]["memory_key"]
                == active[link.target_id]["memory_key"]
                and active[link.source_id]["content"].strip().casefold()
                == active[link.target_id]["content"].strip().casefold()
            )
        ]

    def _earlier_versions(
        self,
        connection: sqlite3.Connection,
        memory_ids: Sequence[str],
        viewers: Optional[tuple[str, ...]] = None,
    ) -> set[str]:
        """Every memory that the given memories replaced, directly or not."""
        visible, visible_parameters = self._visible(viewers)
        earlier = set()
        for start in range(0, len(memory_ids), self._ID_CHUNK):
            chunk = tuple(memory_ids[start:start + self._ID_CHUNK])
            placeholders = ",".join("?" for _ in chunk)
            earlier.update(
                row["id"]
                for row in connection.execute(
                    """
                    WITH RECURSIVE versions(id) AS (
                        SELECT id FROM memories
                        WHERE superseded_by IN ({0}){1}
                        UNION
                        SELECT memories.id FROM memories
                        JOIN versions ON memories.superseded_by = versions.id{1}
                    )
                    SELECT id FROM versions
                    """.format(placeholders, visible),
                    chunk + tuple(visible_parameters) * 2,
                )
            )
        return earlier

    def _current_versions(
        self,
        connection: sqlite3.Connection,
        memory_ids: Iterable[str],
        viewers: Optional[tuple[str, ...]] = None,
    ) -> dict[str, str]:
        """Map each replaced memory to the newest version of its fact.

        With ``viewers``, a chain that passes through a hidden version maps to
        nothing.
        """
        memory_ids = list(memory_ids)
        visible, visible_parameters = self._visible(viewers)
        current = {}
        for start in range(0, len(memory_ids), self._ID_CHUNK):
            chunk = tuple(memory_ids[start:start + self._ID_CHUNK])
            placeholders = ",".join("?" for _ in chunk)
            current.update(
                (row["start_id"], row["id"])
                for row in connection.execute(
                    """
                    WITH RECURSIVE chain(start_id, id) AS (
                        SELECT id, superseded_by FROM memories
                        WHERE id IN ({0}) AND superseded_by IS NOT NULL{1}
                        UNION
                        SELECT chain.start_id, memories.superseded_by FROM chain
                        JOIN memories ON memories.id = chain.id
                        WHERE memories.superseded_by IS NOT NULL{1}
                    )
                    SELECT chain.start_id, chain.id FROM chain
                    JOIN memories ON memories.id = chain.id
                    WHERE memories.superseded_by IS NULL{1}
                    """.format(placeholders, visible),
                    chunk + tuple(visible_parameters) * 3,
                )
            )
        return current

    def _links_touching(
        self,
        connection: sqlite3.Connection,
        memory_ids: Iterable[str],
        viewers: Optional[tuple[str, ...]] = None,
    ) -> list[Link]:
        memory_ids = list(memory_ids)
        visible, visible_parameters = self._visible(viewers)
        both_ends_visible = ""
        if viewers is not None:
            both_ends_visible = "".join(
                " AND EXISTS (SELECT 1 FROM memories "
                "WHERE memories.id = relationships.{}{})".format(end, visible)
                for end in ("source_id", "target_id")
            )
        links = set()
        for start in range(0, len(memory_ids), self._ID_CHUNK):
            chunk = tuple(memory_ids[start:start + self._ID_CHUNK])
            placeholders = ",".join("?" for _ in chunk)
            for row in connection.execute(
                """
                SELECT source_id, target_id, relation FROM relationships
                WHERE (source_id IN ({0}) OR target_id IN ({0})){1}
                """.format(placeholders, both_ends_visible),
                chunk + chunk + tuple(visible_parameters) * 2,
            ):
                links.add(
                    Link(
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        relation=Relationship(row["relation"]),
                    )
                )
        return sorted(
            links,
            key=lambda link: (link.source_id, link.target_id, link.relation.value),
        )

    def _embed(self, texts: Sequence[str]) -> list[Optional[array]]:
        vectors = []
        for start in range(0, len(texts), self._EMBED_BATCH):
            batch = list(texts[start:start + self._EMBED_BATCH])
            embedded = list(self.embedder(batch))
            if len(embedded) != len(batch):
                raise ValueError(
                    "embedder returned {} vectors for {} texts".format(
                        len(embedded), len(batch)
                    )
                )
            vectors.extend(self._unit_vector(values) for values in embedded)
        return vectors

    @staticmethod
    def _unit_vector(values: Iterable[float]) -> Optional[array]:
        vector = array("f", (float(value) for value in values))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0 or not math.isfinite(norm):
            return None
        return array("f", (value / norm for value in vector))

    def _store_vector(
        self,
        connection: sqlite3.Connection,
        memory_id: str,
        vector: Optional[array],
    ) -> None:
        # An empty blob records that the text has no usable vector, so it is
        # not re-embedded on every search.
        connection.execute(
            """
            INSERT INTO embeddings (memory_id, model, vector) VALUES (?, ?, ?)
            ON CONFLICT(memory_id, model) DO UPDATE SET vector = excluded.vector
            """,
            (
                memory_id,
                self.embedder.name,
                vector.tobytes() if vector is not None else b"",
            ),
        )

    def _embed_missing(self, viewers: Optional[tuple[str, ...]] = None) -> None:
        """Embed only active memories visible to this search's viewers."""
        visible, visible_parameters = self._visible(viewers)
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT id, content FROM memories
                WHERE archived_at IS NULL{} AND id NOT IN (
                    SELECT memory_id FROM embeddings WHERE model = ?
                )
                """.format(visible),
                visible_parameters + [self.embedder.name],
            ).fetchall()
        if not rows:
            return
        vectors = self._embed([row["content"] for row in rows])
        with self._transaction() as connection:
            for row, vector in zip(rows, vectors):
                self._store_vector(connection, row["id"], vector)

    def _set_activity(
        self,
        memory_id: str,
        *,
        created_at: str,
        last_accessed_at: Optional[str] = None,
        access_count: int = 0,
    ) -> None:
        """Backdate a memory's activity; used by evaluation and tests."""
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE memories
                SET created_at = ?, updated_at = ?, last_accessed_at = ?,
                    access_count = ?
                WHERE id = ?
                """,
                (created_at, created_at, last_accessed_at, access_count, memory_id),
            )

    def prune(
        self,
        *,
        threshold: float = 0.1,
        min_age_days: float = 30,
        decay_days: float = 90,
        decay_days_by_kind: Optional[Mapping[Union[MemoryType, str], float]] = None,
        protect_importance: Optional[float] = 0.9,
        dry_run: bool = True,
        as_of: Optional[datetime] = None,
    ) -> list[PruneCandidate]:
        """Find (and unless ``dry_run``, archive) stale low-value memories.

        ``decay_days_by_kind`` overrides ``decay_days`` for specific memory
        types. Memories at or above ``protect_importance`` are never pruned;
        pass ``None`` to protect nothing.
        """
        self._validate_unit_interval(threshold, "threshold")
        if min_age_days < 0 or not math.isfinite(min_age_days):
            raise ValueError("min_age_days must be a finite non-negative number")
        if decay_days <= 0 or not math.isfinite(decay_days):
            raise ValueError("decay_days must be a finite positive number")
        kind_decay_days = {}
        for kind, days in (decay_days_by_kind or {}).items():
            try:
                kind_value = MemoryType(kind).value
            except ValueError as error:
                raise ValueError("unsupported memory type: {}".format(kind)) from error
            if days <= 0 or not math.isfinite(days):
                raise ValueError(
                    "decay days for {} must be a finite positive number".format(kind_value)
                )
            kind_decay_days[kind_value] = days
        if protect_importance is not None:
            self._validate_unit_interval(protect_importance, "protect_importance")

        current_time = as_of or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        now = current_time.astimezone(timezone.utc).isoformat(timespec="microseconds")
        candidates = []

        with self._transaction(write=not dry_run) as connection:
            rows = connection.execute(
                "SELECT * FROM memories WHERE archived_at IS NULL"
            ).fetchall()
            for row in rows:
                memory = self._memory_from_row(row)
                if protect_importance is not None and memory.importance >= protect_importance:
                    continue
                activity_time = datetime.fromisoformat(
                    memory.last_accessed_at or memory.created_at
                )
                age_days = max(
                    0.0,
                    (current_time - activity_time).total_seconds() / 86400.0,
                )
                if age_days < min_age_days:
                    continue
                score = retention_score(
                    memory.importance,
                    memory.confidence,
                    age_days,
                    memory.access_count,
                    kind_decay_days.get(memory.kind.value, decay_days),
                )
                if score < threshold:
                    candidates.append(PruneCandidate(memory=memory, score=score))

            candidates.sort(key=lambda candidate: (candidate.score, candidate.memory.id))
            if candidates and not dry_run:
                connection.executemany(
                    "UPDATE memories SET archived_at = ?, updated_at = ? WHERE id = ?",
                    [
                        (now, now, candidate.memory.id)
                        for candidate in candidates
                    ],
                )
                candidates = [
                    replace(
                        candidate,
                        memory=replace(
                            candidate.memory, archived_at=now, updated_at=now
                        ),
                    )
                    for candidate in candidates
                ]

        return candidates
