"""SQLite persistence with versioned migrations.

A single database file holds all JARVIS state. SQLite is used because it is
embedded, cross-platform, transactional and needs no separate service, which is
what a personal assistant that must work offline on any OS requires.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MIGRATIONS: list[str] = [
    # 1 - foundation
    """
    CREATE TABLE devices (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,              -- cli | desktop | web | mobile | hardware | core
        role TEXT NOT NULL,              -- owner | client | hardware
        platform TEXT,
        token_hash TEXT NOT NULL UNIQUE,
        max_risk INTEGER NOT NULL DEFAULT 3,
        capabilities_json TEXT NOT NULL DEFAULT '[]',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at REAL NOT NULL,
        last_seen REAL,
        revoked_at REAL
    );
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        device_id TEXT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
        token_hash TEXT NOT NULL UNIQUE,
        created_at REAL NOT NULL,
        expires_at REAL NOT NULL,
        revoked INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE pairing_codes (
        code_hash TEXT PRIMARY KEY,
        created_by TEXT,
        kind TEXT NOT NULL,
        role TEXT NOT NULL,
        max_risk INTEGER NOT NULL,
        name_hint TEXT,
        created_at REAL NOT NULL,
        expires_at REAL NOT NULL,
        used_at REAL
    );
    CREATE TABLE audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        actor TEXT,
        action TEXT NOT NULL,
        target TEXT,
        risk TEXT,
        outcome TEXT NOT NULL,
        details_json TEXT NOT NULL DEFAULT '{}',
        trace_id TEXT
    );
    CREATE INDEX audit_ts ON audit_log(ts);
    CREATE TABLE events (
        id TEXT PRIMARY KEY,
        seq INTEGER NOT NULL,
        ts REAL NOT NULL,
        type TEXT NOT NULL,
        source TEXT,
        data_json TEXT NOT NULL,
        trace_id TEXT
    );
    CREATE INDEX events_seq ON events(seq);
    CREATE TABLE model_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        role TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        latency_ms REAL,
        ok INTEGER NOT NULL,
        error TEXT,
        trace_id TEXT
    );
    CREATE TABLE kv (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at REAL NOT NULL
    );
    """,
    # 2 - conversations, approvals, grants
    """
    CREATE TABLE conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        device_id TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        summarized INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        tool_calls_json TEXT,
        tool_call_id TEXT,
        name TEXT,
        meta_json TEXT NOT NULL DEFAULT '{}',
        created_at REAL NOT NULL
    );
    CREATE INDEX messages_conv ON messages(conversation_id, seq);
    CREATE TABLE approvals (
        id TEXT PRIMARY KEY,
        tool TEXT NOT NULL,
        capability TEXT,
        risk TEXT NOT NULL,
        summary TEXT NOT NULL,
        reason TEXT,
        effects TEXT,
        args_json TEXT NOT NULL,
        requested_by TEXT,
        conversation_id TEXT,
        task_id TEXT,
        status TEXT NOT NULL,           -- pending | approved | denied | expired | cancelled
        decided_by TEXT,
        decision_scope TEXT,
        requested_at REAL NOT NULL,
        decided_at REAL
    );
    CREATE TABLE grants (
        id TEXT PRIMARY KEY,
        tool TEXT NOT NULL,
        scope_type TEXT NOT NULL,       -- conversation | task | global
        scope_id TEXT,
        granted_by TEXT,
        created_at REAL NOT NULL,
        expires_at REAL
    );
    """,
    # 3 - memory
    """
    CREATE TABLE memories (
        id TEXT PRIMARY KEY,
        layer TEXT NOT NULL,            -- long_term | episodic | semantic
        kind TEXT NOT NULL DEFAULT 'fact',
        content TEXT NOT NULL,
        tags TEXT NOT NULL DEFAULT '',
        source TEXT,
        importance REAL NOT NULL DEFAULT 0.5,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        last_accessed REAL,
        access_count INTEGER NOT NULL DEFAULT 0,
        embedding BLOB,
        embedding_model TEXT,
        hlc TEXT NOT NULL,
        origin TEXT NOT NULL
    );
    CREATE INDEX memories_layer ON memories(layer);
    CREATE VIRTUAL TABLE memories_fts USING fts5(content, tags, content='memories', content_rowid='rowid', tokenize='porter unicode61');
    CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, content, tags) VALUES (new.rowid, new.content, new.tags);
    END;
    CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content, tags) VALUES ('delete', old.rowid, old.content, old.tags);
    END;
    CREATE TRIGGER memories_au AFTER UPDATE OF content, tags ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content, tags) VALUES ('delete', old.rowid, old.content, old.tags);
        INSERT INTO memories_fts(rowid, content, tags) VALUES (new.rowid, new.content, new.tags);
    END;
    CREATE TABLE working_memory (
        scope TEXT NOT NULL,
        key TEXT NOT NULL,
        value_json TEXT NOT NULL,
        updated_at REAL NOT NULL,
        expires_at REAL,
        PRIMARY KEY (scope, key)
    );
    CREATE TABLE preferences (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        hlc TEXT NOT NULL,
        origin TEXT NOT NULL,
        updated_at REAL NOT NULL
    );
    """,
    # 4 - tasks
    """
    CREATE TABLE tasks (
        id TEXT PRIMARY KEY,
        goal TEXT NOT NULL,
        status TEXT NOT NULL,           -- pending | planning | running | waiting_approval | completed | failed | cancelled | interrupted
        conversation_id TEXT,
        device_id TEXT,
        plan_json TEXT,
        result TEXT,
        verification_json TEXT,
        error TEXT,
        progress REAL NOT NULL DEFAULT 0,
        replans INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        finished_at REAL
    );
    CREATE TABLE task_nodes (
        task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
        node_id TEXT NOT NULL,
        title TEXT NOT NULL,
        instruction TEXT NOT NULL,
        depends_on_json TEXT NOT NULL DEFAULT '[]',
        role TEXT,
        status TEXT NOT NULL,           -- pending | running | completed | failed | skipped | cancelled
        attempts INTEGER NOT NULL DEFAULT 0,
        output TEXT,
        error TEXT,
        tool_log_json TEXT NOT NULL DEFAULT '[]',
        started_at REAL,
        finished_at REAL,
        PRIMARY KEY (task_id, node_id)
    );
    """,
    # 5 - sync
    """
    CREATE TABLE changes (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        entity TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        op TEXT NOT NULL,               -- upsert | delete
        hlc TEXT NOT NULL,
        origin TEXT NOT NULL,
        data_json TEXT
    );
    CREATE INDEX changes_entity ON changes(entity, entity_id);
    CREATE TABLE tombstones (
        entity TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        hlc TEXT NOT NULL,
        origin TEXT NOT NULL,
        PRIMARY KEY (entity, entity_id)
    );
    CREATE TABLE sync_peers (
        peer_id TEXT PRIMARY KEY,
        url TEXT,
        last_pulled_seq INTEGER NOT NULL DEFAULT 0,
        last_pushed_seq INTEGER NOT NULL DEFAULT 0,
        last_sync_at REAL
    );
    CREATE TABLE sync_conflicts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        local_json TEXT,
        remote_json TEXT,
        resolution TEXT NOT NULL,
        ts REAL NOT NULL
    );
    """,
    # 6 - plugins
    """
    CREATE TABLE plugins (
        name TEXT PRIMARY KEY,
        enabled INTEGER NOT NULL DEFAULT 0,
        granted_permissions_json TEXT NOT NULL DEFAULT '[]',
        version TEXT,
        updated_at REAL NOT NULL
    );
    """,
    # 7 - conversation kinds (chat vs internal task-step conversations)
    """
    ALTER TABLE conversations ADD COLUMN kind TEXT NOT NULL DEFAULT 'chat';
    """,
]


class Database:
    """Thread-safe wrapper around one SQLite connection.

    Queries are short and local, so a single serialized connection is simpler
    and safer than a pool; WAL mode keeps readers unblocked.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys=ON")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self.migrate()

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> None:
        with self._lock:
            current = self.schema_version
            for idx, script in enumerate(MIGRATIONS[current:], start=current + 1):
                self._conn.execute("BEGIN")
                try:
                    for statement in _split_sql(script):
                        self._conn.execute(statement)
                    self._conn.execute(f"PRAGMA user_version={idx}")
                    self._conn.execute("COMMIT")
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, seq)

    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def integrity_check(self) -> str:
        return str(self.query_one("PRAGMA integrity_check")["integrity_check"])  # type: ignore[index]

    # --- small key/value store for runtime state -------------------------------
    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value_json FROM kv WHERE key=?", (key,))
        return json.loads(row["value_json"]) if row else default

    def kv_set(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO kv(key, value_json, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
            (key, json.dumps(value), time.time()),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _split_sql(script: str) -> list[str]:
    """Split a migration script into statements, keeping trigger bodies intact."""
    statements: list[str] = []
    buf: list[str] = []
    in_trigger = False
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        upper = stripped.upper()
        if upper.startswith("CREATE TRIGGER"):
            in_trigger = True
        if in_trigger:
            if upper == "END;":
                statements.append("\n".join(buf))
                buf = []
                in_trigger = False
        elif stripped.endswith(";"):
            statements.append("\n".join(buf))
            buf = []
    if buf:
        statements.append("\n".join(buf))
    return statements
