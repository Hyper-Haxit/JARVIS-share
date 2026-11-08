"""Memory system.

Layers:
  short-term  - the conversation itself (ConversationStore)
  working     - scoped scratch state for a conversation / task / project, with TTL
  long_term   - durable facts and preferences about the user
  episodic    - summaries of past interactions and events
  semantic    - chunks of documents the user asked JARVIS to index

Retrieval is hybrid: SQLite FTS5 (BM25) keyword search plus embedding cosine
similarity when an embedding model is reachable. Nothing is stored silently:
every write emits MEMORY_CREATED/UPDATED so clients can show it, and deletion is
a hard delete (row, FTS index entry and embedding) - "forget" really forgets.
"""

from __future__ import annotations

import array
import json
import math
import re
import time
import uuid
from pathlib import Path
from typing import Any

from jarvis.core.changelog import ChangeLog
from jarvis.core.events import EventBus, EventType
from jarvis.core.observability import get_logger

log = get_logger("memory")

LAYERS = ("long_term", "episodic", "semantic")
_STOP = set("""a an and are as at be but by for from has have i i'm in is it its me my of on or our so that the
this to was we what when where which who why will with you your about can could do does did how am just
please tell remember know""".split())
_TEXT_EXT = {".txt", ".md", ".rst", ".py", ".js", ".ts", ".json", ".toml", ".yaml", ".yml", ".csv", ".html", ".css",
             ".java", ".go", ".rs", ".c", ".cpp", ".h", ".cs", ".sh", ".ps1", ".ini", ".cfg", ".log", ".sql"}


def _pack(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    a = array.array("f")
    a.frombytes(blob)
    return a.tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def fts_query(text: str) -> str | None:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


class MemoryManager:
    def __init__(self, db: Any, events: EventBus, changelog: ChangeLog, router: Any | None, config: Any) -> None:
        self.db = db
        self.events = events
        self.changelog = changelog
        self.router = router
        self.config = config  # MemoryConfig
        self._embed_failed_at = 0.0
        self.embedding_status: str = "unknown"

    # --------------------------------------------------------------- embedding
    async def _embed(self, texts: list[str]) -> tuple[list[list[float]] | None, str | None]:
        if self.router is None or not texts:
            return None, None
        # Back off for a minute after a failure so an absent model doesn't slow every write.
        if time.time() - self._embed_failed_at < 60:
            return None, None
        try:
            vectors, model = await self.router.embed(texts)
            self.embedding_status = f"ok ({model})"
            return vectors, model
        except Exception as e:  # noqa: BLE001
            self._embed_failed_at = time.time()
            self.embedding_status = f"unavailable: {e}"
            log.warning("embedding unavailable; using keyword search only (%s)", str(e)[:200])
            return None, None

    # ------------------------------------------------------------------- write
    async def add(self, content: str, layer: str = "long_term", kind: str = "fact", tags: list[str] | None = None,
                  source: str | None = None, importance: float = 0.5, *, embed: bool = True,
                  dedupe: bool = True) -> dict[str, Any]:
        content = content.strip()
        if not content:
            raise ValueError("memory content is empty")
        if layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}")
        if dedupe:
            existing = self.db.query_one("SELECT id FROM memories WHERE layer=? AND lower(content)=lower(?)", (layer, content))
            if existing:
                return self.get(existing["id"])  # type: ignore[return-value]
        vectors, model = await self._embed([content]) if embed else (None, None)
        mid = "mem_" + uuid.uuid4().hex[:12]
        now = time.time()
        tag_str = ",".join(sorted({t.strip().lower() for t in (tags or []) if t.strip()}))
        hlc = self.changelog.clock.now()
        self.db.execute(
            "INSERT INTO memories(id, layer, kind, content, tags, source, importance, created_at, updated_at, embedding, "
            "embedding_model, hlc, origin) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, layer, kind, content, tag_str, source, float(importance), now, now,
             _pack(vectors[0]) if vectors else None, model, hlc, self.changelog.node_id),
        )
        record = self.get(mid)
        assert record is not None
        if layer != "semantic":  # documents are re-indexable locally; facts sync
            self.changelog.record("memory", mid, "upsert", self._sync_payload(record), hlc=hlc)
        await self.events.publish(EventType.MEMORY_CREATED, {"id": mid, "layer": layer, "kind": kind, "content": content[:300],
                                                             "source": source})
        return record

    async def update(self, mid: str, content: str | None = None, tags: list[str] | None = None,
                     importance: float | None = None) -> dict[str, Any]:
        current = self.get(mid)
        if current is None:
            raise KeyError(mid)
        sets, params = [], []
        if content is not None and content.strip() != current["content"]:
            vectors, model = await self._embed([content.strip()])
            sets += ["content=?", "embedding=?", "embedding_model=?"]
            params += [content.strip(), _pack(vectors[0]) if vectors else None, model]
        if tags is not None:
            sets.append("tags=?")
            params.append(",".join(sorted({t.strip().lower() for t in tags if t.strip()})))
        if importance is not None:
            sets.append("importance=?")
            params.append(float(importance))
        if not sets:
            return current
        hlc = self.changelog.clock.now()
        sets += ["updated_at=?", "hlc=?", "origin=?"]
        params += [time.time(), hlc, self.changelog.node_id, mid]
        self.db.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id=?", params)
        record = self.get(mid)
        assert record is not None
        if record["layer"] != "semantic":
            self.changelog.record("memory", mid, "upsert", self._sync_payload(record), hlc=hlc)
        await self.events.publish(EventType.MEMORY_UPDATED, {"id": mid, "content": record["content"][:300]})
        return record

    async def delete(self, mid: str, reason: str = "user request") -> bool:
        row = self.db.query_one("SELECT layer FROM memories WHERE id=?", (mid,))
        if row is None:
            return False
        self.db.execute("DELETE FROM memories WHERE id=?", (mid,))
        if row["layer"] != "semantic":
            self.changelog.record("memory", mid, "delete", None)
            # Remove the content from prior change records too: forgetting must not
            # leave a copy behind in the sync log.
            self.db.execute("UPDATE changes SET data_json=NULL WHERE entity='memory' AND entity_id=? AND op='upsert'", (mid,))
        await self.events.publish(EventType.MEMORY_DELETED, {"id": mid, "reason": reason})
        return True

    async def forget(self, ids: list[str] | None = None, query: str | None = None, limit: int = 5) -> list[str]:
        targets = list(ids or [])
        if query and not targets:
            targets = [h["id"] for h in await self.search(query, limit=limit, min_score=0.35)]
        deleted = [mid for mid in targets if await self.delete(mid)]
        return deleted

    # -------------------------------------------------------------------- read
    def get(self, mid: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM memories WHERE id=?", (mid,))
        return self._public(row) if row else None

    def list(self, layer: str | None = None, limit: int = 100, offset: int = 0, source: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM memories WHERE 1=1", []
        if layer:
            sql += " AND layer=?"
            params.append(layer)
        if source:
            sql += " AND source=?"
            params.append(source)
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        return [self._public(r) for r in self.db.query(sql, params)]

    def count(self) -> dict[str, int]:
        return {r["layer"]: r["n"] for r in self.db.query("SELECT layer, COUNT(*) AS n FROM memories GROUP BY layer")}

    async def search(self, query: str, limit: int = 8, layers: list[str] | None = None, min_score: float = 0.0,
                     touch: bool = False) -> list[dict[str, Any]]:
        layers = layers or list(LAYERS)
        layer_sql = ",".join("?" for _ in layers)
        scores: dict[str, float] = {}
        q = fts_query(query)
        if q:
            rows = self.db.query(
                f"SELECT m.id, bm25(memories_fts) AS rank FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
                f"WHERE memories_fts MATCH ? AND m.layer IN ({layer_sql}) ORDER BY rank LIMIT ?",
                [q, *layers, limit * 4],
            )
            for r in rows:
                s = max(0.0, -float(r["rank"]))
                scores[r["id"]] = 0.5 * (s / (s + 3.0)) + 0.15
        vectors, _ = await self._embed([query])
        if vectors:
            qv = vectors[0]
            rows = self.db.query(
                f"SELECT id, embedding FROM memories WHERE embedding IS NOT NULL AND layer IN ({layer_sql}) "
                "ORDER BY updated_at DESC LIMIT 5000", layers)
            for r in rows:
                sim = _cosine(qv, _unpack(r["embedding"]) or [])
                if sim > 0.45:
                    # Rescale typical embedding similarities (0.45-0.9) into 0-0.6.
                    sem = min(0.6, (sim - 0.45) * 1.4)
                    scores[r["id"]] = scores.get(r["id"], 0.0) + sem
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for mid, score in ranked:
            if score < min_score:
                continue
            rec = self.get(mid)
            if rec is None:
                continue
            rec["score"] = round(score, 3)
            out.append(rec)
            if len(out) >= limit:
                break
        if touch and out:
            now = time.time()
            self.db.executemany("UPDATE memories SET last_accessed=?, access_count=access_count+1 WHERE id=?",
                                [(now, r["id"]) for r in out])
        return out

    # -------------------------------------------------------------- working mem
    def working_set(self, scope: str, key: str, value: Any, ttl_s: float | None = None) -> None:
        self.db.execute(
            "INSERT INTO working_memory(scope, key, value_json, updated_at, expires_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(scope, key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at, "
            "expires_at=excluded.expires_at",
            (scope, key, json.dumps(value, default=str), time.time(), time.time() + ttl_s if ttl_s else None),
        )

    def working_get(self, scope: str) -> dict[str, Any]:
        now = time.time()
        self.db.execute("DELETE FROM working_memory WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
        rows = self.db.query("SELECT key, value_json FROM working_memory WHERE scope=? ORDER BY updated_at", (scope,))
        return {r["key"]: json.loads(r["value_json"]) for r in rows}

    def working_clear(self, scope: str, key: str | None = None) -> None:
        if key is None:
            self.db.execute("DELETE FROM working_memory WHERE scope=?", (scope,))
        else:
            self.db.execute("DELETE FROM working_memory WHERE scope=? AND key=?", (scope, key))

    # ----------------------------------------------------------- import/export
    def export(self, include_semantic: bool = False) -> dict[str, Any]:
        layers = LAYERS if include_semantic else ("long_term", "episodic")
        rows = self.db.query(
            f"SELECT * FROM memories WHERE layer IN ({','.join('?' for _ in layers)}) ORDER BY created_at", list(layers))
        return {"format": "jarvis-memory-export", "version": 1, "exported_at": time.time(),
                "memories": [self._public(r) for r in rows]}

    async def import_(self, data: dict[str, Any]) -> dict[str, int]:
        if data.get("format") != "jarvis-memory-export":
            raise ValueError("not a JARVIS memory export")
        added = skipped = 0
        for m in data.get("memories", []):
            before = self.db.query_one("SELECT COUNT(*) AS n FROM memories")["n"]
            await self.add(m["content"], layer=m.get("layer", "long_term"), kind=m.get("kind", "fact"),
                           tags=[t for t in (m.get("tags") or []) if t], source=m.get("source") or "import",
                           importance=m.get("importance", 0.5))
            after = self.db.query_one("SELECT COUNT(*) AS n FROM memories")["n"]
            if after > before:
                added += 1
            else:
                skipped += 1
        return {"added": added, "skipped": skipped}

    # --------------------------------------------------------- semantic ingest
    async def ingest(self, path: Path, max_files: int = 500, max_bytes: int = 2_000_000) -> dict[str, Any]:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        files = [path] if path.is_file() else [p for p in sorted(path.rglob("*")) if p.is_file()
                                               and p.suffix.lower() in _TEXT_EXT
                                               and not any(part.startswith(".") or part in ("node_modules", "__pycache__", ".venv")
                                                           for part in p.relative_to(path).parts)]
        files = files[:max_files]
        chunks_added = 0
        skipped: list[str] = []
        size = self.config.semantic_chunk_chars
        for f in files:
            try:
                if f.stat().st_size > max_bytes:
                    skipped.append(f"{f} (too large)")
                    continue
                text = f.read_text("utf-8")
            except (OSError, UnicodeDecodeError):
                skipped.append(f"{f} (unreadable)")
                continue
            # Re-ingesting a file replaces its previous chunks.
            for old in self.db.query("SELECT id FROM memories WHERE layer='semantic' AND source=?", (str(f),)):
                await self.delete(old["id"], reason="re-ingest")
            chunks = _chunk(text, size)
            vectors, model = await self._embed([f"{f.name}: {c}" for c in chunks]) if chunks else (None, None)
            for i, chunk in enumerate(chunks):
                mid = "mem_" + uuid.uuid4().hex[:12]
                now = time.time()
                self.db.execute(
                    "INSERT INTO memories(id, layer, kind, content, tags, source, importance, created_at, updated_at, "
                    "embedding, embedding_model, hlc, origin) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (mid, "semantic", "document_chunk", chunk, f.suffix.lstrip(".").lower(), str(f), 0.4, now, now,
                     _pack(vectors[i]) if vectors else None, model, self.changelog.clock.now(), self.changelog.node_id),
                )
                chunks_added += 1
        await self.events.publish(EventType.MEMORY_CREATED, {"layer": "semantic", "source": str(path), "chunks": chunks_added})
        return {"files": len(files) - len(skipped), "chunks": chunks_added, "skipped": skipped,
                "embedded": self.embedding_status.startswith("ok")}

    # --------------------------------------------------------------- episodic
    async def summarize_conversation(self, conversations: Any, cid: str) -> dict[str, Any] | None:
        """Create an episodic memory summarizing a finished conversation."""
        if self.router is None:
            return None
        from jarvis.models.base import Message  # noqa: PLC0415

        history = [m for m in conversations.history(cid, 60) if m.role in ("user", "assistant") and m.content.strip()]
        if len(history) < 2:
            conversations.mark_summarized(cid)
            return None
        transcript = "\n".join(f"{m.role.upper()}: {m.content[:800]}" for m in history)
        prompt = [
            Message(role="system", content="Summarize this conversation in 1-3 sentences for the assistant's long-term "
                                           "episodic memory. Include what the user wanted and the outcome. Do not invent "
                                           "details. Omit secrets, passwords and financial data."),
            Message(role="user", content=transcript[:12000]),
        ]
        resp, _ = await self.router.chat(prompt, role="fast", max_tokens=200)
        summary = resp.message.content.strip()
        conversations.mark_summarized(cid)
        if not summary:
            return None
        return await self.add(summary, layer="episodic", kind="conversation_summary", source=f"conversation:{cid}",
                              importance=0.4)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _public(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"], "layer": row["layer"], "kind": row["kind"], "content": row["content"],
            "tags": [t for t in (row["tags"] or "").split(",") if t], "source": row["source"],
            "importance": row["importance"], "created_at": row["created_at"], "updated_at": row["updated_at"],
            "last_accessed": row["last_accessed"], "access_count": row["access_count"],
            "embedded": row["embedding"] is not None, "embedding_model": row["embedding_model"],
        }

    def _sync_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        return {k: record[k] for k in ("layer", "kind", "content", "tags", "source", "importance", "created_at")}

    def apply_remote(self, entity_id: str, op: str, data: dict[str, Any] | None, hlc: str, origin: str) -> None:
        """Apply a change received from a sync peer (no events re-emitted to peers)."""
        if op == "delete":
            self.db.execute("DELETE FROM memories WHERE id=?", (entity_id,))
            self.changelog.record("memory", entity_id, "delete", None, hlc=hlc, origin=origin)
            return
        assert data is not None
        now = time.time()
        self.db.execute(
            "INSERT INTO memories(id, layer, kind, content, tags, source, importance, created_at, updated_at, hlc, origin) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET content=excluded.content, tags=excluded.tags, "
            "importance=excluded.importance, updated_at=excluded.updated_at, hlc=excluded.hlc, origin=excluded.origin, "
            "embedding=NULL, embedding_model=NULL",
            (entity_id, data["layer"], data.get("kind", "fact"), data["content"], ",".join(data.get("tags") or []),
             data.get("source"), data.get("importance", 0.5), data.get("created_at", now), now, hlc, origin),
        )
        self.changelog.record("memory", entity_id, "upsert", data, hlc=hlc, origin=origin)

    async def backfill_embeddings(self, batch: int = 32) -> int:
        rows = self.db.query("SELECT id, content FROM memories WHERE embedding IS NULL LIMIT ?", (batch,))
        if not rows:
            return 0
        vectors, model = await self._embed([r["content"] for r in rows])
        if not vectors:
            return 0
        self.db.executemany("UPDATE memories SET embedding=?, embedding_model=? WHERE id=?",
                            [(_pack(v), model, r["id"]) for v, r in zip(vectors, rows, strict=True)])
        return len(rows)


def _chunk(text: str, size: int) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    paras = re.split(r"\n\s*\n", text)
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
        while len(p) > size:
            chunks.append(p[:size])
            p = p[size:]
        buf = p
    if buf:
        chunks.append(buf)
    return chunks
