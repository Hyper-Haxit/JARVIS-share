"""Conversation storage (short-term memory).

Messages are stored verbatim so a conversation can be replayed append-only to
the model. Image bytes are not kept in history: an image is current only for
the turn it was supplied in, so later turns see a textual marker noting when it
was captured instead of a stale picture.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from jarvis.models.base import Message, ToolCall


class ConversationStore:
    def __init__(self, db: Any) -> None:
        self.db = db

    def create(self, device_id: str | None = None, title: str | None = None, kind: str = "chat") -> str:
        cid = "conv_" + uuid.uuid4().hex[:12]
        now = time.time()
        self.db.execute(
            "INSERT INTO conversations(id, title, device_id, created_at, updated_at, kind) VALUES(?,?,?,?,?,?)",
            (cid, title, device_id, now, now, kind),
        )
        return cid

    def exists(self, cid: str) -> bool:
        return self.db.query_one("SELECT 1 FROM conversations WHERE id=?", (cid,)) is not None

    def append(self, cid: str, message: Message, meta: dict[str, Any] | None = None) -> str:
        mid = "msg_" + uuid.uuid4().hex[:12]
        now = time.time()
        meta = dict(meta or {})
        if message.provider_state:
            meta["provider_state"] = message.provider_state
        if message.images:
            meta["images"] = [{"source": i.source, "captured_at": i.captured_at, "media_type": i.media_type} for i in message.images]
        if message.is_error:
            meta["is_error"] = True
        with self.db.transaction() as conn:
            row = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id=?", (cid,)).fetchone()
            conn.execute(
                "INSERT INTO messages(id, conversation_id, seq, role, content, tool_calls_json, tool_call_id, name, meta_json, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (mid, cid, row[0], message.role, message.content,
                 json.dumps([tc.model_dump() for tc in message.tool_calls]) if message.tool_calls else None,
                 message.tool_call_id, message.name, json.dumps(meta, default=str), now),
            )
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, cid))
            if message.role == "user":
                conn.execute(
                    "UPDATE conversations SET title=? WHERE id=? AND title IS NULL",
                    (_title(message.content), cid),
                )
        return mid

    def history(self, cid: str, limit: int = 30) -> list[Message]:
        rows = self.db.query(
            "SELECT * FROM (SELECT * FROM messages WHERE conversation_id=? ORDER BY seq DESC LIMIT ?) ORDER BY seq",
            (cid, limit),
        )
        msgs = [self._message(r) for r in rows]
        # Never start history with orphaned tool results or tool-call turns.
        while msgs and (msgs[0].role == "tool" or (msgs[0].role == "assistant" and msgs[0].tool_calls)):
            msgs.pop(0)
        return msgs

    @staticmethod
    def _message(r: dict[str, Any]) -> Message:
        meta = json.loads(r["meta_json"] or "{}")
        content = r["content"]
        if meta.get("images"):
            notes = []
            for img in meta["images"]:
                when = img.get("captured_at")
                stamp = time.strftime("%H:%M:%S", time.localtime(when)) if when else "unknown time"
                notes.append(f"[earlier image from {img.get('source', 'upload')} at {stamp} - no longer attached]")
            content = (content + "\n" + "\n".join(notes)).strip()
        return Message(
            role=r["role"], content=content,
            tool_calls=[ToolCall(**tc) for tc in json.loads(r["tool_calls_json"])] if r["tool_calls_json"] else [],
            tool_call_id=r["tool_call_id"], name=r["name"], is_error=bool(meta.get("is_error")),
            provider_state=meta.get("provider_state") or {},
        )

    def list(self, limit: int = 50, offset: int = 0, kind: str | None = "chat") -> list[dict[str, Any]]:
        where = "WHERE c.kind=?" if kind else ""
        params: tuple[Any, ...] = (kind, limit, offset) if kind else (limit, offset)
        return self.db.query(
            "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count "
            f"FROM conversations c {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            params,
        )

    def get(self, cid: str) -> dict[str, Any] | None:
        conv = self.db.query_one("SELECT * FROM conversations WHERE id=?", (cid,))
        if conv is None:
            return None
        rows = self.db.query("SELECT * FROM messages WHERE conversation_id=? ORDER BY seq", (cid,))
        conv["messages"] = [
            {
                "id": r["id"], "seq": r["seq"], "role": r["role"], "content": r["content"],
                "tool_calls": json.loads(r["tool_calls_json"]) if r["tool_calls_json"] else [],
                "tool_call_id": r["tool_call_id"], "name": r["name"], "created_at": r["created_at"],
                "meta": {k: v for k, v in json.loads(r["meta_json"] or "{}").items() if k != "provider_state"},
            }
            for r in rows
        ]
        return conv

    def rename(self, cid: str, title: str) -> None:
        self.db.execute("UPDATE conversations SET title=? WHERE id=?", (title[:200], cid))

    def delete(self, cid: str) -> bool:
        return self.db.execute("DELETE FROM conversations WHERE id=?", (cid,)).rowcount > 0

    def purge_older_than(self, days: int) -> int:
        if days <= 0:
            return 0
        cutoff = time.time() - days * 86400
        return self.db.execute("DELETE FROM conversations WHERE updated_at < ?", (cutoff,)).rowcount

    def unsummarized(self, idle_s: float = 1800, min_messages: int = 4) -> list[str]:
        rows = self.db.query(
            "SELECT c.id FROM conversations c WHERE c.summarized=0 AND c.kind='chat' AND c.updated_at < ? AND "
            "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) >= ?",
            (time.time() - idle_s, min_messages),
        )
        return [r["id"] for r in rows]

    def mark_summarized(self, cid: str) -> None:
        self.db.execute("UPDATE conversations SET summarized=1 WHERE id=?", (cid,))


def _title(text: str) -> str:
    text = " ".join(text.split())
    return (text[:60] + "...") if len(text) > 60 else (text or "Conversation")
