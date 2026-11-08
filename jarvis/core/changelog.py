"""Change log for synchronizable entities.

Every mutation of a synced entity (memories, preferences/identity) is recorded
here with a hybrid-logical-clock timestamp. The sync service ships these rows
between JARVIS nodes; deletions leave tombstones so they propagate too.
"""

from __future__ import annotations

import json
from typing import Any

from jarvis.core.hlc import HybridClock

SYNCED_ENTITIES = ("memory", "preference")


class ChangeLog:
    def __init__(self, db: Any, clock: HybridClock) -> None:
        self.db = db
        self.clock = clock

    @property
    def node_id(self) -> str:
        return self.clock.node_id

    def record(self, entity: str, entity_id: str, op: str, data: dict[str, Any] | None, hlc: str | None = None,
               origin: str | None = None) -> str:
        hlc = hlc or self.clock.now()
        origin = origin or self.node_id
        self.db.execute(
            "INSERT INTO changes(entity, entity_id, op, hlc, origin, data_json) VALUES(?,?,?,?,?,?)",
            (entity, entity_id, op, hlc, origin, json.dumps(data, default=str) if data is not None else None),
        )
        if op == "delete":
            self.db.execute(
                "INSERT INTO tombstones(entity, entity_id, hlc, origin) VALUES(?,?,?,?) "
                "ON CONFLICT(entity, entity_id) DO UPDATE SET hlc=excluded.hlc, origin=excluded.origin",
                (entity, entity_id, hlc, origin),
            )
        else:
            self.db.execute("DELETE FROM tombstones WHERE entity=? AND entity_id=?", (entity, entity_id))
        return hlc

    def since(self, seq: int, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM changes WHERE seq > ? ORDER BY seq LIMIT ?", (seq, limit))
        for r in rows:
            r["data"] = json.loads(r.pop("data_json")) if r.get("data_json") else None
        return rows

    def latest_seq(self) -> int:
        row = self.db.query_one("SELECT MAX(seq) AS s FROM changes")
        return int(row["s"] or 0) if row else 0
