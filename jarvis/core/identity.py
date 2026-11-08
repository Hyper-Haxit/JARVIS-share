"""JARVIS identity and personality.

The identity belongs to the core (never to a UI) and is synchronized across
nodes, so every interface - desktop, phone, voice-only device, suit - talks to
the same JARVIS.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field

from jarvis.core.changelog import ChangeLog

IDENTITY_KEY = "identity"


class Identity(BaseModel):
    name: str = "JARVIS"
    user_name: str | None = None
    tone: str = "calm, precise and quietly confident"
    formality: str = "professional but warm"
    verbosity: str = "concise - lead with the answer, expand only when asked"
    humor: str = "dry and sparing"
    response_style: str = "plain language, short paragraphs, no filler or flattery"
    voice: str | None = None
    language: str = "en"
    behaviors: list[str] = Field(default_factory=list)


class IdentityService:
    def __init__(self, db: Any, changelog: ChangeLog) -> None:
        self.db = db
        self.changelog = changelog

    def get(self) -> Identity:
        row = self.db.query_one("SELECT value_json FROM preferences WHERE key=?", (IDENTITY_KEY,))
        if row is None:
            return Identity()
        try:
            return Identity.model_validate(json.loads(row["value_json"]))
        except ValueError:
            return Identity()

    def update(self, patch: dict[str, Any]) -> Identity:
        current = self.get().model_dump()
        unknown = set(patch) - set(current)
        if unknown:
            raise ValueError(f"unknown identity fields: {sorted(unknown)}")
        current.update(patch)
        identity = Identity.model_validate(current)
        self._store(identity.model_dump())
        return identity

    def _store(self, value: dict[str, Any], hlc: str | None = None, origin: str | None = None) -> None:
        hlc = self.changelog.record("preference", IDENTITY_KEY, "upsert", value, hlc=hlc, origin=origin)
        self.db.execute(
            "INSERT INTO preferences(key, value_json, hlc, origin, updated_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, hlc=excluded.hlc, origin=excluded.origin, "
            "updated_at=excluded.updated_at",
            (IDENTITY_KEY, json.dumps(value), hlc, origin or self.changelog.node_id, time.time()),
        )

    def persona_prompt(self) -> str:
        ident = self.get()
        lines = [
            f"You are {ident.name}, a personal AI system that runs across the user's computers, phone, voice devices "
            "and wearable hardware. The same identity, memory and permissions apply on every device.",
            f"Personality - tone: {ident.tone}; formality: {ident.formality}; verbosity: {ident.verbosity}; "
            f"humor: {ident.humor}; style: {ident.response_style}.",
        ]
        if ident.user_name:
            lines.append(f"The user's name is {ident.user_name}.")
        if ident.behaviors:
            lines.append("Standing preferences from the user:\n" + "\n".join(f"- {b}" for b in ident.behaviors))
        return "\n".join(lines)
