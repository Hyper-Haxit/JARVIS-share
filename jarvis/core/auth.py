"""Device identity, pairing and sessions.

Every interface to JARVIS (CLI, desktop, browser, phone, suit component) is a
*device* with its own credential. Credentials are random 256-bit tokens; only
their SHA-256 hashes are stored, so a leaked database does not leak tokens.

Flow:
  * On first start the core creates a local ``owner`` device for the host CLI
    and writes its token to a permission-restricted file in JARVIS_HOME.
  * New devices are added by redeeming a short-lived, single-use pairing code
    created by an already-authenticated owner device.
  * Device tokens can be exchanged for expiring session tokens (used by browsers).
  * Revoking a device invalidates its token and all its sessions immediately.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.config.paths import restrict_permissions
from jarvis.core.events import EventBus, EventType
from jarvis.core.observability import get_logger, redactor
from jarvis.core.permissions import Principal, Risk

log = get_logger("auth")

DEVICE_PREFIX = "jvd_"
SESSION_PREFIX = "jvs_"
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I ambiguity

VALID_KINDS = {"cli", "desktop", "web", "mobile", "hardware", "core", "wearable"}
VALID_ROLES = {"owner", "client", "hardware"}


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalize_code(code: str) -> str:
    return "".join(c for c in code.upper() if c.isalnum())


class AuthError(Exception):
    pass


@dataclass
class Device:
    id: str
    name: str
    kind: str
    role: str
    platform: str | None
    max_risk: Risk
    capabilities: list[Any]
    metadata: dict[str, Any]
    created_at: float
    last_seen: float | None
    revoked_at: float | None

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "platform": self.platform,
            "max_risk": self.max_risk.name,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
        }

    def principal(self, session_id: str | None = None) -> Principal:
        return Principal(self.id, self.name, self.kind, self.role, self.max_risk, session_id)


class AuthService:
    def __init__(self, db: Any, events: EventBus, session_ttl_hours: float = 12.0, pairing_ttl_minutes: float = 10.0) -> None:
        self.db = db
        self.events = events
        self.session_ttl = session_ttl_hours * 3600
        self.pairing_ttl = pairing_ttl_minutes * 60
        self._last_seen_written: dict[str, float] = {}

    # ---------------------------------------------------------------- devices
    @staticmethod
    def _device(row: dict[str, Any]) -> Device:
        return Device(
            id=row["id"], name=row["name"], kind=row["kind"], role=row["role"], platform=row["platform"],
            max_risk=Risk(int(row["max_risk"])), capabilities=json.loads(row["capabilities_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}"), created_at=row["created_at"],
            last_seen=row["last_seen"], revoked_at=row["revoked_at"],
        )

    def create_device(
        self,
        name: str,
        kind: str,
        role: str = "client",
        platform: str | None = None,
        max_risk: Risk = Risk.SENSITIVE,
        capabilities: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Device, str]:
        if kind not in VALID_KINDS:
            raise AuthError(f"invalid device kind '{kind}'")
        if role not in VALID_ROLES:
            raise AuthError(f"invalid device role '{role}'")
        name = (name or kind).strip()[:80]
        token = DEVICE_PREFIX + secrets.token_urlsafe(32)
        redactor.register(token)
        device_id = "dev_" + uuid.uuid4().hex[:12]
        now = time.time()
        self.db.execute(
            "INSERT INTO devices(id, name, kind, role, platform, token_hash, max_risk, capabilities_json, metadata_json, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (device_id, name, kind, role, platform, hash_token(token), int(max_risk), json.dumps(capabilities or []), json.dumps(metadata or {}), now),
        )
        self.events.emit(EventType.DEVICE_REGISTERED, {"device_id": device_id, "name": name, "kind": kind, "role": role})
        self._audit(device_id, "device.register", device_id, "ok", {"kind": kind, "role": role})
        device = self.get_device(device_id)
        assert device is not None
        return device, token

    def get_device(self, device_id: str) -> Device | None:
        row = self.db.query_one("SELECT * FROM devices WHERE id=?", (device_id,))
        return self._device(row) if row else None

    def list_devices(self, include_revoked: bool = True) -> list[Device]:
        sql = "SELECT * FROM devices" + ("" if include_revoked else " WHERE revoked_at IS NULL") + " ORDER BY created_at"
        return [self._device(r) for r in self.db.query(sql)]

    def update_device(self, device_id: str, *, name: str | None = None, max_risk: Risk | None = None,
                      capabilities: list[Any] | None = None, metadata: dict[str, Any] | None = None) -> Device:
        device = self.get_device(device_id)
        if device is None:
            raise AuthError("unknown device")
        if name is not None:
            self.db.execute("UPDATE devices SET name=? WHERE id=?", (name.strip()[:80], device_id))
        if max_risk is not None:
            self.db.execute("UPDATE devices SET max_risk=? WHERE id=?", (int(max_risk), device_id))
        if capabilities is not None:
            self.db.execute("UPDATE devices SET capabilities_json=? WHERE id=?", (json.dumps(capabilities), device_id))
        if metadata is not None:
            self.db.execute("UPDATE devices SET metadata_json=? WHERE id=?", (json.dumps(metadata), device_id))
        return self.get_device(device_id)  # type: ignore[return-value]

    def revoke_device(self, device_id: str, by: str | None = None) -> bool:
        cur = self.db.execute("UPDATE devices SET revoked_at=? WHERE id=? AND revoked_at IS NULL", (time.time(), device_id))
        if cur.rowcount == 0:
            return False
        self.db.execute("UPDATE sessions SET revoked=1 WHERE device_id=?", (device_id,))
        self.events.emit(EventType.DEVICE_REVOKED, {"device_id": device_id, "by": by})
        self._audit(by, "device.revoke", device_id, "ok")
        return True

    def owner_count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM devices WHERE role='owner' AND revoked_at IS NULL")
        return int(row["n"]) if row else 0

    # ------------------------------------------------------- local host device
    def ensure_local_device(self, token_file: Path, platform: str | None = None) -> str:
        """Return the host CLI's device token, creating the device if needed."""
        if token_file.exists():
            token = token_file.read_text("utf-8").strip()
            principal = self.authenticate(token)
            if principal is not None:
                redactor.register(token)
                return token
        _, token = self.create_device("Local host (CLI)", "cli", role="owner", platform=platform, max_risk=Risk.DANGEROUS,
                                      metadata={"local": True})
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(token, "utf-8")
        restrict_permissions(token_file)
        return token

    # ----------------------------------------------------------------- pairing
    def create_pairing_code(self, created_by: Principal, kind: str = "web", role: str = "client",
                            max_risk: Risk = Risk.SENSITIVE, name_hint: str | None = None) -> dict[str, Any]:
        if not created_by.is_owner:
            raise AuthError("only owner devices can create pairing codes")
        if role == "owner" and max_risk > created_by.max_risk:
            raise AuthError("cannot grant more authority than the creating device has")
        if max_risk > created_by.max_risk:
            max_risk = created_by.max_risk
        raw = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(10))
        code = f"{raw[:5]}-{raw[5:]}"
        redactor.register(raw)
        redactor.register(code)
        now = time.time()
        self.db.execute(
            "INSERT INTO pairing_codes(code_hash, created_by, kind, role, max_risk, name_hint, created_at, expires_at) VALUES(?,?,?,?,?,?,?,?)",
            (hash_token(_normalize_code(code)), created_by.device_id, kind, role, int(max_risk), name_hint, now, now + self.pairing_ttl),
        )
        self._audit(created_by.device_id, "pairing.create", kind, "ok", {"role": role, "max_risk": max_risk.name})
        return {"code": code, "expires_at": now + self.pairing_ttl, "kind": kind, "role": role, "max_risk": max_risk.name}

    def redeem_pairing_code(self, code: str, name: str, kind: str | None = None, platform: str | None = None,
                            capabilities: list[Any] | None = None) -> tuple[Device, str]:
        code_hash = hash_token(_normalize_code(code))
        now = time.time()
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM pairing_codes WHERE code_hash=?", (code_hash,)).fetchone()
            if row is None or row["used_at"] is not None or row["expires_at"] < now:
                raise AuthError("invalid or expired pairing code")
            conn.execute("UPDATE pairing_codes SET used_at=? WHERE code_hash=?", (now, code_hash))
        device_kind = kind if kind in VALID_KINDS else row["kind"]
        # The code determines authority, not the redeeming client.
        return self.create_device(name, device_kind, role=row["role"], platform=platform,
                                  max_risk=Risk(int(row["max_risk"])), capabilities=capabilities,
                                  metadata={"paired_by": row["created_by"]})

    # ---------------------------------------------------------------- sessions
    def create_session(self, device_token: str) -> dict[str, Any]:
        principal = self.authenticate(device_token, allow_session=False)
        if principal is None:
            raise AuthError("invalid device credential")
        token = SESSION_PREFIX + secrets.token_urlsafe(32)
        redactor.register(token)
        sid = "ses_" + uuid.uuid4().hex[:12]
        now = time.time()
        self.db.execute(
            "INSERT INTO sessions(id, device_id, token_hash, created_at, expires_at) VALUES(?,?,?,?,?)",
            (sid, principal.device_id, hash_token(token), now, now + self.session_ttl),
        )
        return {"session_token": token, "session_id": sid, "expires_at": now + self.session_ttl, "device_id": principal.device_id}

    def end_session(self, session_id: str) -> None:
        self.db.execute("UPDATE sessions SET revoked=1 WHERE id=?", (session_id,))

    def authenticate(self, token: str | None, allow_session: bool = True) -> Principal | None:
        if not token:
            return None
        token = token.strip()
        now = time.time()
        if token.startswith(SESSION_PREFIX) and allow_session:
            row = self.db.query_one(
                "SELECT s.id AS sid, s.expires_at, s.revoked, d.* FROM sessions s JOIN devices d ON d.id = s.device_id WHERE s.token_hash=?",
                (hash_token(token),),
            )
            if row is None or row["revoked"] or row["expires_at"] < now or row["revoked_at"] is not None:
                return None
            device = self._device(row)
            self._touch(device.id, now)
            return device.principal(row["sid"])
        if token.startswith(DEVICE_PREFIX):
            row = self.db.query_one("SELECT * FROM devices WHERE token_hash=?", (hash_token(token),))
            if row is None or row["revoked_at"] is not None:
                return None
            device = self._device(row)
            self._touch(device.id, now)
            return device.principal()
        return None

    def diagnose(self, token: str | None) -> str:
        """Why a credential was rejected (for server logs; never returned to callers)."""
        if not token:
            return "no credential sent"
        token = token.strip()
        if token.startswith(SESSION_PREFIX):
            row = self.db.query_one("SELECT s.expires_at, s.revoked, d.revoked_at FROM sessions s JOIN devices d ON d.id = s.device_id "
                                    "WHERE s.token_hash=?", (hash_token(token),))
            if row is None:
                return "unknown session"
            if row["revoked_at"] is not None:
                return "device revoked"
            if row["revoked"]:
                return "session ended"
            return "session expired" if row["expires_at"] < time.time() else "ok"
        if token.startswith(DEVICE_PREFIX):
            row = self.db.query_one("SELECT revoked_at FROM devices WHERE token_hash=?", (hash_token(token),))
            if row is None:
                return "unknown device credential (from another JARVIS installation?)"
            return "device revoked" if row["revoked_at"] is not None else "ok"
        return "malformed credential"

    def _touch(self, device_id: str, now: float) -> None:
        if now - self._last_seen_written.get(device_id, 0) > 30:
            self._last_seen_written[device_id] = now
            self.db.execute("UPDATE devices SET last_seen=? WHERE id=?", (now, device_id))

    def purge_expired(self) -> None:
        now = time.time()
        self.db.execute("DELETE FROM sessions WHERE expires_at < ? OR revoked=1", (now - 86400,))
        self.db.execute("DELETE FROM pairing_codes WHERE expires_at < ?", (now - 86400,))

    def _audit(self, actor: str | None, action: str, target: str | None, outcome: str, details: dict[str, Any] | None = None) -> None:
        self.db.execute(
            "INSERT INTO audit_log(ts, actor, action, target, outcome, details_json) VALUES(?,?,?,?,?,?)",
            (time.time(), actor, action, target, outcome, json.dumps(details or {})),
        )
