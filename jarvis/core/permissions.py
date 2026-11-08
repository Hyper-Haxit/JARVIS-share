"""Permission system.

This module is the enforcement point that sits *outside* the model. Every tool
invocation, device command and plugin action passes through
:class:`PermissionGate` before any code with side effects runs. The model can
request actions; it cannot approve them, cannot lower their risk level and
cannot see or forge approval tokens.

Rules that cannot be relaxed by configuration:
  * an action whose risk exceeds the requesting device's ``max_risk`` is denied;
  * DANGEROUS actions are never auto-allowed and never receive standing grants;
    each one needs its own explicit human approval;
  * only devices with role ``owner`` or ``client`` and a sufficient ``max_risk``
    may decide an approval; hardware devices can never approve anything.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from jarvis.core.events import EventBus, EventType
from jarvis.core.observability import get_logger, redact

log = get_logger("permissions")


class Risk(IntEnum):
    SAFE = 0
    LOW_RISK = 1
    SENSITIVE = 2
    DANGEROUS = 3

    @classmethod
    def parse(cls, value: Risk | str | int) -> Risk:
        if isinstance(value, Risk):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value).upper()]


@dataclass
class Principal:
    """The authenticated device on whose behalf something happens."""

    device_id: str
    name: str
    kind: str
    role: str  # owner | client | hardware | system
    max_risk: Risk = Risk.DANGEROUS
    session_id: str | None = None

    @property
    def can_approve(self) -> bool:
        return self.role in ("owner", "client")

    @property
    def is_owner(self) -> bool:
        return self.role == "owner"

    def to_dict(self) -> dict[str, Any]:
        return {"device_id": self.device_id, "name": self.name, "kind": self.kind, "role": self.role, "max_risk": self.max_risk.name}


SYSTEM_PRINCIPAL = Principal(device_id="system", name="JARVIS Core", kind="core", role="system", max_risk=Risk.DANGEROUS)


@dataclass
class ActionRequest:
    tool: str
    capability: str
    risk: Risk
    summary: str  # what JARVIS wants to do
    effects: str  # what will happen
    reason: str = ""  # why (the model's stated rationale or the user's request)
    args: dict[str, Any] = field(default_factory=dict)
    conversation_id: str | None = None
    task_id: str | None = None
    requested_by: str | None = None

    def public(self) -> dict[str, Any]:
        return redact(
            {
                "tool": self.tool,
                "capability": self.capability,
                "risk": self.risk.name,
                "summary": self.summary,
                "effects": self.effects,
                "reason": self.reason,
                "args": self.args,
                "conversation_id": self.conversation_id,
                "task_id": self.task_id,
                "requested_by": self.requested_by,
            }
        )


@dataclass
class GateResult:
    allowed: bool
    decision: str  # allow | approved | denied | expired | policy_denied
    detail: str = ""
    approval_id: str | None = None


class PermissionDenied(Exception):
    pass


class PermissionGate:
    def __init__(self, db: Any, events: EventBus, config: Any) -> None:
        self.db = db
        self.events = events
        self.config = config  # PermissionsConfig
        self._waiters: dict[str, asyncio.Future[tuple[bool, str, str]]] = {}
        # Approvals left pending by a previous run can no longer be honoured.
        self.db.execute("UPDATE approvals SET status='expired', decided_at=? WHERE status='pending'", (time.time(),))

    # ---------------------------------------------------------------- policy
    def policy_decision(self, req: ActionRequest, principal: Principal) -> tuple[str, str]:
        if req.risk > principal.max_risk:
            return "deny", f"device '{principal.name}' is limited to {principal.max_risk.name} actions"
        override = self.config.tool_overrides.get(req.tool)
        if override == "deny":
            return "deny", f"tool '{req.tool}' is disabled by policy"
        if req.risk >= Risk.DANGEROUS:
            return "ask", "dangerous actions always require explicit approval"
        if override == "allow":
            return "allow", "allowed by tool override"
        if self._has_grant(req):
            return "allow", "allowed by standing grant"
        if override == "ask":
            return "ask", "tool override requires approval"
        default = self.config.defaults.get(req.risk.name, "ask")
        return default, f"default policy for {req.risk.name}"

    def _has_grant(self, req: ActionRequest) -> bool:
        now = time.time()
        rows = self.db.query(
            "SELECT scope_type, scope_id FROM grants WHERE tool=? AND (expires_at IS NULL OR expires_at > ?)",
            (req.tool, now),
        )
        for r in rows:
            if r["scope_type"] == "global":
                return True
            if r["scope_type"] == "conversation" and req.conversation_id and r["scope_id"] == req.conversation_id:
                return True
            if r["scope_type"] == "task" and req.task_id and r["scope_id"] == req.task_id:
                return True
        return False

    # ----------------------------------------------------------------- gate
    async def check(self, req: ActionRequest, principal: Principal, timeout: float | None = None) -> GateResult:
        decision, detail = self.policy_decision(req, principal)
        if decision == "allow":
            return GateResult(True, "allow", detail)
        if decision == "deny":
            await self.events.publish(EventType.PERMISSION_DENIED, {**req.public(), "detail": detail, "by": "policy"})
            self.audit(principal, "permission.policy_denied", req, "denied", detail)
            return GateResult(False, "policy_denied", detail)
        return await self._ask(req, principal, timeout)

    async def _ask(self, req: ActionRequest, principal: Principal, timeout: float | None) -> GateResult:
        approval_id = uuid.uuid4().hex[:12]
        now = time.time()
        public = req.public()
        self.db.execute(
            "INSERT INTO approvals(id, tool, capability, risk, summary, reason, effects, args_json, requested_by, "
            "conversation_id, task_id, status, requested_at) VALUES(?,?,?,?,?,?,?,?,?,?,?, 'pending', ?)",
            (
                approval_id, req.tool, req.capability, req.risk.name, req.summary, req.reason, req.effects,
                json.dumps(public["args"], default=str), principal.device_id, req.conversation_id, req.task_id, now,
            ),
        )
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[bool, str, str]] = loop.create_future()
        self._waiters[approval_id] = fut
        timeout = self.config.approval_timeout_s if timeout is None else timeout
        await self.events.publish(
            EventType.PERMISSION_REQUESTED,
            {**public, "approval_id": approval_id, "expires_at": now + timeout},
        )
        try:
            approved, decided_by, scope = await asyncio.wait_for(fut, timeout)
        except TimeoutError:
            self._finish(approval_id, "expired", None, None)
            await self.events.publish(EventType.PERMISSION_DENIED, {**public, "approval_id": approval_id, "detail": "approval timed out"})
            self.audit(principal, "permission.expired", req, "denied", "no decision before timeout")
            return GateResult(False, "expired", "no decision was made before the approval timed out", approval_id)
        except asyncio.CancelledError:
            self._finish(approval_id, "cancelled", None, None)
            await self.events.publish(EventType.PERMISSION_DENIED, {**public, "approval_id": approval_id, "detail": "cancelled"})
            raise
        finally:
            self._waiters.pop(approval_id, None)
        if approved:
            return GateResult(True, "approved", f"approved by {decided_by} ({scope})", approval_id)
        return GateResult(False, "denied", f"denied by {decided_by}", approval_id)

    def _finish(self, approval_id: str, status: str, decided_by: str | None, scope: str | None) -> None:
        self.db.execute(
            "UPDATE approvals SET status=?, decided_by=?, decision_scope=?, decided_at=? WHERE id=? AND status='pending'",
            (status, decided_by, scope, time.time(), approval_id),
        )

    async def decide(self, approval_id: str, approve: bool, principal: Principal, scope: str = "once") -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if row is None:
            raise KeyError(approval_id)
        if row["status"] != "pending":
            raise ValueError(f"approval is already {row['status']}")
        risk = Risk.parse(row["risk"])
        if not principal.can_approve:
            raise PermissionDenied(f"device '{principal.name}' ({principal.role}) cannot decide approvals")
        if approve and risk > principal.max_risk:
            raise PermissionDenied(f"device '{principal.name}' may not approve {risk.name} actions")
        if scope not in ("once", "conversation", "task", "global"):
            raise ValueError("scope must be once, conversation, task or global")
        if risk >= Risk.DANGEROUS:
            scope = "once"
        if scope == "conversation" and not row["conversation_id"]:
            scope = "once"
        if scope == "task" and not row["task_id"]:
            scope = "once"
        if scope == "global" and not principal.is_owner:
            raise PermissionDenied("only the owner device can create global grants")
        status = "approved" if approve else "denied"
        self._finish(approval_id, status, principal.device_id, scope)
        if approve and scope != "once":
            scope_id = row["conversation_id"] if scope == "conversation" else row["task_id"] if scope == "task" else None
            self.db.execute(
                "INSERT INTO grants(id, tool, scope_type, scope_id, granted_by, created_at) VALUES(?,?,?,?,?,?)",
                (uuid.uuid4().hex[:12], row["tool"], scope, scope_id, principal.device_id, time.time()),
            )
        fut = self._waiters.get(approval_id)
        if fut is not None and not fut.done():
            fut.set_result((approve, principal.name, scope))
        evt = EventType.PERMISSION_GRANTED if approve else EventType.PERMISSION_DENIED
        await self.events.publish(evt, {"approval_id": approval_id, "tool": row["tool"], "risk": risk.name, "scope": scope, "by": principal.name})
        self.db.execute(
            "INSERT INTO audit_log(ts, actor, action, target, risk, outcome, details_json) VALUES(?,?,?,?,?,?,?)",
            (time.time(), principal.device_id, "permission.decide", row["tool"], risk.name, status, json.dumps({"approval_id": approval_id, "scope": scope})),
        )
        return {"id": approval_id, "status": status, "scope": scope}

    def cancel_pending(self, task_id: str | None = None, conversation_id: str | None = None) -> int:
        n = 0
        for approval_id, fut in list(self._waiters.items()):
            row = self.db.query_one("SELECT task_id, conversation_id FROM approvals WHERE id=?", (approval_id,))
            if row is None:
                continue
            if (task_id and row["task_id"] == task_id) or (conversation_id and row["conversation_id"] == conversation_id):
                if not fut.done():
                    fut.set_result((False, "cancellation", "once"))
                    n += 1
        return n

    # ---------------------------------------------------------------- queries
    def pending(self) -> list[dict[str, Any]]:
        return [self._row(r) for r in self.db.query("SELECT * FROM approvals WHERE status='pending' ORDER BY requested_at")]

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        return [self._row(r) for r in self.db.query("SELECT * FROM approvals ORDER BY requested_at DESC LIMIT ?", (limit,))]

    def grants(self) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM grants ORDER BY created_at DESC")

    def revoke_grant(self, grant_id: str) -> bool:
        return self.db.execute("DELETE FROM grants WHERE id=?", (grant_id,)).rowcount > 0

    @staticmethod
    def _row(r: dict[str, Any]) -> dict[str, Any]:
        r = dict(r)
        r["args"] = json.loads(r.pop("args_json") or "{}")
        return r

    def audit(self, principal: Principal, action: str, req: ActionRequest, outcome: str, detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO audit_log(ts, actor, action, target, risk, outcome, details_json) VALUES(?,?,?,?,?,?,?)",
            (time.time(), principal.device_id, action, req.tool, req.risk.name, outcome, json.dumps(redact({"detail": detail, "args": req.args}), default=str)),
        )
