"""Setting up a device on the computer JARVIS runs on, without a pairing code.

The first browser (or any later one) on this computer asks to be set up; the
person at this computer confirms it, either in a small confirmation window
the core shows on this computer's desktop or with `jarvis devices approve`
in a terminal (which acts with this computer's own credential). Approval
creates an ordinary owner device through AuthService, so revocation, sessions
and auditing work exactly as for a paired device.

What keeps this safe:
  * Requests are accepted only from this computer, addressed to it by a
    loopback name, and (from browsers) from JARVIS's own page: another website,
    another computer, or a DNS-rebinding trick cannot create one. (Enforced by
    the API layer.)
  * Nothing is granted until a person on this computer approves. The request
    shows a short code in both places so the right browser is approved.
  * The requester proves it is the same requester with a random secret; only
    its hash is kept. The device credential is handed over once, then dropped.
  * Requests expire after five minutes; one at a time; rate limited.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from jarvis.core.auth import VALID_KINDS, AuthError, hash_token
from jarvis.core.observability import get_logger, redactor
from jarvis.core.permissions import Principal, Risk

log = get_logger("auth.setup")

REQUEST_TTL_S = 300
MAX_REQUESTS_PER_MINUTE = 6


@dataclass
class SetupRequest:
    id: str
    secret_hash: str
    code: str
    name: str
    kind: str
    platform: str | None
    capabilities: list[Any]
    created_at: float
    expires_at: float
    status: str = "pending"            # pending | approved | denied | expired | cancelled | claimed
    decided_by: str | None = None
    device_id: str | None = None
    token: str | None = None           # held only between approval and claim
    decided: threading.Event = field(default_factory=threading.Event)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "code": self.code, "name": self.name, "kind": self.kind, "platform": self.platform,
                "created_at": self.created_at, "expires_at": self.expires_at, "status": self.status}


class LocalEnrollment:
    def __init__(self, auth: Any, prompt: Any = None) -> None:
        self.auth = auth
        self.prompt = prompt
        self._requests: dict[str, SetupRequest] = {}
        self._recent: list[float] = []
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- state
    def first_run(self) -> bool:
        """No device has ever been set up besides this computer's own credential."""
        return not any(not d.metadata.get("local") and d.kind != "core" for d in self.auth.list_devices())

    def approval_method(self) -> str:
        return "window" if self.prompt is not None and self.prompt.available() else "terminal"

    def _expire(self) -> None:
        now = time.time()
        for r in self._requests.values():
            if r.status in ("pending", "approved") and r.expires_at < now:
                r.status = "expired"
                r.token = None
                r.decided.set()
        for rid in [k for k, r in self._requests.items() if r.expires_at < now - 600]:
            del self._requests[rid]

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            self._expire()
            return [r.public() for r in self._requests.values() if r.status == "pending"]

    # -------------------------------------------------------------- requests
    def request(self, name: str, kind: str, platform: str | None, capabilities: list[Any]) -> tuple[SetupRequest, str]:
        now = time.time()
        with self._lock:
            self._expire()
            self._recent = [t for t in self._recent if now - t < 60]
            if len(self._recent) >= MAX_REQUESTS_PER_MINUTE:
                raise AuthError("too many setup requests; wait a minute and try again")
            self._recent.append(now)
            # One at a time: a new request (e.g. after a reload) replaces the waiting one.
            for r in self._requests.values():
                if r.status == "pending":
                    r.status = "cancelled"
                    r.decided.set()
                    if self.prompt is not None:
                        self.prompt.dismiss(r.id)
            secret = secrets.token_urlsafe(32)
            redactor.register(secret)
            req = SetupRequest(id="setup_" + uuid.uuid4().hex[:12], secret_hash=hash_token(secret),
                               code=f"{secrets.randbelow(10000):04d}", name=(name or "Browser").strip()[:80],
                               kind=kind if kind in VALID_KINDS else "web", platform=platform,
                               capabilities=capabilities[:20], created_at=now, expires_at=now + REQUEST_TTL_S)
            self._requests[req.id] = req
        self.auth._audit(None, "setup.request", req.id, "pending", {"name": req.name, "kind": req.kind})
        log.info("setup requested by %s on this computer (%s, code %s)", req.name, req.id, req.code)
        return req, secret

    def _get(self, rid: str) -> SetupRequest:
        self._expire()
        req = self._requests.get(rid)
        if req is None:
            raise KeyError(rid)
        return req

    def check_secret(self, rid: str, secret: str) -> SetupRequest:
        with self._lock:
            req = self._get(rid)
        if not secrets.compare_digest(req.secret_hash, hash_token(secret or "")):
            raise PermissionError("not your request")
        return req

    def approve(self, rid: str, by: Principal | None, via: str) -> SetupRequest:
        with self._lock:
            req = self._get(rid)
            if req.status != "pending":
                raise ValueError(f"the request is {req.status}")
            device, token = self.auth.create_device(
                req.name, req.kind, role="owner", platform=req.platform, max_risk=Risk.DANGEROUS,
                capabilities=req.capabilities,
                metadata={"enrolled": "local approval", "approved_via": via, "approved_by": by.device_id if by else None})
            req.status, req.device_id, req.token = "approved", device.id, token
            req.decided_by = by.device_id if by else via
            req.decided.set()
        self.auth._audit(by.device_id if by else None, "setup.approve", device.id, "ok", {"request": rid, "via": via})
        log.info("setup of %s approved (%s) via %s", req.name, device.id, via)
        if self.prompt is not None and via != "window":
            self.prompt.dismiss(rid)
        return req

    def deny(self, rid: str, by: Principal | None, via: str) -> SetupRequest:
        with self._lock:
            req = self._get(rid)
            if req.status != "pending":
                raise ValueError(f"the request is {req.status}")
            req.status = "denied"
            req.decided_by = by.device_id if by else via
            req.decided.set()
        self.auth._audit(by.device_id if by else None, "setup.deny", rid, "ok", {"via": via})
        log.info("setup of %s denied via %s", req.name, via)
        if self.prompt is not None and via != "window":
            self.prompt.dismiss(rid)
        return req

    def cancel(self, rid: str, secret: str) -> None:
        req = self.check_secret(rid, secret)
        with self._lock:
            if req.status == "pending":
                req.status = "cancelled"
                req.decided.set()
        if self.prompt is not None:
            self.prompt.dismiss(rid)

    async def claim(self, rid: str, secret: str, wait_s: float = 25.0) -> dict[str, Any]:
        """Wait (up to ``wait_s``) for a decision; an approved request hands over its credential once."""
        req = self.check_secret(rid, secret)
        if req.status == "pending":
            await asyncio.to_thread(req.decided.wait, wait_s)
        with self._lock:
            self._expire()
            if req.status == "approved" and req.token:
                token, req.token, req.status = req.token, None, "claimed"
                session = self.auth.create_session(token)
                device = self.auth.get_device(req.device_id)
                self.auth._audit(req.device_id, "setup.claim", req.device_id, "ok", {"request": rid})
                return {"status": "approved", "device": device.to_dict() if device else None, "device_token": token, **session}
            return {"status": req.status, "code": req.code, "expires_at": req.expires_at}

    # ------------------------------------------------------- local confirmation
    def ask_locally(self, req: SetupRequest, loop: asyncio.AbstractEventLoop) -> None:
        """Show the confirmation window on this computer, if it can show one, and act on the answer."""
        if self.prompt is None or not self.prompt.available():
            return

        def run() -> None:
            answer = self.prompt.ask(req.id, "JARVIS: set up a device",
                                     f"{req.name} on this computer wants to use JARVIS with full access.\n\n"
                                     f"The browser shows the code {req.code}.\n\n"
                                     "Allow it? Choose No if you didn't just ask for this.", REQUEST_TTL_S)
            try:
                if answer is True:
                    self.approve(req.id, None, "window")
                elif answer is False:
                    self.deny(req.id, None, "window")
            except (KeyError, ValueError):
                pass  # already decided elsewhere, cancelled or expired

        threading.Thread(target=run, name="jarvis-setup-prompt", daemon=True).start()
