"""Shared API dependencies: authentication, authorization, rate limiting."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from jarvis.core.observability import get_logger
from jarvis.core.permissions import Principal

log = get_logger("api.auth")


def runtime(request: Request) -> Any:
    return request.app.state.runtime


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return None


_auth_log_at: dict[tuple[str, str], float] = {}


def _log_auth_failure(rt: Any, request: Request, token: str | None) -> None:
    """Record why a request was refused, without the credential, at most once a minute per client and reason."""
    reason = rt.auth.diagnose(token)
    client = request.client.host if request.client else "unknown"
    key = (client, reason)
    now = time.monotonic()
    if now - _auth_log_at.get(key, -1e9) >= 60:
        _auth_log_at[key] = now
        log.warning("refused %s %s from %s: %s", request.method, request.url.path, client, reason)


async def principal(request: Request) -> Principal:
    rt = request.app.state.runtime
    token = _bearer(request)
    p = rt.auth.authenticate(token)
    if p is None:
        _log_auth_failure(rt, request, token)
        # The response stays generic: callers learn nothing about which credentials exist.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "authentication required", headers={"WWW-Authenticate": "Bearer"})
    request.state.principal = p
    return p


async def interactive(p: Principal = Depends(principal)) -> Principal:
    """Endpoints that only interface devices (not pure hardware) may use."""
    if p.role == "hardware":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "hardware devices cannot use this endpoint")
    return p


async def owner(p: Principal = Depends(principal)) -> Principal:
    if not p.is_owner:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "owner device required")
    return p


class RateLimiter:
    """Token bucket per (client, bucket)."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))

    def allow(self, key: tuple[str, str], per_minute: int) -> bool:
        capacity = float(per_minute)
        rate = per_minute / 60.0
        tokens, last = self._buckets.get(key, (capacity, time.monotonic()))
        now = time.monotonic()
        tokens = min(capacity, tokens + (now - last) * rate)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False
        self._buckets[key] = (tokens - 1.0, now)
        return True
