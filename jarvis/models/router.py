"""Model routing.

Maps a request to a *role* (fast, reasoning, coding, vision, embedding,
fallback), then walks that role's ordered candidate list across providers,
skipping providers that are unhealthy or disallowed by the routing mode, and
falling back to the next candidate when one fails before producing output.

Users can override routing per request (explicit role or "provider/model") or
persistently (role overrides stored in the database).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from jarvis.core.events import EventBus, EventType
from jarvis.core.observability import get_logger, get_trace_id
from jarvis.models.base import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelProvider,
    ProviderError,
    ProviderHealth,
    StreamChunk,
    ToolSchema,
)

log = get_logger("router")

ROLES = ("fast", "reasoning", "coding", "vision", "embedding", "fallback")
OVERRIDES_KEY = "routing.role_overrides"
MODE_KEY = "routing.mode"


def parse_ref(ref: str) -> tuple[str, str]:
    if "/" not in ref:
        raise ValueError(f"model reference must be 'provider/model', got '{ref}'")
    provider, model = ref.split("/", 1)
    return provider, model


@dataclass
class RouteAttempt:
    provider: str
    model: str
    outcome: str  # ok | skipped | failed
    detail: str = ""


@dataclass
class RouteResult:
    role: str
    provider: str | None = None
    model: str | None = None
    local: bool | None = None
    attempts: list[RouteAttempt] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return any(a.outcome != "ok" for a in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "provider": self.provider, "model": self.model, "local": self.local,
            "degraded": self.degraded,
            "attempts": [a.__dict__ for a in self.attempts],
        }


class NoModelAvailable(ProviderError):
    def __init__(self, role: str, attempts: list[RouteAttempt]) -> None:
        reasons = "; ".join(f"{a.provider}/{a.model}: {a.outcome} ({a.detail})" for a in attempts) or "no candidates configured"
        super().__init__(f"no model available for role '{role}': {reasons}", retryable=False)
        self.role = role
        self.attempts = attempts


class ModelRouter:
    def __init__(self, config: Any, providers: dict[str, ModelProvider], db: Any, events: EventBus) -> None:
        self.config = config  # JarvisConfig
        self.providers = providers
        self.db = db
        self.events = events
        self._health: dict[str, ProviderHealth] = {}
        self._health_locks: dict[str, asyncio.Lock] = {}

    # ----------------------------------------------------------- configuration
    @property
    def mode(self) -> str:
        return self.db.kv_get(MODE_KEY) or self.config.routing.mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("auto", "local_only", "cloud_preferred"):
            raise ValueError("mode must be auto, local_only or cloud_preferred")
        self.db.kv_set(MODE_KEY, mode)

    def role_candidates(self, role: str) -> list[str]:
        if role not in ROLES:
            raise ValueError(f"unknown role '{role}'")
        overrides = self.db.kv_get(OVERRIDES_KEY) or {}
        if overrides.get(role):
            return list(overrides[role])
        return list(getattr(self.config.roles, role))

    def set_role_override(self, role: str, refs: list[str] | None) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role '{role}'")
        for ref in refs or []:
            parse_ref(ref)
        overrides = self.db.kv_get(OVERRIDES_KEY) or {}
        if refs:
            overrides[role] = refs
        else:
            overrides.pop(role, None)
        self.db.kv_set(OVERRIDES_KEY, overrides)

    def routing_table(self) -> dict[str, Any]:
        overrides = self.db.kv_get(OVERRIDES_KEY) or {}
        return {
            "mode": self.mode,
            "roles": {r: {"candidates": self.role_candidates(r), "overridden": r in overrides} for r in ROLES},
        }

    # ------------------------------------------------------------ classification
    def classify(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> str:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        if last_user is None:
            return "fast"
        if any(m.images for m in messages[-4:]):
            return "vision"
        text = last_user.content.lower()
        r = self.config.routing
        if "```" in last_user.content or any(k in text for k in r.coding_keywords):
            return "coding"
        if len(last_user.content) > r.long_message_chars or any(k in text for k in r.complex_keywords):
            return "reasoning"
        if len(re.findall(r"^\s*\d+[.)]\s", last_user.content, flags=re.M)) >= 3:
            return "reasoning"
        return "fast"

    # ------------------------------------------------------------------ health
    async def provider_health(self, name: str, force: bool = False) -> ProviderHealth:
        provider = self.providers.get(name)
        if provider is None:
            return ProviderHealth(ok=False, provider=name, local=False, error="provider not configured")
        cached = self._health.get(name)
        ttl = self.config.routing.health_cache_s if (cached and cached.ok) else min(5.0, self.config.routing.health_cache_s)
        if cached and not force and time.time() - cached.checked_at < ttl:
            return cached
        lock = self._health_locks.setdefault(name, asyncio.Lock())
        async with lock:
            cached = self._health.get(name)
            if cached and not force and time.time() - cached.checked_at < ttl:
                return cached
            health = await provider.health()
            changed = cached is None or cached.ok != health.ok
            self._health[name] = health
            if changed:
                await self.events.publish(EventType.PROVIDER_STATUS, health.model_dump())
            return health

    def mark_unhealthy(self, name: str, error: str) -> None:
        provider = self.providers.get(name)
        self._health[name] = ProviderHealth(ok=False, provider=name, local=bool(provider and provider.local), error=error)

    async def all_health(self, force: bool = False) -> list[ProviderHealth]:
        return list(await asyncio.gather(*(self.provider_health(n, force) for n in self.providers)))

    # ----------------------------------------------------------------- planning
    def _ordered(self, role: str, include_fallback: bool = True) -> list[str]:
        refs = self.role_candidates(role)
        if include_fallback and role != "embedding":
            refs += [r for r in self.role_candidates("fallback") if r not in refs]
        if self.mode == "cloud_preferred":
            def is_local(ref: str) -> bool:
                p = self.providers.get(parse_ref(ref)[0])
                return bool(p and p.local)
            refs = sorted(refs, key=is_local)  # stable: cloud first, then local
        return refs

    async def _eligible(self, ref: str, needs_images: bool) -> tuple[ModelProvider | None, str, str]:
        pname, model = parse_ref(ref)
        provider = self.providers.get(pname)
        if provider is None:
            return None, model, "provider not configured"
        if self.mode == "local_only" and not provider.local:
            return None, model, "cloud provider disabled by local_only mode"
        health = await self.provider_health(pname)
        if not health.ok:
            return None, model, f"provider unhealthy: {health.error}"
        if needs_images and pname in self.providers and hasattr(provider, "_capabilities"):
            caps = await provider._capabilities(model)  # type: ignore[attr-defined]
            if caps and "vision" not in caps:
                return None, model, "model lacks vision capability"
        return provider, model, ""

    # ---------------------------------------------------------------- execution
    async def stream(
        self,
        messages: list[Message],
        *,
        role: str | None = None,
        model_ref: str | None = None,
        tools: list[ToolSchema] | None = None,
        think: bool = False,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        route: RouteResult | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream from the first working candidate.

        ``route`` (if given) is filled in with the routing decision so callers can
        report which model actually answered and whether routing degraded.
        """
        role = role or self.classify(messages, tools)
        route = route if route is not None else RouteResult(role=role)
        route.role = role
        refs = [model_ref] if model_ref else self._ordered(role)
        needs_images = any(m.images for m in messages)
        for ref in refs:
            try:
                provider, model, why = await self._eligible(ref, needs_images)
            except ValueError as e:
                route.attempts.append(RouteAttempt(ref, "", "skipped", str(e)))
                continue
            if provider is None:
                route.attempts.append(RouteAttempt(parse_ref(ref)[0], model, "skipped", why))
                continue
            request = ChatRequest(messages=messages, model=model, tools=tools or [], think=think,
                                  json_schema=json_schema, max_tokens=max_tokens, temperature=temperature)
            started = time.perf_counter()
            await self.events.publish(EventType.MODEL_REQUEST_STARTED, {"provider": provider.name, "model": model, "role": role})
            produced = False
            try:
                async for chunk in provider.stream(request):
                    if chunk.type == "done":
                        resp = chunk.response
                        assert resp is not None
                        route.provider, route.model, route.local = provider.name, model, provider.local
                        route.attempts.append(RouteAttempt(provider.name, model, "ok"))
                        self._record(provider.name, model, role, resp.usage.input_tokens, resp.usage.output_tokens,
                                     resp.latency_ms, True, None)
                        await self.events.publish(EventType.MODEL_RESPONSE_COMPLETED, {
                            "provider": provider.name, "model": model, "role": role, "latency_ms": resp.latency_ms,
                            "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                            "stop_reason": resp.stop_reason,
                        })
                    elif not produced:
                        produced = True
                        await self.events.publish(EventType.MODEL_RESPONSE_STARTED, {"provider": provider.name, "model": model})
                    yield chunk
                return
            except ProviderError as e:
                latency = round((time.perf_counter() - started) * 1000, 1)
                self._record(provider.name, model, role, None, None, latency, False, str(e))
                await self.events.publish(EventType.MODEL_REQUEST_FAILED, {"provider": provider.name, "model": model, "error": str(e)})
                route.attempts.append(RouteAttempt(provider.name, model, "failed", str(e)))
                if e.status is None and e.retryable:
                    self.mark_unhealthy(provider.name, str(e))
                if produced:
                    # Output already reached the user; restarting on another model
                    # would duplicate or contradict it.
                    raise
                log.warning("model candidate failed, trying next", extra={"data": {"ref": ref, "error": str(e)}})
                continue
        raise NoModelAvailable(role, route.attempts)

    async def chat(self, messages: list[Message], **kwargs: Any) -> tuple[ChatResponse, RouteResult]:
        route = RouteResult(role=kwargs.get("role") or "")
        final: ChatResponse | None = None
        async for chunk in self.stream(messages, route=route, **kwargs):
            if chunk.type == "done":
                final = chunk.response
        assert final is not None
        return final, route

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], str]:
        """Embed texts with the first working embedding model. Returns (vectors, 'provider/model')."""
        attempts: list[RouteAttempt] = []
        for ref in self._ordered("embedding", include_fallback=False):
            try:
                provider, model, why = await self._eligible(ref, False)
            except ValueError as e:
                attempts.append(RouteAttempt(ref, "", "skipped", str(e)))
                continue
            if provider is None:
                attempts.append(RouteAttempt(parse_ref(ref)[0], model, "skipped", why))
                continue
            started = time.perf_counter()
            try:
                vectors = await provider.embed(texts, model)
                self._record(provider.name, model, "embedding", None, None, round((time.perf_counter() - started) * 1000, 1), True, None)
                return vectors, f"{provider.name}/{model}"
            except ProviderError as e:
                attempts.append(RouteAttempt(provider.name, model, "failed", str(e)))
        raise NoModelAvailable("embedding", attempts)

    def _record(self, provider: str, model: str, role: str, inp: int | None, out: int | None,
                latency: float, ok: bool, error: str | None) -> None:
        try:
            self.db.execute(
                "INSERT INTO model_usage(ts, provider, model, role, input_tokens, output_tokens, latency_ms, ok, error, trace_id) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (time.time(), provider, model, role, inp, out, latency, int(ok), (error or "")[:500] or None, get_trace_id()),
            )
        except Exception:  # pragma: no cover
            log.exception("failed to record model usage")

    def usage_summary(self, since_s: float = 86400) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT provider, model, role, COUNT(*) AS requests, SUM(ok) AS ok, SUM(COALESCE(input_tokens,0)) AS input_tokens, "
            "SUM(COALESCE(output_tokens,0)) AS output_tokens, ROUND(AVG(latency_ms),1) AS avg_latency_ms "
            "FROM model_usage WHERE ts > ? GROUP BY provider, model, role ORDER BY requests DESC",
            (time.time() - since_s,),
        )

    async def close(self) -> None:
        for p in self.providers.values():
            try:
                await p.close()
            except Exception:  # pragma: no cover
                pass
