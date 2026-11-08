"""Core API routes: auth, chat, conversations, system, models, memory, tools,
permissions, devices, capabilities and events."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from jarvis import __version__
from jarvis.api.deps import interactive, owner, principal, runtime
from jarvis.capabilities.base import ToolContext
from jarvis.core.auth import AuthError
from jarvis.core.context import DeviceContext
from jarvis.core.diagnostics import run_diagnostics
from jarvis.core.observability import get_logger
from jarvis.core.permissions import PermissionDenied, Principal, Risk
from jarvis.models.base import ImageInput, ToolCall

log = get_logger("api")
router = APIRouter(prefix="/api/v1")


# ------------------------------------------------------------------- health
@router.get("/health", tags=["system"])
async def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "time": time.time()}


# --------------------------------------------------------------------- auth
class PairBody(BaseModel):
    code: str = Field(min_length=6, max_length=32)
    name: str = Field(min_length=1, max_length=80)
    kind: str | None = None
    platform: str | None = None
    capabilities: list[Any] = Field(default_factory=list)


class SessionBody(BaseModel):
    device_token: str


class PairingCodeBody(BaseModel):
    kind: str = "web"
    role: Literal["client", "owner", "hardware"] = "client"
    max_risk: Literal["SAFE", "LOW_RISK", "SENSITIVE", "DANGEROUS"] = "SENSITIVE"
    name_hint: str | None = None


@router.post("/auth/pair", tags=["auth"])
async def pair(body: PairBody, rt: Any = Depends(runtime)) -> dict[str, Any]:
    try:
        device, token = rt.auth.redeem_pairing_code(body.code, body.name, body.kind, body.platform, body.capabilities)
    except AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e)) from e
    session = rt.auth.create_session(token)
    return {"device": device.to_dict(), "device_token": token, **session}


@router.post("/auth/session", tags=["auth"])
async def session(body: SessionBody, rt: Any = Depends(runtime)) -> dict[str, Any]:
    try:
        return rt.auth.create_session(body.device_token)
    except AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e)) from e


@router.post("/auth/logout", tags=["auth"])
async def logout(p: Principal = Depends(principal), rt: Any = Depends(runtime)) -> dict[str, Any]:
    if p.session_id:
        rt.auth.end_session(p.session_id)
    return {"ok": True}


@router.post("/auth/device/forget", tags=["auth"])
async def forget_this_device(p: Principal = Depends(principal), rt: Any = Depends(runtime)) -> dict[str, Any]:
    """A device revokes its own credential (e.g. `jarvis auth logout`).

    The host's own owner credential is managed by the core and can't be revoked
    this way, so the computer running JARVIS can never lock itself out.
    """
    device = rt.auth.get_device(p.device_id)
    if device is None:
        raise HTTPException(404, "device not found")
    if device.metadata.get("local"):
        raise HTTPException(409, "this computer's own credential is managed by JARVIS and can't be revoked here")
    rt.auth.revoke_device(p.device_id, by=p.device_id)
    gateway = rt.extensions.get("gateway")
    if gateway:
        await gateway.disconnect(p.device_id, "revoked")
    return {"revoked": p.device_id}


@router.get("/auth/whoami", tags=["auth"])
async def whoami(p: Principal = Depends(principal), rt: Any = Depends(runtime)) -> dict[str, Any]:
    device = rt.auth.get_device(p.device_id)
    return {"principal": p.to_dict(), "device": device.to_dict() if device else None, "session_id": p.session_id}


@router.post("/auth/pairing-codes", tags=["auth"])
async def create_pairing_code(body: PairingCodeBody, request: Request, p: Principal = Depends(owner),
                              rt: Any = Depends(runtime)) -> dict[str, Any]:
    try:
        out = rt.auth.create_pairing_code(p, body.kind, body.role, Risk[body.max_risk], body.name_hint)
    except AuthError as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e)) from e
    base = str(request.base_url).rstrip("/")
    out["url"] = f"{base}/#pair={out['code']}"
    return out


# --------------------------------------------------------- set up this device
# A browser on the computer JARVIS runs on can be set up without a pairing
# code: it asks, and the person at this computer approves (core/enrollment.py).
class SetupBody(BaseModel):
    name: str = Field(default="Browser", min_length=1, max_length=80)
    kind: str = "web"
    platform: str | None = Field(default=None, max_length=80)
    capabilities: list[Any] = Field(default_factory=list, max_length=20)


class SetupSecretBody(BaseModel):
    secret: str = Field(min_length=16, max_length=200)


LOOPBACK_CLIENTS = {"127.0.0.1", "::1"}
LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1"}


def _this_computer(request: Request) -> bool:
    """The request comes from this computer, to JARVIS by a loopback name, and (from a browser) from JARVIS's own page.

    The Host check defeats DNS rebinding (another site's name pointed at 127.0.0.1);
    the Origin check stops other websites open in the same browser.
    """
    client = request.client.host if request.client else ""
    host_header = request.headers.get("host") or ""
    name = host_header.rsplit(":", 1)[0] if not host_header.startswith("[") else host_header[1:].split("]", 1)[0]
    if client not in LOOPBACK_CLIENTS or name.lower() not in LOOPBACK_NAMES:
        return False
    origin = request.headers.get("origin")
    return not origin or origin.rstrip("/") == f"{request.url.scheme}://{host_header}"


def _require_this_computer(request: Request) -> None:
    if not _this_computer(request):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "setting up without a code only works in a browser on the computer "
                                                       "JARVIS runs on; pair this device with a code instead")


@router.get("/auth/setup", tags=["auth"])
async def setup_status(request: Request, rt: Any = Depends(runtime)) -> dict[str, Any]:
    local = _this_computer(request)
    out: dict[str, Any] = {"this_computer": local}
    if local:
        out["first_run"] = rt.enrollment.first_run()
        out["approval"] = rt.enrollment.approval_method()
    return out


@router.post("/auth/setup/requests", tags=["auth"])
async def setup_request(body: SetupBody, request: Request, rt: Any = Depends(runtime)) -> dict[str, Any]:
    _require_this_computer(request)
    try:
        req, secret = rt.enrollment.request(body.name, body.kind, body.platform, body.capabilities)
    except AuthError as e:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(e)) from e
    rt.enrollment.ask_locally(req, asyncio.get_running_loop())
    return {"id": req.id, "secret": secret, "code": req.code, "expires_at": req.expires_at,
            "approval": rt.enrollment.approval_method()}


@router.post("/auth/setup/requests/{rid}/claim", tags=["auth"])
async def setup_claim(rid: str, body: SetupSecretBody, request: Request, rt: Any = Depends(runtime)) -> dict[str, Any]:
    """Waits up to 25 s for the decision, then reports it; an approval hands over the device credential once."""
    _require_this_computer(request)
    try:
        return await rt.enrollment.claim(rid, body.secret)
    except KeyError as e:
        raise HTTPException(404, "unknown or expired setup request") from e
    except PermissionError as e:
        raise HTTPException(403, "not your setup request") from e


@router.post("/auth/setup/requests/{rid}/cancel", tags=["auth"])
async def setup_cancel(rid: str, body: SetupSecretBody, request: Request, rt: Any = Depends(runtime)) -> dict[str, Any]:
    _require_this_computer(request)
    try:
        rt.enrollment.cancel(rid, body.secret)
    except KeyError as e:
        raise HTTPException(404, "unknown or expired setup request") from e
    except PermissionError as e:
        raise HTTPException(403, "not your setup request") from e
    return {"ok": True}


@router.get("/auth/setup/requests", tags=["auth"])
async def setup_pending(p: Principal = Depends(owner), rt: Any = Depends(runtime)) -> list[dict[str, Any]]:
    return rt.enrollment.pending()


@router.post("/auth/setup/requests/{rid}/{decision}", tags=["auth"])
async def setup_decide(rid: str, decision: Literal["approve", "deny"], p: Principal = Depends(owner),
                       rt: Any = Depends(runtime)) -> dict[str, Any]:
    try:
        req = (rt.enrollment.approve if decision == "approve" else rt.enrollment.deny)(rid, p, f"device {p.device_id}")
    except KeyError as e:
        raise HTTPException(404, "unknown or expired setup request") from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    return req.public() | {"device_id": req.device_id}


# --------------------------------------------------------------------- chat
class ImageBody(BaseModel):
    data_b64: str
    media_type: str = "image/png"
    source: str = "upload"
    captured_at: float | None = None


class DeviceBody(BaseModel):
    voice: bool = False
    capabilities: list[str] = Field(default_factory=list)
    location: str | None = None


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=100_000)
    conversation_id: str | None = None
    images: list[ImageBody] = Field(default_factory=list, max_length=8)
    role: Literal["fast", "reasoning", "coding", "vision", "fallback"] | None = None
    model: str | None = None
    stream: bool = True
    tool_categories: list[str] = Field(default_factory=list)
    device: DeviceBody = Field(default_factory=DeviceBody)


def _device_context(p: Principal, body: DeviceBody, rt: Any) -> DeviceContext:
    device = rt.auth.get_device(p.device_id)
    caps = body.capabilities or ([c if isinstance(c, str) else c.get("name", "") for c in device.capabilities] if device else [])
    return DeviceContext(device_id=p.device_id, name=p.name, kind=p.kind, capabilities=[c for c in caps if c],
                         voice=body.voice, location=body.location)


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()


@router.post("/chat", tags=["chat"])
async def chat(body: ChatBody, request: Request, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> Any:
    images = []
    for img in body.images:
        try:
            raw = base64.b64decode(img.data_b64, validate=True)
        except ValueError as e:
            raise HTTPException(422, "image data is not valid base64") from e
        if len(raw) > 15_000_000:
            raise HTTPException(413, "image too large")
        images.append(ImageInput(data_b64=img.data_b64, media_type=img.media_type, source=img.source,
                                 captured_at=img.captured_at or time.time()))
    conv_id = body.conversation_id if body.conversation_id and rt.conversations.exists(body.conversation_id) else \
        rt.conversations.create(p.device_id)
    cancel = asyncio.Event()
    active: dict[str, asyncio.Event] = request.app.state.active_runs
    active[conv_id] = cancel
    device = _device_context(p, body.device, rt)
    kwargs = dict(text=body.message, principal=p, conversation_id=conv_id, images=images, device=device,
                  role=body.role, model_ref=body.model, tool_categories=set(body.tool_categories), cancel=cancel)

    if not body.stream:
        try:
            result = await rt.agent.run(**kwargs)
        finally:
            active.pop(conv_id, None)
        return result.to_dict()

    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def emit(event: str, data: dict[str, Any]) -> None:
        await queue.put(_sse(event, data))

    approvals = rt.events.subscribe(["PERMISSION_*"])

    async def forward_approvals() -> None:
        # Approval requests raised during this run are surfaced inline in the stream.
        mine: set[str] = set()
        async for evt in approvals:
            aid = evt.data.get("approval_id")
            if evt.type == "PERMISSION_REQUESTED" and evt.data.get("conversation_id") == conv_id and aid:
                mine.add(aid)
                await queue.put(_sse("approval", evt.data))
            elif evt.type in ("PERMISSION_GRANTED", "PERMISSION_DENIED") and aid in mine:
                await queue.put(_sse("approval_decided", {**evt.data, "approved": evt.type == "PERMISSION_GRANTED"}))

    async def runner() -> None:
        fwd = asyncio.create_task(forward_approvals())
        try:
            await rt.agent.run(emit=emit, **kwargs)
        except Exception as e:  # noqa: BLE001
            log.exception("chat run failed")
            await queue.put(_sse("error", {"error": f"{type(e).__name__}: {e}"}))
        finally:
            fwd.cancel()
            approvals.close()
            active.pop(conv_id, None)
            await queue.put(None)

    task = asyncio.create_task(runner())
    request.app.state.background.add(task)
    task.add_done_callback(request.app.state.background.discard)

    async def stream() -> Any:
        yield _sse("start", {"conversation_id": conv_id})
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/chat/{conversation_id}/cancel", tags=["chat"])
async def cancel_chat(conversation_id: str, request: Request, p: Principal = Depends(interactive)) -> dict[str, Any]:
    ev = request.app.state.active_runs.get(conversation_id)
    if ev is None:
        return {"cancelled": False, "detail": "no active run"}
    ev.set()
    return {"cancelled": True}


@router.get("/conversations", tags=["chat"])
async def list_conversations(limit: int = 50, offset: int = 0, p: Principal = Depends(interactive),
                             rt: Any = Depends(runtime)) -> list[dict[str, Any]]:
    return rt.conversations.list(min(limit, 200), offset)


@router.get("/conversations/{cid}", tags=["chat"])
async def get_conversation(cid: str, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    conv = rt.conversations.get(cid)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    return conv


class RenameBody(BaseModel):
    title: str = Field(min_length=1, max_length=200)


@router.patch("/conversations/{cid}", tags=["chat"])
async def rename_conversation(cid: str, body: RenameBody, p: Principal = Depends(interactive),
                              rt: Any = Depends(runtime)) -> dict[str, Any]:
    rt.conversations.rename(cid, body.title)
    return {"ok": True}


@router.delete("/conversations/{cid}", tags=["chat"])
async def delete_conversation(cid: str, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    if not rt.conversations.delete(cid):
        raise HTTPException(404, "conversation not found")
    return {"deleted": cid}


# ------------------------------------------------------------------- system
@router.get("/system/status", tags=["system"])
async def system_status(p: Principal = Depends(principal), rt: Any = Depends(runtime)) -> dict[str, Any]:
    return await rt.status()


@router.get("/system/identity", tags=["system"])
async def get_identity(p: Principal = Depends(principal), rt: Any = Depends(runtime)) -> dict[str, Any]:
    return rt.identity.get().model_dump()


@router.put("/system/identity", tags=["system"])
async def put_identity(patch: dict[str, Any], p: Principal = Depends(owner), rt: Any = Depends(runtime)) -> dict[str, Any]:
    try:
        return rt.identity.update(patch).model_dump()
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.get("/system/diagnostics", tags=["system"])
async def diagnostics(p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> list[dict[str, Any]]:
    return await run_diagnostics(rt)


@router.get("/system/audit", tags=["system"])
async def audit(limit: int = 100, p: Principal = Depends(owner), rt: Any = Depends(runtime)) -> list[dict[str, Any]]:
    rows = rt.db.query("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (min(limit, 1000),))
    for r in rows:
        r["details"] = json.loads(r.pop("details_json") or "{}")
    return rows


@router.get("/system/metrics", tags=["system"])
async def metrics(hours: float = 24, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    tools = rt.db.query(
        "SELECT target AS tool, COUNT(*) AS calls, SUM(outcome='ok') AS ok FROM audit_log WHERE action='tool.execute' "
        "AND ts > ? GROUP BY target ORDER BY calls DESC", (time.time() - hours * 3600,))
    return {"models": rt.router.usage_summary(hours * 3600), "tools": tools}


@router.get("/system/logs", tags=["system"])
async def logs(lines: int = 200, p: Principal = Depends(owner), rt: Any = Depends(runtime)) -> dict[str, Any]:
    path: Path = rt.paths.logs / "jarvis.log"
    if not path.exists():
        return {"lines": []}
    data = path.read_text("utf-8", errors="replace").splitlines()[-min(lines, 2000):]
    return {"lines": data}


# ------------------------------------------------------------------- models
@router.get("/models", tags=["models"])
async def list_models(p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    out = []
    for name, provider in rt.router.providers.items():
        health = await rt.router.provider_health(name)
        try:
            models = [m.model_dump() for m in await provider.list_models()] if health.ok else []
            err = None
        except Exception as e:  # noqa: BLE001
            models, err = [], str(e)
        out.append({"name": name, "type": provider.type, "local": provider.local, "health": health.model_dump(),
                    "models": models, "error": err})
    return {"providers": out, "routing": rt.router.routing_table(), "provider_errors": rt.provider_errors}


class RoutingBody(BaseModel):
    mode: Literal["auto", "local_only", "cloud_preferred"] | None = None
    roles: dict[str, list[str] | None] = Field(default_factory=dict)


@router.put("/models/routing", tags=["models"])
async def put_routing(body: RoutingBody, p: Principal = Depends(owner), rt: Any = Depends(runtime)) -> dict[str, Any]:
    try:
        if body.mode:
            rt.router.set_mode(body.mode)
        for role, refs in body.roles.items():
            rt.router.set_role_override(role, refs)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return rt.router.routing_table()


@router.post("/models/health", tags=["models"])
async def refresh_health(p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> list[dict[str, Any]]:
    return [h.model_dump() for h in await rt.router.all_health(force=True)]


# ------------------------------------------------------------------- memory
class MemoryBody(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    layer: Literal["long_term", "episodic"] = "long_term"
    kind: str = "fact"
    tags: list[str] = Field(default_factory=list)
    importance: float = Field(0.5, ge=0, le=1)


class MemoryPatch(BaseModel):
    content: str | None = None
    tags: list[str] | None = None
    importance: float | None = Field(None, ge=0, le=1)


class ForgetBody(BaseModel):
    ids: list[str] = Field(default_factory=list)
    query: str | None = None


class IngestBody(BaseModel):
    path: str


@router.get("/memory", tags=["memory"])
async def memory_list(q: str | None = None, layer: str | None = None, limit: int = 100, offset: int = 0,
                      p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    if q:
        items = await rt.memory.search(q, limit=min(limit, 100), layers=[layer] if layer else None)
    else:
        items = rt.memory.list(layer, min(limit, 500), offset)
    return {"items": items, "counts": rt.memory.count(), "embedding": rt.memory.embedding_status}


@router.post("/memory", tags=["memory"])
async def memory_add(body: MemoryBody, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    return await rt.memory.add(body.content, body.layer, body.kind, body.tags, source=f"device:{p.device_id}",
                               importance=body.importance)


@router.patch("/memory/{mid}", tags=["memory"])
async def memory_patch(mid: str, body: MemoryPatch, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    try:
        return await rt.memory.update(mid, body.content, body.tags, body.importance)
    except KeyError as e:
        raise HTTPException(404, "memory not found") from e


@router.delete("/memory/{mid}", tags=["memory"])
async def memory_delete(mid: str, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    if not await rt.memory.delete(mid):
        raise HTTPException(404, "memory not found")
    return {"deleted": mid}


@router.post("/memory/forget", tags=["memory"])
async def memory_forget(body: ForgetBody, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    if not body.ids and not body.query:
        raise HTTPException(422, "provide ids or query")
    return {"deleted": await rt.memory.forget(body.ids or None, body.query)}


@router.get("/memory/export", tags=["memory"])
async def memory_export(include_semantic: bool = False, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> JSONResponse:
    return JSONResponse(rt.memory.export(include_semantic),
                        headers={"Content-Disposition": "attachment; filename=jarvis-memory-export.json"})


@router.post("/memory/import", tags=["memory"])
async def memory_import(data: dict[str, Any], p: Principal = Depends(owner), rt: Any = Depends(runtime)) -> dict[str, Any]:
    try:
        return await rt.memory.import_(data)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/memory/ingest", tags=["memory"])
async def memory_ingest(body: IngestBody, p: Principal = Depends(owner), rt: Any = Depends(runtime)) -> dict[str, Any]:
    try:
        return await rt.memory.ingest(Path(body.path))
    except FileNotFoundError as e:
        raise HTTPException(404, f"path not found: {e}") from e


# -------------------------------------------------------------------- tools
@router.get("/tools", tags=["tools"])
async def list_tools(p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> list[dict[str, Any]]:
    return [t.info() for t in rt.tools.all()]


class InvokeBody(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = "manual invocation"


@router.post("/tools/{name}/invoke", tags=["tools"])
async def invoke_tool(name: str, body: InvokeBody, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    ctx = ToolContext(runtime=rt, principal=p, reason=body.reason, device_id=p.device_id, workspace=rt.paths.workspace)
    res, record = await rt.executor.execute(ToolCall(name=name, arguments=body.args), ctx)
    return {"result": res.to_public(), "record": record.to_dict()}


# -------------------------------------------------------------- permissions
class DecisionBody(BaseModel):
    approve: bool
    scope: Literal["once", "conversation", "task", "global"] = "once"


class PolicyBody(BaseModel):
    defaults: dict[str, Literal["allow", "ask", "deny"]] | None = None
    tool_overrides: dict[str, Literal["allow", "ask", "deny"] | None] | None = None
    approval_timeout_s: float | None = Field(None, ge=10, le=3600)


@router.get("/permissions/policy", tags=["permissions"])
async def get_policy(p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    cfg = rt.config.permissions
    return {"defaults": cfg.defaults, "tool_overrides": cfg.tool_overrides, "approval_timeout_s": cfg.approval_timeout_s,
            "levels": [r.name for r in Risk],
            "invariants": ["DANGEROUS actions always require an explicit, single-use approval",
                           "actions above a device's max risk are denied",
                           "hardware devices can never approve actions"]}


@router.put("/permissions/policy", tags=["permissions"])
async def put_policy(body: PolicyBody, p: Principal = Depends(owner), rt: Any = Depends(runtime)) -> dict[str, Any]:
    cfg = rt.config.permissions
    if body.defaults:
        for level, decision in body.defaults.items():
            if level not in Risk.__members__:
                raise HTTPException(422, f"unknown level {level}")
            if level == "DANGEROUS" and decision == "allow":
                raise HTTPException(422, "DANGEROUS actions cannot be auto-allowed")
            cfg.defaults[level] = decision
    if body.tool_overrides:
        for tool_name, decision in body.tool_overrides.items():
            if decision is None:
                cfg.tool_overrides.pop(tool_name, None)
            else:
                cfg.tool_overrides[tool_name] = decision
    if body.approval_timeout_s:
        cfg.approval_timeout_s = body.approval_timeout_s
    rt.persist_config()
    return await get_policy(p, rt)


@router.get("/permissions/approvals", tags=["permissions"])
async def approvals(state: Literal["pending", "all"] = "pending", p: Principal = Depends(interactive),
                    rt: Any = Depends(runtime)) -> list[dict[str, Any]]:
    return rt.gate.pending() if state == "pending" else rt.gate.history()


@router.post("/permissions/approvals/{approval_id}", tags=["permissions"])
async def decide(approval_id: str, body: DecisionBody, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    try:
        return await rt.gate.decide(approval_id, body.approve, p, body.scope)
    except KeyError as e:
        raise HTTPException(404, "approval not found") from e
    except PermissionDenied as e:
        raise HTTPException(403, str(e)) from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/permissions/grants", tags=["permissions"])
async def grants(p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> list[dict[str, Any]]:
    return rt.gate.grants()


@router.delete("/permissions/grants/{grant_id}", tags=["permissions"])
async def revoke_grant(grant_id: str, p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    if not rt.gate.revoke_grant(grant_id):
        raise HTTPException(404, "grant not found")
    return {"revoked": grant_id}


# ------------------------------------------------------------------ devices
class DevicePatch(BaseModel):
    name: str | None = Field(None, max_length=80)
    max_risk: Literal["SAFE", "LOW_RISK", "SENSITIVE", "DANGEROUS"] | None = None


@router.get("/devices", tags=["devices"])
async def list_devices(p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> list[dict[str, Any]]:
    gateway = rt.extensions.get("gateway")
    online = gateway.online_ids() if gateway else set()
    out = []
    for d in rt.auth.list_devices():
        item = d.to_dict()
        item["online"] = d.id in online
        item["current"] = d.id == p.device_id
        out.append(item)
    return out


@router.patch("/devices/{device_id}", tags=["devices"])
async def patch_device(device_id: str, body: DevicePatch, p: Principal = Depends(owner), rt: Any = Depends(runtime)) -> dict[str, Any]:
    try:
        device = rt.auth.update_device(device_id, name=body.name, max_risk=Risk[body.max_risk] if body.max_risk else None)
    except AuthError as e:
        raise HTTPException(404, str(e)) from e
    return device.to_dict()


@router.post("/devices/{device_id}/revoke", tags=["devices"])
async def revoke_device(device_id: str, p: Principal = Depends(owner), rt: Any = Depends(runtime)) -> dict[str, Any]:
    if device_id == p.device_id:
        raise HTTPException(409, "a device cannot revoke itself; use another owner device")
    if not rt.auth.revoke_device(device_id, by=p.device_id):
        raise HTTPException(404, "device not found or already revoked")
    gateway = rt.extensions.get("gateway")
    if gateway:
        await gateway.disconnect(device_id, "revoked")
    return {"revoked": device_id}


@router.get("/capabilities", tags=["devices"])
async def capabilities(p: Principal = Depends(interactive), rt: Any = Depends(runtime)) -> dict[str, Any]:
    host: dict[str, list[dict[str, Any]]] = {}
    for t in rt.tools.all():
        info = t.info()
        host.setdefault(t.category, []).append({k: info[k] for k in ("name", "capability", "risk", "available", "unavailable_reason")})
    gateway = rt.extensions.get("gateway")
    return {"host": host, "devices": gateway.capability_graph() if gateway else []}


# ------------------------------------------------------------------- events
@router.get("/events/recent", tags=["events"])
async def recent_events(after: int = 0, limit: int = 100, types: str | None = None, p: Principal = Depends(interactive),
                        rt: Any = Depends(runtime)) -> list[dict[str, Any]]:
    patterns = [t.strip() for t in types.split(",")] if types else None
    return rt.events.recent(min(limit, 1000), patterns, after)


HARDWARE_VISIBLE = ["EMERGENCY_STOP*", "HUD_UPDATE", "DEVICE_REVOKED", "SYSTEM_*"]


@router.websocket("/events")
async def events_ws(ws: WebSocket) -> None:
    rt = ws.app.state.runtime
    await ws.accept()
    try:
        first = await asyncio.wait_for(ws.receive_json(), timeout=10)
    except (TimeoutError, WebSocketDisconnect, ValueError):
        await ws.close(code=4401)
        return
    p = rt.auth.authenticate(first.get("token") if isinstance(first, dict) else None)
    if p is None:
        await ws.send_json({"type": "error", "error": "authentication failed"})
        await ws.close(code=4401)
        return
    patterns = first.get("patterns") or ["*"]
    if p.role == "hardware":
        patterns = HARDWARE_VISIBLE
    sub = rt.events.subscribe(patterns, maxsize=2000)
    await ws.send_json({"type": "ready", "principal": p.to_dict()})
    revoked = asyncio.Event()

    async def reader() -> None:
        try:
            while True:
                msg = await ws.receive_json()
                if isinstance(msg, dict) and msg.get("type") == "subscribe" and p.role != "hardware":
                    sub.patterns = [str(x) for x in msg.get("patterns") or ["*"]][:50]
                elif isinstance(msg, dict) and msg.get("type") == "ping":
                    await ws.send_json({"type": "pong", "ts": time.time()})
        except (WebSocketDisconnect, RuntimeError, ValueError):
            revoked.set()

    reader_task = asyncio.create_task(reader())
    try:
        while not revoked.is_set():
            # Wake up as soon as the client goes away so its subscription is released immediately.
            getter = asyncio.ensure_future(sub.get(timeout=15))
            await asyncio.wait({getter, reader_task}, return_when=asyncio.FIRST_COMPLETED)
            if not getter.done():
                getter.cancel()
                break
            evt = getter.result()
            if evt is None:
                if rt.auth.authenticate(first.get("token")) is None:
                    break
                await ws.send_json({"type": "heartbeat", "ts": time.time()})
                continue
            if evt.type == "DEVICE_REVOKED" and evt.data.get("device_id") == p.device_id:
                await ws.send_json({"type": "event", "event": evt.to_dict()})
                break
            await ws.send_json({"type": "event", "event": evt.to_dict()})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        sub.close()
        reader_task.cancel()
        try:
            await ws.close()
        except RuntimeError:
            pass
