"""Tool (capability) definitions, registry, schema validation and the executor.

:class:`ToolExecutor.execute` is the *only* path by which a model-requested
action runs. It validates arguments, computes the effective risk (a tool may
escalate risk based on its arguments, never lower it below its declared base),
asks the permission gate, enforces timeouts and cancellation, and writes events
and audit records. Nothing a model emits can skip these steps.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.core.events import EventBus, EventType
from jarvis.core.observability import Timer, get_logger, redact
from jarvis.core.permissions import ActionRequest, PermissionGate, Principal, Risk
from jarvis.models.base import ImageInput, ToolCall, ToolSchema

log = get_logger("tools")


@dataclass
class ToolContext:
    runtime: Any
    principal: Principal
    conversation_id: str | None = None
    task_id: str | None = None
    reason: str = ""
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    device_id: str | None = None
    workspace: Path | None = None


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    data: Any = None
    images: list[ImageInput] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def fail(cls, error: str, data: Any = None) -> ToolResult:
        return cls(ok=False, content=f"ERROR: {error}", error=error, data=data)

    def to_public(self) -> dict[str, Any]:
        return {"ok": self.ok, "content": self.content, "data": redact(self.data), "error": self.error,
                "images": [{"source": i.source, "captured_at": i.captured_at, "media_type": i.media_type} for i in self.images]}


Handler = Callable[[dict[str, Any], ToolContext], Awaitable[Any] | Any]
RiskFn = Callable[[dict[str, Any], ToolContext], Risk | None]
DescribeFn = Callable[[dict[str, Any]], str]
AvailableFn = Callable[[], tuple[bool, str]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    risk: Risk = Risk.SAFE
    capability: str = "general"
    category: str = "core"
    risk_fn: RiskFn | None = None
    describe: DescribeFn | None = None
    available: AvailableFn | None = None
    timeout_s: float = 60.0
    source: str = "builtin"
    keywords: tuple[str, ...] = ()

    def schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description, parameters=self.parameters)

    def availability(self) -> tuple[bool, str]:
        if self.available is None:
            return True, ""
        try:
            return self.available()
        except Exception as e:  # noqa: BLE001
            return False, f"availability check failed: {e}"

    def info(self) -> dict[str, Any]:
        ok, why = self.availability()
        return {
            "name": self.name, "description": self.description, "parameters": self.parameters,
            "risk": self.risk.name, "capability": self.capability, "category": self.category,
            "source": self.source, "available": ok, "unavailable_reason": why or None, "timeout_s": self.timeout_s,
        }


def tool(name: str, description: str, parameters: dict[str, Any] | None = None, **kwargs: Any) -> Callable[[Handler], Tool]:
    """Decorator turning an async function into a :class:`Tool`."""

    def wrap(fn: Handler) -> Tool:
        return Tool(name=name, description=description,
                    parameters=parameters or {"type": "object", "properties": {}}, handler=fn, **kwargs)

    return wrap


# ----------------------------------------------------------------- validation
class ValidationError(Exception):
    pass


_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "object": (dict,), "array": (list,), "null": (type(None),),
}


def _coerce(value: Any, typ: str) -> Any:
    """Lenient coercion for common small-model mistakes (e.g. "5" for 5)."""
    if isinstance(value, str):
        v = value.strip()
        if typ == "integer":
            try:
                return int(v)
            except ValueError:
                return value
        if typ == "number":
            try:
                return float(v)
            except ValueError:
                return value
        if typ == "boolean" and v.lower() in ("true", "false"):
            return v.lower() == "true"
        if typ in ("array", "object") and v[:1] in "[{":
            try:
                return json.loads(v)
            except ValueError:
                return value
    if typ == "number" and isinstance(value, bool):
        return value
    if typ == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if typ == "array" and not isinstance(value, list) and value is not None and not isinstance(value, dict):
        return [value]
    return value


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> Any:
    """Validate (and leniently coerce) ``value`` against a JSON-schema subset."""
    types = schema.get("type")
    if types is not None:
        type_list = types if isinstance(types, list) else [types]
        if not any(_matches(value, t) for t in type_list):
            for t in type_list:
                coerced = _coerce(value, t)
                if _matches(coerced, t):
                    value = coerced
                    break
            else:
                raise ValidationError(f"{path}: expected {'/'.join(type_list)}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValidationError(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValidationError(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValidationError(f"{path}: above maximum {schema['maximum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value or value[req] is None:
                raise ValidationError(f"{path}.{req}: required")
        out = {}
        for k, v in value.items():
            if k in props:
                out[k] = validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                continue  # silently drop unknown keys rather than fail on model noise
            else:
                out[k] = v
        for k, sub in props.items():
            if k not in out and "default" in sub:
                out[k] = sub["default"]
        return out
    if isinstance(value, list) and "items" in schema:
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValidationError(f"{path}: more than {schema['maxItems']} items")
        return [validate(v, schema["items"], f"{path}[{i}]") for i, v in enumerate(value)]
    return value


def _matches(value: Any, typ: str) -> bool:
    expected = _TYPES.get(typ)
    if expected is None:
        return True
    if typ in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


# ------------------------------------------------------------------- registry
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, t: Tool, replace: bool = False) -> None:
        if not t.name.replace("_", "").isalnum():
            raise ValueError(f"invalid tool name '{t.name}'")
        if t.name in self._tools and not replace:
            raise ValueError(f"tool '{t.name}' already registered")
        self._tools[t.name] = t

    def register_all(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register(t)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def unregister_source(self, source: str) -> None:
        for name in [n for n, t in self._tools.items() if t.source == source]:
            del self._tools[name]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def categories(self) -> dict[str, list[str]]:
        cats: dict[str, list[str]] = {}
        for t in self.all():
            cats.setdefault(t.category, []).append(t.name)
        return cats

    def select(self, text: str, always: set[str] | None = None, categories: set[str] | None = None,
               max_risk: Risk = Risk.DANGEROUS, limit: int = 40) -> list[Tool]:
        """Choose the tools to offer the model for a request.

        Small local models degrade badly with very large tool lists, so tools
        are offered by category: core tools always, others when the request
        mentions their keywords or the category was explicitly enabled.
        """
        text = text.lower()
        always = always or {"core"}
        chosen: list[Tool] = []
        for t in self.all():
            if t.risk > max_risk or not t.availability()[0]:
                continue
            if t.category in always or (categories and t.category in categories) or any(k in text for k in t.keywords):
                chosen.append(t)
        return chosen[:limit]


# ------------------------------------------------------------------- executor
@dataclass
class ExecutionRecord:
    tool: str
    args: dict[str, Any]
    ok: bool
    risk: str
    decision: str
    duration_ms: float
    summary: str
    error: str | None = None
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": redact(self.args), "ok": self.ok, "risk": self.risk, "decision": self.decision,
                "duration_ms": self.duration_ms, "summary": self.summary, "error": self.error}


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, gate: PermissionGate, events: EventBus, db: Any,
                 max_result_chars: int = 12000) -> None:
        self.registry = registry
        self.gate = gate
        self.events = events
        self.db = db
        self.max_result_chars = max_result_chars

    async def execute(self, call: ToolCall, ctx: ToolContext) -> tuple[ToolResult, ExecutionRecord]:
        timer = Timer()
        t = self.registry.get(call.name)
        if t is None:
            known = ", ".join(x.name for x in self.registry.all()[:60])
            res = ToolResult.fail(f"unknown tool '{call.name}'. Available tools: {known}")
            return res, ExecutionRecord(call.name, call.arguments, False, "-", "unknown", timer.ms, "unknown tool", res.error)
        if call.parse_error:
            res = ToolResult.fail(f"could not parse arguments: {call.parse_error}")
            return res, ExecutionRecord(t.name, {}, False, t.risk.name, "invalid", timer.ms, "invalid arguments", res.error)
        try:
            args = validate(call.arguments or {}, t.parameters)
        except ValidationError as e:
            res = ToolResult.fail(f"invalid arguments for {t.name}: {e}")
            return res, ExecutionRecord(t.name, call.arguments, False, t.risk.name, "invalid", timer.ms, "invalid arguments", res.error)
        ok, why = t.availability()
        if not ok:
            res = ToolResult.fail(f"{t.name} is unavailable on this system: {why}")
            return res, ExecutionRecord(t.name, args, False, t.risk.name, "unavailable", timer.ms, "unavailable", res.error)

        risk = t.risk
        if t.risk_fn is not None:
            try:
                dynamic = t.risk_fn(args, ctx)
            except Exception as e:  # noqa: BLE001 - a failing risk assessment must fail closed
                dynamic = Risk.DANGEROUS
                log.warning("risk_fn failed for %s: %s", t.name, e)
            if dynamic is not None and dynamic > risk:
                risk = dynamic
        effects = _describe(t, args)
        request = ActionRequest(
            tool=t.name, capability=t.capability, risk=risk, summary=f"{t.name}: {effects}", effects=effects,
            reason=ctx.reason[:500], args=args, conversation_id=ctx.conversation_id, task_id=ctx.task_id,
            requested_by=ctx.principal.device_id,
        )
        gate_result = await self.gate.check(request, ctx.principal)
        if not gate_result.allowed:
            res = ToolResult.fail(f"permission not granted ({gate_result.decision}): {gate_result.detail}. "
                                  "Do not retry this action unless the user asks; explain what was blocked.")
            self._audit(ctx, t, args, risk, "blocked", gate_result.decision, timer.ms, res.error)
            return res, ExecutionRecord(t.name, args, False, risk.name, gate_result.decision, timer.ms, "blocked", res.error)

        await self.events.publish(EventType.TOOL_STARTED, {
            "tool": t.name, "args": args, "risk": risk.name, "conversation_id": ctx.conversation_id, "task_id": ctx.task_id,
        })
        try:
            res = await self._run(t, args, ctx)
        except TimeoutError:
            res = ToolResult.fail(f"{t.name} timed out after {t.timeout_s:.0f}s")
        except asyncio.CancelledError:
            res = ToolResult.fail(f"{t.name} was cancelled")
            await self._finish(ctx, t, args, risk, gate_result.decision, timer, res)
            raise
        except Exception as e:  # noqa: BLE001 - tool failures are reported, never hidden
            log.exception("tool %s raised", t.name)
            res = ToolResult.fail(f"{t.name} failed: {type(e).__name__}: {e}")
        record = await self._finish(ctx, t, args, risk, gate_result.decision, timer, res)
        return res, record

    async def _run(self, t: Tool, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        async def call() -> Any:
            out = t.handler(args, ctx)
            if inspect.isawaitable(out):
                out = await out
            return out

        task = asyncio.ensure_future(call())
        cancel_wait = asyncio.ensure_future(ctx.cancel.wait())
        try:
            done, _ = await asyncio.wait({task, cancel_wait}, timeout=t.timeout_s, return_when=asyncio.FIRST_COMPLETED)
            if task not in done:
                task.cancel()
                if cancel_wait in done:
                    raise asyncio.CancelledError()
                raise TimeoutError()
            out = task.result()
        finally:
            cancel_wait.cancel()
        res = _normalize(out)
        if len(res.content) > self.max_result_chars:
            res.content = res.content[: self.max_result_chars] + f"\n...[truncated {len(res.content) - self.max_result_chars} chars]"
        return res

    async def _finish(self, ctx: ToolContext, t: Tool, args: dict[str, Any], risk: Risk, decision: str,
                      timer: Timer, res: ToolResult) -> ExecutionRecord:
        summary = (res.content or "").strip().splitlines()[0][:200] if res.content else ""
        evt = EventType.TOOL_COMPLETED if res.ok else EventType.TOOL_FAILED
        await self.events.publish(evt, {
            "tool": t.name, "ok": res.ok, "duration_ms": timer.ms, "summary": summary, "error": res.error,
            "conversation_id": ctx.conversation_id, "task_id": ctx.task_id,
        })
        self._audit(ctx, t, args, risk, "ok" if res.ok else "error", decision, timer.ms, res.error)
        return ExecutionRecord(t.name, args, res.ok, risk.name, decision, timer.ms, summary, res.error, res.data)

    def _audit(self, ctx: ToolContext, t: Tool, args: dict[str, Any], risk: Risk, outcome: str, decision: str,
               ms: float, error: str | None) -> None:
        try:
            self.db.execute(
                "INSERT INTO audit_log(ts, actor, action, target, risk, outcome, details_json, trace_id) VALUES(?,?,?,?,?,?,?,?)",
                (time.time(), ctx.principal.device_id, "tool.execute", t.name, risk.name, outcome,
                 json.dumps(redact({"args": args, "decision": decision, "duration_ms": ms, "error": error,
                                    "conversation_id": ctx.conversation_id, "task_id": ctx.task_id}), default=str)[:8000],
                 None),
            )
        except Exception:  # pragma: no cover
            log.exception("audit write failed")


def _describe(t: Tool, args: dict[str, Any]) -> str:
    if t.describe is not None:
        try:
            return t.describe(args)
        except Exception:  # noqa: BLE001
            pass
    shown = ", ".join(f"{k}={json.dumps(v, default=str)[:80]}" for k, v in redact(args).items())
    return f"run {t.name}({shown})"


def _normalize(out: Any) -> ToolResult:
    if isinstance(out, ToolResult):
        return out
    if out is None:
        return ToolResult(ok=True, content="done")
    if isinstance(out, str):
        return ToolResult(ok=True, content=out)
    try:
        return ToolResult(ok=True, content=json.dumps(out, indent=1, default=str), data=out)
    except (TypeError, ValueError):
        return ToolResult(ok=True, content=str(out))
