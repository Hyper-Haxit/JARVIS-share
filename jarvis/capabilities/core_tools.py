"""Always-available core tools: time, arithmetic, system information, memory."""

from __future__ import annotations

import ast
import math
import operator
import platform
import time
from typing import Any

import psutil

from jarvis.capabilities.base import Tool, ToolContext, ToolResult
from jarvis.core.permissions import Risk

# ------------------------------------------------------------------ calculator
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {name: getattr(math, name) for name in ("sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "log", "log10",
                                                  "log2", "exp", "floor", "ceil", "factorial", "radians", "degrees", "hypot")}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def safe_eval(expr: str) -> float | int:
    """Evaluate an arithmetic expression without exec/eval of arbitrary code."""
    node = ast.parse(expr.replace("^", "**"), mode="eval")

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _BIN[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONSTS:
            return _CONSTS[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            args = [ev(a) for a in n.args]
            if n.func.id == "factorial" and args and args[0] > 1000:
                raise ValueError("factorial argument too large")
            return _FUNCS[n.func.id](*args)
        raise ValueError(f"unsupported expression element: {type(n).__name__}")

    return ev(node)


async def _calculate(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    try:
        value = safe_eval(args["expression"])
    except ZeroDivisionError:
        return ToolResult.fail("division by zero")
    except (ValueError, SyntaxError, TypeError, OverflowError) as e:
        return ToolResult.fail(f"cannot evaluate: {e}")
    return ToolResult(ok=True, content=f"{args['expression']} = {value}", data={"value": value})


async def _get_time(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    now = time.time()
    lt = time.localtime(now)
    return ToolResult(ok=True, content=time.strftime("%A %d %B %Y %H:%M:%S %Z", lt),
                      data={"epoch": now, "iso": time.strftime("%Y-%m-%dT%H:%M:%S", lt), "timezone": time.strftime("%Z", lt)})


async def _system_info(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ctx.runtime.paths.home))
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "hostname": platform.node(),
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "memory_total_gb": round(vm.total / 2**30, 1),
        "memory_used_percent": vm.percent,
        "disk_free_gb": round(disk.free / 2**30, 1),
        "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
    }
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    if battery is not None:
        info["battery_percent"] = battery.percent
        info["on_ac_power"] = battery.power_plugged
    lines = [f"{k}: {v}" for k, v in info.items()]
    return ToolResult(ok=True, content="\n".join(lines), data=info)


async def _enable_tools(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    cats = ctx.runtime.tools.categories()
    requested = [c for c in args.get("categories", []) if c]
    valid = [c for c in requested if c in cats]
    invalid = [c for c in requested if c not in cats]
    lines = [f"enabled: {', '.join(valid) or 'none'}"]
    if invalid:
        lines.append(f"unknown categories: {', '.join(invalid)}")
    lines.append("available categories: " + "; ".join(f"{c} ({', '.join(n)})" for c, n in sorted(cats.items())))
    return ToolResult(ok=bool(valid) or not requested, content="\n".join(lines), data={"enabled": valid})


# ---------------------------------------------------------------------- memory
async def _memory_save(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    record = await ctx.runtime.memory.add(args["content"], layer="long_term", kind=args.get("kind", "fact"),
                                          tags=args.get("tags", []), source=f"conversation:{ctx.conversation_id}",
                                          importance=args.get("importance", 0.6))
    return ToolResult(ok=True, content=f"Saved memory {record['id']}: {record['content']}", data=record)


async def _memory_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    hits = await ctx.runtime.memory.search(args["query"], limit=args.get("limit", 8),
                                           layers=args.get("layers") or None)
    if not hits:
        return ToolResult(ok=True, content="No matching memories.", data=[])
    return ToolResult(ok=True, content="\n".join(f"{h['id']} [{h['layer']}, score {h['score']}]: {h['content'][:400]}" for h in hits),
                      data=hits)


async def _memory_forget(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    ids = args.get("ids") or []
    if not ids:
        return ToolResult.fail("provide the ids of the memories to forget (use memory_search first)")
    found = [i for i in ids if ctx.runtime.memory.get(i) is not None]
    missing = [i for i in ids if i not in found]
    deleted = await ctx.runtime.memory.forget(ids=found)
    msg = f"Permanently deleted {len(deleted)} memory record(s): {', '.join(deleted) or 'none'}."
    if missing:
        msg += f" Not found: {', '.join(missing)}."
    return ToolResult(ok=bool(deleted), content=msg, data={"deleted": deleted, "missing": missing})


async def _working_note(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    scope = f"conversation:{ctx.conversation_id}"
    ctx.runtime.memory.working_set(scope, args["key"], args["value"], ttl_s=args.get("ttl_hours", 24) * 3600)
    return ToolResult(ok=True, content=f"Noted {args['key']} for this conversation.")


def core_tools() -> list[Tool]:
    return [
        Tool("get_time", "Get the current local date, time and timezone.", {"type": "object", "properties": {}},
             _get_time, Risk.SAFE, "system.clock", "core"),
        Tool("calculate", "Evaluate an arithmetic expression exactly (supports + - * / ** %, sqrt, log, sin, pi...).",
             {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. (3+4)*2^5"}},
              "required": ["expression"]},
             _calculate, Risk.SAFE, "compute.math", "core"),
        Tool("system_info", "Report this computer's OS, CPU, memory, disk, uptime and battery (real measurements).",
             {"type": "object", "properties": {}}, _system_info, Risk.SAFE, "system.info", "core",
             keywords=("cpu", "memory usage", "ram", "disk", "battery", "system", "uptime")),
        Tool("enable_tools", "Load additional tool categories when the current tools are not enough "
                             "(e.g. files, shell, web, browser, computer, vision, coding, home, devices, tasks, calendar).",
             {"type": "object", "properties": {"categories": {"type": "array", "items": {"type": "string"}}},
              "required": ["categories"]},
             _enable_tools, Risk.SAFE, "meta.tools", "core"),
        Tool("memory_save", "Store a durable fact or preference about the user in long-term memory. Use when the user "
                            "asks you to remember something or states a lasting preference.",
             {"type": "object", "properties": {
                 "content": {"type": "string", "description": "the fact, written as a standalone sentence"},
                 "kind": {"type": "string", "enum": ["fact", "preference", "person", "project", "routine"], "default": "fact"},
                 "tags": {"type": "array", "items": {"type": "string"}},
                 "importance": {"type": "number", "minimum": 0, "maximum": 1}},
              "required": ["content"]},
             _memory_save, Risk.LOW_RISK, "memory.write", "core",
             describe=lambda a: f"remember: \"{a['content'][:120]}\""),
        Tool("memory_search", "Search long-term, episodic and document memory.",
             {"type": "object", "properties": {
                 "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 8},
                 "layers": {"type": "array", "items": {"type": "string", "enum": ["long_term", "episodic", "semantic"]}}},
              "required": ["query"]},
             _memory_search, Risk.SAFE, "memory.read", "core"),
        Tool("memory_forget", "Permanently delete memories by id (find ids with memory_search first). Use when the user "
                              "says to forget something.",
             {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}}, "required": ["ids"]},
             _memory_forget, Risk.LOW_RISK, "memory.delete", "core",
             describe=lambda a: f"permanently delete memories {', '.join(a.get('ids', []))}"),
        Tool("working_note", "Keep a short-lived note for the current conversation (working memory), e.g. the project "
                             "being discussed.",
             {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"},
                                               "ttl_hours": {"type": "number", "default": 24}},
              "required": ["key", "value"]},
             _working_note, Risk.SAFE, "memory.working", "core"),
    ]
