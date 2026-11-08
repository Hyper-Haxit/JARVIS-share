"""Recovery of tool calls that a model wrote as text.

Small local models sometimes print a tool call (``memory_save({...})``,
``<tool_call>{...}</tool_call>`` or a JSON object with name/arguments) instead
of using the structured tool-calling channel. Left alone, the model then says
"done" although nothing ran. These calls are recovered into real
:class:`ToolCall` objects, which still go through schema validation and the
permission gate like any other call. Only names of tools offered in the current
turn are recognised.
"""

from __future__ import annotations

import json
import re
from typing import Any

from jarvis.models.base import ToolCall

_TAGGED = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_FENCED = re.compile(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", re.S)


def _balanced_json(text: str, start: int) -> tuple[str, int] | None:
    """Return the balanced {...} substring starting at ``start``."""
    if start >= len(text) or text[start] != "{":
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1], i + 1
    return None


def _from_obj(obj: Any, names: set[str]) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or (obj.get("function") or {}).get("name")
    args = obj.get("arguments", obj.get("parameters", (obj.get("function") or {}).get("arguments", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if name in names and isinstance(args, dict):
        return ToolCall(name=name, arguments=args)
    return None


def recover_tool_calls(text: str, tool_names: set[str]) -> tuple[list[ToolCall], str]:
    """Extract textual tool calls. Returns (calls, text with the calls removed)."""
    if not text or not tool_names:
        return [], text
    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []

    for pattern in (_TAGGED, _FENCED):
        for m in pattern.finditer(text):
            try:
                call = _from_obj(json.loads(m.group(1)), tool_names)
            except ValueError:
                call = None
            if call:
                calls.append(call)
                spans.append(m.span())

    # name({...}) style
    for name in tool_names:
        for m in re.finditer(rf"(?<![\w.]){re.escape(name)}\s*\(\s*", text):
            if any(s <= m.start() < e for s, e in spans):
                continue
            found = _balanced_json(text, m.end())
            if not found:
                continue
            body, end = found
            close = re.match(r"\s*\)", text[end:])
            if not close:
                continue
            try:
                args = json.loads(body)
            except ValueError:
                continue
            if isinstance(args, dict):
                calls.append(ToolCall(name=name, arguments=args))
                spans.append((m.start(), end + close.end()))

    # bare JSON object {"name": ..., "arguments": ...} occupying the whole reply
    if not calls:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                call = _from_obj(json.loads(stripped), tool_names)
            except ValueError:
                call = None
            if call:
                return [call], ""

    if not calls:
        return [], text
    remaining = text
    for s, e in sorted(spans, reverse=True):
        remaining = remaining[:s] + remaining[e:]
    return calls, remaining.strip()
