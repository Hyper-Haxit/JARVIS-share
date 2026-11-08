"""Planner: turns a goal into a validated task graph.

The model proposes; code validates. A plan is only accepted if ids are unique,
every dependency exists, the graph is acyclic and within size limits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from jarvis.core.observability import get_logger
from jarvis.models.base import Message

log = get_logger("planner")

MAX_STEPS = 10

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "instruction": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "tools": {"type": "array", "items": {"type": "string"}},
                    "role": {"type": "string", "enum": ["fast", "reasoning", "coding", "vision"]},
                },
                "required": ["id", "title", "instruction", "depends_on"],
            },
        },
    },
    "required": ["summary", "steps"],
}

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "achieved": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["achieved", "issues", "summary"],
}


@dataclass
class PlanStep:
    id: str
    title: str
    instruction: str
    depends_on: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Plan:
    summary: str
    steps: list[PlanStep]
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "steps": [s.to_dict() for s in self.steps], "fallback_reason": self.fallback_reason}


class PlanError(ValueError):
    pass


def extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        return json.loads(fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("no JSON object found")


def validate_plan(data: Any, known_categories: set[str] | None = None) -> Plan:
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise PlanError("plan must be an object with a 'steps' list")
    raw = data["steps"]
    if not raw:
        raise PlanError("plan has no steps")
    if len(raw) > MAX_STEPS:
        raise PlanError(f"plan has {len(raw)} steps; the limit is {MAX_STEPS}")
    steps: list[PlanStep] = []
    seen: set[str] = set()
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            raise PlanError(f"step {i} is not an object")
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(s.get("id") or f"s{i + 1}"))[:32]
        if sid in seen:
            raise PlanError(f"duplicate step id '{sid}'")
        seen.add(sid)
        instruction = str(s.get("instruction") or "").strip()
        if not instruction:
            raise PlanError(f"step '{sid}' has no instruction")
        tools = [str(t) for t in s.get("tools") or [] if isinstance(t, str)]
        if known_categories is not None:
            tools = [t for t in tools if t in known_categories]
        role = s.get("role") if s.get("role") in ("fast", "reasoning", "coding", "vision") else None
        deps = [re.sub(r"[^A-Za-z0-9_-]", "_", str(d))[:32] for d in s.get("depends_on") or []]
        steps.append(PlanStep(sid, str(s.get("title") or sid)[:120], instruction[:4000], deps, tools, role))
    ids = {s.id for s in steps}
    for s in steps:
        missing = [d for d in s.depends_on if d not in ids]
        if missing:
            raise PlanError(f"step '{s.id}' depends on unknown step(s) {missing}")
        if s.id in s.depends_on:
            raise PlanError(f"step '{s.id}' depends on itself")
    # Kahn's algorithm: reject cycles.
    indeg = {s.id: len(set(s.depends_on)) for s in steps}
    children: dict[str, list[str]] = {s.id: [] for s in steps}
    for s in steps:
        for d in set(s.depends_on):
            children[d].append(s.id)
    queue = [sid for sid, n in indeg.items() if n == 0]
    visited = 0
    while queue:
        cur = queue.pop()
        visited += 1
        for c in children[cur]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if visited != len(steps):
        raise PlanError("plan contains a dependency cycle")
    return Plan(summary=str(data.get("summary") or "")[:500], steps=steps)


class Planner:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def _categories(self) -> dict[str, list[str]]:
        cats = self.runtime.tools.categories()
        return {c: n for c, n in cats.items() if c != "core"}

    async def plan(self, goal: str, context: str = "") -> Plan:
        cats = self._categories()
        cat_text = "\n".join(f"- {c}: {', '.join(n)}" for c, n in sorted(cats.items())) or "- (only core tools)"
        system = (
            "You plan multi-step work for a personal AI assistant. Break the goal into 1 to "
            f"{MAX_STEPS} concrete steps. Each step's instruction must be self-contained (another agent "
            "executes it with only that instruction plus the outputs of the steps it depends on). Use "
            "depends_on to express ordering; independent steps may run in parallel. Put the tool categories a "
            "step needs in 'tools'. Do not invent tools. If the goal is simple, use one step. The last step "
            "should produce the final deliverable for the user.\n"
            f"Available tool categories (core tools such as memory, time and calculator are always available):\n{cat_text}\n"
            'Respond with JSON only: {"summary": str, "steps": [{"id": str, "title": str, "instruction": str, '
            '"depends_on": [str], "tools": [str], "role": "fast|reasoning|coding|vision"}]}'
        )
        user = f"Goal: {goal}" + (f"\n\nContext:\n{context}" if context else "")
        errors: list[str] = []
        for attempt in range(2):
            msgs = [Message(role="system", content=system), Message(role="user", content=user)]
            if errors:
                msgs.append(Message(role="user", content=f"Your previous plan was invalid: {errors[-1]}. Return a corrected plan."))
            try:
                resp, _ = await self.runtime.router.chat(msgs, role="reasoning", json_schema=PLAN_SCHEMA, max_tokens=2000)
                return validate_plan(extract_json(resp.message.content), set(cats))
            except (PlanError, ValueError) as e:
                errors.append(str(e))
                log.warning("plan attempt %d invalid: %s", attempt + 1, e)
        # Honest fallback: one step carrying the whole goal, with the reason recorded.
        return Plan(summary=goal[:200], steps=[PlanStep("s1", "Complete the goal", goal, [], list(cats)[:4])],
                    fallback_reason="; ".join(errors))

    async def verify(self, goal: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        lines = []
        for r in results:
            actions = ", ".join(f"{a['tool']}({'ok' if a['ok'] else 'FAILED'})" for a in r.get("actions", [])) or "none"
            lines.append(f"## {r['title']} [{r['status']}]\nActions actually executed: {actions}\nOutput: {(r.get('output') or '')[:2500]}")
        msgs = [
            Message(role="system", content=(
                "You verify whether work achieved its goal. Judge only from the step outputs and the list of actions "
                "actually executed. If a step claims something happened but no corresponding action ran, treat the "
                "claim as unverified. Respond with JSON only: "
                '{"achieved": bool, "issues": [str], "summary": str}. The summary is the final answer for the user: '
                "state what was done and what was not, plainly.")),
            Message(role="user", content=f"Goal: {goal}\n\n" + "\n\n".join(lines)),
        ]
        try:
            resp, _ = await self.runtime.router.chat(msgs, role="reasoning", json_schema=VERIFY_SCHEMA, max_tokens=1500)
            data = extract_json(resp.message.content)
            return {"achieved": bool(data.get("achieved")), "issues": [str(i) for i in data.get("issues") or []],
                    "summary": str(data.get("summary") or "").strip(), "verified_by": f"{resp.provider}/{resp.model}"}
        except Exception as e:  # noqa: BLE001
            return {"achieved": None, "issues": [f"verification could not run: {e}"], "summary": "", "verified_by": None}
