"""Task engine: executes validated task graphs as background work.

Features: dependency ordering with bounded parallelism, per-node timeout and
retries, cancellation (including withdrawal of pending approvals), progress
events, checkpoints (each node's outcome is persisted the moment it finishes, so
a restarted core resumes from the last checkpoint), one bounded re-plan after
failures, and a final verification pass that judges results against the goal
using the record of actions that actually ran.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from jarvis.capabilities.base import Tool, ToolContext, ToolResult
from jarvis.core.context import DeviceContext
from jarvis.core.events import EventType
from jarvis.core.observability import get_logger
from jarvis.core.permissions import Principal, Risk
from jarvis.core.planning import Plan, PlanError, Planner, validate_plan

log = get_logger("tasks")

TERMINAL = {"completed", "failed", "cancelled"}


class CreateTaskBody(BaseModel):
    goal: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None
    steps: list[dict[str, Any]] | None = None


class TaskEngine:
    def __init__(self, runtime: Any) -> None:
        self.rt = runtime
        self.db = runtime.db
        self.planner = Planner(runtime)
        self._running: dict[str, asyncio.Task[Any]] = {}
        self._cancel: dict[str, asyncio.Event] = {}
        self._principals: dict[str, Principal] = {}

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        # Anything that was running when the core stopped is interrupted; it can be resumed.
        n = self.db.execute(
            "UPDATE tasks SET status='interrupted', updated_at=? WHERE status IN ('pending','planning','running')",
            (time.time(),)).rowcount
        self.db.execute("UPDATE task_nodes SET status='pending' WHERE status='running'")
        if n:
            log.info("%d task(s) were interrupted by a restart and can be resumed", n)

    async def stop(self) -> None:
        for tid, t in list(self._running.items()):
            self._cancel.get(tid, asyncio.Event()).set()
            t.cancel()
        self._running.clear()

    def status(self) -> dict[str, Any]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status")
        return {"active": len(self._running), "by_status": {r["status"]: r["n"] for r in rows}}

    # --------------------------------------------------------------- create
    async def create(self, goal: str, principal: Principal, conversation_id: str | None = None,
                     steps: list[dict[str, Any]] | None = None, start: bool = True) -> dict[str, Any]:
        goal = goal.strip()
        if not goal:
            raise ValueError("goal is empty")
        plan: Plan | None = None
        if steps:
            plan = validate_plan({"summary": goal, "steps": steps}, set(self.rt.tools.categories()))
        tid = "task_" + uuid.uuid4().hex[:10]
        now = time.time()
        self.db.execute(
            "INSERT INTO tasks(id, goal, status, conversation_id, device_id, plan_json, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (tid, goal, "pending", conversation_id, principal.device_id, json.dumps(plan.to_dict()) if plan else None, now, now),
        )
        if plan:
            self._store_nodes(tid, plan)
        await self.rt.events.publish(EventType.TASK_CREATED, {"task_id": tid, "goal": goal[:300], "device_id": principal.device_id})
        if start:
            self._launch(tid, principal)
        return self.get(tid)  # type: ignore[return-value]

    def _store_nodes(self, tid: str, plan: Plan, replace_pending: bool = False) -> None:
        if replace_pending:
            self.db.execute("DELETE FROM task_nodes WHERE task_id=? AND status!='completed'", (tid,))
        for s in plan.steps:
            self.db.execute(
                "INSERT OR REPLACE INTO task_nodes(task_id, node_id, title, instruction, depends_on_json, role, status) "
                "VALUES(?,?,?,?,?,?, 'pending')",
                (tid, s.id, s.title, s.instruction, json.dumps({"deps": s.depends_on, "tools": s.tools}), s.role),
            )

    def _launch(self, tid: str, principal: Principal) -> None:
        cancel = asyncio.Event()
        self._cancel[tid] = cancel
        self._principals[tid] = principal
        task = asyncio.create_task(self._run(tid, principal, cancel), name=f"jarvis-{tid}")
        self._running[tid] = task
        task.add_done_callback(lambda _t: (self._running.pop(tid, None), self._cancel.pop(tid, None)))

    # ------------------------------------------------------------- running
    def _set(self, tid: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE tasks SET {cols} WHERE id=?", [*fields.values(), tid])

    async def _run(self, tid: str, principal: Principal, cancel: asyncio.Event) -> None:
        task = self.db.query_one("SELECT * FROM tasks WHERE id=?", (tid,))
        assert task is not None
        try:
            if not self.db.query_one("SELECT 1 FROM task_nodes WHERE task_id=?", (tid,)):
                self._set(tid, status="planning")
                await self.rt.events.publish(EventType.TASK_PROGRESS, {"task_id": tid, "phase": "planning", "progress": 0})
                plan = await self.planner.plan(task["goal"])
                self._store_nodes(tid, plan)
                self._set(tid, plan_json=json.dumps(plan.to_dict()))
            self._set(tid, status="running")
            await self.rt.events.publish(EventType.TASK_STARTED, {"task_id": tid, "goal": task["goal"][:300]})
            await self._execute_graph(tid, principal, cancel)
            nodes = self.nodes(tid)
            failed = [n for n in nodes if n["status"] in ("failed", "skipped")]
            replans = self.db.query_one("SELECT replans FROM tasks WHERE id=?", (tid,))["replans"]
            if failed and replans < 1 and not cancel.is_set():
                await self._replan(tid, task["goal"], nodes)
                await self._execute_graph(tid, principal, cancel)
                nodes = self.nodes(tid)
            if cancel.is_set():
                raise asyncio.CancelledError()
            await self._finish(tid, task["goal"], nodes)
        except asyncio.CancelledError:
            self.db.execute("UPDATE task_nodes SET status='cancelled' WHERE task_id=? AND status IN ('pending','running')", (tid,))
            self._set(tid, status="cancelled", finished_at=time.time())
            self.rt.gate.cancel_pending(task_id=tid)
            await self.rt.events.publish(EventType.TASK_CANCELLED, {"task_id": tid})
        except Exception as e:  # noqa: BLE001 - surface every failure
            log.exception("task %s crashed", tid)
            self._set(tid, status="failed", error=f"{type(e).__name__}: {e}", finished_at=time.time())
            await self.rt.events.publish(EventType.TASK_FAILED, {"task_id": tid, "error": str(e)})

    def _ready(self, nodes: list[dict[str, Any]], running: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        status = {n["node_id"]: n["status"] for n in nodes}
        ready, blocked = [], []
        for n in nodes:
            if n["status"] != "pending" or n["node_id"] in running:
                continue
            deps = n["deps"]
            if any(status.get(d) in ("failed", "skipped", "cancelled") for d in deps):
                blocked.append(n)
            elif all(status.get(d) == "completed" for d in deps):
                ready.append(n)
        return ready, blocked

    async def _execute_graph(self, tid: str, principal: Principal, cancel: asyncio.Event) -> None:
        limit = max(1, self.rt.config.agent.task_max_parallel)
        inflight: dict[str, asyncio.Task[Any]] = {}
        while True:
            if cancel.is_set():
                for t in inflight.values():
                    t.cancel()
                raise asyncio.CancelledError()
            nodes = self.nodes(tid)
            ready, blocked = self._ready(nodes, set(inflight))
            for n in blocked:
                self.db.execute("UPDATE task_nodes SET status='skipped', error=? WHERE task_id=? AND node_id=?",
                                ("a step it depends on did not complete", tid, n["node_id"]))
            for n in ready[: max(0, limit - len(inflight))]:
                inflight[n["node_id"]] = asyncio.create_task(self._run_node(tid, n, principal, cancel))
            if not inflight:
                if blocked:
                    continue
                return
            done, _ = await asyncio.wait(inflight.values(), return_when=asyncio.FIRST_COMPLETED)
            for node_id in [k for k, t in inflight.items() if t in done]:
                inflight.pop(node_id)
            nodes = self.nodes(tid)
            finished = sum(1 for n in nodes if n["status"] in ("completed", "failed", "skipped"))
            progress = round(finished / max(1, len(nodes)), 3)
            self._set(tid, progress=progress)
            await self.rt.events.publish(EventType.TASK_PROGRESS, {"task_id": tid, "progress": progress,
                                                                   "completed": [n["node_id"] for n in nodes if n["status"] == "completed"]})

    async def _run_node(self, tid: str, node: dict[str, Any], principal: Principal, cancel: asyncio.Event) -> None:
        nid = node["node_id"]
        deps_out = []
        for d in node["deps"]:
            dep = self.db.query_one("SELECT title, output FROM task_nodes WHERE task_id=? AND node_id=?", (tid, d))
            if dep:
                deps_out.append(f"### Result of '{dep['title']}'\n{(dep['output'] or '')[:6000]}")
        goal = self.db.query_one("SELECT goal FROM tasks WHERE id=?", (tid,))["goal"]
        prompt = node["instruction"] + ("\n\n" + "\n\n".join(deps_out) if deps_out else "")
        extra = (f"You are executing one step of a larger task. Overall goal: {goal}\n"
                 f"This step: {node['title']}. Do only this step, then report its result concisely and factually. "
                 "If you could not complete it, say exactly what failed. Never claim actions that no tool performed.")
        retries = self.rt.config.agent.task_node_retries
        timeout = self.rt.config.agent.task_node_timeout_s
        last_error = None
        for attempt in range(retries + 1):
            started = time.time()
            self.db.execute("UPDATE task_nodes SET status='running', attempts=?, started_at=? WHERE task_id=? AND node_id=?",
                            (attempt + 1, started, tid, nid))
            conv = self.rt.conversations.create(principal.device_id, f"[task {tid}] {node['title']}", kind="task")
            try:
                result = await asyncio.wait_for(self.rt.agent.run(
                    text=prompt, principal=principal, conversation_id=conv, task_id=tid, role=node.get("role"),
                    tool_categories=set(node["tools"]), extra_system=extra, cancel=cancel,
                    device=DeviceContext(device_id=principal.device_id, name="task engine", kind="core"),
                ), timeout=timeout)
                if result.stop_reason == "cancelled":
                    raise asyncio.CancelledError()
                if result.error or result.stop_reason in ("error", "max_steps", "refusal"):
                    last_error = result.error or f"step ended with {result.stop_reason}: {result.text[:300]}"
                else:
                    self.db.execute(
                        "UPDATE task_nodes SET status='completed', output=?, error=NULL, tool_log_json=?, finished_at=? "
                        "WHERE task_id=? AND node_id=?",
                        (result.text, json.dumps(result.actions), time.time(), tid, nid))
                    return
            except TimeoutError:
                last_error = f"step timed out after {timeout:.0f}s"
            except asyncio.CancelledError:
                self.db.execute("UPDATE task_nodes SET status='cancelled' WHERE task_id=? AND node_id=?", (tid, nid))
                raise
            log.warning("task %s node %s attempt %d failed: %s", tid, nid, attempt + 1, last_error)
            if attempt < retries:
                await asyncio.sleep(min(10, 2 ** attempt))
        self.db.execute("UPDATE task_nodes SET status='failed', error=?, finished_at=? WHERE task_id=? AND node_id=?",
                        (last_error, time.time(), tid, nid))

    async def _replan(self, tid: str, goal: str, nodes: list[dict[str, Any]]) -> None:
        done = "\n".join(f"- {n['title']}: {(n['output'] or '')[:800]}" for n in nodes if n["status"] == "completed")
        failed = "\n".join(f"- {n['title']}: {n['error']}" for n in nodes if n["status"] in ("failed", "skipped"))
        plan = await self.planner.plan(goal, context=f"Already completed:\n{done or '(nothing)'}\n\nFailed or skipped:\n{failed}\n"
                                                    "Plan only the remaining work, using a different approach where a step failed.")
        for s in plan.steps:
            s.id = f"r{s.id}"
            s.depends_on = [f"r{d}" for d in s.depends_on]
        try:
            validate_plan({"summary": plan.summary, "steps": [s.to_dict() for s in plan.steps]})
        except PlanError as e:
            log.warning("replan invalid: %s", e)
            return
        self.db.execute("DELETE FROM task_nodes WHERE task_id=? AND status IN ('failed','skipped')", (tid,))
        self._store_nodes(tid, plan)
        self.db.execute("UPDATE tasks SET replans = replans + 1 WHERE id=?", (tid,))
        await self.rt.events.publish(EventType.TASK_PROGRESS, {"task_id": tid, "phase": "replanned", "steps": len(plan.steps)})

    async def _finish(self, tid: str, goal: str, nodes: list[dict[str, Any]]) -> None:
        results = [{"title": n["title"], "status": n["status"], "output": n["output"], "actions": n["actions"]} for n in nodes]
        all_ok = all(n["status"] == "completed" for n in nodes)
        verification = await self.planner.verify(goal, results)
        achieved = verification.get("achieved")
        status = "completed" if all_ok and achieved is not False else "failed"
        summary = verification.get("summary") or "\n\n".join(f"{r['title']}: {r['output'] or r['status']}" for r in results)
        ledger = [a for r in results for a in r["actions"]]
        self._set(tid, status=status, result=summary, verification_json=json.dumps({**verification, "actions_executed": len(ledger),
                                                                                    "actions_failed": sum(1 for a in ledger if not a["ok"])}),
                  progress=1.0, finished_at=time.time(),
                  error=None if status == "completed" else "; ".join(verification.get("issues") or ["one or more steps failed"]))
        evt = EventType.TASK_COMPLETED if status == "completed" else EventType.TASK_FAILED
        await self.rt.events.publish(evt, {"task_id": tid, "goal": goal[:200], "achieved": achieved, "summary": summary[:500]})
        await self.rt.events.publish(EventType.NOTIFICATION, {
            "title": "TASK COMPLETE" if status == "completed" else "TASK NEEDS ATTENTION",
            "body": goal[:120], "level": "info" if status == "completed" else "warn", "task_id": tid})

    # -------------------------------------------------------------- control
    async def cancel(self, tid: str) -> bool:
        ev = self._cancel.get(tid)
        if ev is None:
            row = self.db.query_one("SELECT status FROM tasks WHERE id=?", (tid,))
            if row and row["status"] not in TERMINAL:
                self._set(tid, status="cancelled", finished_at=time.time())
                return True
            return False
        ev.set()
        self.rt.gate.cancel_pending(task_id=tid)
        return True

    async def resume(self, tid: str, principal: Principal) -> dict[str, Any]:
        row = self.db.query_one("SELECT status FROM tasks WHERE id=?", (tid,))
        if row is None:
            raise KeyError(tid)
        if tid in self._running:
            raise ValueError("task is already running")
        if row["status"] == "completed":
            raise ValueError("task already completed")
        self.db.execute("UPDATE task_nodes SET status='pending', error=NULL WHERE task_id=? AND status IN ('failed','skipped','cancelled','running')", (tid,))
        self._set(tid, status="pending", error=None, finished_at=None)
        self._launch(tid, principal)
        return self.get(tid)  # type: ignore[return-value]

    def delete(self, tid: str) -> bool:
        if tid in self._running:
            raise ValueError("cancel the task before deleting it")
        return self.db.execute("DELETE FROM tasks WHERE id=?", (tid,)).rowcount > 0

    # -------------------------------------------------------------- queries
    def nodes(self, tid: str) -> list[dict[str, Any]]:
        out = []
        for r in self.db.query("SELECT * FROM task_nodes WHERE task_id=? ORDER BY rowid", (tid,)):
            meta = json.loads(r["depends_on_json"] or "{}")
            if isinstance(meta, list):
                meta = {"deps": meta, "tools": []}
            out.append({"node_id": r["node_id"], "title": r["title"], "instruction": r["instruction"], "deps": meta.get("deps", []),
                        "tools": meta.get("tools", []), "role": r["role"], "status": r["status"], "attempts": r["attempts"],
                        "output": r["output"], "error": r["error"], "actions": json.loads(r["tool_log_json"] or "[]"),
                        "started_at": r["started_at"], "finished_at": r["finished_at"]})
        return out

    def get(self, tid: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM tasks WHERE id=?", (tid,))
        if row is None:
            return None
        row["plan"] = json.loads(row.pop("plan_json")) if row.get("plan_json") else None
        row["verification"] = json.loads(row.pop("verification_json")) if row.get("verification_json") else None
        row["nodes"] = self.nodes(tid)
        row["running"] = tid in self._running
        return row

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT id, goal, status, progress, created_at, updated_at, finished_at, error FROM tasks "
                             "ORDER BY created_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["running"] = r["id"] in self._running
        return rows

    # ----------------------------------------------------------------- API
    @property
    def api_router(self) -> APIRouter:
        from jarvis.api.deps import interactive, runtime  # noqa: PLC0415

        router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

        @router.get("")
        async def list_tasks(limit: int = 50, p: Principal = Depends(interactive)) -> list[dict[str, Any]]:
            return self.list(min(limit, 200))

        @router.post("")
        async def create_task(body: CreateTaskBody, p: Principal = Depends(interactive)) -> dict[str, Any]:
            try:
                return await self.create(body.goal, p, body.conversation_id, body.steps)
            except (ValueError, PlanError) as e:
                raise HTTPException(422, str(e)) from e

        @router.get("/{tid}")
        async def get_task(tid: str, p: Principal = Depends(interactive)) -> dict[str, Any]:
            task = self.get(tid)
            if task is None:
                raise HTTPException(404, "task not found")
            return task

        @router.post("/{tid}/cancel")
        async def cancel_task(tid: str, p: Principal = Depends(interactive)) -> dict[str, Any]:
            return {"cancelled": await self.cancel(tid)}

        @router.post("/{tid}/resume")
        async def resume_task(tid: str, p: Principal = Depends(interactive)) -> dict[str, Any]:
            try:
                return await self.resume(tid, p)
            except KeyError as e:
                raise HTTPException(404, "task not found") from e
            except ValueError as e:
                raise HTTPException(409, str(e)) from e

        @router.delete("/{tid}")
        async def delete_task(tid: str, p: Principal = Depends(interactive)) -> dict[str, Any]:
            try:
                if not self.delete(tid):
                    raise HTTPException(404, "task not found")
            except ValueError as e:
                raise HTTPException(409, str(e)) from e
            return {"deleted": tid}

        _ = runtime
        return router


# ------------------------------------------------------------------ agent tools
async def _start_task(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    engine: TaskEngine = ctx.runtime.extensions["tasks"]
    task = await engine.create(args["goal"], ctx.principal, ctx.conversation_id)
    return ToolResult(ok=True, content=f"Started background task {task['id']}. It will plan and execute the steps; "
                                       "progress is visible in the Tasks panel. Do not claim it is finished.",
                      data={"task_id": task["id"]})


async def _task_status(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    engine: TaskEngine = ctx.runtime.extensions["tasks"]
    task = engine.get(args["task_id"])
    if task is None:
        return ToolResult.fail("no such task")
    lines = [f"{task['id']}: {task['status']} ({int(task['progress'] * 100)}%) - {task['goal'][:200]}"]
    for n in task["nodes"]:
        lines.append(f"- [{n['status']}] {n['title']}" + (f": {n['error']}" if n["error"] else ""))
    if task.get("result"):
        lines.append(f"Result: {task['result'][:2000]}")
    return ToolResult(ok=True, content="\n".join(lines), data={"status": task["status"], "progress": task["progress"]})


def register(runtime: Any) -> None:
    engine = TaskEngine(runtime)
    runtime.extensions["tasks"] = engine
    runtime.tools.register_all([
        Tool("start_task", "Start a long-running background task for goals with several distinct steps (research, "
                           "comparisons, multi-file work). Returns immediately with a task id.",
             {"type": "object", "properties": {"goal": {"type": "string", "description": "the full goal, self-contained"}},
              "required": ["goal"]},
             _start_task, Risk.LOW_RISK, "tasks.create", "tasks",
             describe=lambda a: f"start a background task: {a['goal'][:150]}",
             keywords=("task", "background", "compare", "research", "step by step", "and then", "save the results")),
        Tool("task_status", "Get the status, step progress and result of a background task.",
             {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]},
             _task_status, Risk.SAFE, "tasks.read", "tasks", keywords=("task",)),
    ])
