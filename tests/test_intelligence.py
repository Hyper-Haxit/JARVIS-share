"""Phase 2: tool executor, agent loop, router fallback, planner validation, task engine."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from jarvis.capabilities.base import Tool, ToolContext, ToolResult, ValidationError, validate
from jarvis.config.paths import JarvisPaths
from jarvis.config.settings import JarvisConfig, ProviderConfig
from jarvis.core.events import EventType
from jarvis.core.permissions import Risk
from jarvis.core.planning import PlanError, validate_plan
from jarvis.core.toolparse import recover_tool_calls
from jarvis.models.base import Message, ProviderError, ToolCall
from jarvis.runtime import JarvisRuntime
from tests.conftest import ScriptedProvider


def make_runtime(tmp_path: Path, script: list[Any] | None = None, **cfg: Any) -> tuple[JarvisRuntime, ScriptedProvider]:
    provider = ScriptedProvider("local", script or [])
    config = JarvisConfig(providers=[ProviderConfig(name="local", type="ollama")])
    for role in ("fast", "reasoning", "coding", "vision", "fallback"):
        setattr(config.roles, role, ["local/m"])
    config.roles.embedding = ["local/embed"]
    config.permissions.approval_timeout_s = cfg.pop("approval_timeout_s", 2)
    for k, v in cfg.items():
        setattr(config.agent, k, v)
    rt = JarvisRuntime(paths=JarvisPaths(tmp_path / "home"), config=config, providers={"local": provider},
                       configure_logging=False, load_plugins=False)
    return rt, provider


@pytest.fixture
async def rt_env(tmp_path: Path):
    rt, provider = make_runtime(tmp_path)
    await rt.start()
    principal = rt.auth.authenticate(rt.local_token)
    yield rt, provider, principal
    await rt.stop()


# ------------------------------------------------------------- validation
def test_schema_validation_and_coercion() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}, "flag": {"type": "boolean"},
                                               "tags": {"type": "array", "items": {"type": "string"}},
                                               "mode": {"type": "string", "enum": ["a", "b"], "default": "a"}},
              "required": ["n"]}
    assert validate({"n": "5", "flag": "true", "tags": "x"}, schema) == {"n": 5, "flag": True, "tags": ["x"], "mode": "a"}
    for bad in ({}, {"n": 0}, {"n": "five"}, {"n": 1, "mode": "c"}):
        with pytest.raises(ValidationError):
            validate(bad, schema)


def test_textual_tool_call_recovery() -> None:
    names = {"memory_save", "calculate"}
    calls, rest = recover_tool_calls('Sure.\nmemory_save({"content": "likes tea", "tags": ["drink"]})', names)
    assert [c.name for c in calls] == ["memory_save"] and calls[0].arguments["content"] == "likes tea"
    assert rest == "Sure."
    calls, _ = recover_tool_calls('<tool_call>{"name": "calculate", "arguments": {"expression": "2+2"}}</tool_call>', names)
    assert calls[0].arguments == {"expression": "2+2"}
    calls, _ = recover_tool_calls('{"name": "calculate", "arguments": {"expression": "1"}}', names)
    assert calls and calls[0].name == "calculate"
    # Unknown tool names and plain prose are left alone.
    assert recover_tool_calls('delete_everything({"x": 1})', names)[0] == []
    assert recover_tool_calls("The calculate function adds numbers.", names)[0] == []


# ---------------------------------------------------------------- executor
async def test_executor_gates_validates_and_audits(rt_env) -> None:
    rt, _, owner = rt_env
    ran = []

    async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ran.append(args)
        return ToolResult(ok=True, content="did it")

    rt.tools.register(Tool("risky", "test", {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                           handler, Risk.SAFE, "test.risky", "test",
                           risk_fn=lambda a, c: Risk.SENSITIVE if a["x"] > 10 else None))
    ctx = ToolContext(runtime=rt, principal=owner, conversation_id="c1")
    res, rec = await rt.executor.execute(ToolCall(name="risky", arguments={"x": 1}), ctx)
    assert res.ok and rec.decision == "allow"
    res, rec = await rt.executor.execute(ToolCall(name="risky", arguments={}), ctx)
    assert not res.ok and rec.decision == "invalid"
    res, rec = await rt.executor.execute(ToolCall(name="nope", arguments={}), ctx)
    assert not res.ok and rec.decision == "unknown"
    # Argument-dependent escalation -> requires approval -> times out -> blocked, handler never ran.
    rt.config.permissions.approval_timeout_s = 0.2
    res, rec = await rt.executor.execute(ToolCall(name="risky", arguments={"x": 99}), ctx)
    assert not res.ok and rec.decision == "expired" and rec.risk == "SENSITIVE"
    assert ran == [{"x": 1}]
    audit = rt.db.query("SELECT target, outcome FROM audit_log WHERE action='tool.execute'")
    assert ("risky", "blocked") in {(a["target"], a["outcome"]) for a in audit}


async def test_executor_timeout_and_exception(rt_env) -> None:
    rt, _, owner = rt_env

    async def slow(args, ctx):
        await asyncio.sleep(5)

    async def boom(args, ctx):
        raise RuntimeError("kaput")

    rt.tools.register(Tool("slow", "s", {"type": "object", "properties": {}}, slow, timeout_s=0.2))
    rt.tools.register(Tool("boom", "b", {"type": "object", "properties": {}}, boom))
    ctx = ToolContext(runtime=rt, principal=owner)
    res, _ = await rt.executor.execute(ToolCall(name="slow"), ctx)
    assert not res.ok and "timed out" in res.error
    res, _ = await rt.executor.execute(ToolCall(name="boom"), ctx)
    assert not res.ok and "kaput" in res.error


# ------------------------------------------------------------------- agent
async def test_agent_tool_loop_and_persistence(rt_env) -> None:
    rt, provider, owner = rt_env
    provider.push({"tool_calls": [{"name": "calculate", "arguments": {"expression": "6*7"}}]},
                  lambda req: {"text": "The answer is " + req.messages[-1].content.split("= ")[-1]})
    result = await rt.agent.run(text="what is 6*7", principal=owner)
    assert result.text == "The answer is 42"
    assert [a["tool"] for a in result.actions] == ["calculate"] and result.actions[0]["ok"]
    roles = [m["role"] for m in rt.conversations.get(result.conversation_id)["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]


async def test_agent_recovers_textual_tool_calls(rt_env) -> None:
    rt, provider, owner = rt_env
    provider.push({"text": 'memory_save({"content": "User likes green tea"})'}, {"text": "Saved."})
    result = await rt.agent.run(text="remember I like green tea", principal=owner)
    assert [a["tool"] for a in result.actions] == ["memory_save"]
    assert any("green tea" in m["content"] for m in rt.memory.list())


async def test_agent_never_runs_tools_from_refused_or_truncated_turns(rt_env) -> None:
    rt, provider, owner = rt_env
    provider.push({"tool_calls": [{"name": "calculate", "arguments": {"expression": "1"}}], "stop": "max_tokens"})
    r1 = await rt.agent.run(text="x", principal=owner)
    assert r1.actions == [] and "did not run" in r1.text
    provider.push({"text": "", "tool_calls": [{"name": "calculate", "arguments": {"expression": "1"}}], "stop": "refusal"})
    r2 = await rt.agent.run(text="y", principal=owner)
    assert r2.actions == [] and r2.stop_reason == "refusal"


async def test_agent_step_limit_is_reported_honestly(tmp_path: Path) -> None:
    rt, provider, = make_runtime(tmp_path, max_steps=2)
    await rt.start()
    try:
        owner = rt.auth.authenticate(rt.local_token)
        provider.push(*[{"tool_calls": [{"name": "get_time"}]} for _ in range(5)])
        r = await rt.agent.run(text="loop", principal=owner)
        assert r.stop_reason == "max_steps" and "stopped after 2" in r.text
    finally:
        await rt.stop()


async def test_agent_reports_model_unavailable(rt_env) -> None:
    rt, provider, owner = rt_env
    provider.healthy = False
    r = await rt.agent.run(text="hello", principal=owner)
    assert r.stop_reason == "error" and "can't reach any language model" in r.text


async def test_router_falls_back_and_records_attempts(tmp_path: Path) -> None:
    rt, primary = make_runtime(tmp_path)
    backup = ScriptedProvider("backup", [{"text": "from backup"}])
    rt.router.providers["backup"] = backup
    rt.config.roles.fast = ["local/m", "backup/m"]
    primary.push(ProviderError("boom", retryable=False))
    resp, route = await rt.router.chat([Message(role="user", content="hi")], role="fast")
    assert resp.message.content == "from backup"
    assert [a.outcome for a in route.attempts] == ["failed", "ok"] and route.degraded
    rt.router.set_mode("local_only")
    backup.local = False
    primary.push(ProviderError("boom again", retryable=False))
    with pytest.raises(ProviderError):
        await rt.router.chat([Message(role="user", content="hi")], role="fast")
    await rt.router.close()


# ---------------------------------------------------------------- planning
def test_plan_validation_rejects_bad_graphs() -> None:
    ok = validate_plan({"summary": "s", "steps": [
        {"id": "a", "title": "A", "instruction": "do a", "depends_on": []},
        {"id": "b", "title": "B", "instruction": "do b", "depends_on": ["a"]}]})
    assert [s.id for s in ok.steps] == ["a", "b"]
    bad = [
        {"steps": []},
        {"steps": [{"id": "a", "instruction": "x", "depends_on": ["zzz"]}]},
        {"steps": [{"id": "a", "instruction": "x", "depends_on": ["b"]}, {"id": "b", "instruction": "y", "depends_on": ["a"]}]},
        {"steps": [{"id": "a", "instruction": "x", "depends_on": []}, {"id": "a", "instruction": "y", "depends_on": []}]},
        {"steps": [{"id": "a", "instruction": "", "depends_on": []}]},
        {"steps": [{"id": f"s{i}", "instruction": "x", "depends_on": []} for i in range(20)]},
    ]
    for b in bad:
        with pytest.raises(PlanError):
            validate_plan(b)


# -------------------------------------------------------------------- tasks
async def _wait_task(rt: JarvisRuntime, tid: str, timeout: float = 10) -> dict[str, Any]:
    engine = rt.extensions["tasks"]
    for _ in range(int(timeout * 20)):
        t = engine.get(tid)
        if t["status"] in ("completed", "failed", "cancelled") and not t["running"]:
            return t
        await asyncio.sleep(0.05)
    raise AssertionError(f"task did not finish: {engine.get(tid)}")


async def test_task_engine_plans_runs_parallel_and_verifies(rt_env) -> None:
    rt, provider, owner = rt_env
    plan = {"summary": "two lookups then combine", "steps": [
        {"id": "a", "title": "Get time", "instruction": "get the time", "depends_on": []},
        {"id": "b", "title": "Compute", "instruction": "compute 2+2", "depends_on": []},
        {"id": "c", "title": "Combine", "instruction": "combine", "depends_on": ["a", "b"]}]}

    def respond(req):
        last = req.messages[-1].content
        system = req.messages[0].content
        if "You plan multi-step work" in system:
            return {"text": json.dumps(plan)}
        if "You verify whether work achieved" in system:
            return {"text": json.dumps({"achieved": True, "issues": [], "summary": "Done: time and sum combined."})}
        if "combine" in last.split("\n")[0]:
            assert "Result of 'Get time'" in last and "Result of 'Compute'" in last  # dependency outputs passed
            return {"text": "combined"}
        return {"text": "partial result"}

    provider.push(*[respond] * 10)
    events = rt.events.subscribe(["TASK_*"])
    task = await rt.extensions["tasks"].create("do the thing", owner)
    done = await _wait_task(rt, task["id"])
    assert done["status"] == "completed", done
    assert [n["status"] for n in done["nodes"]] == ["completed"] * 3
    assert done["verification"]["achieved"] is True and done["result"].startswith("Done")
    seen = []
    while (e := await events.get(0.01)) is not None:
        seen.append(e.type)
    assert "TASK_STARTED" in seen and "TASK_COMPLETED" in seen and "TASK_PROGRESS" in seen
    # Step conversations are internal and do not pollute the chat list.
    assert all(not (c["title"] or "").startswith("[task") for c in rt.conversations.list())


async def test_task_failure_skips_dependents_and_is_reported(tmp_path: Path) -> None:
    rt, provider = make_runtime(tmp_path, task_node_retries=0)
    await rt.start()
    try:
        owner = rt.auth.authenticate(rt.local_token)
        steps = [{"id": "a", "title": "A", "instruction": "fail please", "depends_on": []},
                 {"id": "b", "title": "B", "instruction": "after a", "depends_on": ["a"]}]

        def respond(req):
            system = req.messages[0].content
            if "You plan multi-step work" in system:  # replan attempt: return invalid -> fallback single step
                return {"text": "not json"}
            if "You verify" in system:
                return {"text": json.dumps({"achieved": False, "issues": ["step A failed"], "summary": "Could not complete."})}
            raise ProviderError("model crashed", retryable=False)

        provider.push(*[respond] * 20)
        task = await rt.extensions["tasks"].create("goal", owner, steps=steps)
        done = await _wait_task(rt, task["id"])
        assert done["status"] == "failed"
        assert done["error"]
    finally:
        await rt.stop()


async def test_task_cancel_and_resume_from_checkpoint(rt_env) -> None:
    rt, provider, owner = rt_env
    engine = rt.extensions["tasks"]
    gate = asyncio.Event()

    async def slow_handler(args, ctx):
        await gate.wait()
        return "slow done"

    rt.tools.register(Tool("slow_op", "slow", {"type": "object", "properties": {}}, slow_handler, category="test", timeout_s=30))
    steps = [{"id": "a", "title": "Quick", "instruction": "quick", "depends_on": []},
             {"id": "b", "title": "Slow", "instruction": "slow", "depends_on": ["a"], "tools": ["test"]}]

    def respond(req):
        last = req.messages[-1]
        if req.messages[0].content.startswith("You verify"):
            return {"text": json.dumps({"achieved": True, "issues": [], "summary": "ok"})}
        if last.role == "tool":
            return {"text": "step finished"}
        if last.content.startswith("slow"):
            return {"tool_calls": [{"name": "slow_op"}]}
        return {"text": "quick done"}

    provider.push(*[respond] * 20)
    task = await engine.create("g", owner, steps=steps)
    for _ in range(100):
        nodes = engine.nodes(task["id"])
        if nodes[1]["status"] == "running":
            break
        await asyncio.sleep(0.05)
    assert await engine.cancel(task["id"])
    done = await _wait_task(rt, task["id"])
    assert done["status"] == "cancelled"
    assert done["nodes"][0]["status"] == "completed"  # checkpoint kept
    gate.set()
    await engine.resume(task["id"], owner)
    done = await _wait_task(rt, task["id"])
    assert done["status"] == "completed"
    assert done["nodes"][0]["attempts"] == 1  # completed node was not re-run


async def test_interrupted_tasks_marked_on_restart(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    rt.db.execute("INSERT INTO tasks(id, goal, status, created_at, updated_at) VALUES('t1','g','running',0,0)")
    await rt.start()
    try:
        assert rt.extensions["tasks"].get("t1")["status"] == "interrupted"
    finally:
        await rt.stop()


async def test_events_cover_agent_states(rt_env) -> None:
    rt, provider, owner = rt_env
    provider.push({"text": "hi"})
    sub = rt.events.subscribe([EventType.AGENT_STATE])
    await rt.agent.run(text="hello", principal=owner)
    states = []
    while (e := await sub.get(0.01)) is not None:
        states.append(e.data["state"])
    assert states[0] == "thinking" and states[-1] == "idle"
