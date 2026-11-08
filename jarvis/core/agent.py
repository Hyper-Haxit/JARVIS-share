"""The agent loop: reason -> act (through the permission-gated executor) -> observe.

Guarantees enforced here rather than trusted to the model:
  * tool calls from a turn that was refused or truncated are never executed;
  * every executed action is recorded and returned to the client, so what the
    user sees as "done" is what actually ran, not what the model said;
  * the loop is bounded (max steps), cancellable and reports real failures.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from jarvis.capabilities.base import Tool, ToolContext
from jarvis.core.context import ContextBuilder, DeviceContext
from jarvis.core.events import EventType
from jarvis.core.observability import get_logger, get_trace_id, new_trace_id, reset_trace_id, set_trace_id
from jarvis.core.permissions import Principal
from jarvis.core.toolparse import recover_tool_calls
from jarvis.models.base import ImageInput, Message, ProviderError
from jarvis.models.router import NoModelAvailable, RouteResult

log = get_logger("agent")

Emit = Callable[[str, dict[str, Any]], Awaitable[None]]
META_TOOL = "enable_tools"


@dataclass
class AgentResult:
    conversation_id: str
    text: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    route: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = "end"
    steps: int = 0
    error: str | None = None
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    recalled_memory_ids: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    trace_id: str | None = None
    verification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Agent:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.builder = ContextBuilder(runtime)

    async def _state(self, state: str, conversation_id: str, emit: Emit | None, **extra: Any) -> None:
        data = {"state": state, "conversation_id": conversation_id, **extra}
        await self.runtime.events.publish(EventType.AGENT_STATE, data)
        if emit:
            await emit("state", data)

    async def run(
        self,
        *,
        text: str,
        principal: Principal,
        conversation_id: str | None = None,
        images: list[ImageInput] | None = None,
        device: DeviceContext | None = None,
        role: str | None = None,
        model_ref: str | None = None,
        task_id: str | None = None,
        tool_categories: set[str] | None = None,
        tools_override: list[Tool] | None = None,
        extra_system: str | None = None,
        emit: Emit | None = None,
        cancel: asyncio.Event | None = None,
        persist: bool = True,
        max_steps: int | None = None,
    ) -> AgentResult:
        rt = self.runtime
        started = time.perf_counter()
        trace_token = set_trace_id(new_trace_id())
        cancel = cancel or asyncio.Event()
        device = device or DeviceContext(device_id=principal.device_id, name=principal.name, kind=principal.kind)
        if conversation_id is None or not rt.conversations.exists(conversation_id):
            conversation_id = rt.conversations.create(principal.device_id)
        result = AgentResult(conversation_id=conversation_id, text="", trace_id=get_trace_id())
        try:
            categories = set(tool_categories or set())
            role = role or rt.router.classify([Message(role="user", content=text, images=images or [])])
            if role == "coding":
                from jarvis.capabilities.coding import CODING_RULES  # noqa: PLC0415

                categories |= {"coding", "files"}
                extra_system = f"{extra_system}\n\n{CODING_RULES}" if extra_system else CODING_RULES
            turn = await self.builder.build(conversation_id, text, images, device, extra_system)
            result.recalled_memory_ids = turn.recalled_memory_ids
            messages = turn.messages
            if persist:
                rt.conversations.append(conversation_id, messages[-1], {"device_id": principal.device_id})
            await rt.events.publish(EventType.CHAT_MESSAGE, {
                "conversation_id": conversation_id, "role": "user", "content": text, "device_id": principal.device_id,
            })
            max_steps = max_steps or rt.config.agent.max_steps

            for step in range(max_steps + 1):
                if cancel.is_set():
                    raise asyncio.CancelledError()
                tools = tools_override if tools_override is not None else rt.tools.select(
                    text, categories=categories, max_risk=principal.max_risk)
                await self._state("thinking", conversation_id, emit, step=step, role=role)
                route = RouteResult(role=role)
                response = None
                async for chunk in rt.router.stream(messages, role=role, model_ref=model_ref, tools=[t.schema() for t in tools],
                                                    think=(role == "reasoning"), route=route):
                    if cancel.is_set():
                        raise asyncio.CancelledError()
                    if chunk.type == "text":
                        await rt.events.publish(EventType.MODEL_RESPONSE_DELTA, {"conversation_id": conversation_id, "text": chunk.text})
                        if emit:
                            await emit("delta", {"text": chunk.text})
                    elif chunk.type == "thinking" and emit:
                        await emit("thinking", {"text": chunk.text})
                    elif chunk.type == "done":
                        response = chunk.response
                assert response is not None
                result.route = route.to_dict()
                result.usage["input_tokens"] += response.usage.input_tokens
                result.usage["output_tokens"] += response.usage.output_tokens
                assistant = response.message
                result.stop_reason = response.stop_reason
                result.steps = step
                if not assistant.tool_calls and response.stop_reason == "end":
                    recovered, remaining = recover_tool_calls(assistant.content, {t.name for t in tools})
                    if recovered:
                        log.info("recovered %d textual tool call(s) from model output", len(recovered))
                        assistant.tool_calls = recovered
                        assistant.content = remaining
                        response.stop_reason = "tool_use"
                        # Provider replay state no longer matches the edited message.
                        assistant.provider_state = {}

                if response.stop_reason == "refusal":
                    result.text = assistant.content.strip() or "I can't help with that request."
                    if response.refusal_detail:
                        result.text += f" ({response.refusal_detail})"
                    assistant.tool_calls = []
                    assistant.content = result.text
                    break
                if assistant.tool_calls and response.stop_reason == "max_tokens":
                    # A truncated tool call may parse as valid but incomplete input: never run it.
                    assistant.tool_calls = []
                    result.text = (assistant.content.strip() + "\n\n" if assistant.content.strip() else "") + \
                        "My response was cut off before I finished preparing an action, so I did not run it."
                    assistant.content = result.text
                    break
                if not assistant.tool_calls:
                    result.text = assistant.content.strip()
                    break
                if step >= max_steps:
                    result.text = (f"I stopped after {max_steps} tool steps without finishing. "
                                   "Here is where things stand: " + _summarize_actions(result.actions))
                    result.stop_reason = "max_steps"
                    assistant = Message(role="assistant", content=result.text)
                    break

                messages.append(assistant)
                if persist:
                    rt.conversations.append(conversation_id, assistant)
                await self._state("acting", conversation_id, emit, tools=[c.name for c in assistant.tool_calls])
                reason = assistant.content.strip() or f"User request: {text[:300]}"
                ctx = ToolContext(runtime=rt, principal=principal, conversation_id=conversation_id, task_id=task_id,
                                  reason=reason, cancel=cancel, device_id=device.device_id, workspace=rt.paths.workspace)
                for call in assistant.tool_calls:
                    if emit:
                        await emit("tool_start", {"id": call.id, "tool": call.name, "args": call.arguments})
                    res, record = await rt.executor.execute(call, ctx)
                    result.actions.append(record.to_dict())
                    if call.name == META_TOOL and res.ok and isinstance(res.data, dict):
                        categories |= set(res.data.get("enabled", []))
                    if emit:
                        await emit("tool_end", {"id": call.id, **record.to_dict(), "result": res.to_public()})
                    tool_msg = Message(role="tool", content=res.content, tool_call_id=call.id, name=call.name,
                                       images=res.images, is_error=not res.ok)
                    messages.append(tool_msg)
                    if persist:
                        rt.conversations.append(conversation_id, tool_msg)
                    if cancel.is_set():
                        raise asyncio.CancelledError()
            from jarvis.capabilities.coding import verification_statement  # noqa: PLC0415

            statement = verification_statement(result.actions)
            if statement:
                # Generated from what actually ran, never from the model's own claims.
                result.text = f"{result.text}\n\n{statement}".strip()
                result.verification = statement
            final = Message(role="assistant", content=result.text, provider_state=assistant.provider_state)
            if persist:
                rt.conversations.append(conversation_id, final, {"route": result.route, "actions": len(result.actions)})
        except asyncio.CancelledError:
            result.text = result.text or "Cancelled."
            result.stop_reason = "cancelled"
            rt.gate.cancel_pending(task_id=task_id, conversation_id=conversation_id)
            if persist:
                rt.conversations.append(conversation_id, Message(role="assistant", content="(cancelled by user)"), {"cancelled": True})
        except NoModelAvailable as e:
            result.error = str(e)
            result.stop_reason = "error"
            result.text = ("I can't reach any language model right now, so I can't answer. "
                           f"Details: {e}")
            if persist:
                rt.conversations.append(conversation_id, Message(role="assistant", content=result.text), {"error": True})
        except ProviderError as e:
            result.error = str(e)
            result.stop_reason = "error"
            result.text = f"The model failed while answering: {e}"
            if persist:
                rt.conversations.append(conversation_id, Message(role="assistant", content=result.text), {"error": True})
        finally:
            result.duration_ms = round((time.perf_counter() - started) * 1000, 1)
            reset_trace_id(trace_token)
        await rt.events.publish(EventType.CHAT_MESSAGE, {
            "conversation_id": conversation_id, "role": "assistant", "content": result.text,
            "actions": len(result.actions), "route": result.route, "error": result.error,
        })
        await self._state("error" if result.error else "idle", conversation_id, emit,
                          error=result.error)
        if emit:
            await emit("done", result.to_dict())
        return result


def _summarize_actions(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "no actions were completed."
    parts = [f"{a['tool']} ({'ok' if a['ok'] else 'failed'})" for a in actions[-8:]]
    return "; ".join(parts) + "."
