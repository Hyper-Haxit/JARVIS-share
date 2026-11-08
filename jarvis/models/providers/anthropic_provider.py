"""Anthropic (Claude) provider, built on the official ``anthropic`` Python SDK.

Notes on behaviour:
  * The API key and base URL are always passed explicitly from JARVIS config so
    the SDK never silently picks up ``ANTHROPIC_*`` variables meant for other
    tools on the same machine.
  * Requests are streamed. Client tools set ``eager_input_streaming``, so tool
    inputs are *not* validated server-side: the JARVIS tool executor validates
    every input against its schema before running anything, and tool calls from
    a turn that stopped on ``max_tokens`` or ``refusal`` are never executed.
  * Raw assistant content (including thinking blocks and their signatures) is
    kept in ``Message.provider_state['anthropic']`` and replayed unchanged.
  * For models that support it, server-side refusal fallbacks are enabled by
    default (``fallbacks="default"``); disable with provider option
    ``refusal_fallback = false``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from jarvis.models.base import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelInfo,
    ModelProvider,
    ProviderError,
    ProviderHealth,
    StreamChunk,
    ToolCall,
    Usage,
)

DEFAULT_BASE_URL = "https://api.anthropic.com"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-fable-5-1", "claude-opus-5", "claude-sonnet-5-5"}
# Models that take legacy budget-based thinking instead of adaptive thinking.
BUDGET_THINKING_PREFIXES = ("claude-haiku-4-5",)
_STOP = {
    "end_turn": "end",
    "stop_sequence": "end",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "max_tokens",
    "refusal": "refusal",
    "pause_turn": "end",
}
_REPLAY_KEEP = {"text", "thinking", "redacted_thinking", "tool_use"}


def _sanitize_replay(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare stored assistant content for replay.

    After a mid-output fallback, model-internal blocks before the last
    ``fallback`` marker must be omitted; text blocks are kept.
    """
    last_fb = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)
    out = []
    for i, b in enumerate(blocks):
        t = b.get("type")
        if t == "fallback" or t not in _REPLAY_KEEP:
            continue
        if i < last_fb and t != "text":
            continue
        out.append(b)
    return out


def _block_dict(block: Any) -> dict[str, Any] | None:
    t = getattr(block, "type", None)
    if t == "text":
        return {"type": "text", "text": block.text}
    if t == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if t == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    if t == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if t == "fallback":
        return {"type": "fallback"}
    return None


class AnthropicProvider(ModelProvider):
    type = "anthropic"

    def __init__(self, name: str, api_key: str | None, base_url: str | None = None, timeout_s: float = 600.0,
                 options: dict[str, Any] | None = None) -> None:
        super().__init__(name, local=False)
        self.options = options or {}
        self.base_url = base_url or DEFAULT_BASE_URL
        self._api_key = api_key
        self._client = None
        if api_key:
            try:
                from anthropic import AsyncAnthropic  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise ProviderError("the 'anthropic' package is not installed", retryable=False) from e
            self._client = AsyncAnthropic(api_key=api_key, base_url=self.base_url, timeout=timeout_s, max_retries=2)

    def _require(self) -> Any:
        if self._client is None:
            raise ProviderError(f"{self.name}: no API key configured", retryable=False)
        return self._client

    # ------------------------------------------------------------ conversion
    def _convert(self, request: ChatRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        system_blocks: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results() -> None:
            if pending_results:
                messages.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for m in request.messages:
            if m.role == "system":
                if not messages:
                    system_blocks.append({"type": "text", "text": m.content})
                else:
                    # Later system context is delivered as a user-side note to keep
                    # the conversation append-only.
                    flush_results()
                    messages.append({"role": "user", "content": [{"type": "text", "text": f"[system note]\n{m.content}"}]})
                continue
            if m.role == "tool":
                content: list[dict[str, Any]] = [{"type": "text", "text": m.content or "(no output)"}]
                for img in m.images:
                    content.append({"type": "image", "source": {"type": "base64", "media_type": img.media_type, "data": img.data_b64}})
                block: dict[str, Any] = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": content}
                if m.is_error:
                    block["is_error"] = True
                pending_results.append(block)
                continue
            flush_results()
            if m.role == "assistant":
                raw = m.provider_state.get("anthropic", {}).get("content")
                if raw and m.provider_state["anthropic"].get("model") == request.model:
                    blocks = _sanitize_replay(raw)
                else:
                    blocks = []
                    if m.content:
                        blocks.append({"type": "text", "text": m.content})
                    for tc in m.tool_calls:
                        blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                if blocks:
                    messages.append({"role": "assistant", "content": blocks})
                continue
            # user
            parts: list[dict[str, Any]] = [
                {"type": "image", "source": {"type": "base64", "media_type": img.media_type, "data": img.data_b64}}
                for img in m.images
            ]
            parts.append({"type": "text", "text": m.content or "(empty)"})
            messages.append({"role": "user", "content": parts})
        flush_results()
        if system_blocks:
            # Cache the stable leading system prompt.
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}
        return system_blocks, messages

    def _params(self, request: ChatRequest) -> dict[str, Any]:
        system, messages = self._convert(request)
        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or int(self.options.get("max_tokens", 32000)),
            "messages": messages,
        }
        if system:
            params["system"] = system
        if request.tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters, "eager_input_streaming": True}
                for t in sorted(request.tools, key=lambda t: t.name)
            ]
        output_config: dict[str, Any] = {}
        effort = request.options.get("effort") or self.options.get("effort")
        if effort:
            output_config["effort"] = effort
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        if output_config:
            params["output_config"] = output_config
        if request.think:
            if request.model.startswith(BUDGET_THINKING_PREFIXES):
                params["thinking"] = {"type": "enabled", "budget_tokens": int(self.options.get("thinking_budget", 4000))}
            else:
                params["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self.options.get("refusal_fallback", True) and request.model in FALLBACK_MODELS:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    # ---------------------------------------------------------------- stream
    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        import anthropic  # noqa: PLC0415

        client = self._require()
        params = self._params(request)
        started = time.perf_counter()
        attempts = 0
        while True:
            attempts += 1
            try:
                async with client.beta.messages.stream(**params) as stream:
                    async for event in stream:
                        etype = getattr(event, "type", None)
                        if etype == "text":
                            yield StreamChunk(type="text", text=event.text)
                        elif etype == "thinking":
                            yield StreamChunk(type="thinking", text=getattr(event, "thinking", "") or "")
                    final = await stream.get_final_message()
                break
            except ValueError as e:
                # Tool input JSON that the SDK could not parse at all: re-issue (bounded).
                if attempts > 2:
                    raise ProviderError(f"anthropic: unparseable tool input after retries: {e}", retryable=False) from e
                continue
            except anthropic.AuthenticationError as e:
                raise ProviderError("anthropic: authentication failed (check the API key)", retryable=False, status=401) from e
            except anthropic.NotFoundError as e:
                raise ProviderError(f"anthropic: model or endpoint not found: {request.model}", retryable=False, status=404) from e
            except anthropic.RateLimitError as e:
                raise ProviderError("anthropic: rate limited", retryable=True, status=429) from e
            except anthropic.BadRequestError as e:
                raise ProviderError(f"anthropic: bad request: {e.message}", retryable=False, status=400) from e
            except anthropic.APIStatusError as e:
                raise ProviderError(f"anthropic: API error {e.status_code}", retryable=e.status_code >= 500, status=e.status_code) from e
            except anthropic.APIConnectionError as e:
                raise ProviderError(f"anthropic: connection failed: {e}", retryable=True) from e

        raw_blocks = [b for b in (_block_dict(blk) for blk in final.content) if b is not None]
        last_fb = max((i for i, b in enumerate(raw_blocks) if b["type"] == "fallback"), default=-1)
        text = "".join(b["text"] for b in raw_blocks if b["type"] == "text")
        thinking = "".join(b.get("thinking", "") for b in raw_blocks[last_fb + 1:] if b["type"] == "thinking")
        tool_calls = [
            ToolCall(id=b["id"], name=b["name"], arguments=b["input"] if isinstance(b["input"], dict) else {},
                     parse_error=None if isinstance(b["input"], dict) else "tool input was not an object")
            for b in raw_blocks[last_fb + 1:]
            if b["type"] == "tool_use"
        ]
        stop = _STOP.get(final.stop_reason or "end_turn", "end")
        refusal_detail = None
        if stop == "refusal":
            details = getattr(final, "stop_details", None)
            if details is not None:
                refusal_detail = getattr(details, "explanation", None) or getattr(details, "category", None)
        if stop == "end" and tool_calls:
            stop = "tool_use"
        usage = Usage(input_tokens=getattr(final.usage, "input_tokens", 0) or 0,
                      output_tokens=getattr(final.usage, "output_tokens", 0) or 0)
        message = Message(
            role="assistant",
            content=text,
            tool_calls=tool_calls,
            provider_state={"anthropic": {"model": request.model, "content": raw_blocks}},
        )
        yield StreamChunk(
            type="done",
            response=ChatResponse(
                message=message, stop_reason=stop, usage=usage, provider=self.name,
                model=getattr(final, "model", request.model) or request.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 1), thinking=thinking,
                refusal_detail=refusal_detail,
            ),
        )

    async def list_models(self) -> list[ModelInfo]:
        client = self._require()
        try:
            page = await client.models.list(limit=100)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"anthropic: list models failed: {e}") from e
        return [ModelInfo(name=m.id, provider=self.name, local=False, capabilities=["completion", "tools", "vision"])
                for m in page.data]

    async def health(self) -> ProviderHealth:
        if self._client is None:
            return ProviderHealth(ok=False, provider=self.name, local=False, error="no API key configured")
        started = time.perf_counter()
        try:
            await self._client.models.list(limit=1)
            return ProviderHealth(ok=True, provider=self.name, local=False, latency_ms=round((time.perf_counter() - started) * 1000, 1))
        except Exception as e:  # noqa: BLE001
            return ProviderHealth(ok=False, provider=self.name, local=False, error=type(e).__name__ + ": " + str(e)[:200])

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
