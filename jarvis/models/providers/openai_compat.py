"""OpenAI-compatible provider.

Covers any server implementing the OpenAI Chat Completions wire format: OpenAI
itself, LM Studio, llama.cpp server, vLLM, LocalAI, OpenRouter, Groq, etc.
Whether it is treated as local is decided by configuration (or inferred from a
loopback base URL).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx

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

_STOP = {"stop": "end", "tool_calls": "tool_use", "function_call": "tool_use", "length": "max_tokens", "content_filter": "refusal"}


def _is_loopback(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


class OpenAICompatProvider(ModelProvider):
    type = "openai"

    def __init__(self, name: str, base_url: str, api_key: str | None = None, timeout_s: float = 120.0,
                 local: bool | None = None, options: dict[str, Any] | None = None) -> None:
        super().__init__(name, local=_is_loopback(base_url) if local is None else local)
        self.base_url = base_url.rstrip("/")
        self.options = options or {}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=httpx.Timeout(timeout_s, connect=5.0))

    @staticmethod
    def _message(m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""}
        if m.role == "assistant":
            out: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                out["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in m.tool_calls
                ]
            return out
        if m.images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": m.content or ""}]
            for img in m.images:
                parts.append({"type": "image_url", "image_url": {"url": f"data:{img.media_type};base64,{img.data_b64}"}})
            return {"role": m.role, "content": parts}
        return {"role": m.role, "content": m.content or ""}

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [self._message(m) for m in request.messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in request.tools
            ]
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": request.json_schema}}
        body.update(self.options.get("extra_body", {}))

        started = time.perf_counter()
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage = Usage()
        finish = "stop"
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise ProviderError(f"{self.name} {resp.status_code}: {detail}",
                                        retryable=resp.status_code >= 500 or resp.status_code == 429, status=resp.status_code)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    data = json.loads(payload)
                    if data.get("usage"):
                        u = data["usage"]
                        usage = Usage(input_tokens=u.get("prompt_tokens") or 0, output_tokens=u.get("completion_tokens") or 0)
                    for choice in data.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            text_parts.append(delta["content"])
                            yield StreamChunk(type="text", text=delta["content"])
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["args"] += fn["arguments"]
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
        except httpx.ConnectError as e:
            raise ProviderError(f"cannot reach {self.name} at {self.base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name} request timed out: {e}") from e

        tool_calls = []
        for idx in sorted(calls):
            slot = calls[idx]
            try:
                args = json.loads(slot["args"]) if slot["args"].strip() else {}
                err = None if isinstance(args, dict) else "arguments were not a JSON object"
            except ValueError:
                args, err = {}, f"invalid JSON arguments: {slot['args'][:200]}"
            kwargs = {"id": slot["id"]} if slot["id"] else {}
            tool_calls.append(ToolCall(name=slot["name"], arguments=args if isinstance(args, dict) else {}, parse_error=err, **kwargs))
        stop = _STOP.get(finish, "end")
        if tool_calls and stop == "end":
            stop = "tool_use"
        yield StreamChunk(
            type="done",
            response=ChatResponse(
                message=Message(role="assistant", content="".join(text_parts), tool_calls=tool_calls),
                stop_reason=stop, usage=usage, provider=self.name, model=request.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            ),
        )

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        try:
            resp = await self._client.post("/embeddings", json={"model": model, "input": texts})
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name} embed failed: {e}") from e
        if resp.status_code != 200:
            raise ProviderError(f"{self.name} embed {resp.status_code}: {resp.text[:300]}", status=resp.status_code,
                                retryable=resp.status_code >= 500)
        data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    async def list_models(self) -> list[ModelInfo]:
        try:
            resp = await self._client.get("/models", timeout=10)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name} list models failed: {e}") from e
        return [ModelInfo(name=m["id"], provider=self.name, local=self.local) for m in resp.json().get("data", [])]

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            resp = await self._client.get("/models", timeout=5)
            if resp.status_code in (401, 403):
                return ProviderHealth(ok=False, provider=self.name, local=self.local, error="authentication failed")
            resp.raise_for_status()
            return ProviderHealth(ok=True, provider=self.name, local=self.local,
                                  latency_ms=round((time.perf_counter() - started) * 1000, 1))
        except Exception as e:  # noqa: BLE001
            return ProviderHealth(ok=False, provider=self.name, local=self.local, error=str(e) or type(e).__name__)

    async def close(self) -> None:
        await self._client.aclose()
