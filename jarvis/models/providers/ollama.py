"""Ollama provider (local models).

Uses the native Ollama HTTP API: /api/chat (streaming NDJSON, tool calling,
images, structured output), /api/embed, /api/tags and /api/version.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

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


class OllamaProvider(ModelProvider):
    type = "ollama"

    def __init__(self, name: str, base_url: str = "http://127.0.0.1:11434", timeout_s: float = 120.0,
                 options: dict[str, Any] | None = None) -> None:
        super().__init__(name, local=True)
        self.base_url = base_url.rstrip("/")
        self.options = options or {}
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout_s, connect=5.0))
        self._caps: dict[str, list[str]] = {}

    @staticmethod
    def _message(m: Message) -> dict[str, Any]:
        out: dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.images:
            out["images"] = [img.data_b64 for img in m.images]
        if m.tool_calls:
            out["tool_calls"] = [{"function": {"name": tc.name, "arguments": tc.arguments}} for tc in m.tool_calls]
        if m.role == "tool" and m.name:
            out["tool_name"] = m.name
        return out

    async def _capabilities(self, model: str) -> list[str]:
        if model not in self._caps:
            try:
                for info in await self.list_models():
                    self._caps[info.name] = info.capabilities
            except ProviderError:
                return []
        return self._caps.get(model, [])

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        caps = await self._capabilities(request.model)
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [self._message(m) for m in request.messages],
            "stream": True,
        }
        if request.tools:
            if caps and "tools" not in caps:
                raise ProviderError(f"model {request.model} does not support tool calling", retryable=False)
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in request.tools
            ]
        if "thinking" in caps:
            body["think"] = bool(request.think)
        if request.json_schema is not None:
            body["format"] = request.json_schema
        options: dict[str, Any] = {**self.options.get("model_options", {}), **request.options}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            body["options"] = options
        keep_alive = self.options.get("keep_alive")
        if keep_alive is not None:
            body["keep_alive"] = keep_alive

        started = time.perf_counter()
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = Usage()
        done_reason = "stop"
        try:
            async with self._client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise ProviderError(f"ollama {resp.status_code}: {detail}", retryable=resp.status_code >= 500,
                                        status=resp.status_code)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise ProviderError(f"ollama: {data['error']}", retryable=False)
                    msg = data.get("message") or {}
                    if msg.get("thinking"):
                        thinking_parts.append(msg["thinking"])
                        yield StreamChunk(type="thinking", text=msg["thinking"])
                    if msg.get("content"):
                        text_parts.append(msg["content"])
                        yield StreamChunk(type="text", text=msg["content"])
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        args = fn.get("arguments") or {}
                        parse_error = None
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except ValueError:
                                parse_error, args = f"invalid JSON arguments: {args[:200]}", {}
                        tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=args, parse_error=parse_error))
                    if data.get("done"):
                        usage = Usage(input_tokens=data.get("prompt_eval_count") or 0, output_tokens=data.get("eval_count") or 0)
                        done_reason = data.get("done_reason") or "stop"
        except httpx.ConnectError as e:
            raise ProviderError(f"cannot reach Ollama at {self.base_url}: {e}", retryable=True) from e
        except httpx.TimeoutException as e:
            raise ProviderError(f"Ollama request timed out: {e}", retryable=True) from e

        stop = "tool_use" if tool_calls else ("max_tokens" if done_reason == "length" else "end")
        response = ChatResponse(
            message=Message(role="assistant", content="".join(text_parts), tool_calls=tool_calls),
            stop_reason=stop,
            usage=usage,
            provider=self.name,
            model=request.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            thinking="".join(thinking_parts),
        )
        yield StreamChunk(type="done", response=response)

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        try:
            resp = await self._client.post("/api/embed", json={"model": model, "input": texts})
        except httpx.HTTPError as e:
            raise ProviderError(f"ollama embed failed: {e}") from e
        if resp.status_code != 200:
            raise ProviderError(f"ollama embed {resp.status_code}: {resp.text[:300]}", retryable=resp.status_code >= 500,
                                status=resp.status_code)
        return resp.json()["embeddings"]

    async def list_models(self) -> list[ModelInfo]:
        try:
            resp = await self._client.get("/api/tags", timeout=10)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"ollama list failed: {e}") from e
        out = []
        for m in resp.json().get("models", []):
            details = m.get("details") or {}
            caps = list(m.get("capabilities") or [])
            if not caps and "bert" in (details.get("family") or ""):
                caps = ["embedding"]
            out.append(
                ModelInfo(
                    name=m["name"], provider=self.name, local=True, capabilities=caps,
                    size_bytes=m.get("size"), parameter_size=details.get("parameter_size"),
                )
            )
            self._caps[m["name"]] = caps
        return out

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            resp = await self._client.get("/api/version", timeout=3)
            resp.raise_for_status()
            return ProviderHealth(ok=True, provider=self.name, local=True,
                                  latency_ms=round((time.perf_counter() - started) * 1000, 1),
                                  detail={"version": resp.json().get("version")})
        except Exception as e:  # noqa: BLE001 - any failure means unhealthy
            return ProviderHealth(ok=False, provider=self.name, local=True, error=str(e) or type(e).__name__)

    async def close(self) -> None:
        await self._client.aclose()
