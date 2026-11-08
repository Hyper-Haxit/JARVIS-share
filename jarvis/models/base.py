"""Provider-neutral model interface.

JARVIS Core speaks only these types. Each provider adapter translates them to
and from its own wire format, so the agent, planner, memory and voice systems
never depend on a specific vendor.
"""

from __future__ import annotations

import abc
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["end", "tool_use", "max_tokens", "refusal", "error"]


class ImageInput(BaseModel):
    data_b64: str
    media_type: str = "image/png"
    source: str = "upload"  # upload | screenshot | camera | device:<id>
    captured_at: float | None = None
    width: int | None = None
    height: int | None = None

    def age_s(self) -> float | None:
        return None if self.captured_at is None else max(0.0, time.time() - self.captured_at)


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: "call_" + uuid.uuid4().hex[:12])
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Set when the provider could not produce valid JSON for the arguments.
    parse_error: str | None = None


class Message(BaseModel):
    role: Role
    content: str = ""
    images: list[ImageInput] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name for role=tool
    is_error: bool = False
    # Provider-specific state that must be replayed verbatim (e.g. thinking
    # blocks). Keyed by provider name; ignored by other providers.
    provider_state: dict[str, Any] = Field(default_factory=dict)


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str
    tools: list[ToolSchema] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    think: bool = False
    # JSON schema for structured output (provider support varies).
    json_schema: dict[str, Any] | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    message: Message
    stop_reason: StopReason = "end"
    usage: Usage = Field(default_factory=Usage)
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    thinking: str = ""
    refusal_detail: str | None = None


class StreamChunk(BaseModel):
    type: Literal["text", "thinking", "done"]
    text: str = ""
    response: ChatResponse | None = None


class ModelInfo(BaseModel):
    name: str
    provider: str
    local: bool
    capabilities: list[str] = Field(default_factory=list)
    size_bytes: int | None = None
    parameter_size: str | None = None
    context_length: int | None = None


class ProviderHealth(BaseModel):
    ok: bool
    provider: str
    local: bool
    latency_ms: float | None = None
    error: str | None = None
    checked_at: float = Field(default_factory=time.time)
    detail: dict[str, Any] = Field(default_factory=dict)


class ProviderError(Exception):
    def __init__(self, message: str, *, retryable: bool = True, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class ModelProvider(abc.ABC):
    """Base class for all model providers."""

    type: str = "abstract"

    def __init__(self, name: str, local: bool) -> None:
        self.name = name
        self.local = local

    @abc.abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream a response. The final chunk has type 'done' and carries the full response."""

    async def chat(self, request: ChatRequest) -> ChatResponse:
        final: ChatResponse | None = None
        async for chunk in self.stream(request):
            if chunk.type == "done":
                final = chunk.response
        if final is None:
            raise ProviderError(f"{self.name}: stream ended without a response")
        return final

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise ProviderError(f"{self.name} does not support embeddings", retryable=False)

    @abc.abstractmethod
    async def list_models(self) -> list[ModelInfo]: ...

    @abc.abstractmethod
    async def health(self) -> ProviderHealth: ...

    async def close(self) -> None:  # pragma: no cover - default no-op
        return None
