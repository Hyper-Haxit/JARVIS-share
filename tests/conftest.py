from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

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

# Tests never show windows on the desktop (e.g. the setup confirmation).
os.environ["JARVIS_NO_DESKTOP_PROMPTS"] = "1"

OLLAMA_URL = os.environ.get("JARVIS_TEST_OLLAMA", "http://127.0.0.1:11434")


def ollama_available() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/version", timeout=2).status_code == 200
    except Exception:
        return False


def ollama_has(model: str) -> bool:
    try:
        names = [m["name"] for m in httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()["models"]]
        return model in names
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(not ollama_available(), reason="local Ollama server not running")


class ScriptedProvider(ModelProvider):
    """Test double: replays scripted responses. Used only in tests.

    Each script entry is either a ChatResponse-like dict (text / tool_calls) or
    an Exception to raise.
    """

    type = "scripted"

    def __init__(self, name: str = "scripted", script: list[Any] | None = None, local: bool = True,
                 healthy: bool = True, embed_dim: int = 8) -> None:
        super().__init__(name, local)
        self.script = list(script or [])
        self.requests: list[ChatRequest] = []
        self.healthy = healthy
        self.embed_dim = embed_dim

    def push(self, *items: Any) -> None:
        self.script.extend(items)

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        if not self.script:
            raise ProviderError("script exhausted", retryable=False)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(request)
        text = item.get("text", "")
        calls = [ToolCall(name=c["name"], arguments=c.get("arguments", {}), id=c.get("id", f"call_{i}"))
                 for i, c in enumerate(item.get("tool_calls", []))]
        for piece in text.split(" ") if text else []:
            yield StreamChunk(type="text", text=piece + " ")
        stop = item.get("stop", "tool_use" if calls else "end")
        yield StreamChunk(type="done", response=ChatResponse(
            message=Message(role="assistant", content=text, tool_calls=calls), stop_reason=stop,
            usage=Usage(input_tokens=10, output_tokens=5), provider=self.name, model=request.model, latency_ms=1.0,
        ))

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        # Deterministic bag-of-letters vectors so similar texts are similar.
        out = []
        for t in texts:
            v = [0.0] * self.embed_dim
            for ch in t.lower():
                if ch.isalpha():
                    v[ord(ch) % self.embed_dim] += 1.0
            out.append(v)
        return out

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="m", provider=self.name, local=self.local, capabilities=["completion", "tools", "vision"])]

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=self.healthy, provider=self.name, local=self.local, error=None if self.healthy else "down")


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "jarvis-home"
    monkeypatch.setenv("JARVIS_HOME", str(home))
    for key in list(os.environ):
        if key.startswith("JARVIS__") or key.startswith("JARVIS_SECRET_"):
            monkeypatch.delenv(key, raising=False)
    return home
