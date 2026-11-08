"""Provider factory: builds provider instances from configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jarvis.config.settings import ProviderConfig
from jarvis.models.base import ModelProvider

ProviderFactory = Callable[[ProviderConfig, Any], ModelProvider]
_registry: dict[str, ProviderFactory] = {}


def register_provider_type(type_name: str, factory: ProviderFactory) -> None:
    """Allow plugins to add new provider types."""
    _registry[type_name] = factory


def _ollama(cfg: ProviderConfig, secrets: Any) -> ModelProvider:
    from jarvis.models.providers.ollama import OllamaProvider

    return OllamaProvider(cfg.name, cfg.base_url or "http://127.0.0.1:11434", cfg.timeout_s, cfg.options)


def _anthropic(cfg: ProviderConfig, secrets: Any) -> ModelProvider:
    from jarvis.models.providers.anthropic_provider import AnthropicProvider

    key = secrets.get(cfg.api_key_secret or "anthropic_api_key")
    return AnthropicProvider(cfg.name, key, cfg.base_url, cfg.timeout_s, cfg.options)


def _openai(cfg: ProviderConfig, secrets: Any) -> ModelProvider:
    from jarvis.models.providers.openai_compat import OpenAICompatProvider

    key = secrets.get(cfg.api_key_secret) if cfg.api_key_secret else None
    return OpenAICompatProvider(cfg.name, cfg.base_url or "https://api.openai.com/v1", key, cfg.timeout_s, cfg.local, cfg.options)


register_provider_type("ollama", _ollama)
register_provider_type("anthropic", _anthropic)
register_provider_type("openai", _openai)


def build_provider(cfg: ProviderConfig, secrets: Any) -> ModelProvider:
    factory = _registry.get(cfg.type)
    if factory is None:
        raise ValueError(f"unknown provider type '{cfg.type}'")
    provider = factory(cfg, secrets)
    if cfg.local is not None:
        provider.local = cfg.local
    return provider
