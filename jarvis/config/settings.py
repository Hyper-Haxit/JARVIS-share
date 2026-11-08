"""Typed configuration.

Precedence (lowest to highest):
    built-in defaults  <  config.toml  <  environment variables (JARVIS__SECTION__KEY)

Nothing secret lives in this file: providers reference secrets *by name* and the
values are resolved through :mod:`jarvis.config.secrets` at runtime.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, Field

from jarvis.config.paths import JarvisPaths

Decision = Literal["allow", "ask", "deny"]


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7777
    tls_cert: str | None = None
    tls_key: str | None = None
    # Binding to a non-loopback address without TLS is refused unless this is set.
    allow_insecure_lan: bool = False
    cors_origins: list[str] = Field(default_factory=list)
    session_ttl_hours: float = 12.0
    pairing_code_ttl_minutes: float = 10.0
    rate_limit_per_minute: int = 240
    auth_rate_limit_per_minute: int = 20
    # Setting up a browser on this computer is confirmed by the person at this
    # computer: in a confirmation window (when the core runs on a desktop) or
    # with `jarvis devices approve`. Set to false for terminal-only approval.
    setup_confirmation_window: bool = True


class ProviderConfig(BaseModel):
    name: str
    type: Literal["ollama", "anthropic", "openai"]
    base_url: str | None = None
    # Name of the secret holding the API key (never the key itself).
    api_key_secret: str | None = None
    enabled: bool = True
    timeout_s: float = 120.0
    # Cloud providers send data off-device; local ones do not.
    local: bool | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class RolesConfig(BaseModel):
    """Ordered candidate lists ("provider/model") for each model role."""

    fast: list[str] = Field(default_factory=lambda: ["ollama/qwen3:4b-instruct"])
    reasoning: list[str] = Field(default_factory=lambda: ["ollama/qwen2.5:7b"])
    coding: list[str] = Field(default_factory=lambda: ["ollama/qwen2.5-coder:7b"])
    vision: list[str] = Field(default_factory=lambda: ["ollama/gemma3:4b"])
    embedding: list[str] = Field(default_factory=lambda: ["ollama/nomic-embed-text"])
    fallback: list[str] = Field(default_factory=lambda: ["ollama/llama3.2:3b"])


class RoutingConfig(BaseModel):
    # auto: use role candidates in order, skipping unhealthy providers
    # local_only: never send data to cloud providers
    # cloud_preferred: try cloud candidates first, fall back to local
    mode: Literal["auto", "local_only", "cloud_preferred"] = "auto"
    health_cache_s: float = 30.0
    complex_keywords: list[str] = Field(
        default_factory=lambda: [
            "plan", "analyze", "analyse", "compare", "research", "strategy", "prove",
            "step by step", "pros and cons", "trade-off", "tradeoff", "design", "evaluate",
        ]
    )
    coding_keywords: list[str] = Field(
        default_factory=lambda: [
            "code", "function", "bug", "stack trace", "traceback", "compile", "refactor",
            "repository", "repo", "unit test", "python", "javascript", "typescript", "rust",
            "exception", "crash", "git ", "pull request", "regex", "sql",
        ]
    )
    long_message_chars: int = 1200


class PermissionsConfig(BaseModel):
    defaults: dict[str, Decision] = Field(
        default_factory=lambda: {
            "SAFE": "allow",
            "LOW_RISK": "allow",
            "SENSITIVE": "ask",
            "DANGEROUS": "ask",
        }
    )
    # tool name -> decision; can only make a tool stricter for DANGEROUS tools.
    tool_overrides: dict[str, Decision] = Field(default_factory=dict)
    approval_timeout_s: float = 300.0
    # Filesystem roots tools may touch without escalation. Paths outside these
    # roots are escalated to SENSITIVE (read) or DANGEROUS (write/delete).
    allowed_roots: list[str] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    conversation_retention_days: int = 90
    episodic_summaries: bool = True
    auto_recall: bool = True
    recall_limit: int = 6
    semantic_chunk_chars: int = 1200


class AgentConfig(BaseModel):
    max_steps: int = 12
    history_messages: int = 30
    tool_result_max_chars: int = 12000
    task_max_parallel: int = 3
    task_node_timeout_s: float = 600.0
    task_node_retries: int = 1


class VoiceConfig(BaseModel):
    enabled: bool = False
    wake_word: str = "hey_jarvis"
    wake_engine: Literal["openwakeword", "transcript", "none"] = "openwakeword"
    wake_threshold: float = 0.5
    stt_engine: Literal["faster-whisper", "openai"] = "faster-whisper"
    stt_model: str = "base.en"
    # ---- Text-to-speech: the one place JARVIS's voice is chosen.
    #   tts_engine    "kokoro"  local neural voice (Kokoro-82M fp32; model files via `jarvis voice install`)
    #                 "system"  the operating system's voice (SAPI on Windows); "pyttsx3" is its old name
    #                 "none"    no speech
    #   tts_fallback  "system"  speaks whatever the primary voice can't (not installed, failed, still loading); "none"
    tts_engine: Literal["kokoro", "system", "pyttsx3", "none"] = "kokoro"
    tts_voice: str = "af_heart"            # Kokoro voice id: a/b = American/British English, f/m = female/male
    tts_speed: float = Field(default=1.0, ge=0.5, le=2.0)  # Kokoro speaking speed
    tts_model_dir: str | None = None       # Kokoro model folder; default <JARVIS home>/models/tts/kokoro
    tts_fallback: Literal["system", "none"] = "system"
    tts_fallback_voice: str | None = None  # system voice name (part of it is enough); None = system default
    tts_rate: int = 185                    # system voice speaking rate, words per minute
    tts_preload: bool = True               # load Kokoro in the background when JARVIS starts (~1 GB RAM)
    tts_init_wait_s: float = 12.0          # a first utterance waits this long for Kokoro to load, then falls back
    tts_boost_priority: bool = True        # above-normal priority while synthesizing (Windows); see voice/kokoro.py
    input_device: str | int | None = None
    output_device: str | int | None = None
    sample_rate: int = 16000
    vad_threshold_db: float = 12.0
    silence_ms: int = 800
    max_utterance_s: float = 20.0
    # Seconds after a reply during which you can answer without the wake word.
    # Off by default: one wake word is one request, and room conversation can't
    # turn into requests on its own.
    conversation_window_s: float = 0.0
    barge_in: bool = True
    # How you can interrupt JARVIS while it speaks:
    #   wake_word  say the wake word (safe with speakers: JARVIS's own voice can't interrupt it)
    #   voice      any loud speech (only for headsets, where JARVIS's voice can't reach the mic)
    barge_in_mode: Literal["wake_word", "voice"] = "wake_word"
    # Audio never leaves the device unless explicitly allowed.
    allow_cloud_audio: bool = False


class HomeAssistantConfig(BaseModel):
    url: str | None = None
    token_secret: str = "home_assistant_token"
    verify_tls: bool = True
    # domain -> risk level name
    domain_risk: dict[str, str] = Field(
        default_factory=lambda: {
            "light": "LOW_RISK",
            "switch": "LOW_RISK",
            "fan": "LOW_RISK",
            "media_player": "LOW_RISK",
            "scene": "LOW_RISK",
            "climate": "SENSITIVE",
            "cover": "SENSITIVE",
            "vacuum": "SENSITIVE",
            "lock": "DANGEROUS",
            "alarm_control_panel": "DANGEROUS",
            "valve": "DANGEROUS",
            "siren": "SENSITIVE",
        }
    )


class WebConfig(BaseModel):
    search_provider: Literal["duckduckgo", "searxng", "brave"] = "duckduckgo"
    searxng_url: str | None = None
    brave_api_key_secret: str = "brave_api_key"
    user_agent: str = "Mozilla/5.0 (compatible; JARVIS/0.1; personal assistant)"
    max_page_chars: int = 20000


class CalendarConfig(BaseModel):
    # Paths to .ics files or http(s) iCal URLs (e.g. a Google Calendar secret address).
    sources: list[str] = Field(default_factory=list)


class HardwareConfig(BaseModel):
    heartbeat_interval_s: float = 2.0
    watchdog_timeout_s: float = 6.0


class PluginsConfig(BaseModel):
    directories: list[str] = Field(default_factory=list)
    load_builtin: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_logs: bool = True
    diagnostics: bool = False
    event_retention: int = 5000


class JarvisConfig(BaseModel):
    node_name: str | None = None
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: list[ProviderConfig] = Field(
        default_factory=lambda: [
            ProviderConfig(name="ollama", type="ollama", base_url="http://127.0.0.1:11434"),
        ]
    )
    roles: RolesConfig = Field(default_factory=RolesConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    home_assistant: HomeAssistantConfig = Field(default_factory=HomeAssistantConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


ENV_PREFIX = "JARVIS__"


def _coerce(value: str) -> Any:
    low = value.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") or value.startswith("{"):
        import json

        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def _apply_env(data: dict[str, Any], environ: dict[str, str]) -> dict[str, Any]:
    for key, value in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX):].split("__") if p]
        if not parts:
            continue
        cursor = data
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                break
        else:
            cursor[parts[-1]] = _coerce(value)
    return data


def load_config(paths: JarvisPaths, environ: dict[str, str] | None = None) -> JarvisConfig:
    data: dict[str, Any] = {}
    if paths.config_file.exists():
        with paths.config_file.open("rb") as f:
            data = tomllib.load(f)
    data = _apply_env(data, dict(os.environ if environ is None else environ))
    return JarvisConfig.model_validate(data)


def save_config(config: JarvisConfig, path: Path) -> None:
    data = config.model_dump(mode="json", exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_bytes(tomli_w.dumps(data).encode("utf-8"))
    tmp.replace(path)
