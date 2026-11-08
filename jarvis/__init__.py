"""JARVIS - a cross-platform personal AI operating layer.

The package is organised in layers that never import "upwards":

    jarvis.config        configuration, paths, secret storage
    jarvis.core          identity, events, permissions, memory, agent, tasks, auth, sync
    jarvis.models        model provider abstraction and routing
    jarvis.capabilities  tools the agent can use (filesystem, shell, web, vision, ...)
    jarvis.platforms     OS adapters (Windows / macOS / Linux)
    jarvis.voice         audio pipeline (VAD, wake word, STT, TTS)
    jarvis.hardware      device gateway, capability protocol, safety layer
    jarvis.plugins       plugin loader
    jarvis.api           HTTP / WebSocket API
    jarvis.clients       CLI, desktop shell and web client assets
"""

__version__ = "0.1.0"
