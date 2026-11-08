"""Built-in capability registration.

Each capability module exposes ``register(runtime)``. Modules whose optional
dependencies are missing still register their tools; the tools report
themselves unavailable (with the reason) instead of silently disappearing.
"""

from __future__ import annotations

import importlib
from typing import Any

from jarvis.core.observability import get_logger

log = get_logger("capabilities")

BUILTIN_MODULES = [
    "jarvis.capabilities.core_tools",
    "jarvis.core.tasks",
    "jarvis.voice.service",
    "jarvis.capabilities.files",
    "jarvis.capabilities.shell",
    "jarvis.capabilities.computer",
    "jarvis.capabilities.web",
    "jarvis.capabilities.browser",
    "jarvis.capabilities.vision",
    "jarvis.capabilities.coding",
]


def load_builtin_capabilities(runtime: Any) -> None:
    from jarvis.capabilities.core_tools import core_tools

    runtime.tools.register_all(core_tools())
    for mod_name in BUILTIN_MODULES[1:]:
        try:
            mod = importlib.import_module(mod_name)
            mod.register(runtime)
        except Exception:  # noqa: BLE001 - a broken capability must not take down the core
            log.exception("capability module %s failed to register", mod_name)
