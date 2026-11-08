"""JARVIS runtime: constructs and wires every subsystem.

The runtime is the composition root. Subsystems receive what they need through
it; none of them imports a client, a UI or a hardware driver directly.
"""

from __future__ import annotations

import asyncio
import platform
import sys
import time
import uuid
from typing import Any

from jarvis import __version__
from jarvis.capabilities.base import ToolExecutor, ToolRegistry
from jarvis.config.paths import JarvisPaths
from jarvis.config.secrets import SecretStore
from jarvis.config.settings import JarvisConfig, load_config
from jarvis.core.agent import Agent
from jarvis.core.auth import AuthService
from jarvis.core.changelog import ChangeLog
from jarvis.core.conversation import ConversationStore
from jarvis.core.enrollment import LocalEnrollment
from jarvis.core.events import EventBus, EventType
from jarvis.core.hlc import HybridClock
from jarvis.core.identity import IdentityService
from jarvis.core.local_prompt import make_prompt
from jarvis.core.memory.manager import MemoryManager
from jarvis.core.observability import get_logger, setup_logging
from jarvis.core.permissions import PermissionGate
from jarvis.core.storage import Database
from jarvis.models.base import ModelProvider
from jarvis.models.providers import build_provider
from jarvis.models.router import ModelRouter

log = get_logger("runtime")


class JarvisRuntime:
    def __init__(
        self,
        paths: JarvisPaths | None = None,
        config: JarvisConfig | None = None,
        providers: dict[str, ModelProvider] | None = None,
        secrets: SecretStore | None = None,
        configure_logging: bool = True,
        load_plugins: bool = True,
    ) -> None:
        migrated_from = None
        if paths is None:
            from jarvis.config.paths import default_home, migrate_legacy_home  # noqa: PLC0415

            migrated_from = migrate_legacy_home(default_home())
        self.paths = paths or JarvisPaths()
        self.paths.ensure()
        self.config = config or load_config(self.paths)
        if configure_logging:
            setup_logging(self.config.logging.level, self.paths.logs, self.config.logging.json_logs)
        if migrated_from is not None:
            log.warning("moved JARVIS data from %s to %s (the old folder is no longer used)", migrated_from, self.paths.home)
        self.started_at: float | None = None
        self.db = Database(self.paths.database)
        node_id = self.db.kv_get("node_id")
        if not node_id:
            node_id = "node_" + uuid.uuid4().hex[:10]
            self.db.kv_set("node_id", node_id)
        self.node_id: str = node_id
        self.node_name = self.config.node_name or platform.node() or node_id
        self.clock = HybridClock(node_id)
        self.changelog = ChangeLog(self.db, self.clock)
        self.events = EventBus(self.db, self.config.logging.event_retention)
        self.secrets = secrets or SecretStore(self.paths.fallback_secrets_file)
        self.auth = AuthService(self.db, self.events, self.config.server.session_ttl_hours,
                                self.config.server.pairing_code_ttl_minutes)
        self.enrollment = LocalEnrollment(self.auth, make_prompt(self.config.server.setup_confirmation_window))
        self.gate = PermissionGate(self.db, self.events, self.config.permissions)

        self.provider_errors: dict[str, str] = {}
        if providers is None:
            providers = {}
            for pc in self.config.providers:
                if not pc.enabled:
                    continue
                try:
                    providers[pc.name] = build_provider(pc, self.secrets)
                except Exception as e:  # noqa: BLE001 - one bad provider must not stop JARVIS
                    self.provider_errors[pc.name] = str(e)
                    log.error("provider %s could not be created: %s", pc.name, e)
        self.router = ModelRouter(self.config, providers, self.db, self.events)
        self.identity = IdentityService(self.db, self.changelog)
        self.conversations = ConversationStore(self.db)
        self.memory = MemoryManager(self.db, self.events, self.changelog, self.router, self.config.memory)
        self.tools = ToolRegistry()
        self.executor = ToolExecutor(self.tools, self.gate, self.events, self.db, self.config.agent.tool_result_max_chars)
        self.agent = Agent(self)
        self.extensions: dict[str, Any] = {}
        self._background: list[asyncio.Task[Any]] = []
        self.local_token: str | None = None
        self._register_capabilities()
        self._load_plugins = load_plugins

    # ------------------------------------------------------------ capabilities
    def _register_capabilities(self) -> None:
        from jarvis.capabilities import load_builtin_capabilities  # noqa: PLC0415

        load_builtin_capabilities(self)

    # --------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.events.bind_loop(loop)
        self.local_token = self.auth.ensure_local_device(self.paths.local_token_file, platform=sys.platform)
        self.started_at = time.time()
        for name, ext in list(self.extensions.items()):
            start = getattr(ext, "start", None)
            if start is not None:
                try:
                    await start()
                except Exception:  # noqa: BLE001
                    log.exception("extension %s failed to start", name)
        if self._load_plugins:
            plugins = self.extensions.get("plugins")
            if plugins is not None:
                await plugins.load_enabled()
        self._background.append(asyncio.create_task(self._maintenance_loop(), name="jarvis-maintenance"))
        await self.events.publish(EventType.SYSTEM_STARTED, {"version": __version__, "node_id": self.node_id,
                                                             "node_name": self.node_name})
        log.info("JARVIS core started (node %s)", self.node_id)

    async def stop(self) -> None:
        await self.events.publish(EventType.SYSTEM_STOPPING, {})
        for t in self._background:
            t.cancel()
        for name, ext in list(self.extensions.items()):
            stop = getattr(ext, "stop", None)
            if stop is not None:
                try:
                    await stop()
                except Exception:  # noqa: BLE001
                    log.exception("extension %s failed to stop", name)
        await self.router.close()
        self.db.close()

    async def _maintenance_loop(self) -> None:
        """Periodic housekeeping: retention, episodic summaries, embedding backfill."""
        await asyncio.sleep(5)
        while True:
            try:
                self.auth.purge_expired()
                removed = self.conversations.purge_older_than(self.config.memory.conversation_retention_days)
                if removed:
                    log.info("retention: removed %d old conversations", removed)
                if self.config.memory.episodic_summaries:
                    for cid in self.conversations.unsummarized()[:3]:
                        try:
                            await self.memory.summarize_conversation(self.conversations, cid)
                        except Exception as e:  # noqa: BLE001
                            log.warning("episodic summary failed for %s: %s", cid, e)
                            break
                await self.memory.backfill_embeddings()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("maintenance iteration failed")
            await asyncio.sleep(300)

    def persist_config(self) -> None:
        from jarvis.config.settings import save_config  # noqa: PLC0415

        save_config(self.config, self.paths.config_file)

    # ------------------------------------------------------------------ status
    async def service_notices(self) -> list[str]:
        notices = []
        mode = self.router.mode
        if mode == "local_only":
            notices.append("Routing is in local-only mode: cloud models are disabled by the user.")
        for h in await self.router.all_health():
            if not h.ok:
                kind = "local" if h.local else "cloud"
                notices.append(f"{kind} model provider '{h.provider}' is unavailable ({h.error}).")
        return notices

    async def status(self) -> dict[str, Any]:
        health = await self.router.all_health()
        ext_status = {}
        for name, ext in self.extensions.items():
            fn = getattr(ext, "status", None)
            if fn is not None:
                try:
                    res = fn()
                    ext_status[name] = await res if asyncio.iscoroutine(res) else res
                except Exception as e:  # noqa: BLE001
                    ext_status[name] = {"error": str(e)}
        healthy = {h.provider for h in health if h.ok}
        roles = {}
        for role in ("fast", "reasoning", "coding", "vision"):
            cands = self.router.role_candidates(role)
            usable = [c for c in cands if c.split("/", 1)[0] in healthy]
            roles[role] = {"candidates": cands, "active": usable[0] if usable else None}
        ident = self.identity.get()
        return {
            "name": ident.name,
            "user_name": ident.user_name,
            "roles": roles,
            "version": __version__,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "platform": sys.platform,
            "started_at": self.started_at,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else None,
            "routing_mode": self.router.mode,
            "event_subscribers": self.events.subscriber_count,
            "providers": [h.model_dump() for h in health],
            "provider_errors": self.provider_errors,
            "memory": self.memory.count(),
            "embedding": self.memory.embedding_status,
            "pending_approvals": len(self.gate.pending()),
            "tools": len(self.tools.all()),
            "event_subscribers": self.events.subscriber_count,
            "secret_backend": self.secrets.backend_name,
            "extensions": ext_status,
        }
