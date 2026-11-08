"""Central event bus.

Every subsystem publishes what it is doing here; clients, devices, the HUD and
plugins subscribe without the core needing to know they exist. Events are
redacted before they are stored or delivered.
"""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from jarvis.core.observability import get_logger, get_trace_id, redact

log = get_logger("events")


class EventType(StrEnum):
    # voice
    USER_SPEECH_STARTED = "USER_SPEECH_STARTED"
    USER_SPEECH_ENDED = "USER_SPEECH_ENDED"
    WAKE_WORD_DETECTED = "WAKE_WORD_DETECTED"
    TRANSCRIPT = "TRANSCRIPT"
    MIC_STATE = "MIC_STATE"
    AUDIO_LEVEL = "AUDIO_LEVEL"
    SPEECH_STARTED = "SPEECH_STARTED"
    SPEECH_ENDED = "SPEECH_ENDED"
    SPEECH_INTERRUPTED = "SPEECH_INTERRUPTED"
    # models
    MODEL_REQUEST_STARTED = "MODEL_REQUEST_STARTED"
    MODEL_RESPONSE_STARTED = "MODEL_RESPONSE_STARTED"
    MODEL_RESPONSE_DELTA = "MODEL_RESPONSE_DELTA"
    MODEL_RESPONSE_COMPLETED = "MODEL_RESPONSE_COMPLETED"
    MODEL_REQUEST_FAILED = "MODEL_REQUEST_FAILED"
    # agent / conversation
    CHAT_MESSAGE = "CHAT_MESSAGE"
    AGENT_STATE = "AGENT_STATE"
    # tools
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    TOOL_FAILED = "TOOL_FAILED"
    # tasks
    TASK_CREATED = "TASK_CREATED"
    TASK_STARTED = "TASK_STARTED"
    TASK_PROGRESS = "TASK_PROGRESS"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_CANCELLED = "TASK_CANCELLED"
    # devices
    DEVICE_REGISTERED = "DEVICE_REGISTERED"
    DEVICE_CONNECTED = "DEVICE_CONNECTED"
    DEVICE_DISCONNECTED = "DEVICE_DISCONNECTED"
    DEVICE_REVOKED = "DEVICE_REVOKED"
    DEVICE_TELEMETRY = "DEVICE_TELEMETRY"
    DEVICE_WATCHDOG_TIMEOUT = "DEVICE_WATCHDOG_TIMEOUT"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    EMERGENCY_STOP_CLEARED = "EMERGENCY_STOP_CLEARED"
    SAFETY_VIOLATION = "SAFETY_VIOLATION"
    # permissions
    PERMISSION_REQUESTED = "PERMISSION_REQUESTED"
    PERMISSION_GRANTED = "PERMISSION_GRANTED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    # memory
    MEMORY_CREATED = "MEMORY_CREATED"
    MEMORY_UPDATED = "MEMORY_UPDATED"
    MEMORY_DELETED = "MEMORY_DELETED"
    # system
    SYSTEM_STARTED = "SYSTEM_STARTED"
    SYSTEM_STOPPING = "SYSTEM_STOPPING"
    PROVIDER_STATUS = "PROVIDER_STATUS"
    NOTIFICATION = "NOTIFICATION"
    HUD_UPDATE = "HUD_UPDATE"
    PLUGIN_LOADED = "PLUGIN_LOADED"
    PLUGIN_FAILED = "PLUGIN_FAILED"
    SYNC_COMPLETED = "SYNC_COMPLETED"


# High-frequency events are delivered live but not persisted.
EPHEMERAL = {
    EventType.MODEL_RESPONSE_DELTA,
    EventType.AUDIO_LEVEL,
    EventType.DEVICE_TELEMETRY,
    EventType.HUD_UPDATE,
}


@dataclass
class Event:
    type: str
    data: dict[str, Any]
    source: str = "core"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)
    seq: int = 0
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seq": self.seq,
            "type": self.type,
            "ts": self.ts,
            "source": self.source,
            "trace_id": self.trace_id,
            "data": self.data,
        }


Handler = Callable[[Event], Awaitable[None] | None]


class Subscription:
    """A bounded queue of events matching a set of glob patterns."""

    def __init__(self, bus: EventBus, patterns: list[str] | None, maxsize: int) -> None:
        self.bus = bus
        self.patterns = patterns or ["*"]
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self.closed = False

    def matches(self, event: Event) -> bool:
        return any(fnmatch.fnmatchcase(event.type, p) for p in self.patterns)

    def offer(self, event: Event) -> None:
        if self.closed or not self.matches(event):
            return
        if self.queue.full():
            # Drop the oldest so slow consumers always see the newest state.
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(event)

    async def get(self, timeout: float | None = None) -> Event | None:
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except TimeoutError:
            return None

    def __aiter__(self) -> Subscription:
        return self

    async def __anext__(self) -> Event:
        if self.closed:
            raise StopAsyncIteration
        return await self.queue.get()

    def close(self) -> None:
        self.closed = True
        self.bus._remove(self)


class EventBus:
    def __init__(self, db: Any | None = None, retention: int = 5000) -> None:
        self.db = db
        self.retention = retention
        self._subs: list[Subscription] = []
        self._handlers: list[tuple[str, Handler]] = []
        self._recent: deque[Event] = deque(maxlen=500)
        self._seq = 0
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._persist_count = 0
        if db is not None:
            row = db.query_one("SELECT MAX(seq) AS s FROM events")
            self._seq = int(row["s"] or 0) if row else 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, patterns: list[str] | None = None, maxsize: int = 1000) -> Subscription:
        sub = Subscription(self, patterns, maxsize)
        self._subs.append(sub)
        return sub

    def _remove(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)

    def on(self, pattern: str, handler: Handler) -> Callable[[], None]:
        entry = (pattern, handler)
        self._handlers.append(entry)

        def unsubscribe() -> None:
            if entry in self._handlers:
                self._handlers.remove(entry)

        return unsubscribe

    def _make(self, type_: str, data: dict[str, Any] | None, source: str) -> Event:
        with self._lock:
            self._seq += 1
            seq = self._seq
        return Event(type=str(type_), data=redact(data or {}), source=source, seq=seq, trace_id=get_trace_id())

    async def publish(self, type_: str, data: dict[str, Any] | None = None, source: str = "core") -> Event:
        event = self._make(type_, data, source)
        self._dispatch(event)
        for pattern, handler in list(self._handlers):
            if fnmatch.fnmatchcase(event.type, pattern):
                try:
                    result = handler(event)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    log.exception("event handler failed for %s", event.type)
        return event

    def emit(self, type_: str, data: dict[str, Any] | None = None, source: str = "core") -> None:
        """Publish from synchronous code or from another thread."""
        loop = self._loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            running.create_task(self.publish(type_, data, source))
        elif loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.publish(type_, data, source), loop)
        else:
            event = self._make(type_, data, source)
            self._dispatch(event)

    def _dispatch(self, event: Event) -> None:
        self._recent.append(event)
        if self.db is not None and event.type not in EPHEMERAL:
            self._persist(event)
        for sub in list(self._subs):
            try:
                sub.offer(event)
            except Exception:  # pragma: no cover
                log.exception("subscriber failed")

    def _persist(self, event: Event) -> None:
        try:
            self.db.execute(
                "INSERT INTO events(id, seq, ts, type, source, data_json, trace_id) VALUES(?,?,?,?,?,?,?)",
                (event.id, event.seq, event.ts, event.type, event.source, json.dumps(event.data, default=str), event.trace_id),
            )
            self._persist_count += 1
            if self._persist_count % 200 == 0:
                self.db.execute(
                    "DELETE FROM events WHERE seq <= (SELECT MAX(seq) FROM events) - ?", (self.retention,)
                )
        except Exception:  # pragma: no cover - persistence must never break the bus
            log.exception("failed to persist event")

    def recent(self, limit: int = 100, patterns: list[str] | None = None, after_seq: int = 0) -> list[dict[str, Any]]:
        if self.db is not None:
            rows = self.db.query(
                "SELECT * FROM events WHERE seq > ? ORDER BY seq DESC LIMIT ?", (after_seq, max(limit * 4, limit))
            )
            out = []
            for r in rows:
                if patterns and not any(fnmatch.fnmatchcase(r["type"], p) for p in patterns):
                    continue
                out.append(
                    {
                        "id": r["id"], "seq": r["seq"], "type": r["type"], "ts": r["ts"],
                        "source": r["source"], "trace_id": r["trace_id"], "data": json.loads(r["data_json"]),
                    }
                )
                if len(out) >= limit:
                    break
            return list(reversed(out))
        events = [e for e in self._recent if e.seq > after_seq]
        if patterns:
            events = [e for e in events if any(fnmatch.fnmatchcase(e.type, p) for p in patterns)]
        return [e.to_dict() for e in events[-limit:]]

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)
