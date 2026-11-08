"""Phase 1: configuration, secrets, storage, events, auth, permissions."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from jarvis.config.paths import JarvisPaths
from jarvis.config.secrets import SecretStore
from jarvis.config.settings import JarvisConfig, PermissionsConfig, load_config, save_config
from jarvis.core.auth import AuthError, AuthService
from jarvis.core.events import EventBus, EventType
from jarvis.core.hlc import HybridClock
from jarvis.core.observability import REDACTED, redact, redactor
from jarvis.core.permissions import ActionRequest, PermissionDenied, PermissionGate, Principal, Risk
from jarvis.core.storage import MIGRATIONS, Database


# ----------------------------------------------------------------------- config
def test_config_defaults_file_and_env_precedence(tmp_home: Path) -> None:
    paths = JarvisPaths()
    assert paths.home == tmp_home.resolve()
    cfg = load_config(paths, environ={})
    assert cfg.server.host == "127.0.0.1"
    cfg.server.port = 8123
    save_config(cfg, paths.config_file)
    assert load_config(paths, environ={}).server.port == 8123
    env = {"JARVIS__SERVER__PORT": "9001", "JARVIS__ROUTING__MODE": "local_only", "JARVIS__VOICE__ENABLED": "true"}
    cfg2 = load_config(paths, environ=env)
    assert cfg2.server.port == 9001
    assert cfg2.routing.mode == "local_only"
    assert cfg2.voice.enabled is True


def test_config_contains_no_secret_values() -> None:
    dumped = json.dumps(JarvisConfig().model_dump())
    assert "api_key\":" not in dumped.replace("api_key_secret", "")


# ---------------------------------------------------------------------- secrets
def test_secret_store_file_fallback_and_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SecretStore(tmp_path / "secrets.json", use_keyring=False)
    assert not store.secure
    assert store.get("demo") is None
    store.set("demo", "super-secret-value-123")
    assert store.get("demo") == "super-secret-value-123"
    # Registered values are redacted from any text.
    assert "super-secret-value-123" not in redactor.redact_text("token is super-secret-value-123")
    monkeypatch.setenv("JARVIS_SECRET_DEMO", "from-env-value-999")
    assert store.get("demo") == "from-env-value-999"
    monkeypatch.delenv("JARVIS_SECRET_DEMO")
    assert store.delete("demo")
    assert store.get("demo") is None


def test_redaction_patterns() -> None:
    text = "key sk-ant-abcdefghijklmnop1234 and Bearer abcdefghijklmnopqrstuv and password=hunter2hunter2"
    out = redactor.redact_text(text)
    assert "sk-ant-abcdefghijklmnop1234" not in out
    assert "abcdefghijklmnopqrstuv" not in out
    assert "hunter2hunter2" not in out
    assert redact({"api_key": "x123456", "nested": {"token": "abcdef"}, "ok": "fine"}) == {
        "api_key": REDACTED, "nested": {"token": REDACTED}, "ok": "fine"}


# ---------------------------------------------------------------------- storage
def test_database_migrates_and_is_idempotent(tmp_path: Path) -> None:
    db = Database(tmp_path / "j.db")
    assert db.schema_version == len(MIGRATIONS)
    db.close()
    db = Database(tmp_path / "j.db")  # re-open: no re-migration errors
    assert db.schema_version == len(MIGRATIONS)
    assert db.integrity_check() == "ok"
    db.kv_set("a", {"b": 1})
    assert db.kv_get("a") == {"b": 1}


def test_transaction_rolls_back(tmp_path: Path) -> None:
    db = Database(tmp_path / "j.db")
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute("INSERT INTO kv(key, value_json, updated_at) VALUES('x','1',0)")
            raise RuntimeError("boom")
    assert db.kv_get("x") is None


# ----------------------------------------------------------------------- events
async def test_event_bus_publish_subscribe_persist_and_redact(tmp_path: Path) -> None:
    db = Database(tmp_path / "j.db")
    bus = EventBus(db)
    sub = bus.subscribe(["TOOL_*"])
    seen = []
    bus.on("MEMORY_*", lambda e: seen.append(e.type))
    await bus.publish(EventType.TOOL_STARTED, {"tool": "x", "api_key": "abc123456"})
    await bus.publish(EventType.MEMORY_CREATED, {"id": "m1"})
    await bus.publish(EventType.MODEL_RESPONSE_DELTA, {"text": "hi"})  # ephemeral
    evt = await sub.get(timeout=1)
    assert evt is not None and evt.type == "TOOL_STARTED"
    assert evt.data["api_key"] == REDACTED
    assert seen == ["MEMORY_CREATED"]
    persisted = [r["type"] for r in db.query("SELECT type FROM events ORDER BY seq")]
    assert persisted == ["TOOL_STARTED", "MEMORY_CREATED"]
    recent = bus.recent(10)
    assert [e["type"] for e in recent] == ["TOOL_STARTED", "MEMORY_CREATED"]
    sub.close()
    assert bus.subscriber_count == 0


async def test_event_bus_slow_subscriber_drops_oldest() -> None:
    bus = EventBus()
    sub = bus.subscribe(maxsize=3)
    for i in range(5):
        await bus.publish("X", {"i": i})
    got = [(await sub.get(0.1)).data["i"] for _ in range(3)]
    assert got == [2, 3, 4]
    assert sub.dropped == 2


async def test_event_bus_emit_from_thread() -> None:
    bus = EventBus()
    bus.bind_loop(asyncio.get_running_loop())
    sub = bus.subscribe()
    await asyncio.to_thread(bus.emit, "FROM_THREAD", {"ok": True})
    evt = await sub.get(timeout=2)
    assert evt is not None and evt.type == "FROM_THREAD"


def test_hlc_monotonic_and_observe() -> None:
    a, b = HybridClock("a"), HybridClock("b")
    t1 = a.now()
    t2 = a.now()
    assert t2 > t1
    future = HybridClock.format(int(time.time() * 1000) + 60_000, 3, "b")
    a.observe(future)
    assert a.now() > future


# ------------------------------------------------------------------------- auth
@pytest.fixture
def auth(tmp_path: Path) -> AuthService:
    db = Database(tmp_path / "j.db")
    return AuthService(db, EventBus(db))


def test_local_device_token_persisted_and_hashed(auth: AuthService, tmp_path: Path) -> None:
    token_file = tmp_path / "local.token"
    token = auth.ensure_local_device(token_file)
    assert token.startswith("jvd_")
    assert auth.ensure_local_device(token_file) == token  # stable across restarts
    principal = auth.authenticate(token)
    assert principal is not None and principal.role == "owner" and principal.max_risk == Risk.DANGEROUS
    rows = auth.db.query("SELECT token_hash FROM devices")
    assert all(token not in r["token_hash"] for r in rows)


def test_pairing_flow_single_use_and_expiry(auth: AuthService, tmp_path: Path) -> None:
    owner = auth.authenticate(auth.ensure_local_device(tmp_path / "t"))
    assert owner is not None
    code = auth.create_pairing_code(owner, kind="mobile", role="client", max_risk=Risk.SENSITIVE)["code"]
    device, token = auth.redeem_pairing_code(code.lower().replace("-", " "), "Phone", platform="android")
    assert device.kind == "mobile" and device.role == "client" and device.max_risk == Risk.SENSITIVE
    with pytest.raises(AuthError):
        auth.redeem_pairing_code(code, "Phone again")
    p = auth.authenticate(token)
    assert p is not None and not p.is_owner
    # Clients cannot mint pairing codes.
    with pytest.raises(AuthError):
        auth.create_pairing_code(p)
    # Expired codes are rejected.
    code2 = auth.create_pairing_code(owner)["code"]
    auth.db.execute("UPDATE pairing_codes SET expires_at=?", (time.time() - 1,))
    with pytest.raises(AuthError):
        auth.redeem_pairing_code(code2, "late")


def test_sessions_and_revocation(auth: AuthService, tmp_path: Path) -> None:
    owner = auth.authenticate(auth.ensure_local_device(tmp_path / "t"))
    device, token = auth.create_device("Browser", "web", role="client")
    session = auth.create_session(token)
    assert auth.authenticate(session["session_token"]) is not None
    # A session token cannot be used to mint further sessions.
    with pytest.raises(AuthError):
        auth.create_session(session["session_token"])
    assert auth.revoke_device(device.id, by=owner.device_id)
    assert auth.authenticate(token) is None
    assert auth.authenticate(session["session_token"]) is None
    # Expired sessions are rejected.
    d2, t2 = auth.create_device("B2", "web")
    s2 = auth.create_session(t2)
    auth.db.execute("UPDATE sessions SET expires_at=? WHERE id=?", (time.time() - 1, s2["session_id"]))
    assert auth.authenticate(s2["session_token"]) is None
    assert auth.authenticate("garbage") is None
    assert auth.authenticate("jvd_forged") is None


# ------------------------------------------------------------------ permissions
@pytest.fixture
def gate(tmp_path: Path) -> PermissionGate:
    db = Database(tmp_path / "j.db")
    return PermissionGate(db, EventBus(db), PermissionsConfig(approval_timeout_s=2))


OWNER = Principal("dev_owner", "Owner", "cli", "owner", Risk.DANGEROUS)
CLIENT = Principal("dev_client", "Phone", "mobile", "client", Risk.SENSITIVE)
HARDWARE = Principal("dev_hw", "Helmet", "hardware", "hardware", Risk.LOW_RISK)


def req(risk: Risk, tool: str = "t", conv: str | None = "c1") -> ActionRequest:
    return ActionRequest(tool=tool, capability="test", risk=risk, summary="do it", effects="it happens",
                         args={"password": "hunter2"}, conversation_id=conv)


async def test_safe_and_low_risk_allowed_by_default(gate: PermissionGate) -> None:
    assert (await gate.check(req(Risk.SAFE), OWNER)).allowed
    assert (await gate.check(req(Risk.LOW_RISK), OWNER)).allowed


async def test_risk_above_device_limit_denied(gate: PermissionGate) -> None:
    result = await gate.check(req(Risk.SENSITIVE), HARDWARE)
    assert not result.allowed and result.decision == "policy_denied"


async def test_sensitive_requires_and_receives_approval(gate: PermissionGate) -> None:
    sub = gate.events.subscribe([EventType.PERMISSION_REQUESTED])

    async def approve() -> None:
        evt = await sub.get(timeout=2)
        assert evt is not None
        assert evt.data["args"]["password"] == REDACTED  # args are redacted in the event
        await gate.decide(evt.data["approval_id"], True, OWNER)

    task = asyncio.create_task(approve())
    result = await gate.check(req(Risk.SENSITIVE), OWNER)
    await task
    assert result.allowed and result.decision == "approved"


async def test_denial_and_timeout(gate: PermissionGate) -> None:
    sub = gate.events.subscribe([EventType.PERMISSION_REQUESTED])

    async def deny() -> None:
        evt = await sub.get(timeout=2)
        await gate.decide(evt.data["approval_id"], False, OWNER)

    task = asyncio.create_task(deny())
    assert (await gate.check(req(Risk.SENSITIVE), OWNER)).decision == "denied"
    await task
    timed = await gate.check(req(Risk.SENSITIVE), OWNER, timeout=0.2)
    assert not timed.allowed and timed.decision == "expired"


async def test_dangerous_never_gets_standing_grant(gate: PermissionGate) -> None:
    gate.config.tool_overrides["nuke"] = "allow"  # an allow override must not bypass DANGEROUS
    sub = gate.events.subscribe([EventType.PERMISSION_REQUESTED])

    async def approve_global() -> dict:
        evt = await sub.get(timeout=2)
        return await gate.decide(evt.data["approval_id"], True, OWNER, scope="global")

    task = asyncio.create_task(approve_global())
    assert (await gate.check(req(Risk.DANGEROUS, "nuke"), OWNER)).allowed
    decision = await task
    assert decision["scope"] == "once"
    assert gate.grants() == []
    # Next call must ask again (times out -> denied).
    assert not (await gate.check(req(Risk.DANGEROUS, "nuke"), OWNER, timeout=0.2)).allowed


async def test_conversation_grant_applies_only_to_that_conversation(gate: PermissionGate) -> None:
    sub = gate.events.subscribe([EventType.PERMISSION_REQUESTED])

    async def approve() -> None:
        evt = await sub.get(timeout=2)
        await gate.decide(evt.data["approval_id"], True, OWNER, scope="conversation")

    task = asyncio.create_task(approve())
    assert (await gate.check(req(Risk.SENSITIVE, "send", "c1"), OWNER)).allowed
    await task
    assert (await gate.check(req(Risk.SENSITIVE, "send", "c1"), OWNER)).decision == "allow"
    assert not (await gate.check(req(Risk.SENSITIVE, "send", "c2"), OWNER, timeout=0.2)).allowed


async def test_who_can_approve(gate: PermissionGate) -> None:
    sub = gate.events.subscribe([EventType.PERMISSION_REQUESTED])
    check = asyncio.create_task(gate.check(req(Risk.DANGEROUS), OWNER, timeout=2))
    evt = await sub.get(timeout=2)
    aid = evt.data["approval_id"]
    with pytest.raises(PermissionDenied):
        await gate.decide(aid, True, HARDWARE)  # hardware can never approve
    with pytest.raises(PermissionDenied):
        await gate.decide(aid, True, CLIENT)  # client limited to SENSITIVE
    await gate.decide(aid, False, CLIENT)  # but a client may deny
    assert (await check).decision == "denied"
    with pytest.raises(ValueError):
        await gate.decide(aid, True, OWNER)  # already decided


async def test_deny_override_blocks_tool(gate: PermissionGate) -> None:
    gate.config.tool_overrides["calc"] = "deny"
    assert (await gate.check(req(Risk.SAFE, "calc"), OWNER)).decision == "policy_denied"


async def test_pending_approvals_expire_on_restart(tmp_path: Path) -> None:
    db = Database(tmp_path / "j.db")
    g1 = PermissionGate(db, EventBus(db), PermissionsConfig())
    task = asyncio.create_task(g1.check(req(Risk.SENSITIVE), OWNER, timeout=5))
    await asyncio.sleep(0.05)
    assert len(g1.pending()) == 1
    task.cancel()
    PermissionGate(db, EventBus(db), PermissionsConfig())  # simulated restart
    assert db.query("SELECT status FROM approvals")[0]["status"] in ("expired", "cancelled")
