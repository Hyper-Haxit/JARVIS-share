"""API and security tests over real HTTP/WebSocket (FastAPI TestClient)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jarvis.api.server import InsecureBindError, check_bind, create_app
from jarvis.capabilities.base import Tool, ToolResult
from jarvis.core.permissions import Risk
from tests.test_intelligence import make_runtime


@pytest.fixture
def env(tmp_path: Path):
    rt, provider = make_runtime(tmp_path)
    app = create_app(rt)
    with TestClient(app) as client:
        owner = {"Authorization": f"Bearer {rt.local_token}"}
        yield rt, provider, client, owner


def sse(resp_text: str) -> list[tuple[str, dict[str, Any]]]:
    out, event = [], "message"
    for line in resp_text.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            out.append((event, json.loads(line[5:])))
    return out


def test_health_is_public_everything_else_requires_auth(env) -> None:
    rt, _, client, owner = env
    assert client.get("/api/v1/health").json()["status"] == "ok"
    for path in ("/api/v1/system/status", "/api/v1/memory", "/api/v1/devices", "/api/v1/tasks", "/api/v1/conversations",
                 "/api/v1/permissions/approvals", "/api/v1/voice/status"):
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers={"Authorization": "Bearer jvd_forged_token_value_123"}).status_code == 401, path
        assert client.get(path, headers=owner).status_code == 200, path


def test_security_headers(env) -> None:
    _, _, client, owner = env
    r = client.get("/api/v1/health")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store"
    assert client.get("/").headers.get("content-security-policy")


def test_pairing_session_and_revocation_flow(env) -> None:
    rt, _, client, owner = env
    code = client.post("/api/v1/auth/pairing-codes", json={"kind": "mobile", "role": "client", "max_risk": "LOW_RISK"}, headers=owner).json()
    assert "#pair=" in code["url"]
    paired = client.post("/api/v1/auth/pair", json={"code": code["code"], "name": "Test phone", "kind": "mobile"}).json()
    phone = {"Authorization": f"Bearer {paired['session_token']}"}
    me = client.get("/api/v1/auth/whoami", headers=phone).json()
    assert me["principal"]["role"] == "client" and me["principal"]["max_risk"] == "LOW_RISK"
    # A client device cannot mint codes, change policy, or revoke devices.
    assert client.post("/api/v1/auth/pairing-codes", json={}, headers=phone).status_code == 403
    assert client.put("/api/v1/permissions/policy", json={"defaults": {"SENSITIVE": "allow"}}, headers=phone).status_code == 403
    assert client.post(f"/api/v1/devices/{me['device']['id']}/revoke", headers=phone).status_code == 403
    # Code is single use.
    assert client.post("/api/v1/auth/pair", json={"code": code["code"], "name": "again"}).status_code == 401
    # Owner revokes; the session stops working immediately.
    assert client.post(f"/api/v1/devices/{me['device']['id']}/revoke", headers=owner).status_code == 200
    assert client.get("/api/v1/auth/whoami", headers=phone).status_code == 401


def test_policy_cannot_auto_allow_dangerous(env) -> None:
    _, _, client, owner = env
    r = client.put("/api/v1/permissions/policy", json={"defaults": {"DANGEROUS": "allow"}}, headers=owner)
    assert r.status_code == 422


def test_chat_non_streaming_and_streaming(env) -> None:
    rt, provider, client, owner = env
    provider.push({"text": "Hello there."}, {"tool_calls": [{"name": "get_time"}]}, {"text": "It is now."})
    r = client.post("/api/v1/chat", json={"message": "hi", "stream": False}, headers=owner).json()
    assert r["text"] == "Hello there." and r["conversation_id"]
    resp = client.post("/api/v1/chat", json={"message": "time?", "conversation_id": r["conversation_id"]}, headers=owner)
    events = sse(resp.text)
    names = [e for e, _ in events]
    assert names[0] == "start" and "tool_start" in names and "tool_end" in names and names[-1] == "done"
    done = events[-1][1]
    assert done["text"] == "It is now." and done["actions"][0]["tool"] == "get_time"
    conv = client.get(f"/api/v1/conversations/{r['conversation_id']}", headers=owner).json()
    assert [m["role"] for m in conv["messages"]].count("user") == 2


def test_chat_approval_round_trip_over_api(env) -> None:
    rt, provider, client, owner = env
    ran = []

    async def handler(args, ctx):
        ran.append(args)
        return ToolResult(ok=True, content="sent")

    rt.tools.register(Tool("send_note", "send", {"type": "object", "properties": {"to": {"type": "string"}}, "required": ["to"]},
                           handler, Risk.SENSITIVE, "messaging.send", "core"))
    provider.push({"tool_calls": [{"name": "send_note", "arguments": {"to": "bob"}}]}, {"text": "Sent it."})

    def approve_when_asked() -> None:
        for _ in range(100):
            pending = client.get("/api/v1/permissions/approvals", headers=owner).json()
            if pending:
                client.post(f"/api/v1/permissions/approvals/{pending[0]['id']}", json={"approve": True}, headers=owner)
                return
            time.sleep(0.05)

    t = threading.Thread(target=approve_when_asked)
    t.start()
    events = sse(client.post("/api/v1/chat", json={"message": "send bob a note"}, headers=owner).text)
    t.join()
    names = [e for e, _ in events]
    assert "approval" in names and "approval_decided" in names
    assert ran == [{"to": "bob"}] and events[-1][1]["text"] == "Sent it."


def test_manual_tool_invocation_is_still_gated(env) -> None:
    rt, _, client, owner = env
    rt.config.permissions.approval_timeout_s = 0.3

    async def handler(args, ctx):
        raise AssertionError("must not run without approval")

    rt.tools.register(Tool("wipe", "x", {"type": "object", "properties": {}}, handler, Risk.DANGEROUS, "fs.delete", "core"))
    r = client.post("/api/v1/tools/wipe/invoke", json={"args": {}}, headers=owner).json()
    assert r["result"]["ok"] is False and r["record"]["decision"] == "expired"


def test_tasks_api(env) -> None:
    rt, provider, client, owner = env
    provider.push(*[{"text": json.dumps({"achieved": True, "issues": [], "summary": "ok"})} if i % 2 else {"text": "step done"} for i in range(6)])
    t = client.post("/api/v1/tasks", json={"goal": "two things", "steps": [
        {"id": "a", "title": "A", "instruction": "do a", "depends_on": []}]}, headers=owner)
    assert t.status_code == 200, t.text
    assert client.post("/api/v1/tasks", json={"goal": "x", "steps": [{"id": "a", "instruction": "x", "depends_on": ["a"]}]},
                       headers=owner).status_code == 422
    assert client.get(f"/api/v1/tasks/{t.json()['id']}", headers=owner).json()["goal"] == "two things"


def test_memory_api_and_forget(env) -> None:
    rt, _, client, owner = env
    m = client.post("/api/v1/memory", json={"content": "The spare key is under the blue pot"}, headers=owner).json()
    assert client.get("/api/v1/memory?q=spare key", headers=owner).json()["items"][0]["id"] == m["id"]
    assert client.post("/api/v1/memory/forget", json={"ids": [m["id"]]}, headers=owner).json()["deleted"] == [m["id"]]
    assert client.get("/api/v1/memory?q=spare key", headers=owner).json()["items"] == []
    exported = client.get("/api/v1/memory/export", headers=owner).json()
    assert "spare key" not in json.dumps(exported)
    # The content is also scrubbed from the sync change log.
    assert all("spare key" not in (r["data_json"] or "") for r in rt.db.query("SELECT data_json FROM changes"))


def test_malformed_requests_are_rejected(env) -> None:
    _, _, client, owner = env
    assert client.post("/api/v1/chat", json={"message": ""}, headers=owner).status_code == 422
    assert client.post("/api/v1/chat", json={"message": "x", "images": [{"data_b64": "%%%"}]}, headers=owner).status_code == 422
    assert client.post("/api/v1/chat", content=b"{not json", headers={**owner, "Content-Type": "application/json"}).status_code == 422
    assert client.post("/api/v1/auth/pair", json={"code": "x"}).status_code == 422


def test_auth_rate_limit(env) -> None:
    rt, _, client, _ = env
    codes = [client.post("/api/v1/auth/pair", json={"code": "AAAAA-BBBBB", "name": "x"}).status_code for _ in range(rt.config.server.auth_rate_limit_per_minute + 3)]
    assert 429 in codes


def test_websocket_requires_auth_and_streams_events(env) -> None:
    rt, provider, client, owner = env
    with client.websocket_connect("/api/v1/events") as ws:
        ws.send_json({"token": "jvd_wrong"})
        assert ws.receive_json()["type"] == "error"
    with client.websocket_connect("/api/v1/events") as ws:
        ws.send_json({"token": rt.local_token, "patterns": ["TOOL_*"]})
        assert ws.receive_json()["type"] == "ready"
        client.post("/api/v1/tools/get_time/invoke", json={"args": {}}, headers=owner)
        msg = ws.receive_json()
        assert msg["type"] == "event" and msg["event"]["type"] == "TOOL_STARTED"


def test_refuses_lan_bind_without_tls(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    check_bind(rt, "127.0.0.1")
    with pytest.raises(InsecureBindError):
        check_bind(rt, "0.0.0.0")
    rt.config.server.tls_cert, rt.config.server.tls_key = "c.pem", "k.pem"
    check_bind(rt, "0.0.0.0")


def test_secrets_never_reach_responses_or_logs(env, tmp_path: Path) -> None:
    rt, _, client, owner = env
    rt.secrets.set("test_api_key", "sk-ant-THISISASECRETVALUE123456")
    assert rt.secrets.get("test_api_key")
    status = client.get("/api/v1/system/status", headers=owner).text
    assert "THISISASECRETVALUE" not in status
    from jarvis.core.observability import get_logger
    get_logger("test").warning("leaking sk-ant-THISISASECRETVALUE123456 by mistake")
    log_file = rt.paths.logs / "jarvis.log"
    if log_file.exists():
        assert "THISISASECRETVALUE" not in log_file.read_text("utf-8")
    events = json.dumps(client.get("/api/v1/events/recent?limit=500", headers=owner).json())
    assert rt.local_token not in events
