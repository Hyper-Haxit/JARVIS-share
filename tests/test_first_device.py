"""First-device setup, pairing and the device lifecycle, against a real running core.

Setting up the computer JARVIS runs on must not need a second device or a
pairing code: a browser on this computer asks, the person at this computer
approves (in a confirmation window or with `jarvis devices approve`), and the
browser becomes an ordinary owner device. Everything else (other computers,
other websites, revoked or logged-out devices) still goes through the existing
authentication.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.test_cli_auth import LiveCore, _free_port, run_cli


class FakePrompt:
    """Stands in for the confirmation window: answers as told, records what it was asked."""

    def __init__(self, answer: bool | None = True, delay: float = 0.0) -> None:
        self.answer, self.delay = answer, delay
        self.asked: list[str] = []
        self.dismissed: list[str] = []

    def available(self) -> bool:
        return True

    def ask(self, key: str, title: str, text: str, timeout_s: float) -> bool | None:
        self.asked.append(text)
        time.sleep(self.delay)
        return self.answer

    def dismiss(self, key: str) -> None:
        self.dismissed.append(key)


@pytest.fixture
def core(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JARVIS_SECRETS_BACKEND", "file")
    monkeypatch.delenv("JARVIS_TOKEN", raising=False)
    c = LiveCore(tmp_path / "server", _free_port())
    monkeypatch.setenv("JARVIS_URL", c.url)
    monkeypatch.setenv("JARVIS_HOME", str(c.rt.paths.home))  # the CLI runs on this computer, with its credential
    yield c
    c.stop()


def browser_headers(core: LiveCore) -> dict[str, str]:
    """What a browser showing JARVIS's own page sends."""
    return {"Origin": core.url}


def request_setup(core: LiveCore, name: str = "Chrome on Windows", **headers: str) -> httpx.Response:
    return httpx.post(f"{core.url}/api/v1/auth/setup/requests", headers={**browser_headers(core), **headers},
                      json={"name": name, "kind": "web", "platform": "Win32", "capabilities": ["display"]})


def claim(core: LiveCore, req: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
    r = httpx.post(f"{core.url}/api/v1/auth/setup/requests/{req['id']}/claim", headers=browser_headers(core),
                   json={"secret": req["secret"]}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------- first run
def test_fresh_install_offers_setup_on_this_computer(core: LiveCore) -> None:
    devices = core.rt.auth.list_devices()
    assert len(devices) == 1 and devices[0].metadata.get("local"), "a fresh install has only this computer's own credential"
    st = httpx.get(f"{core.url}/api/v1/auth/setup", headers=browser_headers(core)).json()
    assert st == {"this_computer": True, "first_run": True, "approval": "terminal"}


def test_first_device_enrollment_end_to_end(core: LiveCore, capsys: pytest.CaptureFixture[str]) -> None:
    r = request_setup(core)
    assert r.status_code == 200
    req = r.json()
    assert len(req["code"]) == 4 and req["approval"] == "terminal"
    pending = httpx.get(f"{core.url}/api/v1/auth/setup/requests", headers=core.owner).json()
    assert [p["id"] for p in pending] == [req["id"]] and "secret" not in pending[0]
    # The person at this computer approves in a terminal.
    rc, out = run_cli(capsys, "devices", "approve", "--yes")
    assert rc == 0 and "Approved" in out
    got = claim(core, req)
    assert got["status"] == "approved" and got["device_token"].startswith("jvd_") and got["session_token"].startswith("jvs_")
    who = httpx.get(f"{core.url}/api/v1/auth/whoami", headers=auth(got["session_token"])).json()
    device = core.rt.auth.get_device(who["principal"]["device_id"])
    assert device.role == "owner" and device.kind == "web" and device.metadata["enrolled"] == "local approval"
    assert httpx.get(f"{core.url}/api/v1/auth/setup", headers=browser_headers(core)).json()["first_run"] is False
    # The credential is handed over once; the request can't be replayed.
    assert claim(core, req)["status"] == "claimed" and "device_token" not in claim(core, req)
    actions = [r["action"] for r in core.rt.db.query("SELECT action FROM audit_log WHERE action LIKE 'setup.%' ORDER BY id")]
    assert actions == ["setup.request", "setup.approve", "setup.claim"]


def test_approval_in_the_confirmation_window(core: LiveCore, capsys: pytest.CaptureFixture[str]) -> None:
    core.rt.enrollment.prompt = FakePrompt(answer=True)
    req = request_setup(core).json()
    assert req["approval"] == "window"
    assert claim(core, req)["status"] == "approved"
    assert "the code " + req["code"] in core.rt.enrollment.prompt.asked[0], "the window shows the same code as the browser"
    core.rt.enrollment.prompt = FakePrompt(answer=False)
    assert claim(core, request_setup(core, "Edge on Windows").json())["status"] == "denied"
    # "Decide later": the request keeps waiting and the terminal can still approve it.
    core.rt.enrollment.prompt = FakePrompt(answer=None)
    req = request_setup(core, "Firefox on Windows").json()
    time.sleep(0.3)
    assert run_cli(capsys, "devices", "approve", "--yes")[0] == 0
    assert claim(core, req)["status"] == "approved"
    assert req["id"] in core.rt.enrollment.prompt.dismissed, "approving in the terminal closes the window"


def test_the_browser_hears_the_decision_as_it_happens(core: LiveCore) -> None:
    req = request_setup(core).json()
    result: dict[str, Any] = {}
    t = threading.Thread(target=lambda: result.update(claim(core, req)))
    started = time.monotonic()
    t.start()
    time.sleep(1.0)
    assert httpx.post(f"{core.url}/api/v1/auth/setup/requests/{req['id']}/approve", headers=core.owner).status_code == 200
    t.join(10)
    assert result["status"] == "approved" and time.monotonic() - started < 3, "no polling delay"


# ------------------------------------------------------------- not from here
def test_other_computers_and_websites_cannot_set_up(core: LiveCore) -> None:
    port = core.url.rsplit(":", 1)[1]
    for headers, why in [({"Host": f"192.168.1.20:{port}"}, "addressed by a network name"),
                         ({"Host": f"jarvis.evil.example:{port}"}, "DNS rebinding"),
                         ({"Origin": "https://evil.example"}, "another website in the same browser")]:
        r = request_setup(core, **headers)
        assert r.status_code == 403, why
        assert "pair this device with a code" in r.json()["detail"]
    st = httpx.get(f"{core.url}/api/v1/auth/setup", headers={"Host": f"192.168.1.20:{port}"}).json()
    assert st == {"this_computer": False}, "nothing about the installation is revealed to other computers"
    from fastapi.testclient import TestClient

    from jarvis.api.server import create_app

    with TestClient(create_app(core.rt, manage_lifecycle=False)) as remote:  # a client that isn't on this computer
        assert remote.post("/api/v1/auth/setup/requests", json={"name": "x"}).status_code == 403
        assert remote.get("/api/v1/auth/setup").json() == {"this_computer": False}
    assert core.rt.enrollment.pending() == []


def test_unknown_device_cannot_use_jarvis(core: LiveCore) -> None:
    assert httpx.get(f"{core.url}/api/v1/auth/whoami").status_code == 401
    assert httpx.get(f"{core.url}/api/v1/auth/whoami", headers=auth("jvd_" + "x" * 43)).status_code == 401
    assert httpx.post(f"{core.url}/api/v1/auth/session", json={"device_token": "jvd_" + "y" * 43}).status_code == 401
    mine = request_setup(core).json()
    r = httpx.post(f"{core.url}/api/v1/auth/setup/requests/{mine['id']}/claim", headers=browser_headers(core),
                   json={"secret": "a-guessed-secret-value-1234"})
    assert r.status_code == 403, "a request can only be claimed by the browser that made it"
    # Only an owner can approve: a client device can't.
    code = core.pairing_code(role="client")
    paired = httpx.post(f"{core.url}/api/v1/auth/pair", json={"code": code, "name": "phone", "kind": "mobile"}).json()
    r = httpx.post(f"{core.url}/api/v1/auth/setup/requests/{mine['id']}/approve", headers=auth(paired["session_token"]))
    assert r.status_code == 403
    r = httpx.post(f"{core.url}/api/v1/auth/setup/requests/{mine['id']}/approve")
    assert r.status_code == 401


def test_another_device_pairs_with_a_code_from_a_set_up_device(core: LiveCore) -> None:
    req = request_setup(core).json()
    httpx.post(f"{core.url}/api/v1/auth/setup/requests/{req['id']}/approve", headers=core.owner)
    browser = claim(core, req)
    # Device A (the browser just set up) pairs device B with a one-time code.
    code = httpx.post(f"{core.url}/api/v1/auth/pairing-codes", headers=auth(browser["session_token"]),
                      json={"kind": "mobile", "role": "client", "max_risk": "SENSITIVE"}).json()
    b = httpx.post(f"{core.url}/api/v1/auth/pair", json={"code": code["code"], "name": "Phone", "kind": "mobile"})
    assert b.status_code == 200
    who = httpx.get(f"{core.url}/api/v1/auth/whoami", headers=auth(b.json()["session_token"])).json()
    assert who["principal"]["role"] == "client"
    again = httpx.post(f"{core.url}/api/v1/auth/pair", json={"code": code["code"], "name": "Phone 2", "kind": "mobile"})
    assert again.status_code == 401, "a pairing code works once"


# ------------------------------------------------------------- lifecycle
def enrolled(core: LiveCore) -> dict[str, Any]:
    req = request_setup(core).json()
    httpx.post(f"{core.url}/api/v1/auth/setup/requests/{req['id']}/approve", headers=core.owner)
    return claim(core, req)


def test_revoked_and_logged_out_devices_are_refused(core: LiveCore) -> None:
    a, b = enrolled(core), enrolled(core)
    httpx.post(f"{core.url}/api/v1/devices/{a['device_id']}/revoke", headers=core.owner).raise_for_status()
    assert httpx.get(f"{core.url}/api/v1/auth/whoami", headers=auth(a["session_token"])).status_code == 401
    assert httpx.post(f"{core.url}/api/v1/auth/session", json={"device_token": a["device_token"]}).status_code == 401
    # Logging out (forgetting this device) works the same way.
    assert httpx.post(f"{core.url}/api/v1/auth/device/forget", headers=auth(b["session_token"])).status_code == 200
    assert httpx.get(f"{core.url}/api/v1/auth/whoami", headers=auth(b["device_token"])).status_code == 401
    # A removed browser on this computer can be set up again, through the same approval.
    st = httpx.get(f"{core.url}/api/v1/auth/setup", headers=browser_headers(core)).json()
    assert st["this_computer"] and st["first_run"] is False


def test_expired_session_is_renewed_from_the_device_credential(core: LiveCore) -> None:
    a = enrolled(core)
    core.rt.db.execute("UPDATE sessions SET expires_at=? WHERE device_id=?", (time.time() - 1, a["device_id"]))
    assert httpx.get(f"{core.url}/api/v1/auth/whoami", headers=auth(a["session_token"])).status_code == 401
    fresh = httpx.post(f"{core.url}/api/v1/auth/session", json={"device_token": a["device_token"]}).json()
    assert httpx.get(f"{core.url}/api/v1/auth/whoami", headers=auth(fresh["session_token"])).status_code == 200


def test_restart_keeps_the_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_SECRETS_BACKEND", "file")
    port = _free_port()
    first = LiveCore(tmp_path / "server", port)
    try:
        a = enrolled(first)
    finally:
        first.stop()
    second = LiveCore(tmp_path / "server", port)
    try:
        s = httpx.post(f"{second.url}/api/v1/auth/session", json={"device_token": a["device_token"]})
        assert s.status_code == 200, "a set-up browser stays set up across restarts"
        assert httpx.get(f"{second.url}/api/v1/auth/setup", headers={"Origin": second.url}).json()["first_run"] is False
    finally:
        second.stop()


def test_one_request_at_a_time_and_rate_limited(core: LiveCore) -> None:
    first = request_setup(core).json()
    second = request_setup(core).json()
    assert claim(core, first)["status"] == "cancelled", "a new request (e.g. after a reload) replaces the old one"
    assert [p["id"] for p in core.rt.enrollment.pending()] == [second["id"]]
    for _ in range(4):
        request_setup(core)
    assert request_setup(core).status_code == 429


def test_denied_and_expired_requests(core: LiveCore, capsys: pytest.CaptureFixture[str]) -> None:
    req = request_setup(core).json()
    assert run_cli(capsys, "devices", "deny")[0] == 0
    assert claim(core, req)["status"] == "denied"
    req = request_setup(core).json()
    core.rt.enrollment._requests[req["id"]].expires_at = time.time() - 1
    assert claim(core, req)["status"] == "expired"
    assert httpx.post(f"{core.url}/api/v1/auth/setup/requests/{req['id']}/approve", headers=core.owner).status_code in (404, 409)


# ------------------------------------------------------- one home, no leaks
def test_cli_and_browser_share_the_canonical_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    """With no override, the core and the CLI both use %USERPROFILE%/.jarvis (never the virtualized AppData)."""
    from jarvis.config.paths import JarvisPaths, default_home
    from jarvis.runtime import JarvisRuntime

    monkeypatch.delenv("JARVIS_HOME", raising=False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    monkeypatch.setenv("HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    monkeypatch.setenv("JARVIS_SECRETS_BACKEND", "file")
    home = default_home()
    if sys.platform == "win32":
        assert home == tmp_path / "profile" / ".jarvis"
    core = LiveCore(tmp_path, _free_port(), runtime=JarvisRuntime(paths=JarvisPaths(home), configure_logging=False, load_plugins=False))
    monkeypatch.setenv("JARVIS_URL", core.url)
    try:
        browser = enrolled(core)
        assert (home / "jarvis.db").exists() and (home / "local_device.token").exists()
        assert not list((tmp_path / "LocalAppData").rglob("*")), "nothing is written to AppData (the packaged-app split)"
        # The CLI on this computer finds the same identity and sees the browser.
        rc, out = run_cli(capsys, "auth", "status")
        assert rc == 0 and "this computer's own credential" in out and str(home) in out
        rc, out = run_cli(capsys, "devices", "list")
        assert rc == 0 and browser["device_id"] in out
    finally:
        core.stop()


@pytest.mark.skipif(sys.platform != "win32", reason="the AppData location only existed on Windows")
def test_setup_after_the_appdata_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from jarvis.config.paths import JarvisPaths, default_home, migrate_legacy_home
    from jarvis.runtime import JarvisRuntime

    legacy = tmp_path / "LocalAppData" / "JARVIS"
    old = JarvisRuntime(paths=JarvisPaths(legacy), configure_logging=False, load_plugins=False)
    asyncio.run(old.start())
    asyncio.run(old.stop())
    monkeypatch.delenv("JARVIS_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    home = default_home()
    assert migrate_legacy_home(home) == legacy
    rt = JarvisRuntime(paths=JarvisPaths(home), configure_logging=False, load_plugins=False)
    assert rt.enrollment.first_run(), "the migrated installation still has no browser: first-run setup is offered"
    req, secret = rt.enrollment.request("Chrome on Windows", "web", "Win32", [])
    rt.enrollment.approve(req.id, rt.auth.authenticate((home / "local_device.token").read_text("utf-8").strip()), "test")
    got = asyncio.run(rt.enrollment.claim(req.id, secret, wait_s=0))
    assert got["status"] == "approved"
    rt.db.close()
    import sqlite3

    rows = sqlite3.connect(home / "jarvis.db").execute("SELECT COUNT(*) FROM devices WHERE kind='web'").fetchone()[0]
    assert rows == 1, "the browser was enrolled in the canonical home"
    rows = sqlite3.connect(legacy / "jarvis.db").execute("SELECT COUNT(*) FROM devices WHERE kind='web'").fetchone()[0]
    assert rows == 0, "the old AppData copy is not used"


def test_no_credentials_in_responses_or_logs(core: LiveCore, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    req = request_setup(core).json()
    listing = httpx.get(f"{core.url}/api/v1/auth/setup/requests", headers=core.owner).text
    assert req["secret"] not in listing
    decided = httpx.post(f"{core.url}/api/v1/auth/setup/requests/{req['id']}/approve", headers=core.owner)
    assert "jvd_" not in decided.text and "jvs_" not in decided.text, "approving doesn't reveal the new credential"
    got = claim(core, req)
    secrets_ = [req["secret"], got["device_token"], got["session_token"], core.rt.local_token]
    log_text = caplog.text + "".join(r.getMessage() for r in caplog.records)
    for s in secrets_:
        assert s not in log_text
    rows = core.rt.db.query("SELECT details_json FROM audit_log")
    assert not any(s in (r["details_json"] or "") for r in rows for s in secrets_)
    events = core.rt.db.query("SELECT data_json FROM events")
    assert not any(s in (r["data_json"] or "") for r in events for s in secrets_)


# ------------------------------------------------------------- in a browser
def test_browser_setup_flow_end_to_end(core: LiveCore) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        try:
            ctx = b.new_context(bypass_csp=True)  # test predicates need eval
            page = ctx.new_page()
            page.goto(core.url)
            page.wait_for_selector("#pairing:not([hidden])")
            page.wait_for_function("document.getElementById('pair-title').textContent === 'Set up this device'")
            assert page.is_visible("#setup-start") and not page.is_visible("#pair-code"), "no code needed here"
            page.click("#setup-start")
            page.wait_for_selector("#setup-wait:not([hidden])")
            code = page.text_content("#setup-code")
            pending = httpx.get(f"{core.url}/api/v1/auth/setup/requests", headers=core.owner).json()
            assert [r["code"] for r in pending] == [code], "the browser shows the code the computer asks about"
            httpx.post(f"{core.url}/api/v1/auth/setup/requests/{pending[0]['id']}/approve", headers=core.owner).raise_for_status()
            page.wait_for_function("window.jarvis?.started === true", timeout=30000)
            assert page.is_hidden("#pairing")
            assert core.rt.auth.get_device(page.evaluate("JSON.parse(sessionStorage.getItem('jarvis.session')).device_id")).kind == "web"
            assert page.evaluate("sessionStorage.getItem('jarvis.setup')") is None, "nothing about the request is left behind"
            # Normal startup afterwards: recognised, straight in.
            page.reload()
            page.wait_for_function("window.jarvis?.started === true", timeout=30000)
            assert page.is_hidden("#pairing")
            # Revoked from elsewhere: the browser is told, and can be set up again.
            device_id = page.evaluate("JSON.parse(sessionStorage.getItem('jarvis.session')).device_id")
            httpx.post(f"{core.url}/api/v1/devices/{device_id}/revoke", headers=core.owner).raise_for_status()
            page.wait_for_selector("#pairing:not([hidden])", timeout=15000)
            page.wait_for_function("document.getElementById('pair-title').textContent === 'Set up this browser'")
            assert "removed" in page.text_content("#pair-note")
            assert "jvd_" not in page.content() and "jvs_" not in page.content(), "no credential is ever shown on the page"
        finally:
            b.close()
