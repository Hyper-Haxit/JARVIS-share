"""Regression tests for CLI authentication against a real running core.

Bug: `jarvis voice start` returned a bare "HTTP 401: authentication required"
when the terminal had no credential (the core had been launched by a packaged
desktop app, so its credential lived in a virtualized AppData folder the
terminal could not see). These tests drive the real CLI entry point against a
real uvicorn server.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from jarvis.api.server import create_app
from jarvis.clients.cli.credentials import PairingStore
from jarvis.clients.cli.main import main
from jarvis.config.paths import JarvisPaths
from tests.test_intelligence import make_runtime

pytest.importorskip("faster_whisper")


class PushSource:
    def start(self, on_frame: Any) -> None:
        self.cb = on_frame

    def stop(self) -> None:
        self.cb = None


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class LiveCore:
    """A real JARVIS core served over HTTP in a background thread."""

    def __init__(self, base: Path, port: int, runtime: Any = None) -> None:
        from jarvis.voice.service import VoiceService

        self.rt = runtime if runtime is not None else make_runtime(base)[0]
        self.rt.extensions["voice"] = VoiceService(self.rt, source_factory=PushSource, dry_run_audio=True)
        self.url = f"http://127.0.0.1:{port}"
        self.server = uvicorn.Server(uvicorn.Config(create_app(self.rt), host="127.0.0.1", port=port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                break
            time.sleep(0.05)
        assert self.server.started, "server did not start"

    @property
    def owner(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.rt.local_token}"}

    def pairing_code(self, role: str = "owner") -> str:
        r = httpx.post(f"{self.url}/api/v1/auth/pairing-codes", headers=self.owner,
                       json={"kind": "cli", "role": role, "max_risk": "DANGEROUS" if role == "owner" else "SENSITIVE"})
        r.raise_for_status()
        return r.json()["code"]

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=15)


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The CLI runs with its own home folder and no inherited credentials."""
    cli_home = tmp_path / "cli-home"
    monkeypatch.setenv("JARVIS_HOME", str(cli_home))
    monkeypatch.setenv("JARVIS_SECRETS_BACKEND", "file")
    monkeypatch.delenv("JARVIS_TOKEN", raising=False)
    return cli_home


@pytest.fixture
def core(tmp_path: Path, cli_env: Path, monkeypatch: pytest.MonkeyPatch):
    port = _free_port()
    c = LiveCore(tmp_path / "server", port)
    monkeypatch.setenv("JARVIS_URL", c.url)
    yield c
    c.stop()


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out + out.err


# ------------------------------------------------------------------ the bug
def test_unauthenticated_voice_start_gives_clear_guidance(core: LiveCore, capsys: pytest.CaptureFixture[str],
                                                          caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger="jarvis.api.auth")
    rc, out = run_cli(capsys, "voice", "start")
    assert rc == 5
    assert "HTTP 401" not in out
    assert "isn't connected to JARVIS" in out and "no device credential was found" in out
    assert "jarvis auth login" in out and "jarvis devices pair --kind cli --role owner" in out and "jarvis start" in out
    assert core.rt.extensions["voice"].pipeline is None, "voice must not start without authentication"
    assert "refused POST /api/v1/voice/start" in caplog.text and "no credential sent" in caplog.text
    assert core.rt.local_token not in caplog.text


def test_paired_cli_voice_start_reaches_voice_subsystem(core: LiveCore, cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out = run_cli(capsys, "auth", "login", "--code", core.pairing_code())
    assert rc == 0 and "Paired" in out
    paired = PairingStore(JarvisPaths(cli_env)).get(core.url)
    assert paired is not None
    token, info = paired
    assert token not in out and core.rt.local_token not in out

    rc, out = run_cli(capsys, "voice", "start")
    assert rc == 0, out
    assert "Listening for" in out
    pipeline = core.rt.extensions["voice"].pipeline
    assert pipeline is not None and pipeline.running and pipeline.state == "standby"

    rc, out = run_cli(capsys, "auth", "status")
    assert rc == 0 and info["device_name"] in out and "connected" in out and token not in out
    rc, _ = run_cli(capsys, "voice", "stop")
    assert rc == 0 and not pipeline.running


def test_revoked_terminal_is_refused_with_guidance(core: LiveCore, cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(capsys, "auth", "login", "--code", core.pairing_code())
    _, info = PairingStore(JarvisPaths(cli_env)).get(core.url)
    r = httpx.post(f"{core.url}/api/v1/devices/{info['device_id']}/revoke", headers=core.owner)
    assert r.status_code == 200
    rc, out = run_cli(capsys, "voice", "status")
    assert rc == 5 and "did not accept" in out and "revoked" in out and "jarvis auth login" in out


def test_logout_revokes_on_the_server_and_forgets_locally(core: LiveCore, cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(capsys, "auth", "login", "--code", core.pairing_code())
    _, info = PairingStore(JarvisPaths(cli_env)).get(core.url)
    rc, out = run_cli(capsys, "auth", "logout")
    assert rc == 0 and "Removed" in out
    assert core.rt.auth.get_device(info["device_id"]).revoked
    assert PairingStore(JarvisPaths(cli_env)).get(core.url) is None
    rc, _ = run_cli(capsys, "voice", "status")
    assert rc == 5


def test_client_role_cannot_start_voice(core: LiveCore, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(capsys, "auth", "login", "--code", core.pairing_code(role="client"))
    rc, out = run_cli(capsys, "voice", "start")
    assert rc == 6 and "owner" in out
    assert core.rt.extensions["voice"].pipeline is None


def test_bad_pairing_code_is_rejected_without_storing_anything(core: LiveCore, cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out = run_cli(capsys, "auth", "login", "--code", "AAAAA-BBBBB")
    assert rc == 5 and "not accepted" in out
    assert PairingStore(JarvisPaths(cli_env)).get(core.url) is None


def test_pairing_survives_core_restart(tmp_path: Path, cli_env: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    port = _free_port()
    first = LiveCore(tmp_path / "server", port)
    monkeypatch.setenv("JARVIS_URL", first.url)
    try:
        assert run_cli(capsys, "auth", "login", "--code", first.pairing_code())[0] == 0
    finally:
        first.stop()
    second = LiveCore(tmp_path / "server", port)  # same home, fresh process state
    try:
        rc, out = run_cli(capsys, "voice", "status")
        assert rc == 0, out
    finally:
        second.stop()


def test_host_credential_works_without_pairing(core: LiveCore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("JARVIS_HOME", str(core.rt.paths.home))  # same computer, same home as the core
    rc, out = run_cli(capsys, "auth", "status")
    assert rc == 0 and "this computer's own credential" in out and core.rt.local_token not in out
    r = httpx.post(f"{core.url}/api/v1/auth/device/forget", headers=core.owner)
    assert r.status_code == 409  # the host can't lock itself out


def test_credential_from_another_installation_is_explained(core: LiveCore, cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli_env.mkdir(parents=True, exist_ok=True)
    (cli_env / "local_device.token").write_text("jvd_" + "x" * 43, "utf-8")
    rc, out = run_cli(capsys, "voice", "status")
    assert rc == 5 and "did not accept this computer's own credential" in out and "jarvis start" in out


@pytest.mark.skipif(sys.platform != "win32", reason="the legacy AppData location only existed on Windows")
def test_legacy_appdata_home_is_migrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from jarvis.config.paths import default_home, migrate_legacy_home
    from jarvis.runtime import JarvisRuntime

    legacy = tmp_path / "LocalAppData" / "JARVIS"
    rt = JarvisRuntime(paths=JarvisPaths(legacy), configure_logging=False, load_plugins=False)
    asyncio.run(rt.start())
    token, device_count = rt.local_token, len(rt.auth.list_devices())
    asyncio.run(rt.stop())
    monkeypatch.delenv("JARVIS_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    home = default_home()
    assert home == tmp_path / "profile" / ".jarvis"
    assert migrate_legacy_home(home) == legacy
    assert (home / "local_device.token").read_text("utf-8").strip() == token
    moved = JarvisRuntime(paths=JarvisPaths(home), configure_logging=False, load_plugins=False)
    assert len(moved.auth.list_devices()) == device_count and moved.auth.authenticate(token) is not None
    moved.db.close()
    assert "no longer used" in (legacy / "MOVED-TO.txt").read_text("utf-8")
    assert migrate_legacy_home(home) is None  # only once
