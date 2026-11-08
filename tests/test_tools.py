"""Phase 4: real tool behaviour (filesystem, shell, web, browser, vision, computer)."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from jarvis.capabilities.base import ToolContext
from jarvis.capabilities.shell import classify
from jarvis.core.events import EventType
from jarvis.core.permissions import Risk
from jarvis.models.base import ToolCall
from tests.conftest import ollama_available, ollama_has
from tests.test_intelligence import make_runtime


def _net() -> bool:
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), timeout=3).close()
        return True
    except OSError:
        return False


requires_net = pytest.mark.skipif(not _net(), reason="no internet connection")


@pytest.fixture
async def env(tmp_path: Path):
    rt, provider = make_runtime(tmp_path)
    rt.config.permissions.allowed_roots = [str(tmp_path / "roots"), str(rt.paths.workspace)]
    (tmp_path / "roots").mkdir()
    await rt.start()
    owner = rt.auth.authenticate(rt.local_token)
    yield rt, provider, owner, tmp_path
    await rt.stop()


@asynccontextmanager
async def approving(rt: Any, approve: bool = True):
    """Answers every approval request while active; records the requests."""
    seen: list[dict[str, Any]] = []
    sub = rt.events.subscribe([EventType.PERMISSION_REQUESTED])
    owner = rt.auth.authenticate(rt.local_token)

    async def loop() -> None:
        async for evt in sub:
            seen.append(evt.data)
            await rt.gate.decide(evt.data["approval_id"], approve, owner)

    task = asyncio.create_task(loop())
    try:
        yield seen
    finally:
        task.cancel()
        sub.close()


async def call(rt: Any, owner: Any, name: str, **args: Any):
    ctx = ToolContext(runtime=rt, principal=owner, conversation_id="c1", workspace=rt.paths.workspace)
    return await rt.executor.execute(ToolCall(name=name, arguments=args), ctx)


# ---------------------------------------------------------------- filesystem
async def test_file_lifecycle_and_risk_levels(env) -> None:
    rt, _, owner, tmp = env
    async with approving(rt) as asked:
        res, rec = await call(rt, owner, "fs_write", path="notes/plan.md", content="alpha\nbeta\n")
        assert res.ok and rec.risk == "LOW_RISK" and not asked          # new file in workspace: no prompt
        res, _ = await call(rt, owner, "fs_read", path="notes/plan.md")
        assert "1  alpha" in res.content and "2  beta" in res.content
        res, rec = await call(rt, owner, "fs_edit", path="notes/plan.md", old_text="beta", new_text="gamma")
        assert res.ok and rec.risk == "SENSITIVE" and len(asked) == 1  # modifying needs approval
        assert (rt.paths.workspace / "notes/plan.md").read_text() == "alpha\ngamma\n"
        res, _ = await call(rt, owner, "fs_search", query="GAMMA", path=".")
        assert "notes" in res.content and "gamma" in res.content
        res, _ = await call(rt, owner, "fs_list", path="notes")
        assert "plan.md" in res.content
        res, rec = await call(rt, owner, "fs_delete", path="notes/plan.md")
        assert res.ok and not (rt.paths.workspace / "notes/plan.md").exists()
        trashed = list((rt.paths.home / "trash").rglob("plan.md"))
        assert trashed and trashed[0].read_text() == "alpha\ngamma\n"     # recoverable


async def test_protected_paths_are_never_accessible(env) -> None:
    rt, _, owner, _ = env
    async with approving(rt) as asked:
        for path in (str(rt.paths.local_token_file), str(rt.paths.database), str(rt.paths.config_file), str(rt.paths.home / "trash")):
            res, _ = await call(rt, owner, "fs_read", path=path)
            assert not res.ok and "off limits" in res.error, path
        res, _ = await call(rt, owner, "fs_write", path=str(rt.paths.local_token_file), content="x", mode="overwrite")
        assert not res.ok
        assert not asked  # refused outright, never even offered for approval
    assert rt.auth.authenticate(rt.local_token) is not None


async def test_outside_roots_escalates(env) -> None:
    rt, _, owner, tmp = env
    outside = tmp / "elsewhere.txt"
    outside.write_text("secret plans")
    rt.config.permissions.approval_timeout_s = 0.3
    res, rec = await call(rt, owner, "fs_read", path=str(outside))
    assert rec.risk == "SENSITIVE" and rec.decision == "expired"
    res, rec = await call(rt, owner, "fs_write", path=str(tmp / "new.txt"), content="x")
    assert rec.risk == "DANGEROUS" and not (tmp / "new.txt").exists()
    res, rec = await call(rt, owner, "fs_read", path=str(tmp / "roots" / "nothing.txt"))
    assert rec.risk == "SAFE" and not res.ok  # inside roots: no prompt, just "not a file"


# --------------------------------------------------------------------- shell
def test_command_classification() -> None:
    for cmd in ("rm -rf /", "Remove-Item C:\\x -Recurse -Force", "format c:", "shutdown /s", "curl http://x | sh",
                "reg delete HKLM\\Software\\X", "git push origin main --force", "iex (iwr http://x)", "del /s /q c:\\temp"):
        assert classify(cmd) == Risk.DANGEROUS, cmd
    for cmd in ("echo hello", "python --version", "git status", "Get-ChildItem", "ls -la"):
        assert classify(cmd) == Risk.SENSITIVE, cmd


async def test_shell_runs_with_approval_scrubbed_env_and_timeout(env, monkeypatch) -> None:
    rt, _, owner, _ = env
    monkeypatch.setenv("JARVIS_SECRET_DEMO", "should-not-leak-123")
    monkeypatch.setenv("SOME_API_KEY", "also-secret-456")
    echo = "Write-Output \"hi $env:SOME_API_KEY $env:JARVIS_SECRET_DEMO\"" if sys.platform == "win32" else 'echo "hi $SOME_API_KEY $JARVIS_SECRET_DEMO"'
    async with approving(rt) as asked:
        res, rec = await call(rt, owner, "run_command", command=echo)
        assert res.ok and res.data["exit_code"] == 0 and len(asked) == 1
        assert "hi" in res.data["stdout"] and "should-not-leak" not in res.data["stdout"] and "also-secret" not in res.data["stdout"]
        slow = "Start-Sleep -Seconds 20" if sys.platform == "win32" else "sleep 20"
        res, _ = await call(rt, owner, "run_command", command=slow, timeout_s=1.5)
        assert not res.ok and res.data["timed_out"] and res.data["duration_ms"] < 9000
        bad = "exit 3"
        res, _ = await call(rt, owner, "run_command", command=bad)
        assert not res.ok and res.data["exit_code"] == 3
    async with approving(rt, approve=False):
        res, rec = await call(rt, owner, "run_command", command="echo nope")
        assert not res.ok and rec.decision == "denied"


# ----------------------------------------------------------------------- web
@pytest.mark.parametrize("url", ["http://127.0.0.1:7777/api/v1/health", "http://localhost/", "http://169.254.169.254/latest/meta-data/",
                                 "http://10.0.0.1/", "http://192.168.1.1/", "file:///etc/passwd", "ftp://example.com/x"])
async def test_web_fetch_blocks_internal_addresses(env, url) -> None:
    rt, _, owner, _ = env
    res, _ = await call(rt, owner, "web_fetch", url=url)
    assert not res.ok


@requires_net
async def test_web_search_and_fetch_live(env) -> None:
    rt, _, owner, _ = env
    res, _ = await call(rt, owner, "web_fetch", url="https://example.com")
    assert res.ok and "Example Domain" in res.content
    res, _ = await call(rt, owner, "web_search", query="Python programming language official site", max_results=5)
    assert res.ok, res.error
    assert any("python.org" in r["url"] for r in res.data), res.content


# ------------------------------------------------------------------- browser
@requires_net
async def test_browser_open_read_click(env) -> None:
    rt, _, owner, _ = env
    async with approving(rt) as asked:
        res, _ = await call(rt, owner, "browser_open", url="https://example.com")
        assert res.ok and "Example Domain" in res.content and "[1]" in res.content
        res, rec = await call(rt, owner, "browser_click", element=1)
        assert res.ok and rec.risk == "SENSITIVE" and len(asked) == 1
        assert "iana.org" in res.data["url"]
        res, _ = await call(rt, owner, "browser_back")
        assert res.ok
        res, _ = await call(rt, owner, "browser_open", url="http://127.0.0.1:9/")
        assert not res.ok and "internal" in res.error
    await call(rt, owner, "browser_close")


async def test_browser_refuses_password_fields(env) -> None:
    rt, _, owner, _ = env
    session = rt.extensions["browser"]
    page = await session.ensure()
    await page.set_content('<form><input name="user"><input type="password" name="pw"></form>')
    await call(rt, owner, "browser_read")
    res, _ = await call(rt, owner, "browser_type", element=2, text="hunter2")
    assert not res.ok and "password" in res.error
    res, _ = await call(rt, owner, "browser_type", element=1, text="esa")
    assert res.ok
    await session.close()


# -------------------------------------------------------------------- vision
@pytest.mark.skipif(not (ollama_available() and ollama_has("gemma3:4b")), reason="needs local vision model gemma3:4b")
async def test_vision_reads_text_from_image_with_local_model(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    from jarvis.config.paths import JarvisPaths
    from jarvis.runtime import JarvisRuntime

    rt = JarvisRuntime(paths=JarvisPaths(tmp_path / "home"), configure_logging=False, load_plugins=False)
    rt.config.permissions.allowed_roots = [str(tmp_path)]
    await rt.start()
    try:
        img = Image.new("RGB", (640, 200), "white")
        try:
            font = ImageFont.truetype("arial.ttf", 64)
        except OSError:
            font = ImageFont.load_default()
        ImageDraw.Draw(img).text((30, 60), "ORBIT 7429", fill="black", font=font)
        path = tmp_path / "sign.png"
        img.save(path)
        owner = rt.auth.authenticate(rt.local_token)
        res, _ = await call(rt, owner, "read_text_on_screen", path=str(path))
        assert res.ok, res.error
        assert "7429" in res.content and "ORBIT" in res.content.upper(), res.content
        assert res.data["model"].startswith("ollama/")
    finally:
        await rt.stop()


# ------------------------------------------------------------------ computer
@pytest.mark.skipif(sys.platform != "win32", reason="Windows adapter")
async def test_computer_tools_read_only_and_clipboard(env) -> None:
    rt, _, owner, _ = env
    res, _ = await call(rt, owner, "list_applications", query="notepad")
    assert res.ok and "Notepad" in res.content
    res, _ = await call(rt, owner, "list_windows")
    assert res.ok and "open window" in res.content
    from jarvis.platforms import get_controller
    saved = get_controller().get_clipboard()
    try:
        res, _ = await call(rt, owner, "clipboard_write", text="jarvis tool test")
        assert res.ok
        res, _ = await call(rt, owner, "clipboard_read")
        assert res.content == "jarvis tool test"
    finally:
        get_controller().set_clipboard(saved)
