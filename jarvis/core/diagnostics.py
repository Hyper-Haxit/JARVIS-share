"""Self-diagnostics (``jarvis doctor``).

Each check returns a status (ok / warn / fail / skip), what was found and, when
something is wrong, an actionable fix. Checks inspect the real environment;
nothing is assumed.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | skip
    detail: str
    fix: str | None = None
    group: str = "general"


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


async def run_diagnostics(rt: Any) -> list[dict[str, Any]]:
    checks: list[Check] = []
    add = checks.append

    # -- configuration & storage
    add(Check("config", "ok" if rt.paths.config_file.exists() else "warn",
              f"config file: {rt.paths.config_file}" if rt.paths.config_file.exists() else "no config.toml; using defaults",
              None if rt.paths.config_file.exists() else "run `jarvis setup` to create one", "core"))
    try:
        integrity = rt.db.integrity_check()
        add(Check("database", "ok" if integrity == "ok" else "fail", f"{rt.paths.database} (schema v{rt.db.schema_version}, integrity {integrity})",
                  None if integrity == "ok" else "restore from backup or delete the database", "core"))
    except Exception as e:  # noqa: BLE001
        add(Check("database", "fail", str(e), "check disk and file permissions", "core"))
    try:
        with tempfile.NamedTemporaryFile(dir=rt.paths.home, delete=True):
            pass
        free = shutil.disk_usage(rt.paths.home).free / 2**30
        add(Check("filesystem", "ok" if free > 1 else "warn", f"home {rt.paths.home} writable; {free:.1f} GB free",
                  None if free > 1 else "free some disk space", "core"))
    except OSError as e:
        add(Check("filesystem", "fail", f"cannot write to {rt.paths.home}: {e}", "fix permissions or set JARVIS_HOME", "core"))
    add(Check("secrets", "ok" if rt.secrets.secure else "warn", f"backend: {rt.secrets.backend_name}",
              None if rt.secrets.secure else "install/enable an OS keyring; secrets are in a plain file", "security"))
    host = rt.config.server.host
    lan = host not in ("127.0.0.1", "localhost", "::1")
    tls = bool(rt.config.server.tls_cert and rt.config.server.tls_key)
    if lan and not tls:
        add(Check("transport", "warn" if rt.config.server.allow_insecure_lan else "fail",
                  f"server binds {host} without TLS", "run `jarvis setup --tls` or bind to 127.0.0.1", "security"))
    else:
        add(Check("transport", "ok", f"binds {host}" + (" with TLS" if tls else " (loopback only)"), None, "security"))
    token_ok = rt.local_token is not None and rt.auth.authenticate(rt.local_token) is not None
    add(Check("authentication", "ok" if token_ok else "warn",
              f"{len(rt.auth.list_devices(include_revoked=False))} active device(s); local CLI credential "
              + ("valid" if token_ok else "not initialised"), None if token_ok else "start the core once", "security"))

    # -- models
    health = await rt.router.all_health(force=True)
    for h in health:
        add(Check(f"provider:{h.provider}", "ok" if h.ok else "fail",
                  ("reachable" + (f" ({h.latency_ms} ms)" if h.latency_ms else "")) if h.ok else f"unreachable: {h.error}",
                  None if h.ok else ("start Ollama (`ollama serve`)" if "ollama" in h.provider else "check URL / API key"),
                  "models"))
    if not health:
        add(Check("providers", "fail", "no model providers configured", "add a provider in config.toml", "models"))
    available: dict[str, set[str]] = {}
    for name, provider in rt.router.providers.items():
        try:
            available[name] = {m.name for m in await provider.list_models()}
        except Exception:  # noqa: BLE001
            available[name] = set()
    for role in ("fast", "reasoning", "coding", "vision", "embedding", "fallback"):
        refs = rt.router.role_candidates(role)
        present = []
        for ref in refs:
            pname, model = ref.split("/", 1)
            models = available.get(pname, set())
            if model in models or f"{model}:latest" in models or (not models and pname in available):
                present.append(ref)
        add(Check(f"role:{role}", "ok" if present else "warn",
                  f"available: {', '.join(present)}" if present else f"none of {refs} are installed/reachable",
                  None if present else f"install one, e.g. `ollama pull {refs[0].split('/', 1)[1]}`" if refs else "configure candidates",
                  "models"))

    # -- audio / vision / devices
    if _has("sounddevice"):
        try:
            import sounddevice as sd  # noqa: PLC0415

            devices = await asyncio.to_thread(sd.query_devices)
            ins = [d for d in devices if d["max_input_channels"] > 0]
            outs = [d for d in devices if d["max_output_channels"] > 0]
            add(Check("microphone", "ok" if ins else "fail", f"{len(ins)} input device(s)" + (f"; default: {sd.query_devices(kind='input')['name']}" if ins else ""),
                      None if ins else "connect a microphone", "voice"))
            add(Check("speakers", "ok" if outs else "fail", f"{len(outs)} output device(s)", None if outs else "connect speakers", "voice"))
        except Exception as e:  # noqa: BLE001
            add(Check("audio", "fail", f"audio subsystem error: {e}", "check audio drivers", "voice"))
    else:
        add(Check("audio", "warn", "sounddevice not installed", "pip install 'jarvis-core[voice]'", "voice"))
    for mod, label, extra in (("faster_whisper", "speech recognition", "voice"), ("openwakeword", "wake word", "voice"),
                              ("kokoro_onnx", "Kokoro voice", "voice"), ("pyttsx3", "system voice", "voice"),
                              ("mss", "screenshots", "vision"),
                              ("PIL", "image processing", "vision"), ("cv2", "camera capture", "camera"),
                              ("playwright", "browser automation", "browser"), ("webview", "desktop shell", "desktop"),
                              ("pyautogui", "keyboard/mouse control", "control"), ("cryptography", "TLS certificates", "tls")):
        add(Check(f"dependency:{label}", "ok" if _has(mod) else "warn", f"{mod} {'installed' if _has(mod) else 'missing'}",
                  None if _has(mod) else f"pip install 'jarvis-core[{extra}]'", "dependencies"))
    if _has("playwright"):
        browsers = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or (
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "ms-playwright") if sys.platform == "win32"
            else os.path.expanduser("~/Library/Caches/ms-playwright") if sys.platform == "darwin"
            else os.path.expanduser("~/.cache/ms-playwright"))
        has_chromium = os.path.isdir(browsers) and any(n.startswith("chromium") for n in os.listdir(browsers))
        add(Check("browser", "ok" if has_chromium else "warn", "Chromium for Playwright " + ("installed" if has_chromium else "not installed"),
                  None if has_chromium else "python -m playwright install chromium", "browser"))

    for name, ext in rt.extensions.items():
        diag = getattr(ext, "diagnostics", None)
        if diag is None:
            continue
        try:
            res = diag()
            res = await res if asyncio.iscoroutine(res) else res
            for item in res:
                add(Check(**item))
        except Exception as e:  # noqa: BLE001
            add(Check(f"{name}", "fail", f"diagnostics failed: {e}", None, name))

    return [asdict(c) for c in checks]
