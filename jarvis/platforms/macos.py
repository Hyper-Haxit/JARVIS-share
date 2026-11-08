"""macOS adapter.

NOT VERIFIED on a Mac in this build (developed on Windows). It uses only stock
macOS facilities: `open -a`, `osascript` (System Events) for windows and
notifications, and `pbcopy`/`pbpaste` for the clipboard. Window listing needs the
Accessibility permission for the process running JARVIS.
"""

from __future__ import annotations

import time
from pathlib import Path

from jarvis.platforms.base import AppInfo, ComputerController, PlatformError, WindowInfo, run

APP_DIRS = [Path("/Applications"), Path("/System/Applications"), Path("/System/Applications/Utilities"),
            Path.home() / "Applications"]


def _osa(script: str, timeout: float = 15) -> str:
    res = run(["osascript", "-e", script], timeout=timeout)
    if res.returncode != 0:
        msg = res.stderr.strip()
        if "not allowed assistive access" in msg or "-1719" in msg:
            raise PlatformError("macOS denied Accessibility access; allow it in System Settings → Privacy & Security → Accessibility")
        raise PlatformError(msg or "osascript failed")
    return res.stdout.strip()


def _q(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


class MacController(ComputerController):
    platform = "macos"

    def __init__(self) -> None:
        self._cache: tuple[float, list[AppInfo]] | None = None

    def list_applications(self) -> list[AppInfo]:
        if self._cache and time.time() - self._cache[0] < 300:
            return self._cache[1]
        apps = {}
        for d in APP_DIRS:
            if d.exists():
                for app in d.glob("*.app"):
                    apps.setdefault(app.stem.lower(), AppInfo(name=app.stem, target=str(app), kind="bundle"))
        result = sorted(apps.values(), key=lambda a: a.name.lower())
        self._cache = (time.time(), result)
        return result

    def launch(self, app: AppInfo) -> None:
        res = run(["open", "-a", app.target])
        if res.returncode != 0:
            raise PlatformError(f"could not open {app.name}: {res.stderr.strip()}")

    def list_windows(self) -> list[WindowInfo]:
        out = _osa('tell application "System Events" to get {name, unix id} of (processes whose background only is false)')
        names_pids = [p.strip() for p in out.split(",")]
        half = len(names_pids) // 2
        front = _osa('tell application "System Events" to get name of first process whose frontmost is true')
        wins = []
        for name, pid in zip(names_pids[:half], names_pids[half:], strict=False):
            wins.append(WindowInfo(id=pid, title=name, pid=int(pid) if pid.isdigit() else None, process=name, focused=name == front))
        return wins

    def focus_window(self, query: str) -> WindowInfo:
        match = next((w for w in self.list_windows() if query.lower() in w.title.lower()), None)
        if match is None:
            raise PlatformError(f"no open application matches '{query}'")
        _osa(f'tell application "{_q(match.title)}" to activate')
        match.focused = True
        return match

    def get_clipboard(self) -> str:
        return run(["pbpaste"]).stdout

    def set_clipboard(self, text: str) -> None:
        res = run(["pbcopy"], input_text=text)
        if res.returncode != 0:
            raise PlatformError("pbcopy failed")

    def notify(self, title: str, message: str) -> None:
        _osa(f'display notification "{_q(message[:400])}" with title "{_q(title[:120])}"')

    def _open_target(self, target: str) -> None:
        res = run(["open", target])
        if res.returncode != 0:
            raise PlatformError(res.stderr.strip())
