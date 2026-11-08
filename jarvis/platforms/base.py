"""Computer control: one interface, one implementation per operating system.

Tools call these methods ("open the app called Spotify") instead of having the
model improvise OS-specific shell commands. Each method either does the real
thing or raises PlatformError explaining why it can't on this system.
"""

from __future__ import annotations

import abc
import os
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil


class PlatformError(RuntimeError):
    pass


@dataclass
class AppInfo:
    name: str
    target: str          # launch target (shortcut path, app id, bundle path, desktop file)
    kind: str            # shortcut | uwp | bundle | desktop | exe

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WindowInfo:
    id: str
    title: str
    pid: int | None
    process: str | None
    focused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run(cmd: list[str], timeout: float = 15, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=input_text, encoding="utf-8",
                          errors="replace", **kwargs)


def _score(query: str, name: str) -> float:
    q, n = query.lower().strip(), name.lower()
    if q == n:
        return 1.0
    if n.startswith(q):
        return 0.9
    if q in n:
        return 0.75
    words = q.split()
    if words and all(w in n for w in words):
        return 0.6
    return 0.0


class ComputerController(abc.ABC):
    platform = "abstract"

    # ---------------------------------------------------------------- apps
    @abc.abstractmethod
    def list_applications(self) -> list[AppInfo]: ...

    def find_application(self, query: str) -> list[AppInfo]:
        scored = [(s, a) for a in self.list_applications() if (s := _score(query, a.name)) > 0]
        scored.sort(key=lambda x: (-x[0], len(x[1].name)))
        return [a for _, a in scored[:8]]

    @abc.abstractmethod
    def launch(self, app: AppInfo) -> None: ...

    def open_application(self, query: str) -> AppInfo:
        matches = self.find_application(query)
        if not matches:
            raise PlatformError(f"no installed application matches '{query}'")
        self.launch(matches[0])
        return matches[0]

    def running_processes(self, query: str) -> list[psutil.Process]:
        q = query.lower().replace(".exe", "")
        out = []
        for p in psutil.process_iter(["name", "exe", "pid"]):
            name = (p.info.get("name") or "").lower().replace(".exe", "")
            if name and (name == q or q in name):
                out.append(p)
        return out

    def close_application(self, query: str, force: bool = False) -> list[dict[str, Any]]:
        procs = self.running_processes(query)
        if not procs:
            raise PlatformError(f"no running application matches '{query}'")
        closed = []
        for p in procs:
            try:
                (p.kill if force else p.terminate)()
                closed.append({"pid": p.pid, "name": p.info.get("name")})
            except psutil.Error as e:
                closed.append({"pid": p.pid, "name": p.info.get("name"), "error": str(e)})
        psutil.wait_procs(procs, timeout=5)
        return closed

    # ------------------------------------------------------------- windows
    @abc.abstractmethod
    def list_windows(self) -> list[WindowInfo]: ...

    @abc.abstractmethod
    def focus_window(self, query: str) -> WindowInfo: ...

    # ----------------------------------------------------------- clipboard
    @abc.abstractmethod
    def get_clipboard(self) -> str: ...

    @abc.abstractmethod
    def set_clipboard(self, text: str) -> None: ...

    # ------------------------------------------------------- notifications
    @abc.abstractmethod
    def notify(self, title: str, message: str) -> None: ...

    # --------------------------------------------------------------- files
    def open_path(self, path: Path) -> None:
        if not path.exists():
            raise PlatformError(f"{path} does not exist")
        self._open_target(str(path))

    def open_url(self, url: str) -> None:
        if not url.startswith(("http://", "https://")):
            raise PlatformError("only http(s) URLs can be opened")
        if not webbrowser.open(url):
            raise PlatformError("no web browser is available")

    @abc.abstractmethod
    def _open_target(self, target: str) -> None: ...

    def capabilities(self) -> dict[str, bool]:
        return {"applications": True, "windows": True, "clipboard": True, "notifications": True, "input": _has_pyautogui()}


def _has_pyautogui() -> bool:
    try:
        import pyautogui  # noqa: F401, PLC0415
        return True
    except Exception:  # noqa: BLE001 - also fails without a display on Linux
        return False


def which(*names: str) -> str | None:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def env_without_secrets() -> dict[str, str]:
    """Environment for child processes with credentials removed."""
    out = {}
    for k, v in os.environ.items():
        ku = k.upper()
        if ku.startswith("JARVIS_") or any(s in ku for s in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")):
            continue
        out[k] = v
    return out
