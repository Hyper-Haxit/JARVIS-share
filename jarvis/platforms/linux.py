"""Linux adapter.

NOT VERIFIED on Linux in this build (developed on Windows). Applications come
from freedesktop .desktop files and launch with `gtk-launch` (or the Exec line);
windows use `wmctrl` (X11) when installed; clipboard uses wl-clipboard (Wayland)
or xclip/xsel (X11); notifications use `notify-send`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from jarvis.platforms.base import AppInfo, ComputerController, PlatformError, WindowInfo, run, which

DESKTOP_DIRS = [Path("/usr/share/applications"), Path("/usr/local/share/applications"),
                Path.home() / ".local/share/applications", Path("/var/lib/flatpak/exports/share/applications"),
                Path.home() / ".local/share/flatpak/exports/share/applications", Path("/var/lib/snapd/desktop/applications")]


def _parse_desktop(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    in_entry = False
    try:
        for line in path.read_text("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("["):
                in_entry = line == "[Desktop Entry]"
                continue
            if in_entry and "=" in line:
                k, v = line.split("=", 1)
                data.setdefault(k.strip(), v.strip())
    except OSError:
        pass
    return data


class LinuxController(ComputerController):
    platform = "linux"

    def __init__(self) -> None:
        self._cache: tuple[float, list[AppInfo]] | None = None

    def list_applications(self) -> list[AppInfo]:
        if self._cache and time.time() - self._cache[0] < 300:
            return self._cache[1]
        apps = {}
        for d in DESKTOP_DIRS:
            if not d.exists():
                continue
            for f in d.glob("*.desktop"):
                e = _parse_desktop(f)
                if e.get("Type") != "Application" or e.get("NoDisplay") == "true" or e.get("Hidden") == "true" or not e.get("Name"):
                    continue
                apps.setdefault(e["Name"].lower(), AppInfo(name=e["Name"], target=str(f), kind="desktop"))
        result = sorted(apps.values(), key=lambda a: a.name.lower())
        self._cache = (time.time(), result)
        return result

    def launch(self, app: AppInfo) -> None:
        desktop_id = Path(app.target).stem
        if which("gtk-launch"):
            res = run(["gtk-launch", desktop_id])
            if res.returncode == 0:
                return
        exec_line = _parse_desktop(Path(app.target)).get("Exec", "")
        argv = [a for a in shlex.split(exec_line) if not a.startswith("%")]
        if not argv:
            raise PlatformError(f"no way to launch {app.name}")
        subprocess.Popen(argv, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def list_windows(self) -> list[WindowInfo]:
        if not which("wmctrl"):
            raise PlatformError("window listing needs wmctrl (X11); it is not installed")
        res = run(["wmctrl", "-lp"])
        out = []
        for line in res.stdout.splitlines():
            parts = line.split(None, 4)
            if len(parts) >= 5:
                out.append(WindowInfo(id=parts[0], title=parts[4], pid=int(parts[2]) if parts[2].isdigit() else None, process=None))
        return out

    def focus_window(self, query: str) -> WindowInfo:
        match = next((w for w in self.list_windows() if query.lower() in w.title.lower()), None)
        if match is None:
            raise PlatformError(f"no open window matches '{query}'")
        run(["wmctrl", "-ia", match.id])
        match.focused = True
        return match

    def _clip_cmds(self) -> tuple[list[str], list[str]]:
        if os.environ.get("WAYLAND_DISPLAY") and which("wl-copy"):
            return ["wl-paste", "--no-newline"], ["wl-copy"]
        if which("xclip"):
            return ["xclip", "-selection", "clipboard", "-o"], ["xclip", "-selection", "clipboard"]
        if which("xsel"):
            return ["xsel", "--clipboard", "--output"], ["xsel", "--clipboard", "--input"]
        raise PlatformError("no clipboard tool found (install wl-clipboard, xclip or xsel)")

    def get_clipboard(self) -> str:
        return run(self._clip_cmds()[0]).stdout

    def set_clipboard(self, text: str) -> None:
        if run(self._clip_cmds()[1], input_text=text).returncode != 0:
            raise PlatformError("setting the clipboard failed")

    def notify(self, title: str, message: str) -> None:
        if not which("notify-send"):
            raise PlatformError("notify-send is not installed")
        run(["notify-send", title[:120], message[:400]])

    def _open_target(self, target: str) -> None:
        if not which("xdg-open"):
            raise PlatformError("xdg-open is not installed")
        subprocess.Popen(["xdg-open", target], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
