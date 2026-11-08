"""Windows adapter (verified on Windows 11).

Applications: Start Menu shortcuts (.lnk) plus packaged/Store apps from
Get-StartApps. Windows: Win32 EnumWindows / SetForegroundWindow via ctypes.
Clipboard: Win32 clipboard API via ctypes. Notifications: Windows toast via
PowerShell and the WinRT notification API.
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from ctypes import wintypes
from pathlib import Path

import psutil

from jarvis.platforms.base import AppInfo, ComputerController, PlatformError, WindowInfo, run

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.GetWindow.argtypes = [wintypes.HWND, ctypes.c_uint]
user32.GetWindow.restype = wintypes.HWND
user32.OpenClipboard.argtypes = [wintypes.HWND]
user32.GetClipboardData.restype = wintypes.HANDLE
user32.GetClipboardData.argtypes = [wintypes.UINT]
user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
user32.SetClipboardData.restype = wintypes.HANDLE
kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalLock.restype = wintypes.LPVOID
kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
SW_RESTORE = 9
GW_OWNER = 4
POWERSHELL_AUMID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


class WindowsController(ComputerController):
    platform = "windows"

    def __init__(self) -> None:
        self._apps_cache: tuple[float, list[AppInfo]] | None = None

    # ------------------------------------------------------------------ apps
    def list_applications(self) -> list[AppInfo]:
        if self._apps_cache and time.time() - self._apps_cache[0] < 300:
            return self._apps_cache[1]
        apps: dict[str, AppInfo] = {}
        roots = [Path(os.environ.get("ProgramData", r"C:\ProgramData")) / r"Microsoft\Windows\Start Menu\Programs",
                 Path(os.environ.get("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs"]
        skip = ("uninstall", "readme", "help", "documentation", "release notes", "website", "license")
        for root in roots:
            if not root.exists():
                continue
            for lnk in root.rglob("*.lnk"):
                name = lnk.stem
                if any(s in name.lower() for s in skip):
                    continue
                apps.setdefault(name.lower(), AppInfo(name=name, target=str(lnk), kind="shortcut"))
        try:
            res = run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                       "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"], timeout=20)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                for item in data if isinstance(data, list) else [data]:
                    name, appid = item.get("Name"), item.get("AppID")
                    if name and appid and name.lower() not in apps and not any(s in name.lower() for s in skip):
                        apps[name.lower()] = AppInfo(name=name, target=appid, kind="uwp")
        except Exception:  # noqa: BLE001 - shortcuts alone are still useful
            pass
        result = sorted(apps.values(), key=lambda a: a.name.lower())
        self._apps_cache = (time.time(), result)
        return result

    def launch(self, app: AppInfo) -> None:
        if app.kind == "uwp":
            res = run(["explorer.exe", f"shell:AppsFolder\\{app.target}"], timeout=15)
            # explorer returns 1 even on success; verify nothing obviously failed
            if res.returncode not in (0, 1):
                raise PlatformError(f"could not start {app.name}: {res.stderr.strip()}")
        else:
            try:
                os.startfile(app.target)  # type: ignore[attr-defined]
            except OSError as e:
                raise PlatformError(f"could not start {app.name}: {e}") from e

    # --------------------------------------------------------------- windows
    def _enum(self) -> list[tuple[int, str, int]]:
        found: list[tuple[int, str, int]] = []

        def cb(hwnd: int, _lp: int) -> bool:
            if user32.IsWindowVisible(hwnd) and not user32.GetWindow(hwnd, GW_OWNER):
                length = user32.GetWindowTextLengthW(hwnd)
                if length:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    pid = wintypes.DWORD()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    found.append((int(hwnd), buf.value, int(pid.value)))
            return True

        user32.EnumWindows(EnumWindowsProc(cb), 0)
        return found

    def list_windows(self) -> list[WindowInfo]:
        fg = user32.GetForegroundWindow()
        out = []
        for hwnd, title, pid in self._enum():
            try:
                pname = psutil.Process(pid).name()
            except psutil.Error:
                pname = None
            out.append(WindowInfo(id=str(hwnd), title=title, pid=pid, process=pname, focused=hwnd == fg))
        return out

    def focus_window(self, query: str) -> WindowInfo:
        q = query.lower()
        wins = self.list_windows()
        match = next((w for w in wins if q in w.title.lower()), None) or \
            next((w for w in wins if w.process and q in w.process.lower()), None)
        if match is None:
            raise PlatformError(f"no open window matches '{query}'")
        hwnd = int(match.id)
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
        # Windows only lets the foreground process change focus; a synthetic Alt
        # press satisfies the rule without side effects.
        user32.keybd_event(0x12, 0, 0, 0)
        ok = user32.SetForegroundWindow(hwnd)
        user32.keybd_event(0x12, 0, 2, 0)
        if not ok:
            raise PlatformError(f"Windows refused to focus '{match.title}'")
        match.focused = True
        return match

    # ------------------------------------------------------------- clipboard
    def _open_clipboard(self) -> None:
        # Other programs (clipboard managers, browsers, remote desktop) hold the
        # clipboard briefly; keep trying for about two seconds before giving up.
        delay = 0.02
        for _ in range(16):
            if user32.OpenClipboard(None):
                return
            time.sleep(delay)
            delay = min(0.2, delay * 1.6)
        raise PlatformError("the clipboard is in use by another application")

    def get_clipboard(self) -> str:
        self._open_clipboard()
        try:
            handle = user32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return ""
            ptr = kernel32.GlobalLock(handle)
            try:
                return ctypes.wstring_at(ptr)
            finally:
                kernel32.GlobalUnlock(handle)
        finally:
            user32.CloseClipboard()

    def set_clipboard(self, text: str) -> None:
        data = ctypes.create_unicode_buffer(text)
        size = ctypes.sizeof(data)
        self._open_clipboard()
        try:
            user32.EmptyClipboard()
            handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
            ptr = kernel32.GlobalLock(handle)
            ctypes.memmove(ptr, ctypes.addressof(data), size)
            kernel32.GlobalUnlock(handle)
            if not user32.SetClipboardData(CF_UNICODETEXT, handle):
                raise PlatformError("SetClipboardData failed")
        finally:
            user32.CloseClipboard()

    # --------------------------------------------------------- notifications
    def notify(self, title: str, message: str) -> None:
        def esc(s: str) -> str:
            return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                    .replace("'", "''"))[:400]

        script = (
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
            "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null;"
            f"$xml = New-Object Windows.Data.Xml.Dom.XmlDocument; $xml.LoadXml('<toast><visual><binding template=\"ToastGeneric\">"
            f"<text>{esc(title)}</text><text>{esc(message)}</text></binding></visual></toast>');"
            f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{POWERSHELL_AUMID}').Show("
            "[Windows.UI.Notifications.ToastNotification]::new($xml))"
        )
        res = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=20)
        if res.returncode != 0:
            raise PlatformError(f"notification failed: {res.stderr.strip()[:300]}")

    def _open_target(self, target: str) -> None:
        try:
            os.startfile(target)  # type: ignore[attr-defined]
        except OSError as e:
            raise PlatformError(str(e)) from e
