"""A yes/no question shown on this computer's own desktop.

Used to confirm setting up a device on this computer. Only offered when the
core runs in an interactive desktop session (not as a service): on Windows a
standard system message box; other platforms answer through the terminal
(`jarvis devices approve`) instead.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from jarvis.core.observability import get_logger

log = get_logger("local_prompt")

IDYES, IDNO, IDCANCEL = 6, 7, 2
MB_YESNOCANCEL, MB_ICONQUESTION, MB_TOPMOST, MB_SETFOREGROUND, MB_SYSTEMMODAL = 0x3, 0x20, 0x40000, 0x10000, 0x1000
WM_CLOSE = 0x0010


class WindowsPrompt:
    def __init__(self) -> None:
        self._titles: dict[str, str] = {}
        self._lock = threading.Lock()

    def available(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            import ctypes  # noqa: PLC0415
            from ctypes import wintypes  # noqa: PLC0415

            user32 = ctypes.windll.user32
            station = user32.GetProcessWindowStation()
            buf = ctypes.create_unicode_buffer(256)
            needed = wintypes.DWORD()
            if not user32.GetUserObjectInformationW(station, 2, buf, ctypes.sizeof(buf), ctypes.byref(needed)):  # UOI_NAME
                return False
            return buf.value.lower() == "winsta0"  # the interactive desktop, not a service session
        except Exception:  # noqa: BLE001
            return False

    def ask(self, key: str, title: str, text: str, timeout_s: float) -> bool | None:
        """Blocks until answered. True = yes, False = no, None = closed or timed out."""
        import ctypes  # noqa: PLC0415

        unique = f"{title} ({key[-4:]})"
        with self._lock:
            self._titles[key] = unique
        timer = threading.Timer(timeout_s, self.dismiss, args=(key,))
        timer.daemon = True
        timer.start()
        try:
            result = ctypes.windll.user32.MessageBoxW(
                None, text + "\n\n[Yes] allow    [No] deny    [Cancel] decide later", unique,
                MB_YESNOCANCEL | MB_ICONQUESTION | MB_TOPMOST | MB_SETFOREGROUND | MB_SYSTEMMODAL)
        finally:
            timer.cancel()
            with self._lock:
                self._titles.pop(key, None)
        return True if result == IDYES else False if result == IDNO else None

    def dismiss(self, key: str) -> None:
        """Close the window for ``key`` if it is still open (decided elsewhere, cancelled or expired)."""
        with self._lock:
            title = self._titles.get(key)
        if not title:
            return
        try:
            import ctypes  # noqa: PLC0415

            hwnd = ctypes.windll.user32.FindWindowW(None, title)
            if hwnd:
                ctypes.windll.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        except Exception:  # noqa: BLE001
            log.debug("could not close the confirmation window", exc_info=True)


def make_prompt(enabled: bool = True) -> Any:
    # JARVIS_NO_DESKTOP_PROMPTS: never show windows (automated tests, headless use).
    if not enabled or sys.platform != "win32" or os.environ.get("JARVIS_NO_DESKTOP_PROMPTS"):
        return None
    return WindowsPrompt()
