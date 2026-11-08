"""Selects the computer-control adapter for the running operating system."""

from __future__ import annotations

import sys

from jarvis.platforms.base import ComputerController, PlatformError

_controller: ComputerController | None = None


def get_controller() -> ComputerController:
    global _controller
    if _controller is None:
        if sys.platform == "win32":
            from jarvis.platforms.windows import WindowsController

            _controller = WindowsController()
        elif sys.platform == "darwin":
            from jarvis.platforms.macos import MacController

            _controller = MacController()
        else:
            from jarvis.platforms.linux import LinuxController

            _controller = LinuxController()
    return _controller


__all__ = ["ComputerController", "PlatformError", "get_controller"]
