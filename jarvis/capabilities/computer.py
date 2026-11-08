"""Computer-control tools, implemented through the platform adapter."""

from __future__ import annotations

import asyncio
from typing import Any

from jarvis.capabilities.base import Tool, ToolContext, ToolResult
from jarvis.core.permissions import Risk
from jarvis.platforms import PlatformError, get_controller


def _wrap(fn):  # noqa: ANN001, ANN202
    async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            return await asyncio.to_thread(fn, args, ctx)
        except PlatformError as e:
            return ToolResult.fail(str(e))
    return handler


def _open_app(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    c = get_controller()
    matches = c.find_application(args["name"])
    if not matches:
        suggestions = [a.name for a in c.list_applications() if args["name"].lower()[:3] in a.name.lower()][:8]
        return ToolResult.fail(f"no installed application matches '{args['name']}'" + (f". Similar: {', '.join(suggestions)}" if suggestions else ""))
    app = matches[0]
    c.launch(app)
    return ToolResult(ok=True, content=f"Opened {app.name}.", data=app.to_dict())


def _list_apps(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    c = get_controller()
    apps = c.find_application(args["query"]) if args.get("query") else c.list_applications()
    return ToolResult(ok=True, content=f"{len(apps)} application(s): " + ", ".join(a.name for a in apps[:150]), data=[a.to_dict() for a in apps[:150]])


def _close_app(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    closed = get_controller().close_application(args["name"], force=args.get("force", False))
    failed = [c for c in closed if c.get("error")]
    return ToolResult(ok=not failed, content=f"Closed {len(closed) - len(failed)} process(es) matching '{args['name']}'"
                      + (f"; {len(failed)} could not be closed" if failed else ""), data=closed)


def _windows(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    wins = get_controller().list_windows()
    lines = [f"{'* ' if w.focused else '  '}{w.title}  [{w.process or '?'}]" for w in wins]
    return ToolResult(ok=True, content=f"{len(wins)} open window(s) (* = focused):\n" + "\n".join(lines), data=[w.to_dict() for w in wins])


def _focus(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    w = get_controller().focus_window(args["query"])
    return ToolResult(ok=True, content=f"Focused “{w.title}”.", data=w.to_dict())


def _clip_get(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    text = get_controller().get_clipboard()
    return ToolResult(ok=True, content=text if text else "(the clipboard has no text)", data={"length": len(text)})


def _clip_set(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    get_controller().set_clipboard(args["text"])
    return ToolResult(ok=True, content=f"Copied {len(args['text'])} characters to the clipboard.")


def _notify(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    get_controller().notify(args.get("title") or "JARVIS", args["message"])
    return ToolResult(ok=True, content="Notification shown.")


def _open_path(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from jarvis.capabilities.files import PathDenied, _policy  # noqa: PLC0415

    try:
        p = _policy(ctx).resolve(args["path"], ctx)
    except PathDenied as e:
        return ToolResult.fail(str(e))
    get_controller().open_path(p)
    return ToolResult(ok=True, content=f"Opened {p} with its default application.")


def _open_url(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    get_controller().open_url(args["url"])
    return ToolResult(ok=True, content=f"Opened {args['url']} in the default browser.")


def _input(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    try:
        import pyautogui  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail(f"keyboard/mouse control is unavailable: {e}")
    pyautogui.FAILSAFE = True  # moving the mouse to a screen corner aborts
    action = args["action"]
    if action == "type":
        pyautogui.write(args.get("text", ""), interval=0.01)
        return ToolResult(ok=True, content=f"Typed {len(args.get('text', ''))} characters.")
    if action == "hotkey":
        keys = [k.strip().lower() for k in args.get("keys", "").split("+") if k.strip()]
        pyautogui.hotkey(*keys)
        return ToolResult(ok=True, content=f"Pressed {'+'.join(keys)}.")
    if action in ("click", "double_click", "right_click", "move"):
        x, y = args.get("x"), args.get("y")
        if x is None or y is None:
            return ToolResult.fail("x and y are required")
        w, h = pyautogui.size()
        if not (0 <= x < w and 0 <= y < h):
            return ToolResult.fail(f"({x}, {y}) is outside the screen ({w}×{h})")
        {"click": lambda: pyautogui.click(x, y), "double_click": lambda: pyautogui.doubleClick(x, y),
         "right_click": lambda: pyautogui.rightClick(x, y), "move": lambda: pyautogui.moveTo(x, y, duration=0.2)}[action]()
        return ToolResult(ok=True, content=f"{action.replace('_', ' ')} at ({x}, {y}).")
    if action == "scroll":
        pyautogui.scroll(int(args.get("amount", -3)) * 100)
        return ToolResult(ok=True, content="Scrolled.")
    return ToolResult.fail(f"unknown input action {action}")


def computer_tools() -> list[Tool]:
    app_kw = ("open", "launch", "start", "close", "quit", "app", "application", "program", "window", "switch to")
    return [
        Tool("open_application", "Open an installed application by name (e.g. 'Spotify', 'Visual Studio Code').",
             {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
             _wrap(_open_app), Risk.LOW_RISK, "computer.apps.open", "computer", keywords=app_kw,
             describe=lambda a: f"open the application “{a['name']}”"),
        Tool("list_applications", "List installed applications, optionally filtered.",
             {"type": "object", "properties": {"query": {"type": "string"}}}, _wrap(_list_apps), Risk.SAFE, "computer.apps.read", "computer",
             keywords=app_kw),
        Tool("close_application", "Close a running application (all its processes).",
             {"type": "object", "properties": {"name": {"type": "string"}, "force": {"type": "boolean", "default": False}}, "required": ["name"]},
             _wrap(_close_app), Risk.SENSITIVE, "computer.apps.close", "computer", keywords=app_kw,
             describe=lambda a: f"{'force-' if a.get('force') else ''}close every process of “{a['name']}” (unsaved work may be lost)"),
        Tool("list_windows", "List open windows and which one is focused.", {"type": "object", "properties": {}},
             _wrap(_windows), Risk.SAFE, "computer.windows.read", "computer", keywords=app_kw),
        Tool("focus_window", "Bring a window to the front by title or application name.",
             {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
             _wrap(_focus), Risk.LOW_RISK, "computer.windows.focus", "computer", keywords=app_kw),
        Tool("clipboard_read", "Read the text on the clipboard.", {"type": "object", "properties": {}},
             _wrap(_clip_get), Risk.LOW_RISK, "computer.clipboard.read", "computer", keywords=("clipboard", "copied", "paste")),
        Tool("clipboard_write", "Put text on the clipboard.", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
             _wrap(_clip_set), Risk.LOW_RISK, "computer.clipboard.write", "computer", keywords=("clipboard", "copy")),
        Tool("notify", "Show a desktop notification on this computer.",
             {"type": "object", "properties": {"title": {"type": "string"}, "message": {"type": "string"}}, "required": ["message"]},
             _wrap(_notify), Risk.LOW_RISK, "computer.notify", "computer", keywords=("notify", "notification", "remind", "alert")),
        Tool("open_path", "Open a file or folder with its default application.",
             {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
             _wrap(_open_path), Risk.LOW_RISK, "computer.open", "computer", keywords=("open", "show me", "folder", "file")),
        Tool("open_url", "Open a web page in the user's default browser.",
             {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
             _wrap(_open_url), Risk.LOW_RISK, "computer.open_url", "computer", keywords=("open", "website", "browser", "http")),
        Tool("keyboard_mouse", "Control the keyboard and mouse: type text, press a hotkey (e.g. 'ctrl+s'), click/move at screen "
                               "coordinates, or scroll. Use a screenshot first to find coordinates.",
             {"type": "object", "properties": {
                 "action": {"type": "string", "enum": ["type", "hotkey", "click", "double_click", "right_click", "move", "scroll"]},
                 "text": {"type": "string"}, "keys": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"},
                 "amount": {"type": "integer"}},
              "required": ["action"]},
             _wrap(_input), Risk.SENSITIVE, "computer.input", "computer",
             describe=lambda a: {"type": f"type “{a.get('text', '')[:80]}”", "hotkey": f"press {a.get('keys')}"}.get(
                 a["action"], f"{a['action']} at ({a.get('x')}, {a.get('y')})") + " with your keyboard/mouse",
             keywords=("type", "click", "press", "keyboard", "mouse", "hotkey", "shortcut")),
    ]


def register(runtime: Any) -> None:
    runtime.tools.register_all(computer_tools())
