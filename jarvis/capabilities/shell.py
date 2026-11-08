"""Shell command execution.

Model-generated commands are untrusted code. Every command needs approval
(SENSITIVE); commands matching destructive or system-altering patterns are
DANGEROUS and always need a fresh, single-use approval. Commands run with a
scrubbed environment (no API keys or tokens), in the JARVIS workspace by
default, with a timeout that kills the whole process tree.

This is not an OS-level sandbox: an approved command runs with your user's
permissions. The approval prompt shows the exact command before it runs.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from jarvis.capabilities.base import Tool, ToolContext, ToolResult
from jarvis.core.permissions import Risk
from jarvis.platforms.base import env_without_secrets

MAX_OUTPUT = 20000

DANGEROUS_PATTERNS = [
    r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\b", r"\brm\s+-[a-z]*r", r"\bdel\s+/[sq]", r"\brmdir\s+/s", r"\brd\s+/s",
    r"remove-item\b.*-recurse", r"\bformat\s+[a-z]:", r"\bmkfs\b", r"\bdd\s+if=", r"\bdiskpart\b", r"\bfdisk\b",
    r"\bshutdown\b", r"\breboot\b", r"\bhalt\b", r"stop-computer", r"restart-computer",
    r"\breg\s+(add|delete)\b", r"set-itemproperty\s+.*hk(lm|cu)", r"\bbcdedit\b", r"\bnetsh\b", r"set-executionpolicy",
    r"\bchmod\s+-r\s+777\b", r"\bchown\s+-r\b", r"\bsudo\b", r"\brunas\b", r"start-process\s+.*-verb\s+runas",
    r"(curl|wget|iwr|invoke-webrequest)\b.*\|\s*(sh|bash|iex|invoke-expression|python)", r"\biex\b", r"invoke-expression",
    r"\bcipher\s+/w", r"vssadmin\s+delete", r"wmic\s+.*delete", r"\btaskkill\s+/f\b.*/im\s+(svchost|csrss|winlogon|lsass)",
    r":\(\)\s*\{", r">\s*/dev/sd[a-z]", r"\bgit\s+push\s+.*--force\b", r"\bgit\s+reset\s+--hard\b", r"\bgit\s+clean\s+-[a-z]*f",
]
_DANGER = re.compile("|".join(DANGEROUS_PATTERNS), re.I)


def classify(command: str) -> Risk:
    return Risk.DANGEROUS if _DANGER.search(command) else Risk.SENSITIVE


def _kill_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.Error:
        pass


def _shell_argv(command: str, shell: str) -> list[str]:
    if sys.platform == "win32":
        if shell == "cmd":
            return ["cmd.exe", "/d", "/s", "/c", command]
        return ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
                "$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[Text.Encoding]::UTF8; " + command]
    return ["/bin/bash", "-lc", command] if Path("/bin/bash").exists() else ["/bin/sh", "-c", command]


async def run_command(command: str, cwd: Path, timeout_s: float, shell: str = "default",
                      cancel: asyncio.Event | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000 | 0x00000200  # no window, new process group
    else:
        kwargs["start_new_session"] = True
    proc = await asyncio.create_subprocess_exec(*_shell_argv(command, shell), cwd=str(cwd), env=env_without_secrets(),
                                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    comm = asyncio.ensure_future(proc.communicate())
    waits = {comm}
    cw = asyncio.ensure_future(cancel.wait()) if cancel is not None else None
    if cw:
        waits.add(cw)
    done, _ = await asyncio.wait(waits, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
    timed_out = comm not in done
    if cw:
        cw.cancel()
    if timed_out:
        _kill_tree(proc.pid)
        try:
            out, err = await asyncio.wait_for(comm, 5)
        except TimeoutError:
            out, err = b"", b""
    else:
        out, err = comm.result()

    def dec(b: bytes) -> str:
        s = b.decode("utf-8", errors="replace")
        return s if len(s) <= MAX_OUTPUT else s[:MAX_OUTPUT // 2] + f"\n...[{len(s) - MAX_OUTPUT} characters omitted]...\n" + s[-MAX_OUTPUT // 2:]

    return {"command": command, "cwd": str(cwd), "exit_code": None if timed_out else proc.returncode, "timed_out": timed_out,
            "cancelled": bool(cancel and cancel.is_set()), "stdout": dec(out), "stderr": dec(err),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1)}


async def _run_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from jarvis.capabilities.files import PathDenied, _policy  # noqa: PLC0415

    try:
        cwd = _policy(ctx).resolve(args.get("cwd") or str(ctx.workspace or ctx.runtime.paths.workspace), ctx)
    except PathDenied as e:
        return ToolResult.fail(str(e))
    if not cwd.is_dir():
        return ToolResult.fail(f"working directory {cwd} does not exist")
    r = await run_command(args["command"], cwd, float(args.get("timeout_s", 120)), args.get("shell", "default"), ctx.cancel)
    status = "timed out" if r["timed_out"] else f"exit code {r['exit_code']}"
    body = [f"$ {r['command']}   ({status}, {r['duration_ms']:.0f} ms, in {r['cwd']})"]
    if r["stdout"].strip():
        body.append("stdout:\n" + r["stdout"].rstrip())
    if r["stderr"].strip():
        body.append("stderr:\n" + r["stderr"].rstrip())
    ok = not r["timed_out"] and r["exit_code"] == 0
    return ToolResult(ok=ok, content="\n".join(body), data=r, error=None if ok else status)


def shell_tools() -> list[Tool]:
    return [
        Tool("run_command", "Run a shell command (PowerShell on Windows, bash elsewhere) and return exit code, stdout and stderr. "
                            "Needs the user's approval. Prefer dedicated tools (files, apps) when they fit.",
             {"type": "object", "properties": {
                 "command": {"type": "string"}, "cwd": {"type": "string", "description": "working directory (default: JARVIS workspace)"},
                 "timeout_s": {"type": "number", "minimum": 1, "maximum": 1800, "default": 120},
                 "shell": {"type": "string", "enum": ["default", "cmd"], "default": "default"}},
              "required": ["command"]},
             _run_tool, Risk.SENSITIVE, "shell.execute", "shell", risk_fn=lambda a, c: classify(a["command"]),
             describe=lambda a: f"run `{a['command'][:300]}`" + (f" in {a['cwd']}" if a.get("cwd") else ""),
             timeout_s=1900, keywords=("command", "terminal", "shell", "powershell", "bash", "script", "install", "pip ", "npm ")),
    ]


def register(runtime: Any) -> None:
    runtime.tools.register_all(shell_tools())
