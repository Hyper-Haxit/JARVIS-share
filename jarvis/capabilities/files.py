"""Filesystem tools.

Risk is computed from the actual paths, in code:
  * JARVIS's own credentials, database, TLS keys and config are never readable
    or writable by tools - not even with approval.
  * Inside the allowed roots (your home folder and the JARVIS workspace):
    reading is SAFE, creating files LOW_RISK, changing or deleting existing
    files SENSITIVE.
  * Outside them: reading is SENSITIVE, any change DANGEROUS.
  * Operating-system directories are always DANGEROUS to change.
Deleting moves items into the JARVIS trash (restorable), never destroys them.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from jarvis.capabilities.base import Tool, ToolContext, ToolResult
from jarvis.core.permissions import Risk

TEXT_LIMIT = 2_000_000
SYSTEM_DIRS_WIN = [r"C:\Windows", r"C:\Program Files", r"C:\Program Files (x86)", r"C:\ProgramData"]
SYSTEM_DIRS_POSIX = ["/bin", "/sbin", "/usr", "/etc", "/boot", "/lib", "/lib64", "/sys", "/proc", "/System", "/Library", "/private/etc"]


class PathDenied(Exception):
    pass


def _is_within(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


class PathPolicy:
    def __init__(self, runtime: Any) -> None:
        self.rt = runtime
        cfg_roots = runtime.config.permissions.allowed_roots
        roots = [Path(r).expanduser() for r in cfg_roots] if cfg_roots else [Path.home(), runtime.paths.workspace]
        self.roots = [r.resolve() for r in roots]
        self.protected = [p.resolve() for p in runtime.paths.protected_paths()] + [(runtime.paths.home / "trash").resolve()]
        sysdirs = SYSTEM_DIRS_WIN if sys.platform == "win32" else SYSTEM_DIRS_POSIX
        self.system = [Path(d).resolve() for d in sysdirs]

    def resolve(self, raw: str, ctx: ToolContext | None = None) -> Path:
        raw = os.path.expandvars(os.path.expanduser(raw.strip().strip('"')))
        p = Path(raw)
        if not p.is_absolute():
            p = (ctx.workspace if ctx and ctx.workspace else self.rt.paths.workspace) / p
        p = p.resolve()  # follows symlinks: the real target is what gets checked
        for prot in self.protected:
            if p == prot or _is_within(p, prot):
                raise PathDenied("that location holds JARVIS's own credentials or data and is off limits to tools")
        return p

    def inside_roots(self, p: Path) -> bool:
        return any(p == r or _is_within(p, r) for r in self.roots)

    def is_system(self, p: Path) -> bool:
        return any(p == s or _is_within(p, s) for s in self.system)

    def read_risk(self, p: Path) -> Risk:
        return Risk.SAFE if self.inside_roots(p) else Risk.SENSITIVE

    def write_risk(self, p: Path, existing: bool) -> Risk:
        if self.is_system(p) or not self.inside_roots(p):
            return Risk.DANGEROUS
        return Risk.SENSITIVE if existing else Risk.LOW_RISK


def _policy(ctx: ToolContext) -> PathPolicy:
    pol = ctx.runtime.extensions.get("path_policy")
    if pol is None:
        pol = ctx.runtime.extensions["path_policy"] = PathPolicy(ctx.runtime)
    return pol


def _risk(kind: str):  # noqa: ANN202
    def fn(args: dict[str, Any], ctx: ToolContext) -> Risk:
        pol = _policy(ctx)
        try:
            if kind == "read":
                return pol.read_risk(pol.resolve(args["path"], ctx))
            if kind == "write":
                p = pol.resolve(args["path"], ctx)
                return pol.write_risk(p, p.exists())
            if kind == "modify":
                return pol.write_risk(pol.resolve(args["path"], ctx), True)
            if kind == "move":
                src, dst = pol.resolve(args["source"], ctx), pol.resolve(args["destination"], ctx)
                return max(pol.write_risk(src, True), pol.write_risk(dst, dst.exists()))
        except PathDenied:
            return Risk.SAFE  # the handler refuses; no approval prompt for a guaranteed refusal
        return Risk.DANGEROUS
    return fn


def _guard(ctx: ToolContext, raw: str) -> Path | ToolResult:
    try:
        return _policy(ctx).resolve(raw, ctx)
    except PathDenied as e:
        return ToolResult.fail(str(e))


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


# --------------------------------------------------------------------- tools
async def fs_list(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    p = _guard(ctx, args.get("path") or ".")
    if isinstance(p, ToolResult):
        return p
    if not p.exists():
        return ToolResult.fail(f"{p} does not exist")
    if p.is_file():
        st = p.stat()
        return ToolResult(ok=True, content=f"{p} is a file ({_human(st.st_size)})")
    pattern = args.get("pattern") or "*"
    limit = args.get("limit", 200)
    it = p.rglob(pattern) if args.get("recursive") else p.glob(pattern)
    rows, total = [], 0
    for child in it:
        if any(part in (".git", "node_modules", "__pycache__", ".venv") for part in child.relative_to(p).parts[:-1]):
            continue
        total += 1
        if len(rows) < limit:
            try:
                st = child.stat()
                rows.append({"name": str(child.relative_to(p)), "dir": child.is_dir(), "size": None if child.is_dir() else st.st_size,
                             "modified": st.st_mtime})
            except OSError:
                continue
    rows.sort(key=lambda r: (not r["dir"], r["name"].lower()))
    lines = [f"{p} ({total} item{'s' if total != 1 else ''}{', showing ' + str(len(rows)) if total > len(rows) else ''})"]
    lines += [f"  {r['name']}{'/' if r['dir'] else ''}{'' if r['dir'] else '  ' + _human(r['size'])}" for r in rows]
    return ToolResult(ok=True, content="\n".join(lines), data={"path": str(p), "items": rows, "total": total})


async def fs_read(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    p = _guard(ctx, args["path"])
    if isinstance(p, ToolResult):
        return p
    if not p.is_file():
        return ToolResult.fail(f"{p} is not a file")
    if p.stat().st_size > TEXT_LIMIT * 5:
        return ToolResult.fail(f"{p} is {_human(p.stat().st_size)}; too large to read as text")
    data = p.read_bytes()
    if _is_binary(data):
        return ToolResult.fail(f"{p} looks like a binary file ({_human(len(data))}); it can't be shown as text")
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    start = max(1, args.get("start_line") or 1)
    end = min(len(lines), args.get("end_line") or len(lines))
    numbered = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(start, end + 1))
    return ToolResult(ok=True, content=f"{p} (lines {start}-{end} of {len(lines)})\n{numbered}",
                      data={"path": str(p), "lines": len(lines)})


async def fs_write(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    p = _guard(ctx, args["path"])
    if isinstance(p, ToolResult):
        return p
    mode = args.get("mode", "create")
    content = args["content"]
    if p.exists() and p.is_dir():
        return ToolResult.fail(f"{p} is a directory")
    if mode == "create" and p.exists():
        return ToolResult.fail(f"{p} already exists; use mode 'overwrite' or 'append' if that is intended")
    p.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append":
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
    else:
        tmp = p.with_name(p.name + ".jarvis-tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(p)
    return ToolResult(ok=True, content=f"{'Appended to' if mode == 'append' else 'Wrote'} {p} ({_human(len(content.encode()))})",
                      data={"path": str(p), "bytes": len(content.encode())})


async def fs_edit(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    p = _guard(ctx, args["path"])
    if isinstance(p, ToolResult):
        return p
    if not p.is_file():
        return ToolResult.fail(f"{p} is not a file")
    text = p.read_text("utf-8")
    old, new = args["old_text"], args["new_text"]
    count = text.count(old)
    if count == 0:
        return ToolResult.fail("the text to replace was not found; read the file again and copy it exactly")
    if count > 1 and not args.get("replace_all"):
        return ToolResult.fail(f"the text to replace occurs {count} times; include more surrounding text or set replace_all")
    updated = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
    tmp = p.with_name(p.name + ".jarvis-tmp")
    tmp.write_text(updated, encoding="utf-8")
    tmp.replace(p)
    return ToolResult(ok=True, content=f"Edited {p}: replaced {count if args.get('replace_all') else 1} occurrence(s)",
                      data={"path": str(p), "replacements": count if args.get("replace_all") else 1})


async def fs_mkdir(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    p = _guard(ctx, args["path"])
    if isinstance(p, ToolResult):
        return p
    p.mkdir(parents=True, exist_ok=True)
    return ToolResult(ok=True, content=f"Folder ready: {p}")


async def fs_move(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    src, dst = _guard(ctx, args["source"]), _guard(ctx, args["destination"])
    for x in (src, dst):
        if isinstance(x, ToolResult):
            return x
    assert isinstance(src, Path) and isinstance(dst, Path)
    if not src.exists():
        return ToolResult.fail(f"{src} does not exist")
    if dst.exists() and not dst.is_dir():
        return ToolResult.fail(f"{dst} already exists")
    final = Path(shutil.move(str(src), str(dst)))
    return ToolResult(ok=True, content=f"Moved {src} → {final}", data={"from": str(src), "to": str(final)})


async def fs_delete(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    p = _guard(ctx, args["path"])
    if isinstance(p, ToolResult):
        return p
    if not p.exists():
        return ToolResult.fail(f"{p} does not exist")
    trash = ctx.runtime.paths.home / "trash" / time.strftime("%Y%m%d-%H%M%S")
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / p.name
    shutil.move(str(p), str(dest))
    (trash / f"{p.name}.origin.txt").write_text(str(p), encoding="utf-8")
    return ToolResult(ok=True, content=f"Moved {p} to the JARVIS trash ({dest}); it can be restored from there.",
                      data={"path": str(p), "trash": str(dest)})


async def fs_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    root = _guard(ctx, args.get("path") or ".")
    if isinstance(root, ToolResult):
        return root
    if not root.is_dir():
        return ToolResult.fail(f"{root} is not a folder")
    try:
        rx = re.compile(args["query"], 0 if args.get("case_sensitive") else re.I) if args.get("regex") else None
    except re.error as e:
        return ToolResult.fail(f"invalid regular expression: {e}")
    needle = args["query"] if args.get("case_sensitive") else args["query"].lower()
    glob = args.get("glob") or "*"
    limit = args.get("limit", 100)
    hits, scanned = [], 0
    deadline = time.time() + 20
    for f in root.rglob("*"):
        if time.time() > deadline or len(hits) >= limit:
            break
        rel = f.relative_to(root)
        if any(part.startswith(".") or part in ("node_modules", "__pycache__", "venv") for part in rel.parts[:-1]):
            continue
        if not f.is_file() or not fnmatch.fnmatch(f.name, glob):
            continue
        try:
            if f.stat().st_size > TEXT_LIMIT:
                continue
            data = f.read_bytes()
        except OSError:
            continue
        if _is_binary(data):
            continue
        scanned += 1
        for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            found = rx.search(line) if rx else (needle in (line if args.get("case_sensitive") else line.lower()))
            if found:
                hits.append({"file": str(rel), "line": n, "text": line.strip()[:240]})
                if len(hits) >= limit:
                    break
    content = "\n".join(f"{h['file']}:{h['line']}: {h['text']}" for h in hits) or "No matches."
    return ToolResult(ok=True, content=f"{len(hits)} match(es) in {scanned} files under {root}\n{content}", data={"hits": hits})


def file_tools() -> list[Tool]:
    kw = ("file", "folder", "directory", "document", "save", "write", "read", "open", "download", ".txt", ".md", ".py", "desktop", "documents")
    return [
        Tool("fs_list", "List a folder's contents (optionally recursive, filtered by a glob pattern).",
             {"type": "object", "properties": {"path": {"type": "string", "description": "folder; relative paths are in the JARVIS workspace"},
                                               "pattern": {"type": "string"}, "recursive": {"type": "boolean", "default": False},
                                               "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 200}}},
             fs_list, Risk.SAFE, "files.read", "files", risk_fn=lambda a, c: _risk("read")({"path": a.get("path") or "."}, c), keywords=kw),
        Tool("fs_read", "Read a text file with line numbers (optionally a line range).",
             {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1},
                                               "end_line": {"type": "integer", "minimum": 1}}, "required": ["path"]},
             fs_read, Risk.SAFE, "files.read", "files", risk_fn=_risk("read"), keywords=kw),
        Tool("fs_write", "Create a text file, or overwrite/append to one.",
             {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                               "mode": {"type": "string", "enum": ["create", "overwrite", "append"], "default": "create"}},
              "required": ["path", "content"]},
             fs_write, Risk.LOW_RISK, "files.write", "files", risk_fn=_risk("write"),
             describe=lambda a: f"{a.get('mode', 'create')} {a['path']} ({len(a['content'])} characters)", keywords=kw),
        Tool("fs_edit", "Replace exact text in a file (copy old_text exactly from fs_read output, without line numbers).",
             {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
                                               "replace_all": {"type": "boolean", "default": False}},
              "required": ["path", "old_text", "new_text"]},
             fs_edit, Risk.SENSITIVE, "files.modify", "files", risk_fn=_risk("modify"),
             describe=lambda a: f"edit {a['path']}: replace {len(a['old_text'])} characters", keywords=kw),
        Tool("fs_mkdir", "Create a folder (and parents).", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
             fs_mkdir, Risk.LOW_RISK, "files.write", "files", risk_fn=_risk("write"), keywords=kw),
        Tool("fs_move", "Move or rename a file or folder.",
             {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}},
              "required": ["source", "destination"]},
             fs_move, Risk.SENSITIVE, "files.modify", "files", risk_fn=_risk("move"),
             describe=lambda a: f"move {a['source']} → {a['destination']}", keywords=kw),
        Tool("fs_delete", "Delete a file or folder by moving it to the JARVIS trash (restorable).",
             {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
             fs_delete, Risk.SENSITIVE, "files.delete", "files", risk_fn=_risk("modify"),
             describe=lambda a: f"move {a['path']} to the JARVIS trash", keywords=kw + ("delete", "remove")),
        Tool("fs_search", "Search the text contents of files under a folder.",
             {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"},
                                               "regex": {"type": "boolean", "default": False}, "case_sensitive": {"type": "boolean", "default": False},
                                               "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100}},
              "required": ["query"]},
             fs_search, Risk.SAFE, "files.read", "files", risk_fn=lambda a, c: _risk("read")({"path": a.get("path") or "."}, c),
             keywords=kw + ("search", "find", "grep")),
    ]


def register(runtime: Any) -> None:
    runtime.paths.workspace.mkdir(parents=True, exist_ok=True)
    runtime.tools.register_all(file_tools())
