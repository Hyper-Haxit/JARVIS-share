"""Software-engineering tools: repository inspection, tests, builds and git.

Test and build runs are recorded in a per-conversation verification ledger.
After any run that changed code, the agent appends a statement generated from
that ledger (not from the model) saying whether the change was verified.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from jarvis.capabilities.base import Tool, ToolContext, ToolResult
from jarvis.capabilities.shell import run_command
from jarvis.core.permissions import Risk

CODE_EXT = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".kt", ".cs", ".cpp", ".cc", ".c", ".h",
            ".hpp", ".rb", ".php", ".swift", ".scala", ".sh", ".ps1", ".css", ".scss", ".html", ".vue", ".svelte", ".sql",
            ".toml", ".yaml", ".yml", ".json", ".lua", ".dart", ".m", ".r"}
LANG = {".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript", ".jsx": "JavaScript",
        ".rs": "Rust", ".go": "Go", ".java": "Java", ".kt": "Kotlin", ".cs": "C#", ".cpp": "C++", ".cc": "C++", ".c": "C", ".h": "C/C++",
        ".rb": "Ruby", ".php": "PHP", ".swift": "Swift", ".sh": "Shell", ".ps1": "PowerShell", ".html": "HTML", ".css": "CSS",
        ".vue": "Vue", ".svelte": "Svelte", ".sql": "SQL", ".dart": "Dart", ".lua": "Lua", ".scala": "Scala"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build", "target", ".next", ".idea", ".vscode",
             ".pytest_cache", ".mypy_cache", "bin", "obj", ".gradle", "coverage"}

CODING_RULES = (
    "Software work rules: inspect before changing (repo_overview, code_search, fs_read). Make the smallest correct change with "
    "fs_edit (copy old_text exactly). After changing code, run the tests or build with run_tests/run_build and report exactly "
    "which commands ran and what they returned. Never say a bug is fixed or code works unless a test or build in this "
    "conversation shows it. If verification fails, say so and show the failure."
)


def _quote(text: str) -> str:
    """Quote a literal argument for the shell run_command uses (no expansion)."""
    if sys.platform == "win32":
        return "'" + text.replace("'", "''") + "'"
    return "'" + text.replace("'", "'\''") + "'"


def _repo(ctx: ToolContext, raw: str | None) -> Path | ToolResult:
    from jarvis.capabilities.files import PathDenied, _policy  # noqa: PLC0415

    try:
        p = _policy(ctx).resolve(raw or ".", ctx)
    except PathDenied as e:
        return ToolResult.fail(str(e))
    if not p.is_dir():
        return ToolResult.fail(f"{p} is not a folder")
    return p


def _git_root(p: Path) -> Path | None:
    for d in (p, *p.parents):
        if (d / ".git").exists():
            return d
    return None


def detect(root: Path) -> dict[str, Any]:
    files = {f.name for f in root.iterdir() if f.is_file()}
    info: dict[str, Any] = {"build": [], "test": None, "build_cmd": None}
    if "pyproject.toml" in files or "setup.py" in files or "requirements.txt" in files:
        info["build"].append("python")
        tests_exist = (root / "tests").is_dir() or any(root.glob("test_*.py"))
        venv_py = next((p for p in (root / ".venv/Scripts/python.exe", root / ".venv/bin/python", root / "venv/Scripts/python.exe",
                                    root / "venv/bin/python") if p.exists()), None)
        py = f'"{venv_py}"' if venv_py else "python"
        if tests_exist:
            info["test"] = f"{py} -m pytest -q"
        info["build_cmd"] = f"{py} -m compileall -q ."
    if "package.json" in files:
        info["build"].append("node")
        try:
            scripts = json.loads((root / "package.json").read_text("utf-8")).get("scripts", {})
        except (OSError, ValueError):
            scripts = {}
        if "test" in scripts and "no test specified" not in scripts["test"]:
            info["test"] = info["test"] or "npm test --silent"
        if "build" in scripts:
            info["build_cmd"] = "npm run build"
    if "Cargo.toml" in files:
        info["build"].append("cargo")
        info["test"] = info["test"] or "cargo test"
        info["build_cmd"] = info["build_cmd"] or "cargo build"
    if "go.mod" in files:
        info["build"].append("go")
        info["test"] = info["test"] or "go test ./..."
        info["build_cmd"] = info["build_cmd"] or "go build ./..."
    if any(f.endswith((".sln", ".csproj")) for f in files):
        info["build"].append("dotnet")
        info["test"] = info["test"] or "dotnet test"
        info["build_cmd"] = info["build_cmd"] or "dotnet build"
    if "pom.xml" in files:
        info["build"].append("maven")
        info["test"] = info["test"] or "mvn -q test"
    if "build.gradle" in files or "build.gradle.kts" in files:
        info["build"].append("gradle")
        info["test"] = info["test"] or ("gradlew.bat test" if (root / "gradlew.bat").exists() else "gradle test")
    if "CMakeLists.txt" in files:
        info["build"].append("cmake")
    if "Makefile" in files:
        info["build"].append("make")
    return info


def _summarize_test_output(out: str) -> str:
    patterns = [r"=+ (.*?(?:passed|failed|error).*?) in [\d.]+s", r"(Tests?:\s+.*?total)", r"(test result: \w+\. \d+ passed; \d+ failed)",
                r"^(ok|FAIL)\s+\S+", r"(\d+ passing.*)", r"(Passed!.*|Failed!.*)"]
    for pat in patterns:
        m = re.search(pat, out, re.M | re.I)
        if m:
            return m.group(1).strip()[:200]
    lines = [line for line in out.strip().splitlines() if line.strip()]
    return lines[-1][:200] if lines else ""


def record(ctx: ToolContext, kind: str, command: str, r: dict[str, Any], summary: str) -> dict[str, Any]:
    entry = {"kind": kind, "command": command, "cwd": r["cwd"], "exit_code": r["exit_code"], "timed_out": r["timed_out"],
             "passed": (not r["timed_out"]) and r["exit_code"] == 0, "summary": summary, "at": time.time()}
    scope = f"conversation:{ctx.conversation_id}" if ctx.conversation_id else f"task:{ctx.task_id}"
    mem = ctx.runtime.memory
    ledger = mem.working_get(scope).get("verification") or []
    ledger.append(entry)
    mem.working_set(scope, "verification", ledger[-20:], ttl_s=7 * 86400)
    return entry


# --------------------------------------------------------------------- tools
async def _overview(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    root = _repo(ctx, args.get("path"))
    if isinstance(root, ToolResult):
        return root
    langs: Counter[str] = Counter()
    n_files = 0
    tree: list[str] = []
    for f in root.rglob("*"):
        rel = f.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts):
            continue
        if f.is_file():
            n_files += 1
            if f.suffix.lower() in LANG:
                langs[LANG[f.suffix.lower()]] += 1
        if len(rel.parts) <= 2 and len(tree) < 80:
            tree.append(("  " * (len(rel.parts) - 1)) + rel.name + ("/" if f.is_dir() else ""))
        if n_files > 20000:
            break
    info = detect(root)
    git_root = _git_root(root)
    git_info = ""
    if git_root and shutil.which("git"):
        r = await run_command("git status --short --branch", git_root, 20)
        git_info = r["stdout"].strip()[:1500]
    readme = next((root / n for n in ("README.md", "README.rst", "README.txt", "README") if (root / n).exists()), None)
    lines = [f"Repository: {root}", f"Files: {n_files}" + (" (stopped counting at 20000)" if n_files > 20000 else ""),
             "Languages: " + (", ".join(f"{k} ({v})" for k, v in langs.most_common(8)) or "none detected"),
             "Build systems: " + (", ".join(info["build"]) or "none detected"),
             f"Test command: {info['test'] or 'none detected'}", f"Build command: {info['build_cmd'] or 'none detected'}",
             f"Git: {git_root if git_root else 'not a git repository'}"]
    if git_info:
        lines.append(git_info)
    lines.append("Layout:\n" + "\n".join(tree))
    if readme:
        lines.append(f"{readme.name} (start):\n" + readme.read_text("utf-8", errors="replace")[:1500])
    return ToolResult(ok=True, content="\n".join(lines), data={"root": str(root), "languages": dict(langs), **info,
                                                              "git_root": str(git_root) if git_root else None})


async def _run_tests(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    root = _repo(ctx, args.get("path"))
    if isinstance(root, ToolResult):
        return root
    cmd = args.get("command") or detect(root)["test"]
    if not cmd:
        return ToolResult.fail("no test command could be detected; pass one explicitly")
    r = await run_command(cmd, root, float(args.get("timeout_s", 600)), cancel=ctx.cancel)
    out = (r["stdout"] + "\n" + r["stderr"]).strip()
    summary = _summarize_test_output(out)
    entry = record(ctx, "test", cmd, r, summary)
    status = "timed out" if r["timed_out"] else ("PASSED" if entry["passed"] else f"FAILED (exit code {r['exit_code']})")
    tail = out[-6000:]
    return ToolResult(ok=entry["passed"], content=f"$ {cmd}\n{status} — {summary}\n\n{tail}", data=entry,
                      error=None if entry["passed"] else status)


async def _run_build(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    root = _repo(ctx, args.get("path"))
    if isinstance(root, ToolResult):
        return root
    cmd = args.get("command") or detect(root)["build_cmd"]
    if not cmd:
        return ToolResult.fail("no build command could be detected; pass one explicitly")
    r = await run_command(cmd, root, float(args.get("timeout_s", 900)), cancel=ctx.cancel)
    out = (r["stdout"] + "\n" + r["stderr"]).strip()
    entry = record(ctx, "build", cmd, r, _summarize_test_output(out) or ("succeeded" if r["exit_code"] == 0 else "failed"))
    status = "timed out" if r["timed_out"] else ("SUCCEEDED" if entry["passed"] else f"FAILED (exit code {r['exit_code']})")
    return ToolResult(ok=entry["passed"], content=f"$ {cmd}\n{status}\n\n{out[-6000:]}", data=entry, error=None if entry["passed"] else status)


async def _git(args: dict[str, Any], ctx: ToolContext, sub: str) -> ToolResult:
    root = _repo(ctx, args.get("path"))
    if isinstance(root, ToolResult):
        return root
    git_root = _git_root(root)
    if git_root is None:
        return ToolResult.fail(f"{root} is not inside a git repository")
    if not shutil.which("git"):
        return ToolResult.fail("git is not installed")
    if sub == "status":
        cmd = "git status --short --branch"
    elif sub == "diff":
        cmd = "git diff --stat" if args.get("stat") else "git diff"
        if args.get("staged"):
            cmd += " --cached"
        if args.get("file"):
            cmd += f' -- "{args["file"]}"'
    elif sub == "log":
        cmd = f"git log --oneline -n {int(args.get('count', 15))}"
    elif sub == "branch":
        name = re.sub(r"[^\w./-]", "-", args["name"])
        cmd = f"git switch -c {name}" if args.get("create", True) else f"git switch {name}"
    elif sub == "commit":
        commit = f"git commit -m {_quote(args['message'])}"
        if args.get("add_all"):
            cmd = f"git add -A; if ($?) {{ {commit} }}" if sys.platform == "win32" else f"git add -A && {commit}"
        else:
            cmd = commit
    else:
        return ToolResult.fail(f"unknown git operation {sub}")
    r = await run_command(cmd, git_root, 60, cancel=ctx.cancel)
    out = (r["stdout"] + ("\n" + r["stderr"] if r["stderr"].strip() else "")).strip() or "(no output)"
    ok = r["exit_code"] == 0
    return ToolResult(ok=ok, content=f"$ {cmd}\n{out[:15000]}", data={"exit_code": r["exit_code"]}, error=None if ok else out[-300:])


def coding_tools() -> list[Tool]:
    kw = ("code", "repo", "repository", "project", "bug", "crash", "error", "exception", "test", "build", "compile", "git", "commit",
          "diff", "branch", "refactor", "function", "class", "stack trace", "traceback", "fix")
    path = {"type": "string", "description": "folder inside the repository (default: workspace)"}
    return [
        Tool("repo_overview", "Inspect a code repository: languages, build system, test command, git state, layout and README.",
             {"type": "object", "properties": {"path": path}}, _overview, Risk.SAFE, "coding.inspect", "coding", keywords=kw),
        Tool("run_tests", "Run the project's tests (auto-detected command, or the one given) and record the result for verification.",
             {"type": "object", "properties": {"path": path, "command": {"type": "string"},
                                               "timeout_s": {"type": "number", "minimum": 10, "maximum": 3600, "default": 600}}},
             _run_tests, Risk.SENSITIVE, "coding.test", "coding", timeout_s=3700, keywords=kw,
             describe=lambda a: f"run the tests{(' with `' + a['command'] + '`') if a.get('command') else ''} in {a.get('path') or 'the workspace'} (runs project code)"),
        Tool("run_build", "Build or compile the project (auto-detected command, or the one given) and record the result.",
             {"type": "object", "properties": {"path": path, "command": {"type": "string"},
                                               "timeout_s": {"type": "number", "minimum": 10, "maximum": 3600, "default": 900}}},
             _run_build, Risk.SENSITIVE, "coding.build", "coding", timeout_s=3700, keywords=kw,
             describe=lambda a: f"build the project{(' with `' + a['command'] + '`') if a.get('command') else ''} (runs project tooling)"),
        Tool("git_status", "Show git branch and changed files.", {"type": "object", "properties": {"path": path}},
             lambda a, c: _git(a, c, "status"), Risk.SAFE, "coding.git.read", "coding", keywords=kw),
        Tool("git_diff", "Show uncommitted changes (or --stat summary, staged changes, one file).",
             {"type": "object", "properties": {"path": path, "stat": {"type": "boolean"}, "staged": {"type": "boolean"}, "file": {"type": "string"}}},
             lambda a, c: _git(a, c, "diff"), Risk.SAFE, "coding.git.read", "coding", keywords=kw),
        Tool("git_log", "Show recent commits.", {"type": "object", "properties": {"path": path, "count": {"type": "integer", "minimum": 1, "maximum": 100}}},
             lambda a, c: _git(a, c, "log"), Risk.SAFE, "coding.git.read", "coding", keywords=kw),
        Tool("git_branch", "Create (default) or switch to a branch.",
             {"type": "object", "properties": {"path": path, "name": {"type": "string"}, "create": {"type": "boolean", "default": True}}, "required": ["name"]},
             lambda a, c: _git(a, c, "branch"), Risk.LOW_RISK, "coding.git.branch", "coding", keywords=kw),
        Tool("git_commit", "Commit staged changes (or all changes with add_all) with a message. Never pushes.",
             {"type": "object", "properties": {"path": path, "message": {"type": "string"}, "add_all": {"type": "boolean", "default": False}},
              "required": ["message"]},
             lambda a, c: _git(a, c, "commit"), Risk.SENSITIVE, "coding.git.commit", "coding", keywords=kw,
             describe=lambda a: f"commit {'all changes' if a.get('add_all') else 'staged changes'} with message “{a['message'][:100]}”"),
    ]


EDIT_TOOLS = {"fs_edit", "fs_write", "fs_move", "fs_delete"}
VERIFY_TOOLS = {"run_tests", "run_build"}


def verification_statement(actions: list[dict[str, Any]]) -> str | None:
    """A factual line about verification, computed from what actually ran."""
    last_edit, last_verify = -1, -1
    for i, a in enumerate(actions):
        path = str(a.get("args", {}).get("path") or a.get("args", {}).get("destination") or "")
        if a["tool"] in EDIT_TOOLS and a["ok"] and Path(path).suffix.lower() in CODE_EXT:
            last_edit = i
        if a["tool"] in VERIFY_TOOLS and a["decision"] in ("allow", "approved"):
            last_verify = i
    if last_edit < 0:
        return None
    if last_verify < last_edit:
        return "Not verified: code was changed but no tests or build ran afterwards."
    v = actions[last_verify]
    cmd = v["args"].get("command") or ("the project's tests" if v["tool"] == "run_tests" else "the build")
    if v["ok"]:
        return f"Verified: {cmd} passed after the last change."
    return f"Verification failed: {cmd} did not pass after the last change ({v.get('error') or 'see output above'})."


def register(runtime: Any) -> None:
    runtime.tools.register_all(coding_tools())
