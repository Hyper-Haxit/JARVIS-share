"""`jarvis` command-line interface.

Most commands talk to a running core over the API (so they work identically
against a remote core). `start`, `doctor --offline`, `setup` and `secret` work
locally without a running server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from jarvis import __version__
from jarvis.clients.cli.client import ApiError, CoreUnavailable, Forbidden, JarvisClient, NotAuthenticated

# ----------------------------------------------------------------- terminal
if sys.platform == "win32":
    os.system("")  # enable ANSI escape processing on Windows consoles
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(t: str) -> str:
    return c(t, "2")


def cyan(t: str) -> str:
    return c(t, "36")


def green(t: str) -> str:
    return c(t, "32")


def yellow(t: str) -> str:
    return c(t, "33")


def red(t: str) -> str:
    return c(t, "31")


def bold(t: str) -> str:
    return c(t, "1")


STATUS_COLORS = {"ok": green, "warn": yellow, "fail": red, "skip": dim}


def table(rows: list[dict[str, Any]], cols: list[str]) -> None:
    if not rows:
        print(dim("(none)"))
        return
    widths = {col: max(len(col), *(len(str(r.get(col, ""))[:60]) for r in rows)) for col in cols}
    print("  ".join(bold(col.upper().ljust(widths[col])) for col in cols))
    for r in rows:
        print("  ".join(str(r.get(col, ""))[:60].ljust(widths[col]) for col in cols))


def ts(t: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "-"


# ----------------------------------------------------------------- commands
def cmd_start(args: argparse.Namespace) -> int:
    from jarvis.api.server import InsecureBindError, serve
    from jarvis.runtime import JarvisRuntime

    rt = JarvisRuntime(load_plugins=not args.no_plugins)
    host = args.host or rt.config.server.host
    port = args.port or rt.config.server.port
    scheme = "https" if rt.config.server.tls_cert else "http"
    url = f"{scheme}://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    print(cyan(bold("JARVIS")) + f" core v{__version__} starting on {url}  " + dim(f"(home: {rt.paths.home})"))
    if rt.enrollment.first_run() and not args.open:
        print(f"First time? Open {url} in your browser and choose \"Set up this device\" "
              "(or run `jarvis open`).")
    if args.open:
        def opener() -> None:
            client = JarvisClient(url, token=None)
            for _ in range(100):
                if client.health():
                    break
                time.sleep(0.2)
            _open_ui(url)

        threading.Thread(target=opener, daemon=True).start()
    try:
        serve(rt, host, port)
    except InsecureBindError as e:
        print(red(str(e)))
        return 2
    except KeyboardInterrupt:
        pass
    return 0


def _open_ui(url: str | None = None, path: str = "/") -> int:
    client = JarvisClient(url)
    try:
        out = client.post("/auth/pairing-codes", {"kind": "web", "role": "owner", "max_risk": "DANGEROUS",
                                                  "name_hint": "Browser on this computer"})
    except (NotAuthenticated, Forbidden) as e:
        print(red(e.guidance()))
        return 5 if isinstance(e, NotAuthenticated) else 6
    except (CoreUnavailable, ApiError) as e:
        print(red(str(e)))
        return 1
    target = f"{client.base_url}{path}#pair={out['code']}"
    webbrowser.open(target)
    print(f"Opened {client.base_url}{path} " + dim("(single-use pairing code expires in a few minutes)"))
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    return _open_ui(path="/hud" if args.hud else "/")


def cmd_status(args: argparse.Namespace) -> int:
    client = JarvisClient()
    st = client.get("/system/status")
    if args.json:
        print(json.dumps(st, indent=2))
        return 0
    print(cyan(bold(st["name"])) + f"  v{st['version']}  node {st['node_name']} ({st['node_id']})")
    print(f"uptime {st['uptime_s']}s   routing {st['routing_mode']}   tools {st['tools']}   "
          f"pending approvals {st['pending_approvals']}   secrets: {st['secret_backend']}")
    for p in st["providers"]:
        state = green("online") if p["ok"] else red(f"offline ({p['error']})")
        print(f"  provider {p['provider']:<12} {'local' if p['local'] else 'cloud':<6} {state}")
    print(f"memory {st['memory']}   embeddings {st['embedding']}")
    for name, ext in st.get("extensions", {}).items():
        print(f"  {name}: {json.dumps(ext)[:160]}")
    return 0


def _render_chat(client: JarvisClient, message: str, conv: str | None, args: argparse.Namespace) -> str | None:
    extra: dict[str, Any] = {}
    if getattr(args, "role", None):
        extra["role"] = args.role
    if getattr(args, "model", None):
        extra["model"] = args.model
    started_text = False
    for event, data in client.chat_stream(message, conv, **extra):
        if event == "start":
            conv = data["conversation_id"]
        elif event == "delta":
            if not started_text:
                print(cyan("JARVIS ") , end="", flush=True)
                started_text = True
            print(data["text"], end="", flush=True)
        elif event == "tool_start":
            if started_text:
                print()
                started_text = False
            print(dim(f"  > {data['tool']} {json.dumps(data.get('args', {}))[:120]}"))
        elif event == "tool_end":
            mark = green("ok") if data["ok"] else red(data.get("decision") if not data["ok"] else "failed")
            print(dim(f"  < {data['tool']} [{mark}{dim(']')} {(data.get('summary') or data.get('error') or '')[:140]}"))
        elif event == "approval":
            if started_text:
                print()
                started_text = False
            print(yellow(f"\n  APPROVAL REQUIRED [{data['risk']}] {data['summary']}"))
            if data.get("reason"):
                print(dim(f"  why: {data['reason'][:200]}"))
            try:
                answer = input(yellow("  approve? [y]es once / [c]onversation / [N]o: ")).strip().lower()
            except EOFError:
                answer = "n"
            approve = answer in ("y", "yes", "c")
            scope = "conversation" if answer == "c" else "once"
            try:
                client.post(f"/permissions/approvals/{data['approval_id']}", {"approve": approve, "scope": scope})
            except ApiError as e:
                print(red(f"  could not record decision: {e}"))
        elif event == "approval_decided":
            pass
        elif event == "error":
            print(red(f"\nerror: {data.get('error')}"))
        elif event == "done":
            if started_text:
                print()
            route = data.get("route") or {}
            if not started_text and data.get("text"):
                print(cyan("JARVIS ") + data["text"])
            meta = f"{route.get('provider')}/{route.get('model')} · {data.get('duration_ms', 0) / 1000:.1f}s"
            if data.get("actions"):
                meta += f" · {len(data['actions'])} action(s)"
            if route.get("degraded"):
                meta += " · " + yellow("degraded routing")
            print(dim(f"  [{meta}]"))
    return conv


def cmd_chat(args: argparse.Namespace) -> int:
    client = JarvisClient()
    conv = args.conversation
    print(dim("Talking to JARVIS. Ctrl+C or /exit to quit, /new for a new conversation."))
    while True:
        try:
            line = input(bold("you  "))
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        line = line.strip()
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/new":
            conv = None
            print(dim("new conversation"))
            continue
        try:
            conv = _render_chat(client, line, conv, args)
        except KeyboardInterrupt:
            if conv:
                client.post(f"/chat/{conv}/cancel")
            print(yellow("\ncancelled"))


def cmd_ask(args: argparse.Namespace) -> int:
    _render_chat(JarvisClient(), " ".join(args.question), args.conversation, args)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    results: list[dict[str, Any]]
    client = JarvisClient()
    if not args.offline and client.health():
        results = client.get("/system/diagnostics")
        source = f"running core at {client.base_url}"
    else:
        import asyncio

        from jarvis.core.diagnostics import run_diagnostics
        from jarvis.runtime import JarvisRuntime

        rt = JarvisRuntime(configure_logging=False, load_plugins=False)
        token_file = rt.paths.local_token_file
        rt.local_token = token_file.read_text("utf-8").strip() if token_file.exists() else None

        async def go() -> list[dict[str, Any]]:
            try:
                return await run_diagnostics(rt)
            finally:
                await rt.router.close()

        results = asyncio.run(go())
        source = "local inspection (core not running)"
    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    print(bold("JARVIS doctor") + dim(f" — {source}"))
    group = None
    for r in results:
        if r["group"] != group:
            group = r["group"]
            print("\n" + bold(group.upper()))
        mark = STATUS_COLORS.get(r["status"], str)(r["status"].upper().ljust(4))
        print(f"  {mark}  {r['name']:<32} {r['detail']}")
        if r.get("fix") and r["status"] in ("warn", "fail"):
            print(dim(f"        fix: {r['fix']}"))
    fails = sum(1 for r in results if r["status"] == "fail")
    warns = sum(1 for r in results if r["status"] == "warn")
    print(f"\n{len(results)} checks: " + (red(f"{fails} failing") if fails else green("0 failing")) + f", {warns} warnings")
    return 1 if fails else 0


def cmd_devices(args: argparse.Namespace) -> int:
    client = JarvisClient()
    action = args.action or "list"
    if action == "list":
        rows = client.get("/devices")
        for r in rows:
            r["status"] = "revoked" if r["revoked"] else ("online" if r["online"] else "offline")
            r["last_seen"] = ts(r["last_seen"])
            r["this"] = "*" if r["current"] else ""
        table(rows, ["this", "id", "name", "kind", "role", "max_risk", "status", "last_seen"])
    elif action == "pair":
        out = client.post("/auth/pairing-codes", {"kind": args.kind, "role": args.role, "max_risk": args.max_risk,
                                                  "name_hint": args.name})
        print(f"Pairing code: {bold(out['code'])}  (role {out['role']}, max risk {out['max_risk']}, "
              f"expires {ts(out['expires_at'])})")
        print(f"Browser/phone URL: {out['url']}")
    elif action == "revoke":
        if not args.target:
            print(red("device id required"))
            return 2
        print(client.post(f"/devices/{args.target}/revoke"))
    elif action in ("requests", "approve", "deny"):
        return _setup_requests(client, action, args)
    return 0


def _setup_requests(client: JarvisClient, action: str, args: argparse.Namespace) -> int:
    """Browsers on this computer waiting to be set up (see "Set up this device" in the web app)."""
    pending = client.get("/auth/setup/requests")
    if not pending:
        print("No browser is waiting to be set up.")
        return 0 if action == "requests" else 1
    if action == "requests" or (not args.target and len(pending) > 1):
        for r in pending:
            print(f"{r['id']}  {r['name']}  code {r['code']}  (asked {ts(r['created_at'])})")
        if action != "requests":
            print(f"Several are waiting: run `jarvis devices {action} <id>`.")
            return 2
        return 0
    req = next((r for r in pending if r["id"] == args.target), None) if args.target else pending[0]
    if req is None:
        print(red(f"No waiting request {args.target}."))
        return 1
    if action == "approve" and not args.yes:
        print(f"{req['name']} on this computer wants full access to JARVIS.")
        print(f"Check that it shows the code {bold(req['code'])}.")
        if input("Approve it? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Not approved.")
            return 1
    client.post(f"/auth/setup/requests/{req['id']}/{action}")
    print(green("Approved: the browser is set up.") if action == "approve" else "Denied: the browser was not set up.")
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    client = JarvisClient()
    a = args.action or "list"
    if a == "list":
        out = client.get("/memory", params={"layer": args.layer} if args.layer else None)
        print(dim(f"counts {out['counts']} · embeddings {out['embedding']}"))
        table(out["items"], ["id", "layer", "kind", "content"])
    elif a == "search":
        out = client.get("/memory", params={"q": " ".join(args.rest)})
        table(out["items"], ["id", "layer", "score", "content"])
    elif a == "add":
        print(client.post("/memory", {"content": " ".join(args.rest)}))
    elif a == "forget":
        print(client.post("/memory/forget", {"ids": args.rest}))
    elif a == "export":
        data = client.get("/memory/export")
        Path(args.rest[0] if args.rest else "jarvis-memory-export.json").write_text(json.dumps(data, indent=2), "utf-8")
        print(f"exported {len(data['memories'])} memories")
    elif a == "import":
        print(client.post("/memory/import", json.loads(Path(args.rest[0]).read_text("utf-8"))))
    elif a == "ingest":
        print(client.post("/memory/ingest", {"path": str(Path(args.rest[0]).resolve())}))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    client = JarvisClient()
    a = args.action or "list"
    if a == "list":
        out = client.get("/models")
        for p in out["providers"]:
            state = green("online") if p["health"]["ok"] else red(f"offline: {p['health']['error']}")
            print(bold(p["name"]) + f" ({p['type']}, {'local' if p['local'] else 'cloud'}) {state}")
            for m in p["models"]:
                print(f"   {m['name']:<28} {', '.join(m.get('capabilities') or [])}")
        print(bold("\nrouting") + f" mode={out['routing']['mode']}")
        for role, info in out["routing"]["roles"].items():
            print(f"   {role:<10} {' -> '.join(info['candidates'])}" + (yellow("  (override)") if info["overridden"] else ""))
    elif a == "mode":
        print(client.put("/models/routing", {"mode": args.rest[0]}))
    elif a == "set-role":
        role, refs = args.rest[0], args.rest[1:]
        print(client.put("/models/routing", {"roles": {role: refs or None}}))
    elif a == "health":
        for h in client.post("/models/health"):
            print(f"{h['provider']:<12} {'ok' if h['ok'] else 'DOWN'} {h.get('latency_ms') or ''} {h.get('error') or ''}")
    return 0


def cmd_permissions(args: argparse.Namespace) -> int:
    client = JarvisClient()
    a = args.action or "pending"
    if a == "pending":
        rows = client.get("/permissions/approvals")
        table(rows, ["id", "risk", "tool", "summary"])
    elif a == "history":
        table(client.get("/permissions/approvals", params={"state": "all"}), ["id", "status", "risk", "tool", "summary"])
    elif a in ("approve", "deny"):
        print(client.post(f"/permissions/approvals/{args.rest[0]}", {"approve": a == "approve", "scope": args.scope}))
    elif a == "policy":
        print(json.dumps(client.get("/permissions/policy"), indent=2))
    elif a == "grants":
        table(client.get("/permissions/grants"), ["id", "tool", "scope_type", "scope_id"])
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    client = JarvisClient()
    after = 0
    patterns = args.types
    first = True
    while True:
        params: dict[str, Any] = {"after": after, "limit": args.limit if first else 200}
        if patterns:
            params["types"] = patterns
        for e in client.get("/events/recent", params=params):
            after = max(after, e["seq"])
            data = json.dumps(e["data"], default=str)
            print(dim(time.strftime("%H:%M:%S", time.localtime(e["ts"]))) + " " + cyan(e["type"].ljust(26)) + " " + data[:200])
        first = False
        if not args.follow:
            return 0
        time.sleep(1.0)


def cmd_identity(args: argparse.Namespace) -> int:
    client = JarvisClient()
    if args.set:
        patch: dict[str, Any] = {}
        for kv in args.set:
            k, _, v = kv.partition("=")
            patch[k.strip()] = [x.strip() for x in v.split(";") if x.strip()] if k.strip() == "behaviors" else v.strip()
        print(json.dumps(client.put("/system/identity", patch), indent=2))
    else:
        print(json.dumps(client.get("/system/identity"), indent=2))
    return 0


def cmd_secret(args: argparse.Namespace) -> int:
    import getpass

    from jarvis.config.paths import JarvisPaths
    from jarvis.config.secrets import SecretStore

    paths = JarvisPaths()
    paths.ensure()
    store = SecretStore(paths.fallback_secrets_file)
    if args.action == "set":
        value = getpass.getpass(f"value for '{args.name}' (input hidden): ")
        if not value:
            print(red("empty value; nothing stored"))
            return 1
        backend = store.set(args.name, value)
        print(f"stored '{args.name}' in {backend}" + ("" if store.secure else yellow(" (plain file - no OS keyring available)")))
    elif args.action == "delete":
        print("deleted" if store.delete(args.name) else "not found")
    elif args.action == "check":
        print("present" if store.has(args.name) else "missing")
    return 0


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jarvis", description="JARVIS personal AI system")
    p.add_argument("--version", action="version", version=f"jarvis {__version__}")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("start", help="start the JARVIS core server")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--open", action="store_true", help="open the web interface once started")
    s.add_argument("--no-plugins", action="store_true")
    s.set_defaults(func=cmd_start)

    s = sub.add_parser("open", help="open the web interface (pairs this browser automatically)")
    s.add_argument("--hud", action="store_true", help="open the HUD view")
    s.set_defaults(func=cmd_open)

    s = sub.add_parser("status", help="show core status")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    for name, fn, hlp in (("chat", cmd_chat, "interactive conversation"), ("ask", cmd_ask, "ask a single question")):
        s = sub.add_parser(name, help=hlp)
        if name == "ask":
            s.add_argument("question", nargs="+")
        s.add_argument("--conversation")
        s.add_argument("--role", choices=["fast", "reasoning", "coding", "vision", "fallback"])
        s.add_argument("--model", help="provider/model override")
        s.set_defaults(func=fn)

    s = sub.add_parser("doctor", help="diagnose the installation")
    s.add_argument("--json", action="store_true")
    s.add_argument("--offline", action="store_true", help="inspect locally even if a core is running")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("devices", help="list, pair, approve or revoke devices")
    s.add_argument("action", nargs="?", choices=["list", "pair", "revoke", "requests", "approve", "deny"])
    s.add_argument("target", nargs="?")
    s.add_argument("--kind", default="mobile")
    s.add_argument("--role", default="client", choices=["client", "owner", "hardware"])
    s.add_argument("--max-risk", default="SENSITIVE", choices=["SAFE", "LOW_RISK", "SENSITIVE", "DANGEROUS"])
    s.add_argument("--name")
    s.add_argument("--yes", action="store_true", help="approve without asking")
    s.set_defaults(func=cmd_devices)

    s = sub.add_parser("memory", help="inspect and manage memory")
    s.add_argument("action", nargs="?", choices=["list", "search", "add", "forget", "export", "import", "ingest"])
    s.add_argument("rest", nargs="*")
    s.add_argument("--layer")
    s.set_defaults(func=cmd_memory)

    s = sub.add_parser("models", help="model providers and routing")
    s.add_argument("action", nargs="?", choices=["list", "mode", "set-role", "health"])
    s.add_argument("rest", nargs="*")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("permissions", help="approvals, grants and policy")
    s.add_argument("action", nargs="?", choices=["pending", "history", "approve", "deny", "policy", "grants"])
    s.add_argument("rest", nargs="*")
    s.add_argument("--scope", default="once", choices=["once", "conversation", "task", "global"])
    s.set_defaults(func=cmd_permissions)

    s = sub.add_parser("logs", help="show the event log")
    s.add_argument("-f", "--follow", action="store_true")
    s.add_argument("--types", help="comma-separated patterns, e.g. TOOL_*,PERMISSION_*")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(func=cmd_logs)

    s = sub.add_parser("identity", help="show or change JARVIS's personality")
    s.add_argument("--set", nargs="+", metavar="KEY=VALUE")
    s.set_defaults(func=cmd_identity)

    s = sub.add_parser("secret", help="store API keys/tokens in the OS credential store")
    s.add_argument("action", choices=["set", "delete", "check"])
    s.add_argument("name")
    s.set_defaults(func=cmd_secret)

    from jarvis.clients.cli import auth_commands, extra_commands, voice_commands  # noqa: F401 - register `auth`, `voice`

    extra_commands.register(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except CoreUnavailable as e:
        print(red(str(e)))
        return 3
    except NotAuthenticated as e:
        print(red(e.guidance()))
        return 5
    except Forbidden as e:
        print(red(e.guidance()))
        return 6
    except ApiError as e:
        print(red(str(e)))
        return 4
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
