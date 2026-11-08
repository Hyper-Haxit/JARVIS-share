"""`jarvis auth ...`: how this terminal identifies itself to JARVIS.

Tokens are never printed. Pairing codes are single-use and short-lived; the
device credential they produce is stored in the OS credential store.
"""

from __future__ import annotations

import argparse
import getpass
import platform

import httpx

from jarvis.clients.cli import extra_commands
from jarvis.clients.cli.client import JarvisClient, NotAuthenticated, default_base_url
from jarvis.clients.cli.credentials import PairingStore, device_name, normalize_url, resolve


def _c(t: str, code: str) -> str:
    import sys  # noqa: PLC0415

    return f"\033[{code}m{t}\033[0m" if sys.stdout.isatty() else t


def cmd_auth(args: argparse.Namespace) -> int:
    url = (args.url or default_base_url()).rstrip("/")
    if args.action == "status":
        return _status(url)
    if args.action == "login":
        return _login(url, args.code, args.name)
    if args.action == "logout":
        return _logout(url)
    return 2


def _status(url: str) -> int:
    cred = resolve(url)
    print(f"JARVIS core   {url}")
    if cred is None:
        print("credential    none found")
        raise NotAuthenticated("no credential", url, None)
    print(f"credential    {cred.source}")
    me = JarvisClient(url).get("/auth/whoami")
    d = me["device"] or {}
    print(f"device        {d.get('name')} ({d.get('id')})")
    print(f"access        {d.get('role')}, up to {str(d.get('max_risk', '')).lower().replace('_', ' ')} actions")
    print(_c("status        connected", "32"))
    return 0


def _login(url: str, code: str | None, name: str | None) -> int:
    store = PairingStore()
    try:
        health = httpx.get(f"{url}/api/v1/health", timeout=5)
        health.raise_for_status()
    except httpx.HTTPError:
        print(_c(f"Can't reach JARVIS at {url}. Start it (or check the address), then try again.", "31"))
        return 3
    if not code:
        print("Get a pairing code from a device that's already connected:")
        print("  jarvis devices pair --kind cli --role owner     (or Settings > Devices in the web app)")
        code = getpass.getpass("Pairing code (input hidden): ").strip()
    if not code:
        print("No code entered; nothing changed.")
        return 1
    resp = httpx.post(f"{url}/api/v1/auth/pair", timeout=15, json={
        "code": code, "name": name or device_name(), "kind": "cli", "platform": platform.system().lower(),
    })
    if resp.status_code == 429:
        print(_c("Too many pairing attempts; wait a minute and try again.", "31"))
        return 1
    if resp.status_code != 200:
        print(_c("That pairing code was not accepted: it may be mistyped, already used, or expired. Create a new one and try again.", "31"))
        return 5
    data = resp.json()
    device = data["device"]
    backend = store.save(url, data["device_token"], device)
    where = "the system credential store" if backend != "file" else f"a user-only file in {store.paths.home}"
    print(_c(f"Paired. This terminal is now '{device['name']}' ({device['role']}, up to "
             f"{device['max_risk'].lower().replace('_', ' ')} actions).", "32"))
    print(f"The credential is stored in {where}; revoke it anytime under Devices or with `jarvis auth logout`.")
    return 0


def _logout(url: str) -> int:
    store = PairingStore()
    paired = store.get(url)
    if paired is None:
        cred = resolve(url)
        if cred is not None and cred.kind == "local":
            print("This terminal uses this computer's own credential, which JARVIS manages itself; there is nothing to log out of.")
        else:
            print(f"This terminal isn't paired with {normalize_url(url)}.")
        return 0
    token, info = paired
    revoked = False
    try:
        r = httpx.post(f"{url}/api/v1/auth/device/forget", headers={"Authorization": f"Bearer {token}"}, timeout=10)
        revoked = r.status_code == 200
    except httpx.HTTPError:
        pass
    store.delete(url)
    print(f"Removed this terminal's credential for {normalize_url(url)}.")
    if not revoked:
        print(_c(f"JARVIS couldn't be told to revoke '{info['device_name']}'; revoke it under Devices to be sure.", "33"))
    return 0


def _register(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser("auth", help="connect this terminal to JARVIS (status, login, logout)")
    s.add_argument("action", choices=["status", "login", "logout"])
    s.add_argument("--url", help="JARVIS address (default: this computer)")
    s.add_argument("--code", help="pairing code (prompted for, hidden, if omitted)")
    s.add_argument("--name", help="device name to register (default: CLI on <computer>)")
    s.set_defaults(func=cmd_auth)


extra_commands.add(_register)
