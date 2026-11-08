"""How the CLI proves who it is to a JARVIS core.

Every request carries a device credential (bearer token). The CLI finds one in
this order:

  1. ``JARVIS_TOKEN`` environment variable (explicit, for scripts);
  2. a credential from ``jarvis auth login`` for that server (pairing with a
     single-use code, exactly like a browser or phone). The token is stored in
     the OS credential store (Windows Credential Manager, macOS Keychain, Secret
     Service) and never printed;
  3. this computer's own owner credential, which the core writes to a
     user-only file in its home folder when it starts. Used only for a core on
     this computer (loopback address) whose home folder this CLI shares.

Nothing here weakens authentication: without a valid credential the core
refuses the request, and the CLI explains how to obtain one.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from jarvis.config.paths import JarvisPaths, restrict_permissions
from jarvis.config.secrets import SecretStore


@dataclass
class Credential:
    token: str
    source: str  # human description of where it came from
    kind: str    # env | paired | local


def normalize_url(url: str) -> str:
    p = urlparse(url.strip().rstrip("/"))
    host = (p.hostname or "").lower()
    if host in ("localhost", "::1"):
        host = "127.0.0.1"
    port = p.port or (443 if p.scheme == "https" else 80)
    return f"{p.scheme}://{host}:{port}"


def is_loopback(url: str) -> bool:
    return (urlparse(url).hostname or "").lower() in ("127.0.0.1", "localhost", "::1")


class PairingStore:
    """Credentials from ``jarvis auth login``, one per server URL."""

    def __init__(self, paths: JarvisPaths | None = None) -> None:
        self.paths = paths or JarvisPaths()
        self.meta_file = self.paths.home / "cli_pairings.json"
        self.secrets = SecretStore(self.paths.fallback_secrets_file, service=f"jarvis-cli:{self.paths.home}")

    def _meta(self) -> dict[str, dict]:
        try:
            return json.loads(self.meta_file.read_text("utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_meta(self, meta: dict[str, dict]) -> None:
        self.meta_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.meta_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2), "utf-8")
        tmp.replace(self.meta_file)

    def get(self, url: str) -> tuple[str, dict] | None:
        key = normalize_url(url)
        info = self._meta().get(key)
        if not info:
            return None
        token = self.secrets.get(f"cli-device:{key}")
        return (token, info) if token else None

    def save(self, url: str, token: str, device: dict) -> str:
        key = normalize_url(url)
        backend = self.secrets.set(f"cli-device:{key}", token)
        meta = self._meta()
        meta[key] = {"device_id": device["id"], "device_name": device["name"], "role": device["role"],
                     "max_risk": device["max_risk"], "paired_at": time.time(), "stored_in": backend}
        self._write_meta(meta)
        if backend == "file":
            restrict_permissions(self.paths.fallback_secrets_file)
        return backend

    def delete(self, url: str) -> dict | None:
        key = normalize_url(url)
        meta = self._meta()
        info = meta.pop(key, None)
        self.secrets.delete(f"cli-device:{key}")
        self._write_meta(meta)
        return info


def resolve(url: str, paths: JarvisPaths | None = None) -> Credential | None:
    paths = paths or JarvisPaths()
    env = os.environ.get("JARVIS_TOKEN")
    if env:
        return Credential(env.strip(), "the JARVIS_TOKEN environment variable", "env")
    paired = PairingStore(paths).get(url)
    if paired:
        token, info = paired
        return Credential(token, f"this terminal's pairing as '{info['device_name']}'", "paired")
    if is_loopback(url) and paths.local_token_file.exists():
        try:
            token = paths.local_token_file.read_text("utf-8").strip()
        except OSError:
            token = ""
        if token:
            return Credential(token, f"this computer's own credential ({paths.local_token_file})", "local")
    return None


def device_name() -> str:
    return f"CLI on {socket.gethostname()}"
