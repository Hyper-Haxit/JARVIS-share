"""Platform-appropriate filesystem locations for JARVIS state.

Everything JARVIS persists lives under a single *home* directory so that it can
be inspected, backed up, exported or wiped as a unit. The location can always be
overridden with the ``JARVIS_HOME`` environment variable.

Windows: the home is ``%USERPROFILE%\\.jarvis``, not ``%LOCALAPPDATA%``. When a
process is started by an MSIX-packaged app (for example a desktop app that
launches the JARVIS server), Windows silently redirects its writes under
AppData into that package's private folder. JARVIS would then keep a separate
database, memory and device credential per launcher, and a terminal outside
the package could not see the server's credential. The user-profile root is not
virtualized, so every process of the user sees the same JARVIS.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
from pathlib import Path


def default_home() -> Path:
    override = os.environ.get("JARVIS_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        return Path(os.environ.get("USERPROFILE") or Path.home()) / ".jarvis"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "JARVIS"
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "share") / "jarvis"


def legacy_homes() -> list[Path]:
    """Earlier default locations that may still hold data."""
    if sys.platform == "win32" and not os.environ.get("JARVIS_HOME"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return [Path(base) / "JARVIS"]
    return []


def package_identity() -> str | None:
    """Full name of the MSIX package this process runs in, or None.

    Processes inside a package see a virtualized AppData; diagnostics report it.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes  # noqa: PLC0415

        length = ctypes.c_uint32(0)
        rc = ctypes.windll.kernel32.GetCurrentPackageFullName(ctypes.byref(length), None)
        if rc == 15700:  # APPMODEL_ERROR_NO_PACKAGE
            return None
        buf = ctypes.create_unicode_buffer(length.value)
        if ctypes.windll.kernel32.GetCurrentPackageFullName(ctypes.byref(length), buf) == 0:
            return buf.value
    except Exception:  # noqa: BLE001 - older Windows: no package APIs, so not packaged
        return None
    return None


def migrate_legacy_home(home: Path) -> Path | None:
    """Copy data from an earlier default location into ``home`` (once).

    Only runs when ``home`` has no database yet. The database is copied with
    SQLite's backup API so the copy is consistent even with a WAL file present.
    The old folder is left in place, with a note saying where the data went.
    Returns the location migrated from, if any.
    """
    if (home / "jarvis.db").exists():
        return None
    for old in legacy_homes():
        if not (old / "jarvis.db").exists() or old.resolve() == home.resolve():
            continue
        home.mkdir(parents=True, exist_ok=True)
        src = sqlite3.connect(f"file:{old / 'jarvis.db'}?mode=ro", uri=True)
        dst = sqlite3.connect(home / "jarvis.db")
        try:
            src.backup(dst)
        finally:
            src.close()
            dst.close()
        for item in old.iterdir():
            if item.name.startswith("jarvis.db"):
                continue
            target = home / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            elif not target.exists():
                shutil.copy2(item, target)
        for sensitive in ("local_device.token", "secrets.json"):
            if (home / sensitive).exists():
                restrict_permissions(home / sensitive)
        try:
            (old / "MOVED-TO.txt").write_text(f"JARVIS now keeps its data in {home}\nThis folder is no longer used.\n", "utf-8")
        except OSError:
            pass
        return old
    return None


class JarvisPaths:
    """Resolved locations inside a JARVIS home directory."""

    def __init__(self, home: Path | None = None) -> None:
        self.home = (home or default_home()).resolve()

    @property
    def config_file(self) -> Path:
        return self.home / "config.toml"

    @property
    def database(self) -> Path:
        return self.home / "jarvis.db"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def plugins(self) -> Path:
        return self.home / "plugins"

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def workspace(self) -> Path:
        return self.home / "workspace"

    @property
    def captures(self) -> Path:
        return self.home / "captures"

    @property
    def tls(self) -> Path:
        return self.home / "tls"

    @property
    def local_token_file(self) -> Path:
        """Credential of the host's own CLI device (restricted permissions)."""
        return self.home / "local_device.token"

    @property
    def fallback_secrets_file(self) -> Path:
        return self.home / "secrets.json"

    def ensure(self) -> None:
        for p in (self.home, self.logs, self.plugins, self.models, self.workspace, self.captures):
            p.mkdir(parents=True, exist_ok=True)

    def protected_paths(self) -> list[Path]:
        """Paths that tools must never read or write (credentials, database)."""
        return [
            self.local_token_file,
            self.fallback_secrets_file,
            self.database,
            self.home / "jarvis.db-wal",
            self.home / "jarvis.db-shm",
            self.tls,
            self.config_file,
        ]


def restrict_permissions(path: Path) -> None:
    """Best-effort: make a file readable only by the current user."""
    try:
        if sys.platform == "win32":
            # On Windows the user profile directory is already private to the
            # user; additionally strip inherited ACLs via icacls when available.
            import subprocess

            user = os.environ.get("USERNAME")
            if user:
                subprocess.run(
                    ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
        else:
            os.chmod(path, 0o600)
    except Exception:  # pragma: no cover - permission hardening is best effort
        pass
