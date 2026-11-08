"""Secret storage.

Resolution order for ``get(name)``:
    1. environment variable ``JARVIS_SECRET_<NAME>``
    2. the OS credential store via ``keyring`` (Windows Credential Manager,
       macOS Keychain, Secret Service on Linux)
    3. a fallback JSON file with restricted permissions (reported as a warning by
       ``jarvis doctor`` because it is not encrypted at rest)

Every value that is read is registered with the log redactor so it can never be
written to logs, events or audit records.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from jarvis.config.paths import restrict_permissions
from jarvis.core.observability import redactor

SERVICE_PREFIX = "jarvis"


def _env_name(name: str) -> str:
    return "JARVIS_SECRET_" + "".join(c if c.isalnum() else "_" for c in name).upper()


class SecretStore:
    def __init__(self, fallback_file: Path, service: str | None = None, use_keyring: bool = True) -> None:
        self.fallback_file = fallback_file
        self.service = service or f"{SERVICE_PREFIX}:{fallback_file.parent}"
        self._lock = threading.Lock()
        self._keyring = None
        # JARVIS_SECRETS_BACKEND=file keeps secrets in the protected file (headless machines, tests).
        if os.environ.get("JARVIS_SECRETS_BACKEND", "").lower() == "file":
            use_keyring = False
        if use_keyring:
            try:
                import keyring  # noqa: PLC0415
                from keyring.backends import fail  # noqa: PLC0415

                backend = keyring.get_keyring()
                if not isinstance(backend, fail.Keyring):
                    self._keyring = keyring
            except Exception:
                self._keyring = None

    @property
    def backend_name(self) -> str:
        if self._keyring is not None:
            return type(self._keyring.get_keyring()).__name__
        return "file"

    @property
    def secure(self) -> bool:
        return self._keyring is not None

    def _read_file(self) -> dict[str, str]:
        if not self.fallback_file.exists():
            return {}
        try:
            return json.loads(self.fallback_file.read_text("utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_file(self, data: dict[str, str]) -> None:
        self.fallback_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.fallback_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), "utf-8")
        restrict_permissions(tmp)
        tmp.replace(self.fallback_file)

    def get(self, name: str) -> str | None:
        value = os.environ.get(_env_name(name))
        if not value and self._keyring is not None:
            try:
                value = self._keyring.get_password(self.service, name)
            except Exception:
                value = None
        if not value:
            with self._lock:
                value = self._read_file().get(name)
        if value:
            redactor.register(value)
        return value or None

    def set(self, name: str, value: str) -> str:
        """Store a secret; returns the backend used."""
        redactor.register(value)
        if self._keyring is not None:
            try:
                self._keyring.set_password(self.service, name, value)
                return self.backend_name
            except Exception:
                pass
        with self._lock:
            data = self._read_file()
            data[name] = value
            self._write_file(data)
        return "file"

    def delete(self, name: str) -> bool:
        removed = False
        if self._keyring is not None:
            try:
                self._keyring.delete_password(self.service, name)
                removed = True
            except Exception:
                pass
        with self._lock:
            data = self._read_file()
            if name in data:
                del data[name]
                self._write_file(data)
                removed = True
        return removed

    def has(self, name: str) -> bool:
        return self.get(name) is not None
