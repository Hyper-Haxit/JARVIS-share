"""Additional CLI command groups registered by later subsystems."""

from __future__ import annotations

from typing import Any

_registrars: list[Any] = []


def add(registrar: Any) -> None:
    _registrars.append(registrar)


def register(sub: Any) -> None:
    for r in _registrars:
        r(sub)
