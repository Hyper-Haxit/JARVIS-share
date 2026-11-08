"""HTTP client used by the CLI (and usable by scripts) to talk to a JARVIS core."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

from jarvis.clients.cli.credentials import Credential, is_loopback
from jarvis.clients.cli.credentials import resolve as resolve_credential
from jarvis.config.paths import JarvisPaths
from jarvis.config.settings import load_config
from jarvis.core.observability import redactor


class CoreUnavailable(RuntimeError):
    pass


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class NotAuthenticated(ApiError):
    """The core did not accept this client's identity (HTTP 401)."""

    def __init__(self, detail: str, base_url: str, credential: Credential | None) -> None:
        super().__init__(401, detail)
        self.base_url = base_url
        self.credential = credential

    def guidance(self) -> str:
        url = self.base_url
        login = "jarvis auth login" + ("" if is_loopback(url) and url.endswith(":7777") else f" --url {url}")
        if self.credential is None:
            lines = [f"This terminal isn't connected to JARVIS at {url}: no device credential was found.",
                     "  - If JARVIS runs on this computer: start it with `jarvis start` (that creates this computer's",
                     f"    credential in {JarvisPaths().home}), then run this command again.",
                     "  - Otherwise pair this terminal: on a device that is already connected run",
                     "    `jarvis devices pair --kind cli --role owner` (or Settings > Devices in the web app),",
                     f"    then run `{login}` here and enter the code."]
        else:
            lines = [f"JARVIS at {url} did not accept {self.credential.source}.",
                     "  It was revoked, has expired, or belongs to a different JARVIS installation."]
            if self.credential.kind == "local":
                lines.append("  If JARVIS was moved or reinstalled, restart it with `jarvis start` so it refreshes this credential,")
                lines.append(f"  or pair this terminal with `{login}`.")
            elif self.credential.kind == "env":
                lines.append("  Unset JARVIS_TOKEN or replace it with a valid device credential.")
            else:
                lines.append(f"  Pair this terminal again with `{login}`.")
        return "\n".join(lines)


class Forbidden(ApiError):
    """Authenticated, but this device may not do that (HTTP 403)."""

    def guidance(self) -> str:
        return (f"This device isn't allowed to do that: {self.detail}.\n"
                "  Owner-only actions need a terminal paired with `--role owner` (see `jarvis auth status`).")


def default_base_url(paths: JarvisPaths | None = None) -> str:
    if os.environ.get("JARVIS_URL"):
        return os.environ["JARVIS_URL"].rstrip("/")
    paths = paths or JarvisPaths()
    cfg = load_config(paths)
    scheme = "https" if cfg.server.tls_cert else "http"
    host = "127.0.0.1" if cfg.server.host in ("0.0.0.0", "::") else cfg.server.host
    return f"{scheme}://{host}:{cfg.server.port}"


class JarvisClient:
    """Talks to a JARVIS core as one authenticated device.

    ``token`` overrides credential discovery; ``token=""`` sends no credential.
    """

    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or default_base_url()).rstrip("/")
        if token is None:
            self.credential = resolve_credential(self.base_url)
        else:
            self.credential = Credential(token, "the credential given to this client", "env") if token else None
        self.token = self.credential.token if self.credential else None
        if self.token:
            redactor.register(self.token)
        verify: bool | str = True
        if self.base_url.startswith("https://127.0.0.1") or self.base_url.startswith("https://localhost"):
            cert = JarvisPaths().tls / "cert.pem"
            verify = str(cert) if cert.exists() else True
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, verify=verify,
                                  headers={"Authorization": f"Bearer {self.token}"} if self.token else {})

    def _check(self, resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            if resp.status_code == 401:
                raise NotAuthenticated(str(detail), self.base_url, self.credential)
            if resp.status_code == 403:
                raise Forbidden(403, str(detail))
            raise ApiError(resp.status_code, str(detail))
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.text

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._http.request(method, f"/api/v1{path}", **kwargs)
        except httpx.ConnectError as e:
            raise CoreUnavailable(f"JARVIS core is not running at {self.base_url} (start it with `jarvis start`)") from e
        return self._check(resp)

    def get(self, path: str, **kw: Any) -> Any:
        return self.request("GET", path, **kw)

    def post(self, path: str, json_body: Any = None, **kw: Any) -> Any:
        return self.request("POST", path, json=json_body, **kw)

    def put(self, path: str, json_body: Any = None) -> Any:
        return self.request("PUT", path, json=json_body)

    def patch(self, path: str, json_body: Any = None) -> Any:
        return self.request("PATCH", path, json=json_body)

    def delete(self, path: str) -> Any:
        return self.request("DELETE", path)

    def health(self) -> dict[str, Any] | None:
        try:
            return self.get("/health")
        except (CoreUnavailable, ApiError):
            return None

    def chat_stream(self, message: str, conversation_id: str | None = None, **extra: Any) -> Iterator[tuple[str, dict[str, Any]]]:
        body = {"message": message, "conversation_id": conversation_id, "stream": True, **extra}
        try:
            with self._http.stream("POST", "/api/v1/chat", json=body, timeout=None) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    self._check(resp)
                event = "message"
                for line in resp.iter_lines():
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        yield event, json.loads(line[5:].strip())
        except httpx.ConnectError as e:
            raise CoreUnavailable(f"JARVIS core is not running at {self.base_url}") from e

    def close(self) -> None:
        self._http.close()
