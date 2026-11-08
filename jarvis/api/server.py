"""FastAPI application factory and server entry point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from jarvis import __version__
from jarvis.api.deps import RateLimiter
from jarvis.core.observability import get_logger

log = get_logger("api.server")

WEB_ROOT = Path(__file__).resolve().parent.parent / "clients" / "web" / "static"

CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
       "media-src 'self' blob:; connect-src 'self' ws: wss:; font-src 'self' data:; frame-ancestors 'none'; "
       "base-uri 'self'; form-action 'self'")


def create_app(runtime: Any, manage_lifecycle: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        if manage_lifecycle:
            await runtime.start()
        try:
            yield
        finally:
            for task in list(app.state.background):
                task.cancel()
            if manage_lifecycle:
                await runtime.stop()

    app = FastAPI(title="JARVIS API", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.state.runtime = runtime
    app.state.active_runs = {}
    app.state.background = set()
    limiter = RateLimiter()
    cfg = runtime.config.server

    if cfg.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_methods=["*"],
                           allow_headers=["Authorization", "Content-Type"], allow_credentials=False)

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Response:
        path = request.url.path
        client = request.client.host if request.client else "unknown"
        if path.startswith("/api/"):
            bucket = "auth" if path.startswith("/api/v1/auth/") else "api"
            limit = cfg.auth_rate_limit_per_minute if bucket == "auth" else cfg.rate_limit_per_minute
            if not limiter.allow((client, bucket), limit):
                return JSONResponse({"detail": "rate limit exceeded"}, status_code=429, headers={"Retry-After": "10"})
        try:
            response: Response = await call_next(request)
        except Exception:  # noqa: BLE001 - never leak internals to clients
            log.exception("unhandled error on %s", path)
            response = JSONResponse({"detail": "internal error (see server logs)"}, status_code=500)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if not path.startswith("/api/docs"):
            response.headers.setdefault("Content-Security-Policy", CSP)
        if path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        elif path.startswith("/static/"):
            # Revalidate (cheap ETag check) so interface updates always reach clients.
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    from jarvis.api.routes_core import router as core_router

    app.include_router(core_router)
    for ext in runtime.extensions.values():
        ext_router = getattr(ext, "api_router", None)
        if ext_router is not None:
            app.include_router(ext_router)

    if WEB_ROOT.exists():
        app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(WEB_ROOT / "index.html", headers={"Cache-Control": "no-cache"})

        @app.get("/hud", include_in_schema=False)
        async def hud() -> FileResponse:
            return FileResponse(WEB_ROOT / "hud.html", headers={"Cache-Control": "no-cache"})

        @app.get("/manifest.webmanifest", include_in_schema=False)
        async def manifest() -> FileResponse:
            return FileResponse(WEB_ROOT / "manifest.webmanifest", media_type="application/manifest+json")

        @app.get("/sw.js", include_in_schema=False)
        async def service_worker() -> FileResponse:
            return FileResponse(WEB_ROOT / "sw.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})

    return app


class InsecureBindError(RuntimeError):
    pass


def check_bind(runtime: Any, host: str) -> None:
    cfg = runtime.config.server
    loopback = host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not (cfg.tls_cert and cfg.tls_key) and not cfg.allow_insecure_lan:
        raise InsecureBindError(
            f"Refusing to listen on {host} without TLS: device credentials would cross the network in clear text. "
            "Run `jarvis setup --tls` to create a certificate, or set server.allow_insecure_lan = true if you accept the risk."
        )


def serve(runtime: Any, host: str | None = None, port: int | None = None) -> None:
    import uvicorn  # noqa: PLC0415

    cfg = runtime.config.server
    host = host or cfg.host
    port = port or cfg.port
    check_bind(runtime, host)
    app = create_app(runtime)
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", access_log=False,
        ssl_certfile=cfg.tls_cert, ssl_keyfile=cfg.tls_key, ws_ping_interval=20, ws_ping_timeout=20,
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
