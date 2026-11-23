"""Localhost guard: the API only answers requests meant for this machine.

The server has no login, so two browser attacks matter:
- DNS rebinding: a page on evil.example re-points its own name at 127.0.0.1 and then reads the API.
  The browser still sends `Host: evil.example`, so checking the Host header stops it.
- Cross-site writes: any page can POST a form or a multipart upload to localhost:8000 without a
  CORS preflight. Browsers mark those with an `Origin` header and `Sec-Fetch-Site: cross-site`.

It is not network access control: a non-browser client that can reach the port can send any Host
header. Keep the port on loopback (uvicorn's default; `-p 127.0.0.1:8000:8000` with Docker).

Only hostnames are compared, never ports, so the Vite dev proxy (:5173 -> :8000) keeps working.
RAG_ALLOWED_HOSTS adds hostnames, comma-separated (e.g. a LAN name you serve the UI under).

With a login password (RAG_PASSWORD, server/auth.py) the Host check is dropped (`any_host=True`):
the session cookie is what keeps other people and DNS-rebinding pages out, since a rebound page is on
another hostname and the browser won't send it the cookie. The cross-site checks stay; an `Origin`
must then match an allowed hostname or the request's own Host.
"""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

Scope = dict[str, Any]
ASGIApp = Callable[[Scope, Any, Any], Awaitable[None]]


def allowed_hosts_from_env() -> frozenset[str]:
    extra = os.environ.get("RAG_ALLOWED_HOSTS", "")
    return LOCAL_HOSTS | {_normalize(h) for h in extra.split(",") if h.strip()}


def _normalize(host: str) -> str:
    host = host.strip().lower().rstrip(".")
    return host[1:-1] if host.startswith("[") and host.endswith("]") else host


def host_header_name(value: str) -> str:
    """The hostname of a Host header: 'localhost:8000' -> 'localhost', '[::1]:8000' -> '::1'."""
    value = value.strip()
    if value.startswith("["):
        return _normalize(value[: value.find("]") + 1])
    if value.count(":") == 1:  # host:port (a bare IPv6 address has several colons)
        value = value.split(":")[0]
    return _normalize(value)


class LocalhostGuard:
    """ASGI middleware that answers 403 to requests from outside this machine's own pages."""

    def __init__(self, app: ASGIApp, allowed: frozenset[str] | None = None, any_host: bool = False):
        self.app = app
        self.allowed = allowed if allowed is not None else allowed_hosts_from_env()
        self.any_host = any_host

    def is_allowed(self, host: str | None) -> bool:
        return bool(host) and (host in self.allowed or host.endswith(".localhost"))

    def reason(self, scope: Scope) -> str | None:
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        host = host_header_name(headers.get("host", ""))
        # Starlette's TestClient sends `Host: testserver` from client "testclient"; a real server
        # always reports the peer's IP address there, so this can't be reached over the network.
        in_process_test = host == "testserver" and (scope.get("client") or ("",))[0] == "testclient"
        if not (self.any_host or self.is_allowed(host) or in_process_test):
            return f"host {host or '(none)'!r} is not allowed (set RAG_ALLOWED_HOSTS to add it)"
        if scope["type"] == "websocket" or scope.get("method", "GET").upper() in UNSAFE_METHODS:
            if headers.get("sec-fetch-site", "").lower() == "cross-site":
                return "cross-site request blocked"
            origin = headers.get("origin")
            if origin is not None:
                origin_host = _origin_host(origin)
                same_host = self.any_host and origin_host is not None and origin_host == host
                if not (self.is_allowed(origin_host) or same_host):
                    return f"origin {origin!r} is not allowed"
        return None

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] in ("http", "websocket") and (why := self.reason(scope)):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            await JSONResponse({"detail": why}, status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _origin_host(origin: str) -> str | None:
    try:
        host = urlsplit(origin.strip()).hostname  # 'null' (sandboxed / file: pages) has none
    except ValueError:
        return None
    return _normalize(host) if host else None
