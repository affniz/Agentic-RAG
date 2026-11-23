"""Optional password login for the web UI and API.

Set RAG_PASSWORD to turn it on. Every /api route then needs a session cookie, which
POST /api/auth/login sets after checking the password. Without RAG_PASSWORD nothing changes: the
server stays localhost-only and open to anyone on this machine (see server/security.py).

Sessions are signed cookies, not server-side records:
- The token is `<expires>.<nonce>.<hmac>`, signed with a key derived from a random secret kept in
  RAG_DATA/session_key and from the password. Restarts keep you logged in; changing RAG_PASSWORD
  or deleting session_key logs every browser out.
- Logging out deletes the cookie in that browser only. A copied cookie stays valid until it expires
  (SESSION_DAYS), so rotate the password if one may have leaked.
- The cookie is HttpOnly and SameSite=Strict, and Secure when the request came over HTTPS (behind a
  TLS proxy, run uvicorn with --proxy-headers so it sees the original scheme).

Failed logins are limited per client address (MAX_FAILURES per FAILURE_WINDOW seconds).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)

COOKIE = "rag_session"
SESSION_DAYS = 30
MAX_FAILURES = 10
FAILURE_WINDOW = 15 * 60
# Reachable without a session: the login flow itself, a liveness probe for Docker's HEALTHCHECK, and
# Google's redirect back after "Connect Google Drive" (a cross-site navigation, so the SameSite=Strict
# cookie isn't sent; it is guarded by its single-use state instead - see server/gdrive_connect.py).
PUBLIC_PATHS = frozenset({"/api/auth/status", "/api/auth/login", "/api/auth/logout", "/api/ping",
                          "/api/gdrive/oauth/callback"})


class Auth:
    def __init__(self, password: str | None, key_file: Path, session_seconds: int = SESSION_DAYS * 86400):
        self.password = password or None
        self.session_seconds = session_seconds
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._key = b""
        if self.password:
            if len(self.password) < 12:
                log.warning("RAG_PASSWORD is shorter than 12 characters; use a long random one")
            pw_hash = hashlib.sha256(self.password.encode()).digest()
            self._key = hmac.new(_secret(key_file), b"rag-session-v1" + pw_hash, hashlib.sha256).digest()

    @property
    def required(self) -> bool:
        return self.password is not None

    # ------------------------------------------------------------ tokens

    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()

    def make_token(self, now: float | None = None) -> str:
        expires = int((now if now is not None else time.time()) + self.session_seconds)
        payload = f"{expires}.{secrets.token_hex(8)}"
        return f"{payload}.{self._sign(payload)}"

    def valid(self, token: str | None, now: float | None = None) -> bool:
        if not self.required:
            return True
        if not token or token.count(".") != 2:
            return False
        payload, sig = token.rsplit(".", 1)
        if not hmac.compare_digest(sig, self._sign(payload)):
            return False
        try:
            expires = int(payload.split(".")[0])
        except ValueError:
            return False
        return (now if now is not None else time.time()) < expires

    # ------------------------------------------------------------ password

    def check_password(self, given: str) -> bool:
        if not self.required:
            return False
        a = hashlib.sha256(given.encode()).digest()  # equal lengths, so compare_digest is constant time
        b = hashlib.sha256(self.password.encode()).digest()  # type: ignore[union-attr]
        return hmac.compare_digest(a, b)

    def retry_after(self, client: str, now: float | None = None) -> int:
        """Seconds until `client` may try again, or 0 if it isn't locked out."""
        now = now if now is not None else time.time()
        with self._lock:
            fails = self._failures.get(client)
            if not fails:
                return 0
            while fails and fails[0] <= now - FAILURE_WINDOW:
                fails.popleft()
            if not fails:
                del self._failures[client]
                return 0
            return int(fails[0] + FAILURE_WINDOW - now) + 1 if len(fails) >= MAX_FAILURES else 0

    def record_failure(self, client: str, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        with self._lock:
            if len(self._failures) > 10_000:  # don't let many addresses grow this without bound
                self._failures = {c: f for c, f in self._failures.items() if f and f[-1] > now - FAILURE_WINDOW}
            self._failures.setdefault(client, deque()).append(now)

    def clear_failures(self, client: str) -> None:
        with self._lock:
            self._failures.pop(client, None)


def _secret(path: Path) -> bytes:
    """A random 32-byte secret kept in `path` (created with owner-only permissions on first use)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        data = path.read_bytes()
        if len(data) >= 32:
            return data
        raise RuntimeError(f"{path} is damaged; delete it to create a new one (this logs everyone out)")
    data = secrets.token_bytes(32)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return data


def _cookie(scope: dict[str, Any]) -> str | None:
    for name, value in scope.get("headers", []):
        if name == b"cookie":
            for part in value.decode("latin-1").split(";"):
                key, _, val = part.strip().partition("=")
                if key == COOKIE:
                    return val
    return None


class AuthGuard:
    """ASGI middleware: with a password set, /api routes other than PUBLIC_PATHS need a valid session.

    The frontend (index.html and its assets) is served to anyone; it holds no data and shows the
    login screen until /api/auth/status says the browser is signed in.
    """

    def __init__(self, app: Any, auth: Auth):
        self.app, self.auth = app, auth

    def needs_login(self, scope: dict[str, Any]) -> bool:
        if not self.auth.required or scope["type"] not in ("http", "websocket"):
            return False
        path = scope.get("path", "")
        if not (path == "/api" or path.startswith("/api/")) or path in PUBLIC_PATHS:
            return False
        return not self.auth.valid(_cookie(scope))

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if self.needs_login(scope):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            await JSONResponse({"detail": "login required"}, status_code=401)(scope, receive, send)
            return
        await self.app(scope, receive, send)


class LoginRequest(BaseModel):
    password: str = Field(max_length=1024)


def make_router(auth: Auth) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    @router.get("/auth/status")
    def status(request: Request) -> dict[str, bool]:
        return {"required": auth.required, "authenticated": auth.valid(request.cookies.get(COOKIE))}

    @router.post("/auth/login")
    def login(req: LoginRequest, request: Request, response: Response) -> dict[str, bool]:
        if not auth.required:
            raise HTTPException(400, "login is not enabled (set RAG_PASSWORD)")
        client = request.client.host if request.client else "?"
        if wait := auth.retry_after(client):
            raise HTTPException(429, f"too many failed logins; try again in {wait // 60 + 1} min",
                                headers={"Retry-After": str(wait)})
        if not auth.check_password(req.password):
            auth.record_failure(client)
            log.warning("failed login from %s", client)
            raise HTTPException(401, "wrong password")
        auth.clear_failures(client)
        response.set_cookie(COOKIE, auth.make_token(), max_age=auth.session_seconds, path="/",
                            httponly=True, samesite="strict", secure=request.url.scheme == "https")
        return {"ok": True}

    @router.post("/auth/logout")
    def logout(request: Request, response: Response) -> dict[str, bool]:
        response.delete_cookie(COOKIE, path="/", httponly=True, samesite="strict",
                               secure=request.url.scheme == "https")
        return {"ok": True}

    return router
