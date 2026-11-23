"""Connect Google Drive from the web UI: "Sign in with Google", then pick the folder to sync.

The Documents screen walks through it; nothing needs an environment variable or a restart:
1. One-time setup: an OAuth client from Google Cloud (Google Auth Platform > Clients > Create client >
   "Web application", with this server's callback URL - shown on the screen - as an authorized
   redirect URI; enable the Google Drive API). Its client ID and secret are entered on the screen, or
   set by RAG_GDRIVE_CLIENT_ID / RAG_GDRIVE_CLIENT_SECRET (which then can't be changed in the UI).
2. "Connect Google Drive" sends the browser to Google's sign-in (authorization code flow with PKCE,
   read-only Drive scope). Google redirects back to GET /api/gdrive/oauth/callback, which trades the
   code for a refresh token.
3. The user picks the folder in a browser on the same screen (My Drive, Shared with me, shared drives,
   or a search by name: GET /api/gdrive/folders), and "Sync now" mirrors it into
   RAG_DOCS/RAG_GDRIVE_DIR (rag/gdrive.py) and indexes it.

The client secret and the refresh token are kept in RAG_DATA/gdrive.json (owner-only permissions)
and never sent to the browser. "Disconnect" revokes the token at Google and forgets it; files
already synced stay in the documents folder.

The callback is reachable without a login (server/auth.py PUBLIC_PATHS): Google's redirect is a
cross-site navigation, so the SameSite=Strict session cookie isn't sent with it. It is guarded instead
by the `state` value, which only POST /api/gdrive/connect (a logged-in, same-site request) creates,
is single-use, expires after STATE_SECONDS, and must match a SameSite=Lax cookie set on the browser
that started the sign-in.

A folder set with RAG_GDRIVE_FOLDER (service account or Application Default Credentials) takes
precedence: the screen then only offers "Sync now".
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from rag import gdrive

log = logging.getLogger(__name__)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
SCOPES = ("openid", "email", gdrive.SCOPE)
CALLBACK = "/api/gdrive/oauth/callback"
STATE_COOKIE = "rag_gdrive_state"
STATE_SECONDS = 600

# (url, form data) -> (status code, JSON body). Swapped out in tests.
PostFn = Callable[[str, dict[str, str]], tuple[int, dict[str, Any]]]


def _post(url: str, data: dict[str, str]) -> tuple[int, dict[str, Any]]:
    import requests

    resp = requests.post(url, data=data, timeout=30)
    try:
        body = resp.json()
    except ValueError:
        body = {}
    return resp.status_code, body


def _id_token_email(id_token: str | None) -> str | None:
    """The `email` claim of an ID token. It came straight from Google's token endpoint over TLS, so
    (as Google's docs allow) its signature isn't checked."""
    try:
        payload = id_token.split(".")[1]  # type: ignore[union-attr]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))).get("email")
    except Exception:
        return None


@dataclass
class Pending:
    verifier: str  # PKCE code_verifier
    redirect_uri: str
    expires: float


class DriveConnection:
    """The saved connection: OAuth client, signed-in account and folder (RAG_DATA/gdrive.json)."""

    def __init__(self, path: Path, env_client_id: str | None = None, env_client_secret: str | None = None,
                 post: PostFn = _post):
        self.path = path
        self.env_client = (env_client_id.strip(), (env_client_secret or "").strip()) if env_client_id else None
        self.post = post
        self._lock = threading.Lock()
        self._pending: dict[str, Pending] = {}
        # One session per (client, refresh token): it keeps its access token for about an hour, so
        # browsing folders doesn't ask Google for a new one on every click.
        self._session: tuple[tuple[str, str, str], gdrive.Session] | None = None

    # ------------------------------------------------------------ storage

    def load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, ValueError):
            return {}

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
        tmp.replace(self.path)

    def update(self, **changes: Any) -> dict[str, Any]:
        with self._lock:
            data = self.load()
            for k, v in changes.items():
                if v is None:
                    data.pop(k, None)
                else:
                    data[k] = v
            self.save(data)
            return data

    def client(self) -> tuple[str, str] | None:
        """(client_id, client_secret), from the environment or the saved setup."""
        if self.env_client:
            return self.env_client
        data = self.load()
        return (data["client_id"], data.get("client_secret", "")) if data.get("client_id") else None

    def session(self) -> gdrive.Session:
        client, data = self.client(), self.load()
        if not client or not data.get("refresh_token"):
            raise gdrive.DriveError("Google Drive isn't connected - connect it on the Documents screen")
        key = (client[0], client[1], data["refresh_token"])
        with self._lock:
            if self._session is None or self._session[0] != key:
                self._session = (key, gdrive.oauth_session(client[0], client[1], data["refresh_token"], TOKEN_URL))
            return self._session[1]

    # ------------------------------------------------------------ OAuth flow

    def start(self, redirect_uri: str) -> tuple[str, str]:
        """Begin a sign-in: (Google's authorization URL, state)."""
        client = self.client()
        if not client:
            raise HTTPException(400, "set up the Google OAuth client first")
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        now = time.time()
        with self._lock:
            self._pending = {s: p for s, p in self._pending.items() if p.expires > now}
            self._pending[state] = Pending(verifier, redirect_uri, now + STATE_SECONDS)
        query = urlencode({
            "client_id": client[0], "redirect_uri": redirect_uri, "response_type": "code",
            "scope": " ".join(SCOPES), "access_type": "offline", "prompt": "consent select_account",
            "include_granted_scopes": "true", "state": state,
            "code_challenge": challenge, "code_challenge_method": "S256",
        })
        return f"{AUTH_URL}?{query}", state

    def finish(self, state: str, cookie_state: str | None, code: str | None, error: str | None) -> str:
        """Complete a sign-in from Google's redirect; returns the account's email. Raises ValueError
        with a message for the user if it didn't work."""
        with self._lock:
            pending = self._pending.pop(state, None)
        if pending is None or pending.expires < time.time():
            raise ValueError("the sign-in link expired or was already used - click Connect Google Drive again")
        if not cookie_state or not hmac.compare_digest(cookie_state, state):
            raise ValueError("the sign-in was started in another browser - click Connect Google Drive again here")
        if error:
            raise ValueError("access was not granted" if error == "access_denied" else f"Google returned an error: {error}")
        client = self.client()
        if not code or not client:
            raise ValueError("Google didn't return an authorization code")
        status, body = self.post(TOKEN_URL, {
            "code": code, "client_id": client[0], "client_secret": client[1], "redirect_uri": pending.redirect_uri,
            "grant_type": "authorization_code", "code_verifier": pending.verifier,
        })
        if status >= 400:
            detail = body.get("error_description") or body.get("error") or f"HTTP {status}"
            raise ValueError(f"Google rejected the sign-in: {detail} - check the client ID and secret")
        if gdrive.SCOPE not in body.get("scope", "").split():
            raise ValueError("Google Drive access wasn't allowed - connect again and tick "
                             "\"See and download all your Google Drive files\"")
        if not body.get("refresh_token"):
            raise ValueError("Google didn't return a refresh token - connect again")
        email = _id_token_email(body.get("id_token")) or "your Google account"
        old = self.load().get("refresh_token")
        self.update(refresh_token=body["refresh_token"], email=email, error=None, connected_at=time.time())
        if old and old != body["refresh_token"]:
            self.revoke(old)
        return email

    def revoke(self, token: str) -> None:
        try:
            self.post(REVOKE_URL, {"token": token})
        except Exception as e:  # best effort: the token is forgotten either way
            log.warning("couldn't revoke the Google token: %s", e)


# ---------------------------------------------------------------- API


class ClientRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=300)
    client_secret: str = Field(min_length=1, max_length=300)


class FolderRequest(BaseModel):
    folder: str = Field(min_length=1, max_length=500)  # a folder id (from the picker) or a folder link


def redirect_uri(request: Request) -> str:
    return f"{str(request.base_url).rstrip('/')}{CALLBACK}"


def make_router(conn: DriveConnection, env_folder: str | None, dest_dir: str,
                last_sync: Callable[[], dict[str, Any] | None]) -> APIRouter:
    """env_folder: the folder id set by RAG_GDRIVE_FOLDER (then this screen only syncs it).
    dest_dir: where in the documents folder the Drive files go, for display."""
    router = APIRouter(prefix="/api/gdrive")

    def status(request: Request) -> dict[str, Any]:
        base = {"dir": dest_dir, "last_sync": last_sync(), "redirect_uri": redirect_uri(request)}
        if env_folder:
            return {**base, "mode": "env", "folder": env_folder, "folder_name": None, "account": None,
                    "client": {"configured": True, "from_env": True, "client_id": None}, "error": None}
        data, client = conn.load(), conn.client()
        return {**base, "mode": "oauth", "folder": data.get("folder"), "folder_name": data.get("folder_name"),
                "account": data.get("email") if data.get("refresh_token") else None,
                "client": {"configured": client is not None, "from_env": conn.env_client is not None,
                           "client_id": client[0] if client else None},
                "error": data.get("error")}

    def oauth_only() -> None:
        if env_folder:
            raise HTTPException(409, "the Google Drive folder is set by the server (RAG_GDRIVE_FOLDER)")

    @router.get("")
    def get_status(request: Request) -> dict[str, Any]:
        return status(request)

    @router.put("/client")
    def set_client(req: ClientRequest, request: Request) -> dict[str, Any]:
        oauth_only()
        if conn.env_client:
            raise HTTPException(409, "the OAuth client is set by the server (RAG_GDRIVE_CLIENT_ID)")
        client_id = req.client_id.strip()
        if not client_id.endswith(".apps.googleusercontent.com"):
            raise HTTPException(400, "that doesn't look like a Google OAuth client ID (it ends in .apps.googleusercontent.com)")
        old = conn.load()
        # A refresh token belongs to the client it was issued to: a new client means signing in again.
        changed = old.get("client_id") != client_id
        if changed and old.get("refresh_token"):
            conn.revoke(old["refresh_token"])
        conn.update(client_id=client_id, client_secret=req.client_secret.strip(), error=None,
                    **({"refresh_token": None, "email": None} if changed else {}))
        return status(request)

    @router.post("/connect")
    def connect(request: Request, response: Response) -> dict[str, str]:
        oauth_only()
        url, state = conn.start(redirect_uri(request))
        response.set_cookie(STATE_COOKIE, state, max_age=STATE_SECONDS, path="/api/gdrive/oauth",
                            httponly=True, samesite="lax", secure=request.url.scheme == "https")
        return {"url": url}

    @router.get("/oauth/callback")
    def callback(request: Request, state: str = "", code: str | None = None,
                 error: str | None = None) -> RedirectResponse:
        cookie_state = request.cookies.get(STATE_COOKIE)
        try:
            conn.finish(state, cookie_state, code, error)
        except ValueError as e:
            # Only a browser that started a sign-in gets its error shown: this route needs no login,
            # so a request without the matching cookie must not write anything.
            if state and cookie_state and hmac.compare_digest(cookie_state, state):
                conn.update(error=str(e))
        resp = RedirectResponse("/#/documents", status_code=303)
        resp.delete_cookie(STATE_COOKIE, path="/api/gdrive/oauth")
        return resp

    @router.get("/folders")
    def folders(parent: str = "root", q: str = "") -> dict[str, Any]:
        """Folders for the picker: inside `parent` (a folder id, "root", "shared" or "drives"), or
        matching the search `q`."""
        oauth_only()
        try:
            return gdrive.browse(conn.session(), parent, q.strip()[:100] or None)
        except ValueError:
            raise HTTPException(400, "invalid folder")
        except gdrive.DriveError as e:
            raise HTTPException(400, str(e))

    @router.put("/folder")
    def set_folder(req: FolderRequest, request: Request) -> dict[str, Any]:
        oauth_only()
        try:
            fid = gdrive.folder_id(req.folder)
        except ValueError:
            raise HTTPException(400, "paste a Google Drive folder link (https://drive.google.com/drive/folders/...) or its id")
        try:
            meta = gdrive.folder_info(conn.session(), fid)
        except gdrive.DriveError as e:
            raise HTTPException(400, str(e))
        conn.update(folder=fid, folder_name=meta.get("name") or fid, error=None)
        return status(request)

    @router.delete("/account")
    def disconnect(request: Request) -> dict[str, Any]:
        oauth_only()
        token = conn.load().get("refresh_token")
        if token:
            conn.revoke(token)
        conn.update(refresh_token=None, email=None, error=None)
        return status(request)

    @router.delete("/error")
    def dismiss_error(request: Request) -> dict[str, Any]:
        conn.update(error=None)
        return status(request)

    return router
