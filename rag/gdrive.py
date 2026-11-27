"""Google Drive connector: mirror a Drive folder into a subfolder of the documents folder.

The mirror is plain files, so the normal ingest indexes them like any other document (incrementally,
with the same chunking, OCR and Documents screen). Google Docs are exported as Markdown (headings
survive, so chunks keep their section paths), Google Slides as plain text; PDFs, .md and .txt files
are downloaded as they are. Other types (Sheets, images, Office files, ...) are skipped and reported.

    report = sync_folder(drive_session(), "1AbC...", Path("docs/gdrive"))

Only files this connector wrote are ever deleted or overwritten (tracked in `<dest>/.gdrive.json`), so
pointing it at a folder that also holds your own files is safe: a Drive file whose name one of yours
already has is saved as "<name> [<id prefix>].<ext>" instead. A file is re-downloaded when its Drive
modifiedTime changes, or when the local copy is missing.

Authentication (read-only scope): a service account key or an authorized-user file passed as
`credentials`, else Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS, or
`gcloud auth application-default login --scopes=https://www.googleapis.com/auth/drive.readonly,...`).
With a service account, share the Drive folder with its client_email.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

SCOPE = "https://www.googleapis.com/auth/drive.readonly"
API = "https://www.googleapis.com/drive/v3/files"
DRIVES_API = "https://www.googleapis.com/drive/v3/drives"
STATE_FILE = ".gdrive.json"
MAX_BYTES = 50 * 1024 * 1024

FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"
# Google-native files: exported, as (export mime type, extension).
EXPORTS = {
    "application/vnd.google-apps.document": ("text/markdown", ".md"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
}
# Uploaded files: downloaded as they are, by extension or (if the name has none we index) by mime type.
EXTENSIONS = {".md", ".txt", ".pdf"}
MIME_EXTENSIONS = {"application/pdf": ".pdf", "text/markdown": ".md", "text/x-markdown": ".md", "text/plain": ".txt"}

_FIELDS = "nextPageToken, incompleteSearch, files(id, name, mimeType, modifiedTime, size)"
# A file name's limit is 255 bytes on most file systems (ext4 in Docker counts bytes, not characters);
# this leaves room for an extension, a " [id]" suffix and the temporary file's prefix and suffix.
MAX_NAME_BYTES = 200
# Transient Drive API answers worth retrying (with backoff), and how often.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
RATE_LIMIT_REASONS = frozenset({"rateLimitExceeded", "userRateLimitExceeded"})  # sent with a 403
MAX_TRIES = 5
_sleep = time.sleep  # replaced in tests
_FOLDER_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")
_UNSAFE = re.compile(r"[/\\\x00-\x1f\x7f]")

ProgressFn = Callable[[str, int, int], None]  # (stage, done, total)


class Session(Protocol):
    """What we need from an HTTP session: google.auth's AuthorizedSession (a requests.Session) fits."""

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any: ...


@dataclass
class SyncReport:
    folder: str
    downloaded: list[str] = field(default_factory=list)  # new or changed files, as paths under dest
    removed: list[str] = field(default_factory=list)  # gone from Drive (or no longer supported), deleted locally
    unchanged: int = 0
    skipped: dict[str, str] = field(default_factory=dict)  # Drive path: why it isn't synced (unsupported type, too big)
    failed: dict[str, str] = field(default_factory=dict)  # Drive path: error; the previous local copy is kept
    finished_at: float = 0.0


class DriveError(RuntimeError):
    pass


class DriveAuthError(DriveError):
    """Google turned down the saved sign-in (expired, revoked, or the OAuth client changed).
    `reconnect`: the refresh token itself is dead, so the account has to sign in again."""

    def __init__(self, message: str, reconnect: bool):
        super().__init__(message)
        self.reconnect = reconnect


def folder_id(value: str) -> str:
    """Accept a folder id or a Drive folder URL (https://drive.google.com/drive/folders/<id>?usp=...)."""
    value = value.strip()
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", value) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", value)
    fid = m.group(1) if m else value
    if not _FOLDER_ID.match(fid):
        raise ValueError(f"not a Google Drive folder id or folder URL: {value!r}")
    return fid


def drive_session(credentials: str | Path | None = None) -> Session:
    """An authorized, read-only Drive session (see the module docstring for where credentials come from)."""
    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
    except ImportError as e:  # pragma: no cover - listed in requirements.txt
        raise DriveError("Google Drive needs the google-auth package: pip install -r requirements.txt") from e
    try:
        if credentials:
            creds, _ = google.auth.load_credentials_from_file(str(credentials), scopes=[SCOPE])
        else:
            creds, _ = google.auth.default(scopes=[SCOPE])
    except google.auth.exceptions.GoogleAuthError as e:
        raise DriveError(f"No usable Google credentials: {str(e).rstrip('.')}. Set RAG_GDRIVE_CREDENTIALS to a service account "
                         "key file (and share the folder with its client_email).") from e
    return AuthorizedSession(creds)


class _OAuthSession:
    """An AuthorizedSession that reports a dead refresh token (revoked, expired) as a DriveAuthError,
    and tells `on_auth_error` about it, so the app can show it where the user connects Drive."""

    def __init__(self, session: Any, on_auth_error: Callable[[DriveAuthError], None] | None = None):
        self._session = session
        self._on_auth_error = on_auth_error

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        from google.auth.exceptions import RefreshError

        try:
            return self._session.get(url, params=params, timeout=timeout)
        except RefreshError as e:
            if getattr(e, "retryable", False):  # Google had a hiccup: worth trying again later
                raise DriveError(f"Couldn't refresh the Google sign-in, try again: {e}") from e
            code = str(e.args[0] if e.args else e).split(":")[0].strip()  # e.g. "invalid_grant"
            if code == "invalid_client":
                err = DriveAuthError("Google rejected the OAuth client ID or secret - disconnect, then enter "
                                     "them again", reconnect=False)
            else:
                err = DriveAuthError(f"Google sign-in has expired or was revoked ({code}) - connect Google "
                                     "Drive again", reconnect=True)
            if self._on_auth_error:
                self._on_auth_error(err)
            raise err from e


def oauth_session(client_id: str, client_secret: str, refresh_token: str, token_uri: str,
                  on_auth_error: Callable[[DriveAuthError], None] | None = None) -> Session:
    """A read-only Drive session for an account that signed in with Google (see server/gdrive_connect.py)."""
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2.credentials import Credentials

    creds = Credentials(None, refresh_token=refresh_token, token_uri=token_uri, client_id=client_id,
                        client_secret=client_secret, scopes=[SCOPE])
    return _OAuthSession(AuthorizedSession(creds), on_auth_error)


def folder_info(session: Session, folder: str) -> dict[str, Any]:
    """The folder's Drive metadata (id, name, mimeType); DriveError if it's missing, not shared or not a folder."""
    fid = folder_id(folder)
    meta = _get_json(session, f"{API}/{fid}", {"fields": "id, name, mimeType", "supportsAllDrives": "true"},
                     f"folder {fid}")
    if meta.get("mimeType") != FOLDER:
        raise DriveError(f"{meta.get('name') or fid} is not a folder ({meta.get('mimeType')})")
    return meta


def synced_paths(dest: Path) -> set[str]:
    """Paths (relative to `dest`, with /) of the files this connector keeps in sync there."""
    state = _load_state(dest)
    return {v["path"] for v in state.get("files", {}).values()} | set(state.get("claimed", []))


def last_sync(dest: Path) -> dict[str, Any] | None:
    """The last sync's report (as saved in the state file), or None if this folder was never synced."""
    state = _load_state(dest)
    return state.get("report")


# ---------------------------------------------------------------- Drive API


def _check(resp: Any, what: str) -> Any:
    if resp.status_code >= 400:
        try:
            msg = resp.json()["error"]["message"]
        except Exception:
            msg = (getattr(resp, "text", "") or "")[:200]
        hint = ""
        if resp.status_code == 404 and what.startswith("folder"):
            hint = " - check the id, and that the folder is shared with the account the server uses"
        elif resp.status_code in (401, 403):
            hint = " - check the credentials and that the folder is shared with that account"
        raise DriveError(f"Drive API error {resp.status_code} for {what}: {msg}{hint}")
    return resp


def _retryable(resp: Any) -> bool:
    if resp.status_code in RETRY_STATUSES:
        return True
    if resp.status_code != 403:
        return False
    try:
        return resp.json()["error"]["errors"][0]["reason"] in RATE_LIMIT_REASONS
    except Exception:
        return False


def _wait(attempt: int, resp: Any = None) -> float:
    """Seconds before the next try: the server's Retry-After, else exponential backoff with jitter."""
    try:
        return min(float((getattr(resp, "headers", None) or {}).get("Retry-After")), 60.0)
    except (TypeError, ValueError):
        return min(2.0**attempt, 16.0) + random.random()


def _get(session: Session, url: str, params: dict[str, Any], timeout: float, what: str) -> Any:
    """GET with retries for rate limits, Drive's transient server errors and dropped connections."""
    import requests

    for attempt in range(MAX_TRIES):
        last = attempt == MAX_TRIES - 1
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            if last:
                raise DriveError(f"Couldn't reach Google Drive for {what}: {type(e).__name__}: {e}") from e
            _sleep(_wait(attempt))
            continue
        if last or not _retryable(resp):
            return _check(resp, what)
        _sleep(_wait(attempt, resp))
    raise AssertionError("unreachable")


def _get_json(session: Session, url: str, params: dict[str, Any], what: str) -> dict[str, Any]:
    return _get(session, url, params, 60, what).json()


def _list(session: Session, q: str, what: str, limit: int | None = None,
          complete: bool = False) -> tuple[list[dict[str, Any]], bool]:
    """files.list over My Drive, shared-with-me and shared drives: (files, whether `limit` cut it short).

    complete: raise DriveError if Drive says its answer may be missing items (it can, when it
    searches several drives), rather than return a partial list. A sync needs that: a file left out
    of the listing would be deleted locally."""
    out: list[dict[str, Any]] = []
    # corpora=allDrives: the default ("user") leaves out items that live in shared drives.
    params: dict[str, Any] = {
        "q": q, "fields": _FIELDS, "pageSize": 1000, "orderBy": "folder, name_natural",
        "supportsAllDrives": "true", "includeItemsFromAllDrives": "true", "corpora": "allDrives",
    }
    while True:
        page = _get_json(session, API, params, what)
        if complete and page.get("incompleteSearch"):
            raise DriveError(f"Google Drive returned an incomplete listing of {what}; nothing was changed - try again")
        out += page.get("files", [])
        if limit is not None and len(out) >= limit:
            return out[:limit], len(out) > limit or bool(page.get("nextPageToken"))
        if not page.get("nextPageToken"):
            return out, False
        params["pageToken"] = page["nextPageToken"]


def _list_children(session: Session, parent: str) -> list[dict[str, Any]]:
    return _list(session, f"'{parent}' in parents and trashed = false", f"folder {parent}", complete=True)[0]


# ---------------------------------------------------------------- choosing a folder

# Where browsing starts. Each lists folders to open or choose; none of them can be synced as a whole.
LOCATIONS = {"root": "My Drive", "shared": "Shared with me", "drives": "Shared drives"}
BROWSE_LIMIT = 500


def _quote(text: str) -> str:
    """A string literal for a Drive query."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def browse(session: Session, parent: str = "root", search: str | None = None) -> dict[str, Any]:
    """What the folder picker shows: the folders in `parent` (a folder id, or a LOCATIONS key), or the
    folders whose name contains `search` anywhere the account can see.

    Inside a folder, `files` counts the files directly in it and how many of those a sync would
    download, so the user can tell they found the right one.
    """
    if search:
        q = f"mimeType = '{FOLDER}' and trashed = false and name contains {_quote(search)}"
        items, more = _list(session, q, "folder search", BROWSE_LIMIT)
        return {"folders": [{"id": f["id"], "name": f["name"]} for f in items], "files": None, "truncated": more}
    if parent == "drives":
        drives: list[dict[str, Any]] = []
        params: dict[str, Any] = {"pageSize": 100, "fields": "nextPageToken, drives(id, name)"}
        while len(drives) < BROWSE_LIMIT:
            page = _get_json(session, DRIVES_API, params, "shared drives")
            drives += page.get("drives", [])
            if not page.get("nextPageToken"):
                break
            params["pageToken"] = page["nextPageToken"]
        drives.sort(key=lambda d: d["name"].casefold())
        return {"folders": [{"id": d["id"], "name": d["name"]} for d in drives[:BROWSE_LIMIT]], "files": None,
                "truncated": len(drives) > BROWSE_LIMIT}
    if parent == "shared":
        q = f"sharedWithMe = true and mimeType = '{FOLDER}' and trashed = false"
    else:
        pid = "root" if parent == "root" else folder_id(parent)
        q = f"'{pid}' in parents and trashed = false"
    items, more = _list(session, q, f"folder {parent}", BROWSE_LIMIT)
    folders = [{"id": f["id"], "name": f["name"]} for f in items if f["mimeType"] == FOLDER]
    files = [f for f in items if f["mimeType"] != FOLDER]
    counts = None if parent in LOCATIONS else {
        "total": len(files), "supported": sum(_extension(f) is not None for f in files)}
    return {"folders": folders, "files": counts, "truncated": more}


def _walk(session: Session, root: str) -> list[tuple[list[str], dict[str, Any]]]:
    """Every non-folder item under `root`, with the names of the folders leading to it."""
    folder_info(session, root)
    items: list[tuple[list[str], dict[str, Any]]] = []
    seen = {root}
    stack: list[tuple[list[str], str]] = [([], root)]
    while stack:
        parents, fid = stack.pop()
        for f in _list_children(session, fid):
            if f["mimeType"] == FOLDER:
                if f["id"] not in seen:  # a folder can have several parents; visit it once
                    seen.add(f["id"])
                    stack.append((parents + [f["name"]], f["id"]))
            else:
                items.append((parents, f))
    return items


def _fetch(session: Session, f: dict[str, Any]) -> bytes:
    if f["mimeType"] in EXPORTS:
        url, params = f"{API}/{f['id']}/export", {"mimeType": EXPORTS[f["mimeType"]][0]}
    else:
        url, params = f"{API}/{f['id']}", {"alt": "media", "supportsAllDrives": "true"}
    return _get(session, url, params, 300, f"file {f['name']!r}").content


# ---------------------------------------------------------------- local mirror


def _safe(name: str) -> str:
    name = _UNSAFE.sub("_", name).strip().lstrip(".")[:150]
    while len(name.encode()) > MAX_NAME_BYTES:  # long names in non-Latin scripts: 3-4 bytes a character
        name = name[:-1]
    return name.rstrip() or "untitled"


def _extension(f: dict[str, Any]) -> str | None:
    if f["mimeType"] in EXPORTS:
        return EXPORTS[f["mimeType"]][1]
    suffix = Path(f["name"]).suffix.lower()
    return suffix if suffix in EXTENSIONS else MIME_EXTENSIONS.get(f["mimeType"])


def _local_name(f: dict[str, Any], ext: str) -> str:
    name = _safe(f["name"])
    return name if name.lower().endswith(ext) else f"{name}{ext}"


def _skip_reason(f: dict[str, Any]) -> str:
    if f["mimeType"] == SHORTCUT:
        return "shortcut - add the original file to the folder instead"
    if f["mimeType"] == "application/vnd.google-apps.spreadsheet":
        return "Google Sheets aren't supported"
    return f"unsupported type ({f['mimeType']}); supported: Google Docs, Google Slides, .pdf, .md, .txt"


def _load_state(dest: Path) -> dict[str, Any]:
    try:
        return json.loads((dest / STATE_FILE).read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.gdrive-tmp")  # dot-prefixed, unsupported suffix: ingest ignores it
    tmp.write_bytes(data)
    tmp.replace(path)


def sync_folder(session: Session, folder: str, dest: str | Path,
                on_progress: ProgressFn | None = None) -> SyncReport:
    """Mirror Drive folder `folder` (id or URL, recursively) into `dest`. Returns what changed.

    Listing errors (bad credentials, unknown folder) raise DriveError and change nothing locally.
    A file that fails to download keeps its previous local copy and is retried on the next sync.
    """
    progress = on_progress or (lambda *_: None)
    fid = folder_id(folder)
    dest = Path(dest)
    report = SyncReport(folder=fid)

    progress("listing", 0, 0)
    items = _walk(session, fid)  # before touching anything local

    # Everything we wrote before, even for another folder id: files not in this folder are removed.
    state = _load_state(dest)
    old: dict[str, dict[str, Any]] = state.get("files", {})
    # Paths a sync that didn't finish may have written (see below): ours too, though not in `old`.
    claimed: list[str] = state.get("claimed", [])
    own = {v["path"] for v in old.values()} | set(claimed)

    # Decide each item's local path. Drive allows duplicate names; later ones (by id) get a suffix.
    # Files in `dest` that this connector didn't write are never overwritten: their names are taken.
    wanted: dict[str, tuple[str, str, dict[str, Any]]] = {}  # drive id -> (local path, Drive path, file)
    taken: set[str] = {rel.lower() for rel in _local_files(dest) if rel not in own}
    for parents, f in sorted(items, key=lambda pf: ([_safe(p) for p in pf[0]], pf[1]["name"], pf[1]["id"])):
        drive_path = "/".join(parents + [f["name"]])
        ext = _extension(f)
        if ext is None:
            report.skipped[drive_path] = _skip_reason(f)
            continue
        if f.get("size") is not None and int(f["size"]) > MAX_BYTES:
            report.skipped[drive_path] = f"larger than {MAX_BYTES // 2**20} MB"
            continue
        rel = "/".join([_safe(p) for p in parents] + [_local_name(f, ext)])
        if rel.lower() in taken:
            stem = rel[: -len(ext)]
            rel = f"{stem} [{f['id'][:8]}]{ext}"
        taken.add(rel.lower())
        wanted[f["id"]] = (rel, drive_path, f)

    dest.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}  # the new state: drive id -> {path, modified, name}
    wanted_paths = {rel for rel, _, _ in wanted.values()}

    todo: list[tuple[str, str, str, dict[str, Any]]] = []
    for i, (rel, drive_path, f) in wanted.items():
        prev = old.get(i, {})
        if prev.get("path") == rel and prev.get("modified") == f.get("modifiedTime") and (dest / rel).is_file():
            files[i] = prev
            report.unchanged += 1
        else:
            todo.append((i, rel, drive_path, f))

    # Record the paths about to be written before writing them, so that if this sync is interrupted,
    # the next one still knows those files are its own (to update or remove), not the user's.
    if todo:
        _write_state(dest, {**state, "folder": fid, "files": old,
                            "claimed": sorted(set(claimed) | {rel for _, rel, _, _ in todo})})

    for n, (i, rel, drive_path, f) in enumerate(todo):
        progress("downloading", n, len(todo))
        try:
            data = _fetch(session, f)
            if len(data) > MAX_BYTES:
                raise DriveError(f"larger than {MAX_BYTES // 2**20} MB")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(target, data)
        except Exception as e:  # one bad file must not stop the rest
            report.failed[drive_path] = str(e) if isinstance(e, DriveError) else f"{type(e).__name__}: {e}"
            # Keep the previous copy (still searchable, retried next time) unless another file now owns its path.
            if i in old and (old[i]["path"] == rel or old[i]["path"] not in wanted_paths):
                files[i] = old[i]
            continue
        files[i] = {"path": rel, "modified": f.get("modifiedTime"), "name": f["name"]}
        report.downloaded.append(rel)
    progress("downloading", len(todo), len(todo))

    # Copies we wrote earlier whose path nothing uses any more: deleted, trashed, moved out or now
    # unsupported in Drive, or the old name of a renamed file. Done last, so a path another file
    # took over in this sync is never deleted.
    live_paths = {v["path"] for v in files.values()}
    for i, entry in old.items():
        if entry["path"] not in live_paths and _remove(dest, entry["path"]) and i not in files:
            report.removed.append(entry["path"])
    for rel in claimed:  # written by an interrupted sync and not wanted any more
        if rel not in live_paths:
            _remove(dest, rel)

    report.finished_at = time.time()
    _write_state(dest, {"folder": fid, "files": files, "report": asdict(report)})
    return report


def _write_state(dest: Path, state: dict[str, Any]) -> None:
    _write_atomic(dest / STATE_FILE, json.dumps(state, indent=1).encode())


def _local_files(dest: Path) -> list[str]:
    """Paths (relative, with /) of the files in `dest`, leaving out our state and temporary files."""
    if not dest.is_dir():
        return []
    return [p.relative_to(dest).as_posix() for p in dest.rglob("*")
            if p.is_file() and not p.name.startswith(".")]


def _remove(dest: Path, rel: str) -> bool:
    """Delete a file this connector wrote, then any folders that left empty. Never leaves `dest`."""
    root = dest.resolve()
    path = (dest / rel).resolve()
    if not path.is_relative_to(root) or path == root or not path.is_file():
        return False
    path.unlink()
    parent = path.parent
    while parent != root and parent.is_relative_to(root) and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent
    return True
