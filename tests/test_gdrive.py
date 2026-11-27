"""Google Drive connector, against an in-memory fake of the Drive v3 API (no network, no credentials)."""

import importlib
import json
import time

import pytest
from fastapi.testclient import TestClient

from rag import gdrive

ROOT = "rootFolder0001"
DOC = "application/vnd.google-apps.document"
SLIDES = "application/vnd.google-apps.presentation"
SHEET = "application/vnd.google-apps.spreadsheet"


class Resp:
    def __init__(self, status: int, body: bytes | dict):
        self.status_code = status
        self.content = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.text = self.content.decode(errors="replace")

    def json(self):
        return json.loads(self.content)


class FakeDrive:
    """files: id -> {name, mimeType, parent, modifiedTime, data, [size]}; folders are files too."""

    def __init__(self):
        self.files = {ROOT: {"name": "Policies", "mimeType": gdrive.FOLDER, "parent": "root"}}
        self.drives = [{"id": "0ADriveEng001", "name": "Engineering"}]  # shared drives
        self.fail: set[str] = set()  # ids whose download returns 500
        self.downloads: list[str] = []
        self.page_size = 2  # force pagination
        self.queries: list[str | None] = []

    def add(self, fid, name, mime, data=b"", parent=ROOT, modified="2026-01-01T00:00:00Z", **extra):
        self.files[fid] = {"name": name, "mimeType": mime, "parent": parent, "modifiedTime": modified,
                           "data": data, **extra}

    def _matches(self, q, f):
        """The subset of Drive's query language the connector uses."""
        if "trashed = false" not in q or ("mimeType = '%s'" % gdrive.FOLDER in q and f["mimeType"] != gdrive.FOLDER):
            return False
        if " in parents" in q:
            return f["parent"] == q.split("'")[1]
        if "sharedWithMe = true" in q:
            return f.get("shared", False)
        if "name contains " in q:
            needle = q.split("name contains ", 1)[1][1:-1].replace("\\'", "'").replace("\\\\", "\\")
            return needle.casefold() in f["name"].casefold()
        raise AssertionError(f"unexpected query {q!r}")

    def get(self, url, params=None, timeout=None):
        params = params or {}
        self.queries.append(params.get("q"))
        if url == gdrive.DRIVES_API:
            return Resp(200, {"drives": self.drives})
        assert url.startswith(gdrive.API)
        rest = url[len(gdrive.API):].strip("/")
        if not rest:  # files.list
            assert params["corpora"] == "allDrives"
            kids = sorted((fid for fid, f in self.files.items() if self._matches(params["q"], f)),
                          key=lambda fid: (self.files[fid]["mimeType"] != gdrive.FOLDER, self.files[fid]["name"]))
            start = int(params.get("pageToken", 0))
            page = kids[start : start + self.page_size]
            body = {"files": [{"id": fid, **{k: v for k, v in self.files[fid].items() if k not in ("data", "parent")}}
                              for fid in page]}
            if start + self.page_size < len(kids):
                body["nextPageToken"] = str(start + self.page_size)
            return Resp(200, body)
        fid, _, action = rest.partition("/")
        if fid not in self.files:
            return Resp(404, {"error": {"message": f"File not found: {fid}."}})
        f = self.files[fid]
        if action == "export":
            assert f["mimeType"] == {DOC: DOC, SLIDES: SLIDES}.get(f["mimeType"])
            assert params["mimeType"] == gdrive.EXPORTS[f["mimeType"]][0]
        elif params.get("alt") != "media":  # files.get metadata
            return Resp(200, {"id": fid, "name": f["name"], "mimeType": f["mimeType"]})
        if fid in self.fail:
            return Resp(500, {"error": {"message": "backend error"}})
        self.downloads.append(fid)
        return Resp(200, f["data"])


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(gdrive, "_sleep", waits.append)
    return waits


def listing(dest):
    return sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file() and p.name != gdrive.STATE_FILE)


@pytest.fixture
def drive():
    d = FakeDrive()
    d.add("doc1", "Vacation Policy", DOC, b"# Vacation\n\nUp to 5 days carry over.")
    d.add("pdf1", "handbook.pdf", "application/pdf", b"%PDF-1.4 fake")
    d.add("txt1", "notes.txt", "text/plain", b"plain notes")
    d.add("sub1", "HR", gdrive.FOLDER)
    d.add("doc2", "Benefits", DOC, b"# Benefits\n\nDental is covered.", parent="sub1")
    d.add("deck", "All hands", SLIDES, b"Slide 1\nQ4 goals")
    d.add("sheet", "Budget", SHEET)
    d.add("img", "logo.png", "image/png", b"\x89PNG")
    d.add("big", "huge.pdf", "application/pdf", b"", size=str(gdrive.MAX_BYTES + 1))
    return d


def test_folder_id_accepts_ids_and_urls():
    assert gdrive.folder_id("1AbCdEfGhIjKlMn") == "1AbCdEfGhIjKlMn"
    assert gdrive.folder_id("https://drive.google.com/drive/folders/1AbCdEfGhIjKlMn?usp=sharing") == "1AbCdEfGhIjKlMn"
    assert gdrive.folder_id("https://drive.google.com/drive/u/0/folders/1AbCdEfGhIjKlMn") == "1AbCdEfGhIjKlMn"
    assert gdrive.folder_id("https://drive.google.com/open?id=1AbCdEfGhIjKlMn") == "1AbCdEfGhIjKlMn"
    for bad in ("", "../etc", "abc", "x' or '1'='1"):
        with pytest.raises(ValueError):
            gdrive.folder_id(bad)


def test_first_sync_exports_downloads_and_reports_skips(drive, tmp_path):
    dest = tmp_path / "docs" / "gdrive"
    r = gdrive.sync_folder(drive, ROOT, dest)
    assert listing(dest) == ["All hands.txt", "HR/Benefits.md", "Vacation Policy.md", "handbook.pdf", "notes.txt"]
    assert (dest / "Vacation Policy.md").read_text().startswith("# Vacation")
    assert (dest / "handbook.pdf").read_bytes() == b"%PDF-1.4 fake"
    assert sorted(r.downloaded) == listing(dest) and r.removed == [] and r.unchanged == 0 and r.failed == {}
    assert set(r.skipped) == {"Budget", "logo.png", "huge.pdf"}
    assert "Sheets" in r.skipped["Budget"] and "MB" in r.skipped["huge.pdf"]
    assert gdrive.last_sync(dest)["downloaded"] == r.downloaded


def test_resync_only_fetches_changes_and_mirrors_deletes(drive, tmp_path):
    dest = tmp_path / "gdrive"
    gdrive.sync_folder(drive, ROOT, dest)
    (dest / "mine.md").write_text("my own file")  # not from Drive: never touched

    drive.downloads.clear()
    r = gdrive.sync_folder(drive, ROOT, dest)
    assert drive.downloads == [] and r.downloaded == [] and r.unchanged == 5

    drive.files["doc1"].update(data=b"# Vacation\n\nUp to 10 days.", modifiedTime="2026-02-01T00:00:00Z")
    drive.files["doc2"]["name"] = "Benefits 2026"  # renamed
    del drive.files["txt1"]  # deleted in Drive
    (dest / "handbook.pdf").unlink()  # deleted locally: comes back
    r = gdrive.sync_folder(drive, ROOT, dest)
    assert sorted(r.downloaded) == ["HR/Benefits 2026.md", "Vacation Policy.md", "handbook.pdf"]
    assert r.removed == ["notes.txt"]
    assert listing(dest) == ["All hands.txt", "HR/Benefits 2026.md", "Vacation Policy.md", "handbook.pdf", "mine.md"]
    assert "10 days" in (dest / "Vacation Policy.md").read_text()

    del drive.files["doc2"]  # the subfolder empties: its local folder goes too
    gdrive.sync_folder(drive, ROOT, dest)
    assert not (dest / "HR").exists() and (dest / "mine.md").exists()


def test_duplicate_and_unsafe_names(tmp_path):
    drive = FakeDrive()
    drive.add("aaaaaaaa1", "Policy", DOC, b"one")
    drive.add("bbbbbbbb2", "Policy", DOC, b"two")
    drive.add("cccccccc3", "../../escape.md", "text/markdown", b"x")
    drive.add("dddddddd4", ".hidden.txt", "text/plain", b"y")
    dest = tmp_path / "gdrive"
    gdrive.sync_folder(drive, ROOT, dest)
    assert listing(dest) == ["Policy [bbbbbbbb].md", "Policy.md", "_.._escape.md", "hidden.txt"]
    assert not (tmp_path / "escape.md").exists()


def test_failed_download_keeps_previous_copy_and_retries(drive, tmp_path):
    dest = tmp_path / "gdrive"
    gdrive.sync_folder(drive, ROOT, dest)
    drive.files["doc1"].update(data=b"new", modifiedTime="2026-03-01T00:00:00Z")
    drive.fail.add("doc1")
    r = gdrive.sync_folder(drive, ROOT, dest)
    assert list(r.failed) == ["Vacation Policy"] and "500" in r.failed["Vacation Policy"]  # keyed by Drive path
    assert (dest / "Vacation Policy.md").read_text().startswith("# Vacation")  # old copy kept
    drive.fail.clear()
    r = gdrive.sync_folder(drive, ROOT, dest)
    assert r.downloaded == ["Vacation Policy.md"] and (dest / "Vacation Policy.md").read_text() == "new"


def test_unknown_folder_changes_nothing(drive, tmp_path):
    dest = tmp_path / "gdrive"
    gdrive.sync_folder(drive, ROOT, dest)
    before = listing(dest)
    with pytest.raises(gdrive.DriveError, match="404.*shared"):
        gdrive.sync_folder(drive, "missingFolder123", dest)
    assert listing(dest) == before
    drive.add("notAFolder01", "x.pdf", "application/pdf", b"x")
    with pytest.raises(gdrive.DriveError, match="not a folder"):
        gdrive.sync_folder(drive, "notAFolder01", dest)
    assert listing(dest) == before


# ---------------------------------------------------------------- server


def _client(tmp_path, monkeypatch, drive=None):
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "local.md").write_text("# Local\nA local document.")
    monkeypatch.setenv("RAG_DOCS", str(docs))
    monkeypatch.setenv("RAG_INDEX", str(tmp_path / "idx"))
    monkeypatch.setenv("RAG_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_FAKE_LLM", "1")
    monkeypatch.setenv("RAG_RERANK", "0")
    for var in ("RAG_PASSWORD", "RAG_GDRIVE_CLIENT_ID", "RAG_GDRIVE_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    if drive is not None:
        monkeypatch.setenv("RAG_GDRIVE_FOLDER", f"https://drive.google.com/drive/folders/{ROOT}")
        monkeypatch.setattr(gdrive, "drive_session", lambda credentials=None: drive)
    else:
        monkeypatch.delenv("RAG_GDRIVE_FOLDER", raising=False)
    import server.app as app_module

    return TestClient(importlib.reload(app_module).app)


def _wait(c):
    for _ in range(200):
        st = c.get("/api/ingest").json()
        if st["state"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_server_sync_not_configured(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/documents").json()["gdrive"] is None
    r = c.post("/api/gdrive/sync")
    assert r.status_code == 404 and "connect it" in r.json()["detail"]


def test_server_sync_downloads_and_indexes(drive, tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, drive)
    info = c.get("/api/documents").json()["gdrive"]
    assert info == {"folder": ROOT, "dir": "gdrive", "last_sync": None}

    assert c.post("/api/gdrive/sync").json()["reason"] == "gdrive"
    st = _wait(c)
    assert st["state"] == "done", st
    assert "Vacation Policy.md" in st["sync"]["downloaded"] and "Budget" in st["sync"]["skipped"]
    assert "gdrive/HR/Benefits.md" in st["report"]["added"] and "local.md" in st["report"]["added"]
    hits = c.get("/api/search", params={"q": "dental", "k": 1, "mode": "bm25"}).json()["results"]
    assert hits[0]["source"] == "gdrive/HR/Benefits.md"
    assert c.get("/api/documents").json()["gdrive"]["last_sync"]["unchanged"] == 0

    # Deleted in Drive: removed locally and dropped from the index in the same job.
    del drive.files["doc2"]
    c.post("/api/gdrive/sync")
    st = _wait(c)
    assert st["sync"]["removed"] == ["HR/Benefits.md"] and st["report"]["removed"] == ["gdrive/HR/Benefits.md"]

    monkeypatch.setattr(gdrive, "sync_folder", lambda *a, **k: (_ for _ in ()).throw(gdrive.DriveError("no access")))
    c.post("/api/gdrive/sync")
    st = _wait(c)
    assert st["state"] == "error" and "no access" in st["error"] and st["reason"] == "gdrive"


# ---------------------------------------------------------------- "Connect Google Drive" (OAuth)

import base64  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import stat  # noqa: E402
from urllib.parse import parse_qs, urlsplit  # noqa: E402

from server import gdrive_connect  # noqa: E402

CLIENT_ID = "1234-abc.apps.googleusercontent.com"
SECRET = "GOCSPX-client-secret"
REFRESH = "1//refresh-token-from-google"


def _id_token(email):
    payload = base64.urlsafe_b64encode(json.dumps({"email": email}).encode()).rstrip(b"=").decode()
    return f"header.{payload}.signature"


class FakeGoogle:
    """Google's token and revoke endpoints."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.challenge: str | None = None  # from the authorization URL, checked against the verifier
        self.scope = f"openid https://www.googleapis.com/auth/userinfo.email {gdrive.SCOPE}"
        self.refresh = REFRESH

    def post(self, url, data):
        self.calls.append((url, data))
        if url == gdrive_connect.REVOKE_URL:
            return 200, {}
        assert url == gdrive_connect.TOKEN_URL and data["grant_type"] == "authorization_code"
        if data["code"] != "good-code" or data["client_secret"] != SECRET:
            return 400, {"error": "invalid_grant", "error_description": "Bad Request"}
        digest = hashlib.sha256(data["code_verifier"].encode()).digest()
        assert base64.urlsafe_b64encode(digest).rstrip(b"=").decode() == self.challenge  # PKCE
        assert data["redirect_uri"] == "http://testserver/api/gdrive/oauth/callback"
        return 200, {"access_token": "ya29", "refresh_token": self.refresh, "scope": self.scope,
                     "id_token": _id_token("pat@example.com")}


@pytest.fixture
def oauth(tmp_path, monkeypatch, drive):
    """The app with nothing configured, a fake Google, and `drive` as what the signed-in account sees."""
    c = _client(tmp_path, monkeypatch)
    import server.app as app_module

    google = FakeGoogle()
    app_module.drive_connection.post = google.post
    sessions = []
    monkeypatch.setattr(gdrive, "oauth_session", lambda *a: sessions.append(a) or drive)
    return c, google, sessions, tmp_path / "data" / "gdrive.json"


def _start(c, google):
    r = c.post("/api/gdrive/connect")
    assert r.status_code == 200
    q = {k: v[0] for k, v in parse_qs(urlsplit(r.json()["url"]).query).items()}
    google.challenge = q["code_challenge"]
    return q


def _callback(c, **params):
    return c.get("/api/gdrive/oauth/callback", params=params, follow_redirects=False)


def test_connect_drive_from_the_ui(oauth, drive):
    c, google, sessions, saved = oauth
    st = c.get("/api/gdrive").json()
    assert st["mode"] == "oauth" and st["client"] == {"configured": False, "from_env": False, "client_id": None}
    assert st["redirect_uri"] == "http://testserver/api/gdrive/oauth/callback" and st["account"] is None
    assert c.post("/api/gdrive/connect").status_code == 400  # no OAuth client yet

    # One-time setup: the OAuth client.
    assert c.put("/api/gdrive/client", json={"client_id": "nope", "client_secret": "x"}).status_code == 400
    r = c.put("/api/gdrive/client", json={"client_id": f" {CLIENT_ID} ", "client_secret": SECRET})
    assert r.json()["client"] == {"configured": True, "from_env": False, "client_id": CLIENT_ID}
    assert SECRET not in r.text

    # Sign in with Google.
    q = _start(c, google)
    assert q["client_id"] == CLIENT_ID and q["access_type"] == "offline" and q["code_challenge_method"] == "S256"
    assert gdrive.SCOPE in q["scope"].split() and q["redirect_uri"].endswith("/api/gdrive/oauth/callback")
    r = _callback(c, state=q["state"], code="good-code")
    assert r.status_code == 303 and r.headers["location"] == "/#/documents"
    st = c.get("/api/gdrive").json()
    assert st["account"] == "pat@example.com" and st["error"] is None
    assert REFRESH not in json.dumps(st) and SECRET not in json.dumps(st)
    assert stat.S_IMODE(os.stat(saved).st_mode) == 0o600 and json.loads(saved.read_text())["refresh_token"] == REFRESH

    # The same state can't be used twice.
    assert _callback(c, state=q["state"], code="good-code").status_code == 303
    assert len([u for u, _ in google.calls if u == gdrive_connect.TOKEN_URL]) == 1

    # Choose the folder: checked against Drive before it's saved.
    assert c.put("/api/gdrive/folder", json={"folder": "not a link!"}).status_code == 400
    r = c.put("/api/gdrive/folder", json={"folder": "https://drive.google.com/drive/folders/missingFolder123"})
    assert r.status_code == 400 and "404" in r.json()["detail"]
    r = c.put("/api/gdrive/folder", json={"folder": f"https://drive.google.com/drive/folders/{ROOT}?usp=sharing"})
    assert r.json()["folder"] == ROOT and r.json()["folder_name"] == "Policies"
    assert sessions[-1][:4] == (CLIENT_ID, SECRET, REFRESH, gdrive_connect.TOKEN_URL) and callable(sessions[-1][4])

    # Sync it, through the same job as before.
    assert c.get("/api/documents").json()["gdrive"]["folder"] == ROOT
    assert c.post("/api/gdrive/sync").json()["reason"] == "gdrive"
    st = _wait(c)
    assert st["state"] == "done" and "gdrive/HR/Benefits.md" in st["report"]["added"]
    assert c.get("/api/gdrive").json()["last_sync"]["downloaded"]

    # Disconnect: the token is revoked and forgotten; synced files stay.
    r = c.delete("/api/gdrive/account")
    assert r.json()["account"] is None and google.calls[-1] == (gdrive_connect.REVOKE_URL, {"token": REFRESH})
    assert "refresh_token" not in json.loads(saved.read_text())
    assert c.post("/api/gdrive/sync").status_code == 404
    assert any(d["name"] == "gdrive/HR/Benefits.md" for d in c.get("/api/documents").json()["documents"])


def test_callback_errors_are_shown_only_to_the_browser_that_signed_in(oauth):
    c, google, _, saved = oauth
    c.put("/api/gdrive/client", json={"client_id": CLIENT_ID, "client_secret": SECRET})

    # Another browser (no state cookie), or a made-up state: nothing is written.
    q = _start(c, google)
    other = TestClient(c.app)
    assert _callback(other, state=q["state"], code="good-code").status_code == 303
    assert _callback(other, state="made-up").status_code == 303
    st = c.get("/api/gdrive").json()
    assert st["account"] is None and st["error"] is None

    # The state was used up by that attempt, so this browser's own callback is now refused, with a message.
    _callback(c, state=q["state"], code="good-code")
    assert "expired or was already used" in c.get("/api/gdrive").json()["error"]

    q = _start(c, google)  # cancelled on Google's screen
    _callback(c, state=q["state"], error="access_denied")
    assert c.get("/api/gdrive").json()["error"] == "access was not granted"

    google.scope = "openid email"  # Drive box unticked on the consent screen
    q = _start(c, google)
    _callback(c, state=q["state"], code="good-code")
    assert "Google Drive access wasn't allowed" in c.get("/api/gdrive").json()["error"]

    q = _start(c, google)  # bad secret
    c.put("/api/gdrive/client", json={"client_id": CLIENT_ID, "client_secret": "wrong"})
    _callback(c, state=q["state"], code="good-code")
    assert "check the client ID and secret" in c.get("/api/gdrive").json()["error"]
    assert c.delete("/api/gdrive/error").json()["error"] is None
    assert "refresh_token" not in json.loads(saved.read_text())


def test_new_client_signs_out_and_reconnect_revokes_old_token(oauth):
    c, google, _, _ = oauth
    c.put("/api/gdrive/client", json={"client_id": CLIENT_ID, "client_secret": SECRET})
    _callback(c, state=_start(c, google)["state"], code="good-code")
    google.refresh = "1//second-token"
    _callback(c, state=_start(c, google)["state"], code="good-code")
    assert google.calls[-1] == (gdrive_connect.REVOKE_URL, {"token": REFRESH})

    other_id = "9999-xyz.apps.googleusercontent.com"
    st = c.put("/api/gdrive/client", json={"client_id": other_id, "client_secret": SECRET}).json()
    assert st["account"] is None and google.calls[-1] == (gdrive_connect.REVOKE_URL, {"token": "1//second-token"})


def test_env_folder_takes_precedence(drive, tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, drive)
    st = c.get("/api/gdrive").json()
    assert st["mode"] == "env" and st["folder"] == ROOT
    assert c.put("/api/gdrive/client", json={"client_id": CLIENT_ID, "client_secret": SECRET}).status_code == 409
    assert c.post("/api/gdrive/connect").status_code == 409


def test_env_client_is_used_and_locked(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)  # sets up the folders (and clears these), then reload with them
    monkeypatch.setenv("RAG_GDRIVE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("RAG_GDRIVE_CLIENT_SECRET", SECRET)
    import server.app as app_module

    c = TestClient(importlib.reload(app_module).app)
    assert c.get("/api/gdrive").json()["client"] == {"configured": True, "from_env": True, "client_id": CLIENT_ID}
    assert c.put("/api/gdrive/client", json={"client_id": CLIENT_ID, "client_secret": "x"}).status_code == 409
    assert c.post("/api/gdrive/connect").status_code == 200


def test_callback_needs_no_login_but_the_rest_does(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    monkeypatch.setenv("RAG_PASSWORD", "correct horse battery staple")
    import server.app as app_module

    c = TestClient(importlib.reload(app_module).app)
    for method, path in [("GET", "/api/gdrive"), ("PUT", "/api/gdrive/client"), ("POST", "/api/gdrive/connect"),
                         ("PUT", "/api/gdrive/folder"), ("DELETE", "/api/gdrive/account")]:
        assert c.request(method, path).status_code == 401, path
    r = _callback(c, state="x", code="y")
    assert r.status_code == 303 and not (tmp_path / "data" / "gdrive.json").exists()


def test_revoked_token_is_reported_as_reconnect():
    from google.auth.exceptions import RefreshError

    class Dead:
        def __init__(self, code):
            self.code = code

        def get(self, *a, **k):  # google-auth raises it with (message, response body)
            raise RefreshError(f"{self.code}: Token has been expired or revoked.", {"error": self.code})

    with pytest.raises(gdrive.DriveError, match=r"expired or was revoked \(invalid_grant\) - connect Google Drive again$"):
        gdrive._OAuthSession(Dead("invalid_grant")).get(gdrive.API)
    with pytest.raises(gdrive.DriveError, match="OAuth client ID or secret"):
        gdrive._OAuthSession(Dead("invalid_client")).get(gdrive.API)


# ---------------------------------------------------------------- choosing the folder in the app


def test_browse_locations_folders_and_search(drive):
    drive.add("shr1", "Team Handbook", gdrive.FOLDER, parent="someoneElse", shared=True)
    drive.add("shr2", "Budgets", gdrive.FOLDER, parent="someoneElse", shared=True)
    drive.add("note", "Shared note.txt", "text/plain", parent="someoneElse", shared=True)  # files aren't listed
    drive.add("quot", "Bob's \\ notes", gdrive.FOLDER, parent="sub1")

    mine = gdrive.browse(drive, "root")
    assert mine == {"folders": [{"id": ROOT, "name": "Policies"}], "files": None, "truncated": False}

    inside = gdrive.browse(drive, ROOT)
    assert inside["folders"] == [{"id": "sub1", "name": "HR"}]
    # 7 files directly in it; by type, all but the sheet and the image can be synced (size isn't checked here).
    assert inside["files"] == {"total": 7, "supported": 5}

    shared = gdrive.browse(drive, "shared")
    assert [f["name"] for f in shared["folders"]] == ["Budgets", "Team Handbook"] and shared["files"] is None

    assert gdrive.browse(drive, "drives")["folders"] == [{"id": "0ADriveEng001", "name": "Engineering"}]

    found = gdrive.browse(drive, search="hand")
    assert [f["name"] for f in found["folders"]] == ["Team Handbook"]
    # Quotes and backslashes are escaped in the query, not able to end the string literal.
    assert gdrive.browse(drive, search="Bob's \\")["folders"] == [{"id": "quot", "name": "Bob's \\ notes"}]
    assert drive.queries[-1].endswith("name contains 'Bob\\'s \\\\'")

    with pytest.raises(ValueError):
        gdrive.browse(drive, "x' or '1'='1")


def test_browse_caps_long_listings(drive, monkeypatch):
    monkeypatch.setattr(gdrive, "BROWSE_LIMIT", 3)
    for i in range(5):
        drive.add(f"many{i:08d}", f"Folder {i}", gdrive.FOLDER, parent="root")
    r = gdrive.browse(drive, "root")
    assert len(r["folders"]) == 3 and r["truncated"] is True


def test_choose_folder_by_browsing(oauth, drive):
    c, google, sessions, saved = oauth
    c.put("/api/gdrive/client", json={"client_id": CLIENT_ID, "client_secret": SECRET})
    assert c.get("/api/gdrive/folders").status_code == 400  # not signed in yet
    q = _start(c, google)
    _callback(c, state=q["state"], code="good-code")

    r = c.get("/api/gdrive/folders").json()  # My Drive
    assert r["folders"] == [{"id": ROOT, "name": "Policies"}]
    r = c.get("/api/gdrive/folders", params={"parent": ROOT}).json()
    assert r["folders"] == [{"id": "sub1", "name": "HR"}] and r["files"]["supported"] == 5
    assert c.get("/api/gdrive/folders", params={"q": "  hr "}).json()["folders"] == [{"id": "sub1", "name": "HR"}]
    assert c.get("/api/gdrive/folders", params={"parent": "../etc"}).status_code == 400

    # Choosing sends the folder's id; the server checks it and remembers its name.
    st = c.put("/api/gdrive/folder", json={"folder": "sub1aaaaaaaa"}).status_code
    assert st == 400  # unknown id
    drive.files["sub1aaaaaaaa"] = drive.files.pop("sub1")  # a realistic-length id
    for f in drive.files.values():
        if f["parent"] == "sub1":
            f["parent"] = "sub1aaaaaaaa"
    st = c.put("/api/gdrive/folder", json={"folder": "sub1aaaaaaaa"}).json()
    assert st["folder"] == "sub1aaaaaaaa" and st["folder_name"] == "HR"
    assert len(sessions) == 1  # one Google session reused across all those calls

    c.post("/api/gdrive/sync")
    st = _wait(c)
    assert st["state"] == "done" and st["sync"]["downloaded"] == ["Benefits.md"]


def test_no_browsing_when_the_server_sets_the_folder(drive, tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, drive)
    assert c.get("/api/gdrive/folders").status_code == 409


def test_files_not_from_drive_are_never_overwritten(drive, tmp_path):
    dest = tmp_path / "gdrive"
    dest.mkdir()
    (dest / "notes.txt").write_text("my own notes")  # same name as Drive's notes.txt
    r = gdrive.sync_folder(drive, ROOT, dest)
    assert (dest / "notes.txt").read_text() == "my own notes"
    assert (dest / "notes [txt1].txt").read_bytes() == b"plain notes"
    assert "notes [txt1].txt" in r.downloaded

    # Later syncs keep it that way, and removing it in Drive removes only the downloaded copy.
    del drive.files["txt1"]
    r = gdrive.sync_folder(drive, ROOT, dest)
    assert r.removed == ["notes [txt1].txt"] and (dest / "notes.txt").read_text() == "my own notes"


def test_files_of_an_interrupted_sync_stay_ours(drive, tmp_path, monkeypatch):
    dest = tmp_path / "gdrive"
    real_write = gdrive._write_atomic

    def crash_after(n):
        """Sync with the process dying after `n` downloaded files have been written."""
        written = []

        def write(path, data):
            if path.name != gdrive.STATE_FILE:
                if len(written) == n:
                    raise KeyboardInterrupt
                written.append(path.name)
            real_write(path, data)

        monkeypatch.setattr(gdrive, "_write_atomic", write)
        with pytest.raises(KeyboardInterrupt):
            gdrive.sync_folder(drive, ROOT, dest)
        monkeypatch.setattr(gdrive, "_write_atomic", real_write)
        return written

    assert len(crash_after(2)) == 2

    # The next sync updates those files in place (no "[id]" duplicates) ...
    r = gdrive.sync_folder(drive, ROOT, dest)
    assert listing(dest) == ["All hands.txt", "HR/Benefits.md", "Vacation Policy.md", "handbook.pdf", "notes.txt"]
    assert "claimed" not in json.loads((dest / gdrive.STATE_FILE).read_text())

    # ... and a file the interrupted sync wrote that is gone from Drive meanwhile is removed.
    drive.files["doc1"].update(data=b"# v2", modifiedTime="2026-03-01T00:00:00Z")
    drive.add("new1", "New.md", "text/markdown", b"new")
    assert crash_after(1) == ["New.md"]
    del drive.files["new1"]
    gdrive.sync_folder(drive, ROOT, dest)
    assert "New.md" not in listing(dest) and (dest / "Vacation Policy.md").read_text() == "# v2"



# ---------------------------------------------------------------- review fixes


class Flaky:
    """A session that answers with `answers` (status codes, or an exception) before passing calls to `inner`."""

    def __init__(self, inner, answers, body=None, headers=None):
        self.inner, self.answers, self.body, self.headers, self.calls = inner, list(answers), body, headers, 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.answers:
            a = self.answers.pop(0)
            if isinstance(a, Exception):
                raise a
            r = Resp(a, self.body or {"error": {"message": "try later"}})
            r.headers = self.headers or {}
            return r
        return self.inner.get(url, params=params, timeout=timeout)


def test_transient_drive_errors_are_retried(drive, tmp_path, no_backoff_sleep):
    import requests

    dest = tmp_path / "gdrive"
    session = Flaky(drive, [429, 503, requests.ConnectionError("reset")], headers={"Retry-After": "3"})
    r = gdrive.sync_folder(session, ROOT, dest)
    assert len(r.downloaded) == 5 and r.failed == {}
    assert no_backoff_sleep[:2] == [3.0, 3.0] and len(no_backoff_sleep) == 3

    rate_limited = {"error": {"message": "slow down", "errors": [{"reason": "userRateLimitExceeded"}]}}
    session = Flaky(drive, [403], body=rate_limited)
    assert gdrive.folder_info(session, ROOT)["name"] == "Policies" and session.calls == 2

    forbidden = {"error": {"message": "no access", "errors": [{"reason": "insufficientFilePermissions"}]}}
    session = Flaky(drive, [403], body=forbidden)  # a real 403 isn't retried
    with pytest.raises(gdrive.DriveError, match="403"):
        gdrive.folder_info(session, ROOT)
    assert session.calls == 1

    session = Flaky(drive, [500] * gdrive.MAX_TRIES)
    with pytest.raises(gdrive.DriveError, match="500"):
        gdrive.folder_info(session, ROOT)
    assert session.calls == gdrive.MAX_TRIES


def test_incomplete_listing_changes_nothing(drive, tmp_path):
    dest = tmp_path / "gdrive"
    gdrive.sync_folder(drive, ROOT, dest)
    before = listing(dest)

    real_get = drive.get

    def incomplete(url, params=None, timeout=None):
        resp = real_get(url, params, timeout)
        if url == gdrive.API and params and params.get("q", "").startswith("'sub1'"):
            body = resp.json()
            body["incompleteSearch"] = True
            body["files"] = []  # Drive left the folder's files out
            return Resp(200, body)
        return resp

    drive.get = incomplete
    with pytest.raises(gdrive.DriveError, match="incomplete listing"):
        gdrive.sync_folder(drive, ROOT, dest)
    assert listing(dest) == before


def test_long_names_fit_the_file_system(tmp_path):
    d = FakeDrive()
    d.add("jp1", "社内規程" * 40, DOC, b"# x\n\nbody")
    d.add("en1", "a" * 300 + ".md", "text/markdown", b"long")
    dest = tmp_path / "gdrive"
    r = gdrive.sync_folder(d, ROOT, dest)
    assert r.failed == {} and len(r.downloaded) == 2
    for name in r.downloaded:
        assert len(name.encode()) <= gdrive.MAX_NAME_BYTES + len(".md") and (dest / name).is_file()


def test_expired_sign_in_is_shown_and_forgotten(oauth, drive):
    from google.auth.exceptions import RefreshError

    c, google, sessions, saved = oauth
    c.put("/api/gdrive/client", json={"client_id": CLIENT_ID, "client_secret": SECRET})
    q = _start(c, google)
    _callback(c, state=q["state"], code="good-code")
    c.put("/api/gdrive/folder", json={"folder": ROOT})
    on_auth_error = sessions[0][4]

    class Dead:
        def get(self, *a, **k):
            raise RefreshError("invalid_grant: Token has been expired or revoked.", {"error": "invalid_grant"})

    with pytest.raises(gdrive.DriveAuthError):
        gdrive._OAuthSession(Dead(), on_auth_error).get(gdrive.API)
    st = c.get("/api/gdrive").json()
    assert st["account"] is None and "expired or was revoked" in st["error"]
    assert st["folder"] == ROOT  # kept: connecting again needs no new folder choice
    assert "refresh_token" not in json.loads(saved.read_text())

    # A retryable refresh failure (Google's own hiccup) is just an error for this attempt.
    class Hiccup:
        def get(self, *a, **k):
            raise RefreshError("temporarily unavailable", retryable=True)

    with pytest.raises(gdrive.DriveError, match="try again") as e:
        gdrive._OAuthSession(Hiccup(), lambda err: pytest.fail("not an auth failure")).get(gdrive.API)
    assert not isinstance(e.value, gdrive.DriveAuthError)


def test_unselect_folder_and_synced_files_are_protected(oauth, drive):
    c, google, sessions, saved = oauth
    c.put("/api/gdrive/client", json={"client_id": CLIENT_ID, "client_secret": SECRET})
    q = _start(c, google)
    _callback(c, state=q["state"], code="good-code")
    c.put("/api/gdrive/folder", json={"folder": ROOT})
    c.post("/api/gdrive/sync")
    assert _wait(c)["state"] == "done"

    docs = {r["name"]: r for r in c.get("/api/documents").json()["documents"]}
    assert docs["gdrive/notes.txt"]["gdrive"] and not any(r["gdrive"] for n, r in docs.items() if not n.startswith("gdrive/"))
    r = c.delete("/api/documents/gdrive/notes.txt")
    assert r.status_code == 409 and "Google Drive" in r.json()["detail"]

    st = c.delete("/api/gdrive/folder").json()
    assert st["folder"] is None and st["folder_name"] is None and st["account"] == "pat@example.com"
    assert c.post("/api/gdrive/sync").status_code == 404  # nothing to sync
    docs = {r["name"]: r for r in c.get("/api/documents").json()["documents"]}
    assert "gdrive/notes.txt" in docs and not docs["gdrive/notes.txt"]["gdrive"]  # kept, now an ordinary file
    assert c.delete("/api/documents/gdrive/notes.txt").status_code == 200
