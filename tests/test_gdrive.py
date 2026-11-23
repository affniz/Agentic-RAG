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
        self.files = {ROOT: {"name": "Policies", "mimeType": gdrive.FOLDER, "parent": None}}
        self.fail: set[str] = set()  # ids whose download returns 500
        self.downloads: list[str] = []
        self.page_size = 2  # force pagination

    def add(self, fid, name, mime, data=b"", parent=ROOT, modified="2026-01-01T00:00:00Z", **extra):
        self.files[fid] = {"name": name, "mimeType": mime, "parent": parent, "modifiedTime": modified,
                           "data": data, **extra}

    def get(self, url, params=None, timeout=None):
        params = params or {}
        assert url.startswith(gdrive.API)
        rest = url[len(gdrive.API):].strip("/")
        if not rest:  # files.list
            parent = params["q"].split("'")[1]
            kids = sorted((fid for fid, f in self.files.items() if f["parent"] == parent),
                          key=lambda fid: self.files[fid]["name"])
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
    monkeypatch.setenv("RAG_FAKE_LLM", "1")
    monkeypatch.setenv("RAG_RERANK", "0")
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
    assert r.status_code == 404 and "RAG_GDRIVE_FOLDER" in r.json()["detail"]


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
