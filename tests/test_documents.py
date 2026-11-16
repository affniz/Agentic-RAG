import importlib
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "vacation.md").write_text("# Vacation\nIntro note.\n\n## Carry-over\nUp to 5 unused days carry over until March 31.")
    monkeypatch.setenv("RAG_DOCS", str(docs))
    monkeypatch.setenv("RAG_INDEX", str(tmp_path / "idx"))
    monkeypatch.setenv("RAG_FAKE_LLM", "1")
    monkeypatch.setenv("RAG_RERANK", "0")
    import server.app as app_module

    app_module = importlib.reload(app_module)
    c = TestClient(app_module.app)
    c.docs = docs
    return c


def wait_ingest(c):
    for _ in range(200):
        st = c.get("/api/ingest").json()
        if st["state"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("ingest did not finish")


def statuses(c):
    return {d["name"]: d["status"] for d in c.get("/api/documents").json()["documents"]}


def test_upload_index_search_delete(client):
    c = client
    assert statuses(c) == {"vacation.md": "new"}
    assert c.post("/api/ingest", json={}).json()["state"] == "running"
    st = wait_ingest(c)
    assert st["state"] == "done" and st["report"]["added"] == ["vacation.md"]
    assert statuses(c) == {"vacation.md": "indexed"}

    r = c.post("/api/documents", files=[
        ("files", ("travel.txt", b"Travel Policy\n\nHotel cap is $350 per night in London.", "text/plain")),
        ("files", ("evil.exe", b"MZ", "application/octet-stream")),
        ("files", (".hidden.md", b"x", "text/markdown")),
        ("files", ("empty.md", b"", "text/markdown")),
    ]).json()
    assert r["saved"] == ["travel.txt"] and r["ingest"]["reason"] == "upload"
    assert {x["name"]: x["reason"].split(" ")[0] for x in r["rejected"]} == {
        "evil.exe": "unsupported", ".hidden.md": "invalid", "empty.md": "empty"}
    wait_ingest(c)
    assert statuses(c) == {"vacation.md": "indexed", "travel.txt": "indexed"}

    s = c.get("/api/search", params={"q": "london hotel", "k": 1, "mode": "bm25"}).json()
    assert s["results"][0]["source"] == "travel.txt" and s["reranked"] is False
    assert c.get("/api/search", params={"q": "x", "mode": "magic"}).status_code == 400

    chunks = c.get("/api/documents/vacation.md/chunks").json()
    assert chunks["doc_note"] == "Intro note." and chunks["chunks"][1]["section"] == "Vacation > Carry-over"

    # Edit on disk -> "changed"; delete -> re-index removes it.
    (c.docs / "travel.txt").write_text("Travel Policy\n\nHotel cap is $400.")
    assert statuses(c)["travel.txt"] == "changed"
    assert c.delete("/api/documents/travel.txt").json()["ingest"]["reason"] == "delete"
    st = wait_ingest(c)
    assert st["report"]["removed"] == ["travel.txt"] and statuses(c) == {"vacation.md": "indexed"}


def test_rejects_path_traversal_and_concurrent_ingest(client):
    c = client
    assert c.delete("/api/documents/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)
    assert c.delete("/api/documents/nope.md").status_code == 404
    r = c.post("/api/documents", files=[("files", ("../../escape.md", b"# hi\ntext", "text/markdown"))]).json()
    assert r["saved"] == ["escape.md"] and (c.docs / "escape.md").exists()  # directory part stripped
    if c.get("/api/ingest").json()["state"] == "running":
        assert c.post("/api/ingest", json={}).status_code == 409
    wait_ingest(c)


def test_scanned_pdf_shows_no_text_status(client):
    from tests.test_store import _blank_pdf

    _blank_pdf(client.docs / "scan.pdf")
    client.post("/api/ingest", json={})
    st = wait_ingest(client)
    assert st["report"]["empty"] == ["scan.pdf"]
    body = client.get("/api/documents").json()
    assert {d["name"]: d["status"] for d in body["documents"]}["scan.pdf"] == "empty"
    assert body["ocr"] is False and body["needs_ingest"] is False


def test_transcribe_and_undo_one_pdf(client):
    from tests.test_store import _blank_pdf

    c = client
    _blank_pdf(c.docs / "scan.pdf", pages=3)
    c.post("/api/ingest", json={})
    wait_ingest(c)
    row = next(d for d in c.get("/api/documents").json()["documents"] if d["name"] == "scan.pdf")
    assert row["status"] == "empty" and row["pages"] == 3 and row["transcribed"] is False
    assert row["transcribe_cost_usd"] == pytest.approx(0.0015)

    st = c.post("/api/documents/scan.pdf/transcribe").json()
    assert st["reason"] == "transcribe" and st["file"] == "scan.pdf"
    st = wait_ingest(c)
    assert st["state"] == "done" and st["report"]["updated"] == ["scan.pdf"]
    row = next(d for d in c.get("/api/documents").json()["documents"] if d["name"] == "scan.pdf")
    assert row["status"] == "indexed" and row["transcribed"] is True and row["chunks"] > 0
    first = c.get("/api/documents/scan.pdf/chunks").json()["chunks"][0]
    assert first["section"] == "Demo transcription of scan.pdf" and "no model was called" in first["text"]

    # Undo: the transcription route must not be mistaken for deleting the document.
    assert c.delete("/api/documents/scan.pdf/transcription").json()["reason"] == "untranscribe"
    wait_ingest(c)
    assert (c.docs / "scan.pdf").exists()
    assert statuses(c)["scan.pdf"] == "empty"
    assert c.delete("/api/documents/scan.pdf/transcription").status_code == 404  # nothing to undo
    assert c.post("/api/documents/vacation.md/transcribe").status_code == 404  # PDFs only


def test_unreadable_pdf_shows_error_status(client):
    from tests.test_store import _text_pdf

    _text_pdf(client.docs / "locked.pdf", "Secret contents that need a password to open.", user_password="pw")
    client.post("/api/ingest", json={})
    st = wait_ingest(client)
    assert st["state"] == "done" and "password-protected" in st["report"]["failed"]["locked.pdf"]
    body = client.get("/api/documents").json()
    row = next(d for d in body["documents"] if d["name"] == "locked.pdf")
    assert row["status"] == "error" and "password-protected" in row["error"]
    assert body["needs_ingest"] is False  # re-indexing again won't help; the reason says what will
    assert next(d for d in body["documents"] if d["name"] == "vacation.md")["status"] == "indexed"


def test_deleting_a_transcribed_pdf_deletes_its_transcription(client):
    import io

    from tests.test_store import _blank_pdf

    c = client
    _blank_pdf(c.docs / "scan.pdf", pages=2)
    data = (c.docs / "scan.pdf").read_bytes()
    c.post("/api/ingest", json={})
    wait_ingest(c)
    c.post("/api/documents/scan.pdf/transcribe")
    wait_ingest(c)
    assert next(d for d in c.get("/api/documents").json()["documents"] if d["name"] == "scan.pdf")["transcribed"]

    c.delete("/api/documents/scan.pdf")
    wait_ingest(c)
    # Upload the very same file again: it must come back as untranscribed, not reuse the old text.
    c.post("/api/documents", files=[("files", ("scan.pdf", io.BytesIO(data), "application/pdf"))])
    wait_ingest(c)
    row = next(d for d in c.get("/api/documents").json()["documents"] if d["name"] == "scan.pdf")
    assert row["transcribed"] is False and row["status"] == "empty"
