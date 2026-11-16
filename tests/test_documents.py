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
