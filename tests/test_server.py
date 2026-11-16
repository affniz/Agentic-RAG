import importlib
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "vacation.md").write_text("# Vacation\n## Carry-over\nUp to 5 unused days carry over until March 31.")
    from rag.store import KnowledgeBase

    KnowledgeBase(tmp_path / "idx", rerank=False).ingest(docs)
    monkeypatch.setenv("RAG_INDEX", str(tmp_path / "idx"))
    monkeypatch.setenv("RAG_FAKE_LLM", "1")
    monkeypatch.setenv("RAG_RERANK", "0")
    import server.app as app_module

    app_module = importlib.reload(app_module)
    app_module.client.delay = 0
    return TestClient(app_module.app)


def events(res):
    return [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ")]


def test_chat_stream_and_conversation_roundtrip(client):
    assert client.get("/api/health").json()["documents"] == 1

    evs = events(client.post("/api/chat", json={"message": "how many days carry over?"}))
    types = [e["type"] for e in evs]
    assert types[0] == "conversation" and types[-1] == "done"
    assert {"tool_call", "tool_result", "text_delta", "answer", "usage"} <= set(types)
    answer = next(e for e in evs if e["type"] == "answer")
    assert answer["citations"][0]["source"] == "vacation.md"
    cid = evs[0]["id"]

    # Follow-up goes to the same conversation and both turns are stored.
    events(client.post("/api/chat", json={"message": "and when?", "conversation_id": cid}))
    conv = client.get(f"/api/conversations/{cid}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    assert conv["messages"][1]["steps"][0]["status"] == "done"

    assert len(client.get("/api/conversations").json()) == 1
    assert client.delete(f"/api/conversations/{cid}").json() == {"ok": True}
    assert client.get(f"/api/conversations/{cid}").status_code == 404


def test_chunk_endpoint_and_errors(client):
    r = client.get("/api/chunks/vacation.md/0").json()
    assert r["chunks"][0]["cited"] and r["chunks"][0]["section"] == "Vacation > Carry-over"
    assert client.get("/api/chunks/vacation.md/99").status_code == 404
    assert client.post("/api/chat", json={"message": "x", "model": "gpt-4"}).status_code == 400
    assert client.post("/api/chat", json={"message": "x", "conversation_id": "nope"}).status_code == 404
