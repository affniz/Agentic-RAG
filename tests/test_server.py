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
    monkeypatch.setenv("RAG_DOCS", str(docs))
    monkeypatch.setenv("RAG_DATA", str(tmp_path / "data"))
    evals = tmp_path / "evals"
    (evals / "mini").mkdir(parents=True)
    (evals / "mini" / "config.json").write_text(json.dumps({"docs": "../../docs", "today": "2026-11-16"}))
    (evals / "mini" / "questions.jsonl").write_text(json.dumps({
        "id": "carry", "category": "simple", "question": "How many unused days carry over?",
        "evidence": ["Up to 5 unused days"], "sources": ["vacation.md"], "reference": "Up to 5 unused days, until March 31."}) + "\n")
    monkeypatch.setenv("RAG_EVALS", str(evals))
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


def test_settings_apply_to_the_agent(client):
    s = client.get("/api/settings").json()
    assert s["max_steps"] == 12 and s["search_k"] is None and s["model"] == "claude-haiku-5-5"
    assert client.put("/api/settings", json={**s, "model": "gpt-4"}).status_code == 400
    assert client.put("/api/settings", json={**s, "max_steps": 0}).status_code == 422
    client.put("/api/settings", json={**s, "search_k": 2, "max_steps": 4, "effort": "high"})
    evs = events(client.post("/api/chat", json={"message": "carry over"}))
    import server.app as app_module

    agent = app_module.conversations[evs[0]["id"]].agent
    assert (agent.max_steps, agent.search.k, agent.effort) == (4, 2, "high")
    result = next(e for e in evs if e["type"] == "tool_result")
    assert result["results"] == 1  # one chunk in the index


def test_feedback_roundtrip_and_export(client, tmp_path):
    evs = events(client.post("/api/chat", json={"message": "how many days carry over?"}))
    cid, turn_id = evs[0]["id"], evs[0]["turn_id"]
    url = f"/api/feedback/{turn_id}"
    assert client.put(url, json={"conversation_id": cid, "rating": "up"}).json()["feedback"]["rating"] == "up"
    assert client.get(f"/api/conversations/{cid}").json()["messages"][1]["feedback"]["rating"] == "up"
    assert client.put("/api/feedback/nope", json={"conversation_id": cid, "rating": "up"}).status_code == 404

    evs2 = events(client.post("/api/chat", json={"message": "and when?", "conversation_id": cid}))
    client.put(f"/api/feedback/{evs2[0]['turn_id']}", json={"conversation_id": cid, "rating": "down"})
    fb = client.get("/api/feedback").json()
    assert {f["rating"] for f in fb} == {"up", "down"} and fb[0]["followup"]

    r = client.post("/api/feedback/export").json()
    assert r["questions"] == 1 and len(r["skipped"]) == 1  # 👎 without a comment can't become a question
    q = json.loads((tmp_path / "evals" / "feedback" / "questions.jsonl").read_text())
    assert q["question"] == "how many days carry over?" and q["sources"] == ["vacation.md"]
    assert "[vacation.md#" not in q["reference"]

    client.put(url, json={"conversation_id": cid, "rating": None})
    assert len(client.get("/api/feedback").json()) == 1


def test_reset_clears_conversations(client):
    events(client.post("/api/chat", json={"message": "carry over"}))
    assert client.delete("/api/conversations").json() == {"kept_busy": 0}
    assert client.get("/api/conversations").json() == []


def wait_for_job(client):
    import time

    for _ in range(200):
        state = client.get("/api/evals/job").json()
        if state["state"] != "running":
            return state
        time.sleep(0.05)
    raise AssertionError("eval did not finish")


def test_eval_dashboard_runs(client):
    sets = client.get("/api/evals/sets").json()
    assert sets[0]["name"] == "mini" and sets[0]["categories"] == {"simple": 1}

    est = client.post("/api/evals/estimate", json={"kind": "answers", "set": "mini"}).json()
    assert est["basis"] == "demo" and est["total_usd"] == 0

    client.post("/api/evals/run", json={"kind": "retrieval", "set": "mini"})
    assert wait_for_job(client)["state"] == "done"
    client.post("/api/evals/run", json={"kind": "answers", "set": "mini", "systems": ["agentic", "naive"]})
    job = wait_for_job(client)
    assert job["state"] == "done" and job["done"] == job["total"] == 2

    runs = client.get("/api/evals/runs").json()
    assert [r["kind"] for r in runs] == ["retrieval", "answers"]
    assert runs[0]["configs"]["hybrid"]["recall@k"] == 1.0
    detail = client.get(f"/api/evals/runs/{job['result_file']}").json()
    assert {r["system"] for r in detail["rows"]} == {"agentic", "naive"}
    assert all(r["judge_reasoning"].startswith("Demo judge") for r in detail["rows"])
    assert all("[vacation.md#0]" in r["answer"] for r in detail["rows"])  # both systems answered from the docs
    assert client.get("/api/evals/runs/missing.json").status_code == 404
    assert client.post("/api/evals/run", json={"kind": "answers", "set": "nope"}).status_code == 404


def test_eval_run_model_override_leaves_settings_alone(client):
    before = client.get("/api/settings").json()
    est = client.post("/api/evals/estimate", json={"kind": "answers", "set": "mini", "model": "claude-sonnet-5-5"}).json()
    assert est["model"] == "claude-sonnet-5-5"
    client.post("/api/evals/run", json={"kind": "answers", "set": "mini", "model": "claude-sonnet-5-5", "effort": "high"})
    job = wait_for_job(client)
    run = client.get(f"/api/evals/runs/{job['result_file']}").json()["summary"]
    assert (run["model"], run["effort"]) == ("claude-sonnet-5-5", "high")
    assert client.get("/api/settings").json() == before
    assert client.post("/api/evals/run", json={"kind": "answers", "set": "mini", "model": "gpt-4"}).status_code == 400
