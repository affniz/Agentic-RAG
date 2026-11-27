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
    # Timings are saved with the answer: one per model call, and per tool call under its step.
    first = conv["messages"][1]
    assert [c["step"] for c in first["timings"]] == [1, 2] and first["steps"][0]["step"] == 1
    assert first["steps"][0]["seconds"] >= 0 and all(c["wait_seconds"] <= c["seconds"] for c in first["timings"])
    assert any(e["type"] == "model_call" for e in evs)

    assert len(client.get("/api/conversations").json()) == 1
    assert client.delete(f"/api/conversations/{cid}").json() == {"ok": True}
    assert client.get(f"/api/conversations/{cid}").status_code == 404



def test_client_that_leaves_before_reading_does_not_lock_the_conversation(client):
    # The browser disconnects (or aborts) before the stream is read: the answer still runs and
    # is saved, and the conversation takes the next question instead of answering 409 forever.
    import asyncio

    import server.app as app_module

    async def ask_and_leave():
        res = await app_module.chat(app_module.ChatRequest(message="how many days carry over?"))
        await res.body_iterator.aclose()  # never read
        conv = next(iter(app_module.conversations.values()))
        for _ in range(500):
            if not conv.lock.locked():
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)  # let the worker's last event reach the loop
        return conv

    conv = asyncio.run(ask_and_leave())
    assert not conv.lock.locked()
    assert [t.role for t in conv.turns] == ["user", "assistant"] and conv.turns[1].citations
    evs = events(client.post("/api/chat", json={"message": "and when?", "conversation_id": conv.id}))
    assert evs[-1]["type"] == "done"
    assert len(client.get(f"/api/conversations/{conv.id}").json()["messages"]) == 4

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
    assert s["max_cost_usd"] == 0.5 and agent.max_cost_usd == 0.5  # default cap applies to chat
    assert client.put("/api/settings", json={**s, "max_cost_usd": 0}).status_code == 422
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


# ---------------------------------------------------------------- localhost guard


def test_bad_host_is_rejected(client):
    r = client.get("/api/health", headers={"Host": "evil.example"})
    assert r.status_code == 403 and "evil.example" in r.json()["detail"]
    # DNS rebinding with a port, or a name that only ends like localhost, is rejected too.
    assert client.get("/api/health", headers={"Host": "evil.example:8000"}).status_code == 403
    assert client.get("/api/health", headers={"Host": "notlocalhost"}).status_code == 403
    for host in ["localhost:8000", "127.0.0.1", "[::1]:8000", "app.localhost:5173"]:
        assert client.get("/api/health", headers={"Host": host}).status_code == 200, host


def test_extra_hosts_from_env(client, monkeypatch):
    import server.app as app_module

    monkeypatch.setenv("RAG_ALLOWED_HOSTS", "rag.lan, Other.Example")
    app_module = importlib.reload(app_module)
    c = TestClient(app_module.app)
    assert c.get("/api/health", headers={"Host": "rag.lan:8000"}).status_code == 200
    assert c.get("/api/health", headers={"Host": "other.example"}).status_code == 200
    assert c.post("/api/chat", json={"message": "x", "model": "gpt-4"},
                  headers={"Host": "rag.lan", "Origin": "http://rag.lan:8000"}).status_code == 400


def test_cross_site_post_is_rejected(client):
    for headers in [{"Origin": "http://evil.example"}, {"Origin": "null"},
                    {"Origin": "http://localhost.evil.example:8000"}, {"Sec-Fetch-Site": "cross-site"}]:
        r = client.post("/api/chat", json={"message": "hi"}, headers={"Host": "localhost:8000", **headers})
        assert r.status_code == 403, headers
        assert client.delete("/api/conversations", headers={"Host": "localhost", **headers}).status_code == 403
    # A multipart upload (no CORS preflight) from another site is the main thing to stop.
    r = client.post("/api/documents", files={"files": ("x.md", b"# x")},
                    headers={"Host": "localhost:8000", "Origin": "http://evil.example"})
    assert r.status_code == 403
    assert client.get("/api/conversations").json() == []  # nothing was created


def test_same_origin_post_passes(client):
    # The Vite dev server proxies :5173 -> :8000; only the hostname is compared.
    for origin, site in [("http://localhost:8000", "same-origin"), ("http://localhost:5173", "same-site"),
                         ("http://127.0.0.1:8000", "same-origin"), ("http://[::1]:8000", "same-origin")]:
        r = client.post("/api/chat", json={"message": "carry over"},
                        headers={"Host": "localhost:8000", "Origin": origin, "Sec-Fetch-Site": site})
        assert r.status_code == 200, origin
        assert events(r)[-1]["type"] == "done"
    assert client.get("/api/conversations", headers={"Origin": "http://evil.example"}).status_code == 200  # reads: Host only


# ---------------------------------------------------------------- saved conversations


def test_conversation_survives_restart(client, tmp_path):
    import server.app as app_module

    cid = events(client.post("/api/chat", json={"message": "how many days carry over?"}))[0]["id"]
    turn_id = app_module.conversations[cid].turns[1].id
    client.put(f"/api/feedback/{turn_id}", json={"conversation_id": cid, "rating": "up"})
    path = tmp_path / "data" / "conversations" / f"{cid}.json"
    saved = json.loads(path.read_text())
    assert saved["model"] == "claude-haiku-5-5" and len(saved["messages"]) == 4  # question, tool_use, results, answer
    assert saved["messages"][1]["content"][0]["type"] == "tool_use"  # fake-client blocks saved as dicts
    before = client.get(f"/api/conversations/{cid}").json()

    app_module = importlib.reload(app_module)  # same RAG_DATA: a server restart
    app_module.client.delay = 0
    c = TestClient(app_module.app)
    assert c.get(f"/api/conversations/{cid}").json() == before
    assert app_module.conversations[cid].agent.messages == saved["messages"]

    # Follow-ups continue the restored history.
    evs = events(c.post("/api/chat", json={"message": "and when?", "conversation_id": cid}))
    assert next(e for e in evs if e["type"] == "answer")["citations"][0]["source"] == "vacation.md"
    assert len(json.loads(path.read_text())["messages"]) == 8
    assert [m["role"] for m in c.get(f"/api/conversations/{cid}").json()["messages"]] == ["user", "assistant"] * 2

    assert c.delete(f"/api/conversations/{cid}").json() == {"ok": True}
    assert not path.exists()
    assert importlib.reload(app_module).conversations == {}


def test_reset_deletes_files_and_bad_files_are_skipped(client, tmp_path):
    import server.app as app_module

    events(client.post("/api/chat", json={"message": "carry over"}))
    folder = tmp_path / "data" / "conversations"
    assert len(list(folder.glob("*.json"))) == 1
    client.delete("/api/conversations")
    assert list(folder.glob("*.json")) == []

    (folder / "broken.json").write_text("{not json")
    (folder / "other.json").write_text(json.dumps({"id": "mismatch"}))
    assert importlib.reload(app_module).conversations == {}


def test_message_content_serialization():
    from types import SimpleNamespace as NS

    from anthropic.types.beta import BetaTextBlock, BetaThinkingBlock, BetaToolUseBlock

    from server.conversations import jsonable

    thinking = BetaThinkingBlock(type="thinking", thinking="Let me look.", signature="c2lnbmF0dXJl")
    content = [thinking, BetaToolUseBlock(type="tool_use", id="t1", name="search", input={"q": "x"}),
               BetaTextBlock(type="text", text="Hi", citations=None), NS(type="text", text="fake", extra=NS(a=[1]))]
    out = jsonable([{"role": "assistant", "content": content}, {"role": "user", "content": "plain"}])
    assert json.loads(json.dumps(out)) == out
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "Let me look.", "signature": "c2lnbmF0dXJl"}
    assert BetaThinkingBlock.model_validate(blocks[0]) == thinking  # round-trips unchanged
    assert blocks[1] == {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}}
    assert blocks[2] == {"type": "text", "text": "Hi"}  # None fields dropped
    assert blocks[3] == {"type": "text", "text": "fake", "extra": {"a": [1]}}
    with pytest.raises(TypeError):
        jsonable(object())


# ---------------------------------------------------------------- login (RAG_PASSWORD)

PASSWORD = "correct horse battery staple"


@pytest.fixture
def locked(client, monkeypatch):
    """The app with RAG_PASSWORD set (same data folders as `client`)."""
    import server.app as app_module

    monkeypatch.setenv("RAG_PASSWORD", PASSWORD)
    app_module = importlib.reload(app_module)
    app_module.client.delay = 0
    return TestClient(app_module.app)


def login(c, password=PASSWORD, **headers):
    return c.post("/api/auth/login", json={"password": password}, headers=headers)


def test_without_password_nothing_changes(client):
    assert client.get("/api/auth/status").json() == {"required": False, "authenticated": True}
    assert client.get("/api/conversations").status_code == 200
    assert login(client).status_code == 400


def test_api_needs_login(locked):
    assert locked.get("/api/auth/status").json() == {"required": True, "authenticated": False}
    for method, path in [("GET", "/api/health"), ("GET", "/api/conversations"), ("GET", "/api/documents"),
                         ("GET", "/api/evals/runs"), ("PUT", "/api/settings"), ("DELETE", "/api/conversations"),
                         ("GET", "/api/search?q=x"), ("GET", "/api/nope")]:
        assert locked.request(method, path).status_code == 401, path
    r = locked.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 401 and r.json() == {"detail": "login required"}
    assert locked.get("/api/ping").json() == {"ok": True}  # Docker's HEALTHCHECK


def test_login_logout_roundtrip(locked):
    r = login(locked, "wrong")
    assert r.status_code == 401 and "rag_session" not in r.cookies
    r = login(locked)
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "secure" not in cookie  # plain http here
    assert locked.get("/api/auth/status").json()["authenticated"]
    assert locked.get("/api/health").status_code == 200
    assert events(locked.post("/api/chat", json={"message": "carry over"}))[-1]["type"] == "done"

    assert locked.post("/api/auth/logout").status_code == 200
    assert locked.get("/api/health").status_code == 401


def test_forged_expired_and_rotated_sessions_are_rejected(locked, monkeypatch):
    import server.app as app_module

    auth = app_module.auth
    token = auth.make_token()
    assert auth.valid(token)
    expires, nonce, sig = token.split(".")
    assert not auth.valid(f"{int(expires) + 999}.{nonce}.{sig}")  # edited expiry
    assert not auth.valid(f"{expires}.{nonce}.{'0' * len(sig)}")
    assert not auth.valid("garbage") and not auth.valid("") and not auth.valid(None)
    assert not auth.valid(auth.make_token(now=0))  # expired long ago
    locked.cookies.set("rag_session", f"{int(expires) + 999}.{nonce}.{sig}")
    assert locked.get("/api/health").status_code == 401

    # A restart keeps sessions (same secret file); a new password logs everyone out.
    locked.cookies.clear()
    login(locked)
    app_module = importlib.reload(app_module)
    assert TestClient(app_module.app, cookies=locked.cookies).get("/api/health").status_code == 200
    monkeypatch.setenv("RAG_PASSWORD", "a different long password")
    app_module = importlib.reload(app_module)
    assert TestClient(app_module.app, cookies=locked.cookies).get("/api/health").status_code == 401


def test_failed_logins_are_limited(locked):
    from server import auth as auth_module

    for _ in range(auth_module.MAX_FAILURES):
        assert login(locked, "nope").status_code == 401
    r = login(locked)  # even the right password waits
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0


def test_lockout_expires(tmp_path):
    from server.auth import FAILURE_WINDOW, MAX_FAILURES, Auth

    auth = Auth(PASSWORD, tmp_path / "key")
    for _ in range(MAX_FAILURES):
        auth.record_failure("1.2.3.4", now=1000)
    assert auth.retry_after("1.2.3.4", now=1001) > 0 and auth.retry_after("5.6.7.8", now=1001) == 0
    assert auth.retry_after("1.2.3.4", now=1000 + FAILURE_WINDOW + 1) == 0
    assert (tmp_path / "key").stat().st_mode & 0o777 == 0o600


def test_password_allows_any_host_but_not_cross_site(locked):
    assert login(locked, Host="rag.lan:8000", Origin="http://rag.lan:8000").status_code == 200
    assert locked.get("/api/health", headers={"Host": "rag.lan:8000"}).status_code == 200
    assert locked.post("/api/chat", json={"message": "x", "model": "gpt-4"},
                       headers={"Host": "rag.lan", "Origin": "http://rag.lan:8000"}).status_code == 400
    # Writes from another site are still refused, logged in or not.
    for headers in [{"Origin": "http://evil.example"}, {"Sec-Fetch-Site": "cross-site"}]:
        r = locked.post("/api/chat", json={"message": "hi"}, headers={"Host": "rag.lan", **headers})
        assert r.status_code == 403, headers
        assert login(locked, Host="rag.lan", **headers).status_code == 403
