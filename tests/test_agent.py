import pytest

from rag.agent import RagAgent
from rag.fake_llm import DemoClient, ScriptedClient, message, text_block, tool_block, usage
from rag.store import KnowledgeBase

FakeClient = ScriptedClient


def text(t):
    return text_block(t)


def tool(id_, name, **inp):
    return tool_block(id_, name, **inp)


@pytest.fixture
def kb(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "vacation.md").write_text("# Vacation\nUp to 5 unused days carry over until March 31.")
    k = KnowledgeBase(tmp_path / "idx", rerank=False)
    k.ingest(d)
    return k


def test_parallel_tool_calls_return_in_one_message(kb):
    client = FakeClient([
        message(stop_reason="tool_use", usage=usage(100, 50), content=[
            tool("t1", "search_knowledge_base", query="carry over", k=3, source=None),
            tool("t2", "read_chunks", source="vacation.md", start=0, end=0),
            tool("t3", "no_such_tool"),
        ]),
        message(stop_reason="end_turn", usage=usage(100, 50), content=[text("5 days [vacation.md#0]")]),
    ])
    events = []
    agent = RagAgent(kb, client=client, on_event=events.append)
    ans = agent.ask("carry over?")

    assert ans.text == "5 days [vacation.md#0]" and ans.stop == "end_turn"
    results = agent.messages[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2", "t3"]
    assert [r["is_error"] for r in results] == [False, False, True]
    assert "[vacation.md#0]" in results[0]["content"]
    assert [m["role"] for m in agent.messages] == ["user", "assistant", "user", "assistant"]
    assert [e["type"] for e in events].count("tool_call") == 3
    assert ans.usage.api_calls == 2 and ans.cost_usd == pytest.approx((200 * 2 + 100 * 10) / 1e6)


def test_step_limit_forces_final_answer(kb):
    looping = [message(stop_reason="tool_use", usage=usage(100, 50), content=[tool(f"t{i}", "list_documents")]) for i in range(3)]
    client = FakeClient(looping + [message(stop_reason="end_turn", usage=usage(100, 50), content=[text("best effort")])])
    ans = RagAgent(kb, client=client, max_steps=3).ask("q")
    assert ans.stop == "step_limit" and ans.text == "best effort"
    assert client.requests[-1]["tool_choice"] == {"type": "none"}


def test_refusal(kb):
    client = FakeClient([message(stop_reason="refusal", usage=usage(100, 50), content=[])])
    ans = RagAgent(kb, client=client).ask("q")
    assert ans.stop == "refusal"


def test_history_carries_across_questions(kb):
    client = FakeClient([
        message(stop_reason="end_turn", usage=usage(100, 50), content=[text("a1")]),
        message(stop_reason="end_turn", usage=usage(100, 50), content=[text("a2")]),
    ])
    agent = RagAgent(kb, client=client)
    agent.ask("q1")
    agent.ask("q2")
    assert len(client.requests[1]["messages"]) == 3  # q1, a1, q2
    assert agent.total.api_calls == 2


def test_streaming_events_and_citations(kb):
    client = FakeClient([NS_end("Up to 5 days [vacation.md#0] and more [nope.md#9].")])
    events = []
    ans = RagAgent(kb, client=client, on_event=events.append).ask("q")
    streamed = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert streamed == ans.text
    assert ans.citations == [
        {"source": "vacation.md", "index": 0, "section": "Vacation", "found": True},
        {"source": "nope.md", "index": 9, "section": None, "found": False},
    ]
    assert [e["type"] for e in events][-2:] == ["answer", "usage"]


def test_demo_client_runs_end_to_end(kb):
    ans = RagAgent(kb, client=DemoClient(delay=0)).ask("how many days carry over")
    assert ans.stop == "end_turn" and "[vacation.md#0]" in ans.text
    assert ans.tool_calls[0]["name"] == "search_knowledge_base"


def NS_end(t):
    return message("end_turn", [text_block(t)], usage())
