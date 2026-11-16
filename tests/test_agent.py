from types import SimpleNamespace as NS

import pytest

from rag.agent import RagAgent
from rag.store import KnowledgeBase


def usage(i=100, o=50):
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def text(t):
    return NS(type="text", text=t)


def tool(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


class FakeClient:
    """Returns scripted responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return self.responses.pop(0)


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
        NS(stop_reason="tool_use", usage=usage(), content=[
            tool("t1", "search_knowledge_base", query="carry over", k=3, source=None),
            tool("t2", "read_chunks", source="vacation.md", start=0, end=0),
            tool("t3", "no_such_tool"),
        ]),
        NS(stop_reason="end_turn", usage=usage(), content=[text("5 days [vacation.md#0]")]),
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
    looping = [NS(stop_reason="tool_use", usage=usage(), content=[tool(f"t{i}", "list_documents")]) for i in range(3)]
    client = FakeClient(looping + [NS(stop_reason="end_turn", usage=usage(), content=[text("best effort")])])
    ans = RagAgent(kb, client=client, max_steps=3).ask("q")
    assert ans.stop == "step_limit" and ans.text == "best effort"
    assert client.requests[-1]["tool_choice"] == {"type": "none"}


def test_refusal(kb):
    client = FakeClient([NS(stop_reason="refusal", usage=usage(), content=[])])
    ans = RagAgent(kb, client=client).ask("q")
    assert ans.stop == "refusal"


def test_history_carries_across_questions(kb):
    client = FakeClient([
        NS(stop_reason="end_turn", usage=usage(), content=[text("a1")]),
        NS(stop_reason="end_turn", usage=usage(), content=[text("a2")]),
    ])
    agent = RagAgent(kb, client=client)
    agent.ask("q1")
    agent.ask("q2")
    assert len(client.requests[1]["messages"]) == 3  # q1, a1, q2
    assert agent.total.api_calls == 2
