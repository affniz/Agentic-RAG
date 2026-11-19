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
    agent = RagAgent(kb, client=client, on_event=events.append, model="claude-sonnet-5-5")
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


def test_requests_carry_context_editing_and_fallback_only_where_supported(kb):
    from rag.agent import CONTEXT_EDITING_BETA, CONTEXT_MANAGEMENT
    from rag.models import FALLBACK_BETA

    for model, has_fallback in (("claude-haiku-5-5", False), ("claude-sonnet-5-5", True), ("claude-opus-5-5", True)):
        client = FakeClient([NS_end("ok")])
        RagAgent(kb, client=client, model=model).ask("q")
        req = client.requests[0]
        assert req["context_management"] == CONTEXT_MANAGEMENT
        assert CONTEXT_EDITING_BETA in req["betas"]
        assert (FALLBACK_BETA in req["betas"]) is has_fallback
        assert ("fallbacks" in req) is has_fallback


def test_baseline_sends_no_fallback_for_haiku(kb):
    from rag.baseline import NaiveRag

    client = FakeClient([NS_end("ok")])
    NaiveRag(kb, client=client, model="claude-haiku-5-5").ask("q")
    assert "fallbacks" not in client.requests[0] and "betas" not in client.requests[0]


def test_haiku_long_prompt_is_priced_at_the_higher_tier():
    from rag.agent import Usage
    from rag.models import call_cost

    assert call_cost("claude-haiku-5-5", 100_000, 1_000) == pytest.approx((100_000 * 0.10 + 1_000 * 0.50) / 1e6)
    # Above 100K prompt tokens (cache reads count toward the prompt) every token costs the long rate.
    assert call_cost("claude-haiku-5-5", 1_000, 1_000, cache_read=100_000) == pytest.approx(
        (1_000 * 0.50 + 1_000 * 2.50 + 100_000 * 0.05) / 1e6)
    assert call_cost("unknown-model", 1, 1) is None

    u = Usage()
    u.add(usage(1_000, 100), "claude-haiku-5-5")
    u.add(usage(200_000, 100), "claude-haiku-5-5")  # each call picks its own tier
    assert u.cost_usd == pytest.approx((1_000 * 0.10 + 100 * 0.50 + 200_000 * 0.50 + 100 * 2.50) / 1e6)
    u.add(usage(1, 1), "unknown-model")
    assert u.cost_usd is None and u.api_calls == 3


def test_cost_uses_the_model_that_served_the_response(kb):
    msg = message("end_turn", [text_block("ok")], usage(100, 50))
    msg.model = "claude-opus-5-5"  # e.g. a refusal fallback answered instead
    ans = RagAgent(kb, client=FakeClient([msg]), model="claude-sonnet-5-5").ask("q")
    assert ans.cost_usd == pytest.approx((100 * 4 + 50 * 20) / 1e6)


def test_cut_off_tool_call_is_closed_before_the_next_question(kb):
    # A response cut off at max_tokens can end in a tool_use that never ran. The API rejects any
    # request with an unanswered tool_use, so without a fix every later question would fail.
    client = FakeClient([
        message(stop_reason="max_tokens", usage=usage(100, 50),
                content=[text("Let me check."), tool("t1", "search_knowledge_base", query="carry")]),
        message(stop_reason="end_turn", usage=usage(100, 50), content=[text("a2")]),
    ])
    agent = RagAgent(kb, client=client)
    assert agent.ask("q1").stop == "max_tokens"
    assert agent.ask("q2").text == "a2"
    sent = client.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user", "user"]
    assert sent[2]["content"][0]["tool_use_id"] == "t1" and sent[2]["content"][0]["is_error"] is True
    assert sent[3]["content"] == "q2"


def test_saved_history_with_an_open_tool_call_is_repaired(kb):
    # Conversations restored from disk hold plain dicts, and may have been saved in the broken state.
    client = FakeClient([message(stop_reason="end_turn", usage=usage(100, 50), content=[text("ok")])])
    agent = RagAgent(kb, client=client)
    agent.messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t9", "name": "list_documents", "input": {}}]},
    ]
    agent.ask("q2")
    assert client.requests[0]["messages"][2]["content"][0]["tool_use_id"] == "t9"
