from types import SimpleNamespace as NS

import pytest

from rag.agent import RagAgent
from rag.baseline import NaiveRag
from rag.evaluate import Verdict, answer_eval, load_set
from rag.fake_llm import DemoClient, ScriptedClient, message, text_block, tool_block, usage
from rag.store import KnowledgeBase


@pytest.fixture
def kb(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "people.md").write_text("# People\n## Priya\nPriya Raman works in Bangalore. Manager: Mei Lin Chua.")
    (d / "offices.md").write_text("# Offices\n## Bangalore\nBangalore is a Tier 3 office.")
    k = KnowledgeBase(tmp_path / "idx", rerank=False)
    k.ingest(d)
    return k


def test_naive_retrieves_once_and_answers_in_one_call(kb):
    client = ScriptedClient([message("end_turn", [text_block("Tier 3 [offices.md#0]")], usage(100, 20))])
    ans = NaiveRag(kb, client=client, k=2, today="2026-11-16").ask("Which tier is Bangalore?")
    assert ans.text == "Tier 3 [offices.md#0]" and ans.usage.api_calls == 1 and ans.tool_calls == []
    assert set(ans.retrieved) <= {"people.md#0", "offices.md#0"} and len(ans.retrieved) == 2
    req = client.requests[0]
    assert "tools" not in req and "Today's date: 2026-11-16" in req["system"]
    assert "<context>" in req["messages"][0]["content"]


def test_agent_records_every_retrieved_chunk(kb):
    client = ScriptedClient([
        message("tool_use", [tool_block("t1", "search_knowledge_base", query="Priya manager", k=1, source=None)]),
        message("tool_use", [tool_block("t2", "read_chunks", source="offices.md", start=0, end=0)]),
        message("end_turn", [text_block("done")]),
    ])
    ans = RagAgent(kb, client=client, today="2026-11-16").ask("q")
    assert ans.retrieved[-1] == "offices.md#0" and len(ans.retrieved) == 2
    assert "Today's date: 2026-11-16" in client.requests[0]["system"]


class FakeAnthropic:
    """Demo client for both systems + a judge that marks answers citing offices.md correct."""

    def __init__(self):
        self.beta = DemoClient(delay=0).beta
        self.messages = NS(parse=self._parse)

    def _parse(self, **kw):
        ok = "offices.md" in kw["messages"][0]["content"]
        v = Verdict(reasoning="-", verdict="correct" if ok else "incorrect")
        return NS(parsed_output=v, stop_reason="end_turn", usage=usage(10, 5))


def test_answer_eval_compares_systems(kb, tmp_path):
    qs = [
        {"id": "a", "category": "simple", "question": "Bangalore tier office", "evidence": ["Tier 3 office"],
         "sources": ["offices.md"], "reference": "Tier 3"},
        {"id": "b", "category": "unanswerable", "question": "salary?", "evidence": [], "sources": [],
         "reference": "unknown"},
    ]
    s = answer_eval(kb, qs, model="claude-haiku-5-5", systems=("agentic", "naive"), today="2026-11-16",
                    workers=2, out_dir=tmp_path, client=FakeAnthropic(), log=lambda _: None)
    assert set(s["systems"]) == {"agentic", "naive"}
    for name in ("agentic", "naive"):
        r = s["systems"][name]
        assert set(r["by_category"]) == {"simple", "unanswerable"}
        assert r["evidence_recall"] == 1.0  # only question "a" has evidence, and it's retrieved
    assert s["systems"]["naive"]["avg_api_calls"] == 1.0
    assert (tmp_path / s["results_file"].split("/")[-1]).exists()


def test_shipped_eval_sets_load():
    for name in ("evals/basic", "evals/hard"):
        es = load_set(name)
        assert es.today == "2026-11-16" and es.docs_dir.is_dir()
        assert all({"id", "question", "evidence", "reference"} <= set(q) for q in es.questions)


def test_agent_prompt_versions(kb):
    from rag.agent import AGENT_INSTRUCTIONS, AGENT_INSTRUCTIONS_V1
    from rag.evaluate import make_system

    assert AGENT_INSTRUCTIONS.startswith(AGENT_INSTRUCTIONS_V1) and "past one" in AGENT_INSTRUCTIONS
    new = make_system("agentic", kb, ScriptedClient([]), "claude-haiku-5-5", "medium", "2026-11-16", 5)
    old = make_system("agentic-v1", kb, ScriptedClient([]), "claude-haiku-5-5", "medium", "2026-11-16", 5)
    assert "past one" in new.system and "past one" not in old.system


def test_evidence_recall_counts_document_notes(tmp_path):
    from rag.evaluate import _evidence_recall

    d = tmp_path / "docs"
    d.mkdir()
    (d / "p.md").write_text("# Policy 2025\nIn force until 2026-02-28.\n\n## Hotels\nTier 1: $320 per night")
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(d)
    # Only the Hotels chunk was retrieved, but its note shows the validity window.
    assert _evidence_recall(kb, ["p.md#1"], ["In force until 2026-02-28", "Tier 1: $320"]) == 1.0
