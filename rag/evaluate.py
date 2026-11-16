"""Evaluation: retrieval metrics (free, no API) and end-to-end answer grading (uses the API)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Literal

import anthropic
from pydantic import BaseModel

from .agent import RagAgent, Usage
from .store import KnowledgeBase

JUDGE_MODEL = "claude-haiku-5-5"


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


# ---------------------------------------------------------------- retrieval


def retrieval_eval(kb: KnowledgeBase, questions: list[dict[str, Any]], k: int = 5) -> dict[str, dict[str, float]]:
    """For each search configuration: how often the evidence lands in the top-k chunks.

    recall@k  - share of evidence snippets found in the top-k (averaged over questions)
    hit@k     - share of questions where *all* evidence was found
    mrr       - mean reciprocal rank of the first chunk containing any evidence
    """
    configs = [("bm25", False), ("vector", False), ("hybrid", False)]
    if kb.reranker is not None:
        configs.append(("hybrid", True))
    scored = [q for q in questions if q.get("evidence")]
    results: dict[str, dict[str, float]] = {}

    for mode, rerank in configs:
        name = f"{mode}+rerank" if rerank else mode
        recall = hit = mrr = 0.0
        for q in scored:
            hits = kb.search(q["question"], k=k, mode=mode, rerank=rerank)
            texts = [_norm(c.indexed_text) for c, _ in hits]
            found = [any(_norm(e) in t for t in texts) for e in q["evidence"]]
            recall += sum(found) / len(found)
            hit += all(found)
            rank = next((i + 1 for i, t in enumerate(texts) if any(_norm(e) in t for e in q["evidence"])), None)
            mrr += 1 / rank if rank else 0
        n = len(scored) or 1
        results[name] = {"recall@k": recall / n, "hit@k": hit / n, "mrr": mrr / n}
        if rerank and kb.reranker is not None and not kb.reranker.available:
            results[name]["note"] = "reranker unavailable - same as hybrid"  # type: ignore[assignment]
    return results


# ---------------------------------------------------------------- answers


class Verdict(BaseModel):
    reasoning: str
    verdict: Literal["correct", "partially_correct", "incorrect"]


JUDGE_PROMPT = """\
You are grading an AI assistant's answer to a question about company policies.

Question: {question}
Reference answer: {reference}
Assistant's answer:
<answer>
{answer}
</answer>

Grade only factual agreement with the reference: "correct" if every key fact in the reference is \
present and nothing contradicts it, "partially_correct" if some key facts are missing, \
"incorrect" if it contradicts the reference or claims facts the reference says are unavailable. \
Ignore citation markers like [file.md#2], wording, and extra detail that doesn't contradict the reference."""


def judge(client: anthropic.Anthropic, q: dict[str, Any], answer: str) -> tuple[Verdict, Any]:
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=4000,
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(question=q["question"], reference=q["reference"], answer=answer),
            }
        ],
        output_format=Verdict,
    )
    if response.parsed_output is None:  # refusal or truncation
        return Verdict(reasoning=f"judge returned no verdict ({response.stop_reason})", verdict="incorrect"), response.usage
    return response.parsed_output, response.usage


def answer_eval(
    kb: KnowledgeBase,
    questions: list[dict[str, Any]],
    model: str,
    effort: str = "medium",
    out_dir: str | Path = "evals/results",
    log: Any = print,
) -> dict[str, Any]:
    client = anthropic.Anthropic()
    rows = []
    agent_usage, judge_usage = Usage(), Usage()

    for i, q in enumerate(questions, 1):
        agent = RagAgent(kb, model=model, effort=effort, client=client)  # fresh history per question
        start = time.time()
        ans = agent.ask(q["question"])
        elapsed = time.time() - start
        for f in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "api_calls"):
            setattr(agent_usage, f, getattr(agent_usage, f) + getattr(ans.usage, f))

        verdict, ju = judge(client, q, ans.text)
        judge_usage.add(ju)
        cited = all(re.search(rf"\[{re.escape(s)}#\d+\]", ans.text) for s in q.get("sources", []))
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer": ans.text,
                "verdict": verdict.verdict,
                "judge_reasoning": verdict.reasoning,
                "cited_expected_sources": cited,
                "tool_calls": len(ans.tool_calls),
                "api_calls": ans.usage.api_calls,
                "cost_usd": ans.cost_usd,
                "seconds": round(elapsed, 1),
                "stop": ans.stop,
            }
        )
        log(f"[{i}/{len(questions)}] {q['id']}: {verdict.verdict}  cited={cited}  "
            f"tools={len(ans.tool_calls)}  ${ans.cost_usd or 0:.4f}  {elapsed:.1f}s")

    n = len(rows) or 1
    summary = {
        "model": model,
        "effort": effort,
        "questions": len(rows),
        "correct": sum(r["verdict"] == "correct" for r in rows) / n,
        "partially_correct": sum(r["verdict"] == "partially_correct" for r in rows) / n,
        "cited_expected_sources": sum(r["cited_expected_sources"] for r in rows) / n,
        "avg_tool_calls": sum(r["tool_calls"] for r in rows) / n,
        "avg_seconds": sum(r["seconds"] for r in rows) / n,
        "agent_cost_usd": agent_usage.cost(model),
        "judge_cost_usd": judge_usage.cost(JUDGE_MODEL),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{model}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    summary["results_file"] = str(path)
    return summary
