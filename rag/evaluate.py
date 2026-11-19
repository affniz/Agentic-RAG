"""Evaluation: retrieval metrics (free, no API) and end-to-end answer grading (uses the API).

An eval set is a directory with `questions.jsonl` and an optional `config.json`:
    {"docs": "docs", "today": "2026-11-16"}   # docs path is relative to the set directory
Each question: {"id", "question", "category"?, "evidence": [...], "sources": [...], "reference"}.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import anthropic
from pydantic import BaseModel

from .agent import AGENT_INSTRUCTIONS_V1, Answer, RagAgent, SearchOptions, Usage
from .baseline import NaiveRag
from .models import request_options
from .store import KnowledgeBase

# A different (stronger) model than the default answering model (Haiku), so the judge isn't
# grading its own answers. Answer evals warn when the judge and answering model are the same.
JUDGE_MODEL = "claude-sonnet-5-5"
SYSTEMS = ("agentic", "agentic-v1", "naive")


@dataclass
class EvalSet:
    name: str
    docs_dir: Path
    today: str | None
    questions: list[dict[str, Any]]


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_set(path: str | Path) -> EvalSet:
    path = Path(path)
    if path.is_file():  # a bare questions file: documents in ./docs
        return EvalSet(path.stem, Path("docs"), None, load_questions(path))
    cfg = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
    return EvalSet(
        cfg.get("name", path.name),
        (path / cfg.get("docs", "docs")).resolve(),
        cfg.get("today"),
        load_questions(path / "questions.jsonl"),
    )


def set_knowledge_base(evalset: EvalSet, rerank: bool = True, index_dir: str | Path | None = None) -> KnowledgeBase:
    """Each eval set gets its own index, kept in sync with its documents."""
    kb = KnowledgeBase(index_dir or f".rag_index_eval_{evalset.name}", rerank=rerank)
    kb.ingest(evalset.docs_dir)
    return kb


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


def _evidence_recall(kb: KnowledgeBase, keys: list[str], evidence: list[str]) -> float | None:
    """Share of evidence snippets that appeared in what the system showed the model."""
    if not evidence:
        return None
    # Everything the model was shown for each chunk: section path, document note and text.
    by_key = {f"{c.source}#{c.index}": _norm(f"{c.indexed_text}\n{c.doc_note}") for c in kb.chunks}
    texts = [by_key[k] for k in keys if k in by_key]
    return sum(any(_norm(e) in t for t in texts) for e in evidence) / len(evidence)


# ---------------------------------------------------------------- retrieval


def retrieval_eval(kb: KnowledgeBase, questions: list[dict[str, Any]], k: int = 5) -> dict[str, dict[str, float]]:
    """Single-query retrieval with the question text, per search configuration.

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


def save_retrieval_run(set_name: str, k: int, results: dict[str, Any], questions: int,
                       out_dir: str | Path = "evals/results") -> Path:
    """Keep retrieval metrics too, so the dashboard can chart them over time."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{set_name}-retrieval.json"
    summary = {"kind": "retrieval", "set": set_name, "created_at": time.time(), "k": k,
               "questions": questions, "configs": results}
    path.write_text(json.dumps({"summary": summary, "rows": []}, indent=2))
    return path


# ---------------------------------------------------------------- answers


class Verdict(BaseModel):
    reasoning: str
    verdict: Literal["correct", "partially_correct", "incorrect"]


JUDGE_PROMPT = """\
You are grading an AI assistant's answer to a question about a company's internal documents.

Question: {question}
Reference answer: {reference}
Assistant's answer:
<answer>
{answer}
</answer>

Grade only factual agreement with the reference: "correct" if every key fact in the reference is \
present and nothing contradicts it, "partially_correct" if some key facts are missing or one part \
is wrong, "incorrect" if the main answer contradicts the reference, or if it states facts the \
reference says are unavailable. When the reference says the premise is false or the information \
is unavailable, the answer must say so to be correct. Ignore citation markers like [file.md#2], \
wording, and extra detail that doesn't contradict the reference."""


def judge(client: Any, q: dict[str, Any], answer: str, model: str = JUDGE_MODEL) -> tuple[Verdict, Any]:
    opts = request_options(model)  # refusal fallback where the model has one (needs the beta endpoint)
    api = client.beta.messages if opts else client.messages
    response = api.parse(
        model=model,
        max_tokens=4000,
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(question=q["question"], reference=q["reference"], answer=answer),
            }
        ],
        output_format=Verdict,
        **opts,
    )
    if response.parsed_output is None:  # refusal or truncation
        return Verdict(reasoning=f"judge returned no verdict ({response.stop_reason})", verdict="incorrect"), response.usage
    return response.parsed_output, response.usage


def make_system(name: str, kb: KnowledgeBase, client: Any, model: str, effort: str, today: str | None,
                naive_k: int, max_steps: int = 12, search_k: int | None = None) -> Any:
    """search_k / max_steps: the agent's settings (results per search, step limit); naive uses naive_k."""
    agent = dict(model=model, effort=effort, client=client, today=today, max_steps=max_steps,
                 search=SearchOptions(k=search_k))
    if name == "agentic":
        return RagAgent(kb, **agent)
    if name == "agentic-v1":  # previous agent instructions, kept for A/B comparison
        return RagAgent(kb, **agent, instructions=AGENT_INSTRUCTIONS_V1)
    if name == "naive":
        return NaiveRag(kb, model=model, effort=effort, k=naive_k, client=client, today=today)
    raise ValueError(f"unknown system {name!r}; choose from {SYSTEMS}")


def answer_eval(
    kb: KnowledgeBase,
    questions: list[dict[str, Any]],
    model: str,
    effort: str = "medium",
    systems: tuple[str, ...] = ("agentic",),
    today: str | None = None,
    naive_k: int = 5,
    judge_model: str = JUDGE_MODEL,
    workers: int = 4,
    out_dir: str | Path = "evals/results",
    label: str = "",
    log: Callable[[str], None] = print,
    client: Any = None,
    repeats: int = 1,
    max_steps: int = 12,
    search_k: int | None = None,
    set_name: str | None = None,
    on_row: Callable[[int, int, dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Answer and grade every question with every system, `repeats` times.

    With repeats > 1 each system's summary holds the mean over runs, the min-max range of the
    accuracy metrics, and `unstable`: questions whose verdict changed between runs.
    """
    client = client or anthropic.Anthropic()
    repeats = max(1, repeats)
    judge_same = judge_model == model
    if judge_same:
        log(f"Warning: the judge ({judge_model}) is the model being evaluated, so it grades its own answers. "
            "Pass a different --judge-model for a fairer score.")
    jobs = [(q, s, r) for r in range(repeats) for q in questions for s in systems]

    def run(job: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
        q, sysname, rep = job
        if should_stop and should_stop():
            raise EvalCancelled()
        system = make_system(sysname, kb, client, model, effort, today, naive_k, max_steps, search_k)  # fresh
        start = time.time()
        ans: Answer | None = None
        try:
            ans = system.ask(q["question"])
            elapsed = time.time() - start
            verdict, judge_usage = judge(client, q, ans.text, judge_model)
        except FATAL_API_ERRORS:
            raise  # a bad key or permission fails every question: stop instead of erroring each one
        except Exception as e:  # one failed question (rate limit, overload, ...) must not lose the run
            return _error_row(q, sysname, rep, ans, time.time() - start, e)
        sources = q.get("sources", [])
        cited = all(re.search(rf"\[{re.escape(s)}#\d+\]", ans.text) for s in sources) if sources else None
        return {
            "id": q["id"],
            "category": q.get("category", "uncategorized"),
            "system": sysname,
            "run": rep + 1,
            "question": q["question"],
            "answer": ans.text,
            "verdict": verdict.verdict,
            "judge_reasoning": verdict.reasoning,
            "cited_expected_sources": cited,
            "evidence_recall": _evidence_recall(kb, ans.retrieved, q.get("evidence", [])),
            "retrieved": ans.retrieved,
            "tool_calls": len(ans.tool_calls),
            "api_calls": ans.usage.api_calls,
            "usage": ans.usage.__dict__,
            "cost_usd": ans.cost_usd or 0.0,
            "judge_cost_usd": _cost(judge_usage, judge_model),
            "seconds": round(elapsed, 1),
            "stop": ans.stop,
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, row in enumerate(pool.map(run, jobs), 1):
            rows.append(row)
            ev = "" if row["evidence_recall"] is None else f"  evidence={row['evidence_recall']:.0%}"
            run_tag = f" run {row['run']}" if repeats > 1 else ""
            err = f"  {row['error']}" if row.get("error") else ""
            log(f"[{i}/{len(jobs)}]{run_tag} {row['system']:<10} {row['id']:<26} {row['verdict']:<17}"
                f"tools={row['tool_calls']}{ev}  ${row['cost_usd']:.4f}  {row['seconds']}s{err}")
            if on_row:
                on_row(i, len(jobs), row)

    summary = {"kind": "answers", "set": set_name or label or None, "created_at": time.time(),
               "model": model, "effort": effort, "questions": len(questions), "naive_k": naive_k,
               "judge_model": judge_model, "judge_same_as_answer": judge_same, "repeats": repeats, "max_steps": max_steps, "search_k": search_k,
               "rerank": kb.reranker is not None,
               "systems": {s: combine_runs([r for r in rows if r["system"] == s], repeats) for s in systems}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    reps = f"-x{repeats}" if repeats > 1 else ""
    path = out / f"{stamp}-{label + '-' if label else ''}{model}-{'-'.join(systems)}{reps}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    summary["results_file"] = str(path)
    return summary


class EvalCancelled(Exception):
    """Raised by answer_eval when should_stop() turns true; nothing is saved."""


# Errors that would hit every question alike, so the run stops at once instead of recording one
# error row per question.
FATAL_API_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, EvalCancelled)


def _error_row(q: dict[str, Any], sysname: str, rep: int, ans: Answer | None, seconds: float,
               e: Exception) -> dict[str, Any]:
    """A question that failed with an error: kept in the results, left out of the accuracy metrics.
    If the answer came back and only the judge failed, its text and cost are kept."""
    return {
        "id": q["id"], "category": q.get("category", "uncategorized"), "system": sysname, "run": rep + 1,
        "question": q["question"], "answer": ans.text if ans else "", "verdict": "error",
        "judge_reasoning": "", "error": f"{type(e).__name__}: {e}", "cited_expected_sources": None,
        "evidence_recall": None, "retrieved": ans.retrieved if ans else [],
        "tool_calls": len(ans.tool_calls) if ans else 0, "api_calls": ans.usage.api_calls if ans else 0,
        "usage": ans.usage.__dict__ if ans else {}, "cost_usd": (ans.cost_usd or 0.0) if ans else 0.0,
        "judge_cost_usd": 0.0, "seconds": round(seconds, 1), "stop": "error",
    }


RANGED = ("correct", "partially_correct", "evidence_recall", "cited_expected_sources")


def combine_runs(rows: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    """One system's results: per-run summaries averaged, with min-max ranges and unstable questions."""
    runs = [summarize([r for r in rows if r.get("run", 1) == k]) for k in range(1, repeats + 1)]
    out = summarize(rows)  # pooled over all runs = the mean of equal-sized runs
    out["cost_usd"] = sum(r["cost_usd"] for r in runs) / len(runs)  # per run, comparable to repeats=1
    out["judge_cost_usd"] = sum(r["judge_cost_usd"] for r in runs) / len(runs)
    out["runs"] = repeats
    out["range"] = {}
    answered = [r for r, k in zip(runs, range(1, repeats + 1))
                if any(row.get("run", 1) == k and row["verdict"] != "error" for row in rows)]
    for key in RANGED:  # runs in which every question errored have no accuracy to range over
        vals = [r[key] for r in answered if r[key] is not None]
        if vals:
            out["range"][key] = (min(vals), max(vals))
    for cat, stats in out["by_category"].items():
        vals = [r["by_category"][cat]["correct"] for r in answered if cat in r["by_category"]]
        stats["range"] = (min(vals), max(vals))
        # Questions in the category (errored ones included), not answers pooled over runs.
        stats["n"] = len({r["id"] for r in rows if r["category"] == cat})
    verdicts: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["verdict"] != "error":
            verdicts[r["id"]].add(r["verdict"])
    out["unstable"] = sorted(qid for qid, v in verdicts.items() if len(v) > 1)
    return out


def summarize(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy and averages over the questions that ran; rows that failed with an error are
    left out of them (and counted in `errors`), and only their spend is added to the cost."""
    rows = [r for r in all_rows if r["verdict"] != "error"]
    n = len(rows) or 1
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    with_src = [r for r in rows if r["cited_expected_sources"] is not None]
    with_ev = [r for r in rows if r["evidence_recall"] is not None]
    return {
        "errors": len(all_rows) - len(rows),
        "correct": sum(r["verdict"] == "correct" for r in rows) / n,
        "partially_correct": sum(r["verdict"] == "partially_correct" for r in rows) / n,
        "by_category": {c: {"n": len(rs), "correct": sum(r["verdict"] == "correct" for r in rs) / len(rs)}
                        for c, rs in sorted(by_cat.items())},
        "cited_expected_sources": sum(bool(r["cited_expected_sources"]) for r in with_src) / len(with_src) if with_src else None,
        "evidence_recall": sum(r["evidence_recall"] for r in with_ev) / len(with_ev) if with_ev else None,
        "avg_tool_calls": sum(r["tool_calls"] for r in rows) / n,
        "avg_api_calls": sum(r["api_calls"] for r in rows) / n,
        "avg_seconds": sum(r["seconds"] for r in rows) / n,
        "cost_usd": sum(r["cost_usd"] for r in all_rows),
        "judge_cost_usd": sum(r["judge_cost_usd"] for r in all_rows),
    }


def _cost(u: Any, model: str) -> float:
    usage = Usage()
    usage.add(u, model)
    return usage.cost_usd or 0.0
