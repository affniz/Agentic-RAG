"""Eval dashboard API: list sets and past runs, estimate cost, run an eval in the background."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag.agent import PRICES
from rag.evaluate import (
    JUDGE_MODEL, EvalCancelled, answer_eval, load_set, retrieval_eval, save_retrieval_run, set_knowledge_base,
)
from rag.store import KnowledgeBase

# Rough tokens per question when there is no past run to go by: (input, output, API calls).
ROUGH_TOKENS = {"agentic": (9000, 600, 3), "agentic-v1": (9000, 600, 3)}
JUDGE_TOKENS = (700, 250)
_NAME = re.compile(r"^[\w.-]+$")


def _naive_tokens(k: int) -> tuple[int, int, int]:
    return 1500 + 300 * k, 300, 1


def _price(model: str, tokens: tuple[int, int, int] | tuple[int, int]) -> float:
    p = PRICES.get(model, PRICES["claude-sonnet-5-5"])
    return (tokens[0] * p[0] + tokens[1] * p[1]) / 1_000_000


class RunRequest(BaseModel):
    kind: Literal["retrieval", "answers"]
    set: str
    systems: list[Literal["agentic", "agentic-v1", "naive"]] = ["agentic"]
    model: str | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    limit: int | None = Field(None, ge=1)
    repeats: int = Field(1, ge=1, le=5)
    naive_k: int = Field(5, ge=1, le=25)
    k: int = Field(5, ge=1, le=25)  # retrieval metrics


def _read_summary(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    s = data.get("summary") or {}
    # Runs saved before the dashboard existed have no kind / set / time: recover them from the file name.
    m = re.match(r"(\d{8}-\d{6})-(?:(.+?)-)?claude-", path.name)
    s.setdefault("kind", "answers" if "systems" in s else "retrieval")
    if not s.get("set") and m and m.group(2):
        s["set"] = m.group(2)
    if not s.get("created_at"):
        stamp = m.group(1) if m else path.name[:15]
        try:
            s["created_at"] = time.mktime(time.strptime(stamp, "%Y%m%d-%H%M%S"))
        except ValueError:
            s["created_at"] = path.stat().st_mtime
    s.pop("results_file", None)
    return {"file": path.name, **s}


class EvalJob:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {"state": "idle"}
        self._lock = threading.Lock()
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self.state.get("state") == "running"

    def start(self, info: dict[str, Any], work: Callable[["EvalJob"], dict[str, Any]]) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise HTTPException(409, "an eval is already running")
        self._stop.clear()
        self.state = {"state": "running", "stage": "preparing", "done": 0, "total": 0, "cost_usd": 0.0,
                      "started_at": time.time(), **info}
        threading.Thread(target=self._run, args=(work,), daemon=True).start()
        return self.state

    def _run(self, work: Callable[["EvalJob"], dict[str, Any]]) -> None:
        try:
            self.state.update(state="done", **work(self))
        except EvalCancelled:
            self.state.update(state="cancelled")
        except Exception as e:  # shown in the UI
            self.state.update(state="error", error=f"{type(e).__name__}: {e}")
        finally:
            self.state["finished_at"] = time.time()
            self._lock.release()

    def cancel(self) -> None:
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()


def make_router(evals_dir: Path, index_root: Path, client_factory: Callable[[], Any],
                settings: Callable[[], Any], demo: bool) -> APIRouter:
    """client_factory: the Anthropic client (or the demo client); settings: the current Settings."""
    router = APIRouter(prefix="/api/evals")
    results_dir = evals_dir / "results"
    job = EvalJob()
    kbs: dict[tuple[str, bool], KnowledgeBase] = {}

    def set_dir(name: str) -> Path:
        path = evals_dir / name
        if not _NAME.match(name) or not (path / "questions.jsonl").is_file():
            raise HTTPException(404, f"eval set {name!r} not found")
        return path

    def open_set(name: str) -> Any:
        try:
            return load_set(set_dir(name))
        except (OSError, ValueError, AttributeError) as e:
            raise HTTPException(400, f"can't read eval set {name!r}: {type(e).__name__}: {e}")

    def runs() -> list[dict[str, Any]]:
        if not results_dir.is_dir():
            return []
        out = [s for p in results_dir.glob("*.json") if (s := _read_summary(p))]
        return sorted(out, key=lambda s: s["created_at"])

    @router.get("/sets")
    def sets() -> list[dict[str, Any]]:
        out = []
        for path in sorted(evals_dir.glob("*/questions.jsonl")):
            try:
                es = load_set(path.parent)
                cats: dict[str, int] = {}
                for q in es.questions:
                    cats[q.get("category", "uncategorized")] = cats.get(q.get("category", "uncategorized"), 0) + 1
                cfg = json.loads((path.parent / "config.json").read_text()) if (path.parent / "config.json").exists() else {}
                out.append({"name": path.parent.name, "questions": len(es.questions), "categories": cats,
                            "description": cfg.get("description", ""), "docs_found": es.docs_dir.is_dir()})
            except (OSError, ValueError, AttributeError) as e:  # bad JSON, a question that isn't an object, ...
                out.append({"name": path.parent.name, "questions": 0, "categories": {}, "docs_found": False,
                            "description": "", "error": f"can't read this set: {type(e).__name__}: {e}"})
        return out

    @router.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        return runs()

    @router.get("/runs/{name}")
    def get_run(name: str) -> dict[str, Any]:
        path = results_dir / name
        if not _NAME.match(name) or not path.is_file():
            raise HTTPException(404, "run not found")
        data = json.loads(path.read_text())
        return {"summary": _read_summary(path), "rows": data.get("rows", [])}

    @router.post("/estimate")
    def estimate(req: RunRequest) -> dict[str, Any]:
        es = open_set(req.set)
        s = settings()
        model = req.model or s.model
        n = min(len(es.questions), req.limit or len(es.questions))
        answers = n * req.repeats
        if req.kind == "retrieval" or demo:
            return {"total_usd": 0.0, "questions": n, "answers": answers, "per_system": {}, "judge_usd": 0.0,
                    "api_calls": 0, "basis": "demo" if demo and req.kind == "answers" else "free", "model": model}
        past = [r for r in runs() if r["kind"] == "answers" and r.get("model") == model]
        per_system, basis, calls = {}, "history", 0.0
        judge_per_q: list[float] = []
        for system in req.systems:
            hist = [(r["systems"][system], r) for r in past if system in r.get("systems", {})]
            same_set = [h for h in hist if h[1].get("set") == req.set] or hist
            if same_set:
                stats, run = same_set[-1]
                answered = run["questions"] * stats.get("runs", 1)
                per_q = stats["cost_usd"] * stats.get("runs", 1) / max(1, answered)
                if run.get("judge_model", "claude-haiku-5-5") == JUDGE_MODEL:  # older runs used a cheaper judge
                    judge_per_q.append(stats["judge_cost_usd"] * stats.get("runs", 1) / max(1, answered))
                calls += stats["avg_api_calls"] * answers
            else:
                basis = "rough"
                tokens = _naive_tokens(req.naive_k) if system == "naive" else ROUGH_TOKENS[system]
                per_q = _price(model, tokens)
                calls += tokens[2] * answers
            per_system[system] = round(per_q * answers, 4)
        judge = (sum(judge_per_q) / len(judge_per_q) if judge_per_q
                 else _price(JUDGE_MODEL, JUDGE_TOKENS)) * answers * len(req.systems)
        total = sum(per_system.values()) + judge
        return {"total_usd": round(total, 4), "questions": n, "answers": answers, "per_system": per_system,
                "judge_usd": round(judge, 4), "api_calls": round(calls + answers * len(req.systems)),
                "basis": basis, "model": model}

    def eval_kb(es: Any, rerank: bool) -> KnowledgeBase:
        key = (es.name, rerank)
        if key not in kbs:
            kbs[key] = set_knowledge_base(es, rerank=rerank, index_dir=index_root / es.name)
        else:
            kbs[key].ingest(es.docs_dir)  # pick up edits to the set's documents
        return kbs[key]

    @router.post("/run")
    def run(req: RunRequest) -> dict[str, Any]:
        es = open_set(req.set)
        if not es.docs_dir.is_dir():
            raise HTTPException(400, f"documents folder for {req.set} not found: {es.docs_dir}")
        s = settings()
        model, effort = req.model or s.model, req.effort or s.effort
        if model not in PRICES:
            raise HTTPException(400, f"unknown model {model}")
        questions = es.questions[: req.limit]

        def work(j: EvalJob) -> dict[str, Any]:
            j.state.update(stage="indexing the set's documents")
            kb = eval_kb(es, s.rerank)
            if req.kind == "retrieval":
                j.state.update(stage="searching", total=len(questions))
                metrics = retrieval_eval(kb, questions, k=req.k)
                path = save_retrieval_run(es.name, req.k, metrics, len(questions), results_dir)
                return {"done": len(questions), "result_file": path.name}

            def on_row(i: int, total: int, row: dict[str, Any]) -> None:
                j.state.update(stage="answering", done=i, total=total,
                               cost_usd=j.state["cost_usd"] + row["cost_usd"] + row["judge_cost_usd"])

            j.state.update(stage="answering", total=len(questions) * len(req.systems) * req.repeats)
            summary = answer_eval(kb, questions, model=model, effort=effort, systems=tuple(req.systems),
                                  today=es.today, naive_k=req.naive_k, out_dir=results_dir, label=es.name,
                                  set_name=es.name, client=client_factory(), repeats=req.repeats,
                                  max_steps=s.max_steps, search_k=s.search_k, on_row=on_row,
                                  should_stop=j.stopped, log=lambda _: None)
            return {"result_file": Path(summary["results_file"]).name}

        return job.start({"kind": req.kind, "set": req.set, "model": model if req.kind == "answers" else None}, work)

    @router.get("/job")
    def job_status() -> dict[str, Any]:
        return job.state

    @router.post("/job/cancel")
    def cancel() -> dict[str, Any]:
        if not job.running:
            raise HTTPException(409, "no eval is running")
        job.cancel()
        return job.state

    return router
