"""Settings and answer feedback, stored as JSON files so they survive restarts.

Feedback (👍/👎 on an answer) can be exported as an eval set: a 👍 answer becomes the reference,
a 👎 needs the user's correction (the comment) to become one.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseModel):
    model: str
    effort: Effort = "medium"
    search_k: int | None = Field(None, ge=1, le=10)  # results per search; None = the model decides
    rerank: bool = True
    max_steps: int = Field(12, ge=1, le=30)
    # Chat only: once one question has cost this much (USD), the agent answers with what it has.
    max_cost_usd: float | None = Field(0.5, gt=0, le=100)


class JsonFile:
    """A small JSON document on disk, written atomically."""

    def __init__(self, path: Path, default: Any):
        self.path, self.default = path, default
        self.lock = threading.Lock()

    def read(self) -> Any:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return json.loads(json.dumps(self.default))

    def write(self, data: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        os.replace(tmp, self.path)


class SettingsStore:
    def __init__(self, path: Path, defaults: Settings):
        self.file, self.defaults = JsonFile(path, {}), defaults

    def get(self) -> Settings:
        try:
            return Settings(**{**self.defaults.model_dump(), **self.file.read()})
        except ValueError:  # file from an older version or edited by hand
            return self.defaults

    def put(self, s: Settings) -> Settings:
        with self.file.lock:
            self.file.write(s.model_dump())
        return s


class FeedbackStore:
    """{turn_id: record}. A record holds everything needed to turn it into an eval question."""

    def __init__(self, path: Path):
        self.file = JsonFile(path, {})

    def all(self) -> list[dict[str, Any]]:
        return sorted(self.file.read().values(), key=lambda r: -r["updated_at"])

    def set(self, turn_id: str, record: dict[str, Any] | None) -> None:
        with self.file.lock:
            data = self.file.read()
            if record is None:
                data.pop(turn_id, None)
            else:
                old = data.get(turn_id, {})
                data[turn_id] = {**record, "created_at": old.get("created_at", time.time()), "updated_at": time.time()}
            self.file.write(data)


CITATION = re.compile(r"\s*\[[^\[\]\n]*#\d+\]")  # one or more citations in a bracket, see rag/agent.py


def to_eval_questions(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(questions, skipped). 👍 → the answer is the reference; 👎 → the comment is, if there is one."""
    questions, skipped = [], []
    for r in sorted(records, key=lambda r: r["created_at"]):
        if r["rating"] == "up":
            reference = CITATION.sub("", r["answer"]).strip()
            sources = sorted({c["source"] for c in r.get("citations", []) if c.get("found", True)})
        elif (r.get("comment") or "").strip():
            reference, sources = r["comment"].strip(), []
        else:
            skipped.append({"id": r["turn_id"], "question": r["question"],
                            "reason": "👎 with no comment: add what the answer should have been"})
            continue
        q = {"id": f"fb-{r['turn_id']}", "category": f"feedback-{r['rating']}", "question": r["question"],
             "evidence": [], "sources": sources, "reference": reference}
        if r.get("followup"):
            q["note"] = "asked as a follow-up; may need the earlier turns to make sense on its own"
        questions.append(q)
    return questions, skipped


def export_eval_set(records: list[dict[str, Any]], set_dir: Path, docs_dir: Path) -> dict[str, Any]:
    questions, skipped = to_eval_questions(records)
    set_dir.mkdir(parents=True, exist_ok=True)
    (set_dir / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions))
    config = {"name": set_dir.name, "docs": os.path.relpath(docs_dir.resolve(), set_dir.resolve()),
              "description": "Questions from 👍/👎 feedback in the chat. Review the references before relying on it."}
    (set_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return {"path": str(set_dir), "questions": len(questions), "skipped": skipped}
