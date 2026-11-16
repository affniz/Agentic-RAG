"""FastAPI backend for the web UI.

    uvicorn server.app:app --reload            # API on :8000 (serves frontend/dist if built)
    RAG_FAKE_LLM=1 uvicorn server.app:app      # demo mode: no API key, no cost

Chat responses stream as Server-Sent Events. Conversations live in memory for now
(phase 1); they are lost when the server restarts.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.agent import MODEL, PRICES, RagAgent, Usage
from rag.store import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = os.environ.get("RAG_INDEX", str(ROOT / ".rag_index"))
FAKE_LLM = os.environ.get("RAG_FAKE_LLM") == "1"
RERANK = os.environ.get("RAG_RERANK", "1") == "1"
DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="Agentic RAG")
kb = KnowledgeBase(INDEX_DIR, rerank=RERANK)


def _make_client() -> Any:
    if FAKE_LLM:
        from rag.fake_llm import DemoClient

        return DemoClient()
    import anthropic

    return anthropic.Anthropic()


client = _make_client()


# ---------------------------------------------------------------- conversations


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    text: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    seconds: float | None = None
    stop: str | None = None


@dataclass
class Conversation:
    id: str
    agent: RagAgent
    title: str = "New chat"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turns: list[Turn] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def summary(self) -> dict[str, Any]:
        cost = sum(t.cost_usd or 0 for t in self.turns)
        return {"id": self.id, "title": self.title, "created_at": self.created_at,
                "updated_at": self.updated_at, "turns": len(self.turns), "cost_usd": cost,
                "model": self.agent.model}


conversations: dict[str, Conversation] = {}


def _get(cid: str) -> Conversation:
    conv = conversations.get(cid)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    return conv


# ---------------------------------------------------------------- API


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = None
    model: str | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "documents": len(kb.sources()),
        "chunks": len(kb.chunks),
        "default_model": MODEL,
        "models": list(PRICES),
        "demo_mode": FAKE_LLM,
        "reranker": bool(kb.reranker and kb.reranker.available),
    }


@app.get("/api/conversations")
def list_conversations() -> list[dict[str, Any]]:
    return sorted((c.summary() for c in conversations.values()), key=lambda s: -s["updated_at"])


@app.get("/api/conversations/{cid}")
def get_conversation(cid: str) -> dict[str, Any]:
    conv = _get(cid)
    return {**conv.summary(), "messages": [asdict(t) for t in conv.turns]}


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str) -> dict[str, bool]:
    _get(cid)
    del conversations[cid]
    return {"ok": True}


@app.get("/api/chunks/{source:path}/{index}")
def get_chunk(source: str, index: int, context: int = 1) -> dict[str, Any]:
    """A cited chunk plus `context` neighbours on each side, for the citation panel."""
    chunks = kb.get_chunks(source, max(0, index - context), index + context)
    if not any(c.index == index for c in chunks):
        raise HTTPException(404, f"chunk {source}#{index} not found")
    return {
        "source": source,
        "index": index,
        "total_chunks": kb.sources().get(source, 0),
        "chunks": [{"index": c.index, "section": c.section, "text": c.text, "cited": c.index == index}
                   for c in chunks],
    }


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    model = req.model or MODEL
    if req.model and req.model not in PRICES:
        raise HTTPException(400, f"unknown model {req.model}")

    if req.conversation_id:
        conv = _get(req.conversation_id)
    else:
        cid = uuid.uuid4().hex[:12]
        conv = Conversation(cid, RagAgent(kb, model=model, effort=req.effort, client=client))
        conv.title = req.message.strip().splitlines()[0][:60]
        conversations[cid] = conv
    if not conv.lock.acquire(blocking=False):
        raise HTTPException(409, "this conversation is already answering a question")

    conv.agent.model, conv.agent.effort = model, req.effort
    conv.turns.append(Turn("user", req.message))
    return StreamingResponse(_run(conv, req.message), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _run(conv: Conversation, question: str) -> AsyncIterator[str]:
    """Run the (blocking) agent in a worker thread and relay its events as SSE."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    turn = Turn("assistant", "")
    steps: dict[str, dict[str, Any]] = {}

    def emit(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def on_event(event: dict[str, Any]) -> None:
        if event["type"] == "tool_call":
            step = {"id": event["id"], "name": event["name"], "args": event["args"], "status": "running"}
            steps[event["id"]] = step
            turn.steps.append(step)
        elif event["type"] == "tool_result":
            steps[event["id"]].update(status="error" if event["is_error"] else "done", chars=event["chars"])
        elif event["type"] == "usage":
            event = {**event, "usage": asdict(event["usage"]) if isinstance(event["usage"], Usage) else event["usage"]}
        emit(event)

    def work() -> None:
        start = time.time()
        history_len = len(conv.agent.messages)
        conv.agent.on_event = on_event
        try:
            ans = conv.agent.ask(question)
            turn.text, turn.citations, turn.stop = ans.text, ans.citations, ans.stop
            turn.usage, turn.cost_usd = asdict(ans.usage), ans.cost_usd
        except Exception as e:  # surface API errors to the UI instead of dropping the stream
            turn.text, turn.stop = f"Error: {type(e).__name__}: {e}", "error"
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            # Roll the API history back to before this question so the next request is valid.
            del conv.agent.messages[history_len:]
        finally:
            turn.seconds = round(time.time() - start, 2)
            conv.turns.append(turn)
            conv.updated_at = time.time()
            conv.lock.release()
            emit(None)

    yield _sse({"type": "conversation", **conv.summary()})
    worker = loop.run_in_executor(None, work)
    while (event := await queue.get()) is not None:
        yield _sse(event)
    await worker
    yield _sse({"type": "done", "seconds": turn.seconds, "conversation": conv.summary()})


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


# ---------------------------------------------------------------- frontend

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        file = (DIST / path).resolve()
        if path and file.is_file() and file.is_relative_to(DIST.resolve()):
            return FileResponse(file)
        return FileResponse(DIST / "index.html")
