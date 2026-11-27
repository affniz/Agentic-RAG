"""FastAPI backend for the web UI.

    python -m uvicorn server.app:app --reload          # API on :8000 (serves frontend/dist if built)
    RAG_FAKE_LLM=1 python -m uvicorn server.app:app    # demo mode: no API key, no cost
    Settings: RAG_DOCS (documents folder, default ./docs), RAG_INDEX, RAG_RERANK=0,
              RAG_OCR=1 (transcribe scanned PDFs with Claude automatically),
              RAG_DATA (settings, feedback, conversations; default ./.rag_data), RAG_EVALS (eval sets, default ./evals),
              RAG_ALLOWED_HOSTS (extra hostnames the API answers to, comma-separated; see server/security.py),
              RAG_PASSWORD (require a login; see server/auth.py),
              RAG_GDRIVE_FOLDER (a Google Drive folder id or URL to sync documents from; see rag/gdrive.py;
                without it, Drive is connected from the Documents screen - see server/gdrive_connect.py),
              RAG_GDRIVE_CLIENT_ID / RAG_GDRIVE_CLIENT_SECRET (the OAuth client for "Connect Google Drive";
                can also be entered on the Documents screen),
              RAG_GDRIVE_CREDENTIALS (service account key file; default: Application Default Credentials),
              RAG_GDRIVE_DIR (where in RAG_DOCS the Drive files go, default "gdrive")

Chat responses stream as Server-Sent Events. Settings, 👍/👎 feedback and conversations are saved in
RAG_DATA, so chats survive a restart. Run a single uvicorn worker per RAG_DATA folder.
Only requests addressed to localhost are answered (server/security.py), unless RAG_PASSWORD is set: then
any hostname is accepted and every API route needs a login (server/auth.py).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import gdrive
from rag.agent import MODEL, PRICES, RagAgent, Usage
from rag.store import KnowledgeBase
from server import auth as auth_module
from server import evals
from server.conversations import ConversationStore
from server.documents import DriveSync, make_router
from server.feedback import FeedbackStore, Settings, SettingsStore, export_eval_set
from server.gdrive_connect import DriveConnection
from server.gdrive_connect import make_router as make_drive_router
from server.security import LocalhostGuard

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = os.environ.get("RAG_INDEX", str(ROOT / ".rag_index"))
FAKE_LLM = os.environ.get("RAG_FAKE_LLM") == "1"
RERANK = os.environ.get("RAG_RERANK", "1") == "1"
OCR = os.environ.get("RAG_OCR") == "1"  # transcribe scanned PDFs with Claude (sends them to the API)
DIST = ROOT / "frontend" / "dist"
DOCS_DIR = Path(os.environ.get("RAG_DOCS", str(ROOT / "docs")))
DATA_DIR = Path(os.environ.get("RAG_DATA", str(ROOT / ".rag_data")))
EVALS_DIR = Path(os.environ.get("RAG_EVALS", str(ROOT / "evals")))

log = logging.getLogger(__name__)

auth = auth_module.Auth(os.environ.get("RAG_PASSWORD"), DATA_DIR / "session_key")
app = FastAPI(title="Agentic RAG")
# The last middleware added runs first: reject other hosts and cross-site writes, then check the login.
app.add_middleware(auth_module.AuthGuard, auth=auth)
app.add_middleware(LocalhostGuard, any_host=auth.required)
app.include_router(auth_module.make_router(auth))


def _make_ocr() -> Any:
    if not OCR:
        return None
    from rag.ocr import claude_pdf_transcriber

    return claude_pdf_transcriber()


def _transcriber() -> Any:
    """For the per-file "Transcribe with Claude" action (always available; automatic OCR needs RAG_OCR=1)."""
    if FAKE_LLM:
        from rag.fake_llm import demo_transcriber

        return demo_transcriber
    from rag.ocr import claude_pdf_transcriber

    return claude_pdf_transcriber()


def _drive_dest() -> Path:
    """Where Drive files are mirrored: RAG_DOCS/RAG_GDRIVE_DIR."""
    dest = (DOCS_DIR / os.environ.get("RAG_GDRIVE_DIR", "gdrive")).resolve()
    if not dest.is_relative_to(DOCS_DIR.resolve()) or dest == DOCS_DIR.resolve():
        raise RuntimeError("RAG_GDRIVE_DIR must be a subfolder of the documents folder (RAG_DOCS)")
    return DOCS_DIR / dest.relative_to(DOCS_DIR.resolve())


DRIVE_DEST = _drive_dest()
_ENV_DRIVE_FOLDER = os.environ.get("RAG_GDRIVE_FOLDER", "").strip()
_env_drive: DriveSync | None = None
if _ENV_DRIVE_FOLDER:  # set by the server: a service account or Application Default Credentials
    _credentials = os.environ.get("RAG_GDRIVE_CREDENTIALS") or None
    _env_drive = DriveSync(gdrive.folder_id(_ENV_DRIVE_FOLDER), DRIVE_DEST,
                           functools.cache(lambda: gdrive.drive_session(_credentials)))
# Otherwise connected from the Documents screen with "Sign in with Google" (server/gdrive_connect.py).
drive_connection = DriveConnection(DATA_DIR / "gdrive.json", os.environ.get("RAG_GDRIVE_CLIENT_ID"),
                                   os.environ.get("RAG_GDRIVE_CLIENT_SECRET"))


def _drive_sync() -> DriveSync | None:
    """The Google Drive folder to sync from right now, if there is one."""
    if _env_drive is not None:
        return _env_drive
    data = drive_connection.load()
    if not (data.get("folder") and data.get("refresh_token") and drive_connection.client()):
        return None
    return DriveSync(data["folder"], DRIVE_DEST, drive_connection.session)


kb = KnowledgeBase(INDEX_DIR, rerank=RERANK, pdf_ocr=_make_ocr())
app.include_router(make_router(kb, DOCS_DIR, _transcriber, _drive_sync))
app.include_router(make_drive_router(drive_connection, _env_drive.folder if _env_drive else None,
                                     str(DRIVE_DEST.relative_to(DOCS_DIR)), lambda: gdrive.last_sync(DRIVE_DEST)))


def _make_client() -> Any:
    if FAKE_LLM:
        from rag.fake_llm import DemoClient

        return DemoClient()
    import anthropic

    return anthropic.Anthropic()


client = _make_client()
settings_store = SettingsStore(DATA_DIR / "settings.json", Settings(model=MODEL, rerank=RERANK))
feedback_store = FeedbackStore(DATA_DIR / "feedback.json")
app.include_router(evals.make_router(EVALS_DIR, DATA_DIR / "eval_index", lambda: client,
                                     settings_store.get, FAKE_LLM))


# ---------------------------------------------------------------- conversations


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    steps: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    seconds: float | None = None
    stop: str | None = None
    model: str | None = None
    feedback: dict[str, Any] | None = None  # {"rating": "up" | "down", "comment": str}
    timings: list[dict[str, Any]] = field(default_factory=list)  # one per model call (see RagAgent._create)


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
conversation_store = ConversationStore(DATA_DIR / "conversations")
TURN_FIELDS = {f.name for f in fields(Turn)}


def _save(conv: Conversation) -> None:
    """Write the conversation to disk, unless it was deleted meanwhile. Errors are logged, not raised."""
    try:
        with conversation_store.lock:
            if conversations.get(conv.id) is conv:
                conversation_store.save({
                    "id": conv.id, "title": conv.title, "created_at": conv.created_at,
                    "updated_at": conv.updated_at, "turns": [asdict(t) for t in conv.turns],
                    "model": conv.agent.model, "effort": conv.agent.effort, "messages": conv.agent.messages,
                })
    except Exception:
        log.exception("could not save conversation %s", conv.id)


def _forget(cid: str) -> None:
    with conversation_store.lock:
        conversations.pop(cid, None)
        conversation_store.delete(cid)


def _restore(data: dict[str, Any]) -> Conversation:
    agent = RagAgent(kb, model=data.get("model") or MODEL, effort=data.get("effort") or "medium", client=client)
    agent.messages = list(data.get("messages") or [])
    turns = [Turn(**{k: v for k, v in t.items() if k in TURN_FIELDS}) for t in data.get("turns") or []]
    now = time.time()
    return Conversation(data["id"], agent, title=data.get("title") or "New chat",
                        created_at=data.get("created_at") or now, updated_at=data.get("updated_at") or now,
                        turns=turns)


for _data in conversation_store.load_all():
    try:
        conversations[_data["id"]] = _restore(_data)
    except Exception as e:  # a file from an older version or edited by hand
        log.warning("skipping conversation %s: %s", _data.get("id"), e)


def _get(cid: str) -> Conversation:
    conv = conversations.get(cid)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    return conv


# ---------------------------------------------------------------- API


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = None
    model: str | None = None  # default: the saved settings
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: Literal["up", "down"] | None  # None removes the feedback
    comment: str = Field("", max_length=5000)


@app.get("/api/settings")
def get_settings() -> Settings:
    return settings_store.get()


@app.put("/api/settings")
def put_settings(s: Settings) -> Settings:
    if s.model not in PRICES:
        raise HTTPException(400, f"unknown model {s.model}")
    return settings_store.put(s)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "documents": len(kb.sources()),
        "chunks": len(kb.chunks),
        "default_model": MODEL,
        "models": list(PRICES),
        "demo_mode": FAKE_LLM,
        "reranker": bool(kb.reranker and kb.reranker.available),
        "ocr": kb.pdf_ocr is not None,
        "reranker_configured": kb.reranker is not None,
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
    _forget(cid)
    return {"ok": True}


@app.delete("/api/conversations")
def reset_conversations() -> dict[str, int]:
    """Clear every conversation (feedback already given is kept)."""
    busy = [c for c in conversations.values() if c.lock.locked()]
    for cid in [cid for cid, c in conversations.items() if c not in busy]:
        _forget(cid)
    return {"kept_busy": len(busy)}


@app.put("/api/feedback/{turn_id}")
def put_feedback(turn_id: str, req: FeedbackRequest) -> dict[str, Any]:
    conv = _get(req.conversation_id)
    pos = next((i for i, t in enumerate(conv.turns) if t.id == turn_id and t.role == "assistant"), None)
    if pos is None:
        raise HTTPException(404, "answer not found")
    turn = conv.turns[pos]
    question = next((t for t in reversed(conv.turns[:pos]) if t.role == "user"), None)
    if req.rating is None:
        turn.feedback = None
        feedback_store.set(turn_id, None)
        _save_if_idle(conv)
        return {"feedback": None}
    turn.feedback = {"rating": req.rating, "comment": req.comment}
    _save_if_idle(conv)
    feedback_store.set(turn_id, {
        "turn_id": turn_id, "conversation_id": conv.id, "rating": req.rating, "comment": req.comment,
        "question": question.text if question else "", "answer": turn.text, "citations": turn.citations,
        "model": turn.model, "followup": pos > 1, "cost_usd": turn.cost_usd,
    })
    return {"feedback": turn.feedback}


def _save_if_idle(conv: Conversation) -> None:
    # A busy conversation is saved when its turn ends; saving now would store a half-finished turn.
    if not conv.lock.locked():
        _save(conv)


@app.get("/api/feedback")
def list_feedback() -> list[dict[str, Any]]:
    return feedback_store.all()


@app.post("/api/feedback/export")
def export_feedback() -> dict[str, Any]:
    """Write all feedback as the eval set evals/feedback (replacing the previous export)."""
    records = feedback_store.all()
    if not records:
        raise HTTPException(400, "no feedback to export yet")
    return export_eval_set(records, EVALS_DIR / "feedback", DOCS_DIR)


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
    settings = settings_store.get()
    model, effort = req.model or settings.model, req.effort or settings.effort
    if model not in PRICES:
        raise HTTPException(400, f"unknown model {model}")

    if req.conversation_id:
        conv = _get(req.conversation_id)
    else:
        cid = uuid.uuid4().hex[:12]
        conv = Conversation(cid, RagAgent(kb, model=model, effort=effort, client=client))
        conv.title = req.message.strip().splitlines()[0][:60]
        conversations[cid] = conv
    if not conv.lock.acquire(blocking=False):
        raise HTTPException(409, "this conversation is already answering a question")

    # Settings apply from the next question on, also in existing conversations.
    conv.agent.model, conv.agent.effort, conv.agent.max_steps = model, effort, settings.max_steps
    conv.agent.max_cost_usd = settings.max_cost_usd
    conv.agent.search.k, conv.agent.search.rerank = settings.search_k, settings.rerank
    conv.turns.append(Turn("user", req.message))
    # The worker starts now, not when the stream is first read: it is what releases the lock, and a
    # client that disconnects early may never read the stream at all. The answer is then still saved.
    return StreamingResponse(_relay(conv, *_start(conv, req.message)), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


EventQueue = asyncio.Queue  # of event dicts, then None when the worker is done


def _start(conv: Conversation, question: str) -> tuple[Turn, EventQueue, asyncio.Future[None]]:
    """Run the (blocking) agent in a worker thread; its events go to the returned queue.
    Call with conv.lock held: the worker releases it when it finishes."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    turn = Turn("assistant", "", model=conv.agent.model)
    steps: dict[str, dict[str, Any]] = {}

    def emit(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def on_event(event: dict[str, Any]) -> None:
        if event["type"] == "tool_call":
            step = {"id": event["id"], "name": event["name"], "args": event["args"], "status": "running",
                    "step": event.get("step")}
            steps[event["id"]] = step
            turn.steps.append(step)
        elif event["type"] == "tool_result":
            steps[event["id"]].update(status="error" if event["is_error"] else "done", chars=event["chars"],
                                      results=event.get("results"), seconds=event.get("seconds"))
        elif event["type"] == "model_call":
            turn.timings.append({k: v for k, v in event.items() if k != "type"})
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
            _save(conv)
            conv.lock.release()
            emit(None)

    try:
        return turn, queue, loop.run_in_executor(None, work)
    except BaseException:  # the worker didn't start, so it can't release the lock
        conv.lock.release()
        raise


async def _relay(conv: Conversation, turn: Turn, queue: EventQueue, worker: asyncio.Future[None]) -> AsyncIterator[str]:
    """Relay the worker's events as SSE. If the client goes away, the worker carries on and saves the answer."""
    yield _sse({"type": "conversation", **conv.summary(), "turn_id": turn.id})
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
