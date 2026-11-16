"""Document management API: list, upload, delete, re-index (background job with progress), search tester."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from rag.store import SEARCH_MODES, SUPPORTED, KnowledgeBase

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_NAME_OK = re.compile(r"^[^/\\\x00-\x1f]+$")


class IngestJob:
    """One re-index at a time, run in a background thread; the UI polls its state."""

    def __init__(self, kb: KnowledgeBase, docs_dir: Path):
        self.kb, self.docs_dir = kb, docs_dir
        self._lock = threading.Lock()
        self.state: dict[str, Any] = {"state": "idle"}

    def start(self, rebuild: bool = False, reason: str = "manual") -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise HTTPException(409, "an ingest is already running")
        self.state = {"state": "running", "stage": "starting", "done": 0, "total": 0, "reason": reason,
                      "rebuild": rebuild, "started_at": time.time()}
        threading.Thread(target=self._run, args=(rebuild,), daemon=True).start()
        return self.state

    def _run(self, rebuild: bool) -> None:
        def progress(stage: str, done: int, total: int) -> None:
            self.state.update(stage=stage, done=done, total=total)

        try:
            r = self.kb.ingest(self.docs_dir, rebuild=rebuild, on_progress=progress)
            self.state.update(state="done", report={"added": r.added, "updated": r.updated, "removed": r.removed,
                                                    "unchanged": r.unchanged, "chunks": r.chunks, "empty": r.empty})
        except Exception as e:  # surfaced in the UI
            self.state.update(state="error", error=f"{type(e).__name__}: {e}")
        finally:
            self.state["finished_at"] = time.time()
            self._lock.release()

    @property
    def running(self) -> bool:
        return self.state.get("state") == "running"


class IngestRequest(BaseModel):
    rebuild: bool = False


def make_router(kb: KnowledgeBase, docs_dir: Path) -> APIRouter:
    docs_dir.mkdir(parents=True, exist_ok=True)
    root = docs_dir.resolve()
    job = IngestJob(kb, docs_dir)
    router = APIRouter(prefix="/api")

    def resolve(name: str) -> Path:
        path = (docs_dir / name).resolve()
        if not path.is_relative_to(root) or path == root:
            raise HTTPException(400, "invalid document name")
        return path

    @router.get("/documents")
    def list_documents() -> dict[str, Any]:
        # Snapshot the job state first: if an ingest finishes while we build the list, the client
        # still sees "running" and polls again, instead of pairing "done" with a stale list.
        ingest_state = dict(job.state)
        indexed = kb.indexed_files()
        counts = kb.sources()
        rows = []
        for path in sorted(docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            rel = str(path.relative_to(docs_dir))
            stat = path.stat()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            status = "new" if rel not in indexed else ("indexed" if indexed[rel] == digest else "changed")
            if status == "indexed" and not counts.get(rel):
                status = "empty"  # indexed, but no text could be extracted
            rows.append({"name": rel, "type": path.suffix.lower().lstrip("."), "size": stat.st_size,
                         "modified": stat.st_mtime, "chunks": counts.get(rel, 0), "status": status})
        on_disk = {r["name"] for r in rows}
        rows += [{"name": rel, "type": Path(rel).suffix.lstrip("."), "size": None, "modified": None,
                  "chunks": counts.get(rel, 0), "status": "deleted"} for rel in sorted(indexed) if rel not in on_disk]
        return {"documents": rows, "chunks": len(kb.chunks), "ingest": ingest_state, "ocr": kb.pdf_ocr is not None,
                "needs_ingest": any(r["status"] not in ("indexed", "empty") for r in rows)}

    @router.post("/documents")
    async def upload(files: list[UploadFile]) -> dict[str, Any]:
        saved, rejected = [], []
        for f in files:
            name = Path(f.filename or "").name
            if not name or name.startswith(".") or not _NAME_OK.match(name):
                rejected.append({"name": f.filename, "reason": "invalid file name"})
                continue
            if Path(name).suffix.lower() not in SUPPORTED:
                rejected.append({"name": name, "reason": f"unsupported type (use {', '.join(sorted(SUPPORTED))})"})
                continue
            data = await f.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                rejected.append({"name": name, "reason": f"larger than {MAX_UPLOAD_BYTES // 2**20} MB"})
                continue
            if not data:
                rejected.append({"name": name, "reason": "empty file"})
                continue
            target = resolve(name)
            tmp = target.with_name(f".{target.name}.upload")
            tmp.write_bytes(data)
            tmp.replace(target)  # atomic: never leave a half-written document
            saved.append(name)
        ingest = None
        if saved:
            ingest = job.state if job.running else job.start(reason="upload")
        return {"saved": saved, "rejected": rejected, "ingest": ingest}

    @router.delete("/documents/{name:path}")
    def delete(name: str) -> dict[str, Any]:
        path = resolve(name)
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            raise HTTPException(404, "document not found")
        path.unlink()
        return {"deleted": name, "ingest": job.state if job.running else job.start(reason="delete")}

    @router.get("/documents/{name:path}/chunks")
    def document_chunks(name: str) -> dict[str, Any]:
        chunks = [c for c in kb.chunks if c.source == name]
        if not chunks:
            raise HTTPException(404, "no indexed chunks for this document")
        return {"name": name, "doc_note": next((c.doc_note for c in chunks if c.doc_note), ""),
                "chunks": [{"index": c.index, "section": c.section, "text": c.text} for c in chunks]}

    @router.get("/ingest")
    def ingest_status() -> dict[str, Any]:
        return job.state

    @router.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        return job.start(rebuild=req.rebuild)

    @router.get("/search")
    def search(q: str, k: int = 8, mode: str = "hybrid", rerank: bool = True,
               source: str | None = None) -> dict[str, Any]:
        if mode not in SEARCH_MODES:
            raise HTTPException(400, f"mode must be one of {SEARCH_MODES}")
        start = time.time()
        hits = kb.search(q, k=max(1, min(k, 25)), mode=mode, rerank=rerank, source=source or None)
        reranked = bool(rerank and kb.reranker and kb.reranker.available)
        return {
            "query": q, "mode": mode, "reranked": reranked, "ms": round((time.time() - start) * 1000),
            "results": [{"rank": i + 1, "score": round(score, 4), "source": c.source, "index": c.index,
                         "section": c.section, "doc_note": c.doc_note, "text": c.text}
                        for i, (c, score) in enumerate(hits)],
        }

    return router
