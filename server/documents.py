"""Document management API: list, upload, delete, re-index and per-file transcription (background jobs
with progress), Google Drive sync, search tester."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from pypdf import PdfReader

from rag import gdrive
from rag.store import SEARCH_MODES, SUPPORTED, KnowledgeBase

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# Rough Claude Haiku cost of transcribing one PDF page (~1.6k input + ~0.6k output tokens).
EST_USD_PER_PAGE = 0.0005
Transcriber = Callable[[Path], str]
Progress = Callable[[str, int, int], None]
# Runs before an ingest (a transcription, a Drive sync); may return fields to add to the job's final state.
BeforeFn = Callable[[Progress], dict[str, Any] | None]
_NAME_OK = re.compile(r"^[^/\\\x00-\x1f]+$")


class IngestJob:
    """One re-index at a time, run in a background thread; the UI polls its state."""

    def __init__(self, kb: KnowledgeBase, docs_dir: Path):
        self.kb, self.docs_dir = kb, docs_dir
        self._lock = threading.Lock()  # held while a job runs
        # Guards the pending flag together with taking/releasing _lock, so a request() that arrives
        # just as a job finishes is either seen by that job or starts a new one - never lost.
        self._guard = threading.Lock()
        self._pending: str | None = None  # reason for one more pass after the running job
        self.state: dict[str, Any] = {"state": "idle"}

    def start(self, rebuild: bool = False, reason: str = "manual", force: tuple[str, ...] = (),
              before: BeforeFn | None = None, file: str | None = None) -> dict[str, Any]:
        """Run `before` (e.g. a transcription), then an ingest, in a background thread."""
        if not self._lock.acquire(blocking=False):
            raise HTTPException(409, "indexing is already running - try again when it finishes")
        return self._launch(rebuild, reason, force, before, file)

    def request(self, reason: str) -> dict[str, Any]:
        """Index the folder's current contents: start an ingest now, or, if one is running, run one more
        pass when it finishes (it may already have scanned past a file that was just added or removed)."""
        with self._guard:
            if self._lock.acquire(blocking=False):
                return self._launch(False, reason, (), None, None)
            self._pending = reason
            return self.state

    def _launch(self, rebuild: bool, reason: str, force: tuple[str, ...], before: BeforeFn | None,
                file: str | None) -> dict[str, Any]:
        """Start the job thread; the caller holds _lock, which the thread releases."""
        self._set_running(reason, rebuild, file)
        threading.Thread(target=self._run, args=(rebuild, force, before), daemon=True).start()
        return self.state

    def _set_running(self, reason: str, rebuild: bool, file: str | None) -> None:
        self.state = {"state": "running", "stage": "starting", "done": 0, "total": 0, "reason": reason,
                      "rebuild": rebuild, "file": file, "started_at": time.time()}

    def _run(self, rebuild: bool, force: tuple[str, ...], before: BeforeFn | None) -> None:
        carried: dict[str, Any] = {}  # what `before` returned (e.g. a Drive sync report), kept across passes
        while True:
            result = self._pass(rebuild, force, before)
            carried.update({k: v for k, v in result.items() if k not in ("state", "report", "error")})
            with self._guard:
                if self._pending is None:
                    # Published only now, so pollers never see "done" between two passes.
                    self.state.update(carried, **result, finished_at=time.time())
                    self._lock.release()
                    return
                reason, self._pending = self._pending, None
            # Files changed while that pass ran: one more incremental pass, still holding the lock.
            rebuild, force, before = False, (), None
            self._set_running(reason, False, None)

    def _pass(self, rebuild: bool, force: tuple[str, ...], before: BeforeFn | None) -> dict[str, Any]:
        """Run one ingest; return the final state fields (not yet applied to self.state)."""
        def progress(stage: str, done: int, total: int) -> None:
            self.state.update(stage=stage, done=done, total=total)

        try:
            extra = (before(progress) or {}) if before is not None else {}
            r = self.kb.ingest(self.docs_dir, rebuild=rebuild, on_progress=progress, force=force)
            return {**extra, "state": "done", "report": {"added": r.added, "updated": r.updated, "removed": r.removed,
                                                "unchanged": r.unchanged, "chunks": r.chunks, "empty": r.empty,
                                                "failed": r.failed, "embedded": r.embedded}}
        except Exception as e:  # surfaced in the UI
            return {"state": "error", "error": f"{type(e).__name__}: {e}"}

    @property
    def running(self) -> bool:
        return self.state.get("state") == "running"


class IngestRequest(BaseModel):
    rebuild: bool = False


@dataclass
class DriveSync:
    """A Google Drive folder mirrored into `dest` (a folder inside the documents folder)."""

    folder: str  # folder id
    dest: Path
    session: Callable[[], gdrive.Session]  # created on first use, so a bad key only fails the sync


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdf_pages(path: Path) -> int | None:
    try:
        return len(PdfReader(path).pages)
    except Exception:  # unreadable or encrypted PDF
        return None


def make_router(kb: KnowledgeBase, docs_dir: Path, transcriber: Callable[[], Transcriber],
                drive_sync: Callable[[], DriveSync | None] = lambda: None) -> APIRouter:
    """transcriber: returns the function that turns a PDF into text (created on first use).
    drive_sync: the Google Drive folder that POST /api/gdrive/sync mirrors right now, if one is set up
    (it can be connected while the server runs, so it's looked up on each request)."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    root = docs_dir.resolve()
    job = IngestJob(kb, docs_dir)
    router = APIRouter(prefix="/api")
    # (rel_path, size, mtime_ns, inode) -> (sha256, pdf pages): the UI polls the list while indexing,
    # so only files that changed since the last listing are re-read. The inode catches a same-size
    # replacement (uploads swap in a new file) within one mtime tick.
    file_info: dict[tuple[str, int, int, int], tuple[str, int | None]] = {}
    file_info_lock = threading.Lock()

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
        drive = drive_sync()
        indexed = kb.indexed_files()
        indexed_at = kb.indexed_times()
        errors = kb.index_errors()
        counts = kb.sources()
        rows = []
        seen: dict[tuple[str, int, int, int], tuple[str, int | None]] = {}
        for path in sorted(docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            rel = str(path.relative_to(docs_dir))
            stat = path.stat()
            is_pdf = path.suffix.lower() == ".pdf"
            key = (rel, stat.st_size, stat.st_mtime_ns, stat.st_ino)
            info = file_info.get(key)
            if info is None:
                info = (_file_digest(path), _pdf_pages(path) if is_pdf else None)
            seen[key] = info
            digest, pages = info
            status = "new" if rel not in indexed else ("indexed" if indexed[rel] == digest else "changed")
            if status == "indexed" and not counts.get(rel):
                status = "empty"  # indexed, but no text could be extracted
            if rel in errors:
                status = "error"  # the last ingest tried and couldn't read it (a previous version may still be indexed)
            rows.append({"name": rel, "type": path.suffix.lower().lstrip("."), "size": stat.st_size,
                         "modified": stat.st_mtime, "indexed_at": indexed_at.get(rel), "error": errors.get(rel) if status == "error" else None, "chunks": counts.get(rel, 0), "status": status,
                         "pages": pages, "transcribed": is_pdf and kb.has_transcription(path),
                         "transcribe_cost_usd": round((pages or 1) * EST_USD_PER_PAGE, 4) if is_pdf else None})
        with file_info_lock:  # keep only files that still exist, as they are now
            file_info.clear()
            file_info.update(seen)
        on_disk = {r["name"] for r in rows}
        rows += [{"name": rel, "type": Path(rel).suffix.lstrip("."), "size": None, "modified": None,
                  "indexed_at": indexed_at.get(rel), "error": None, "chunks": counts.get(rel, 0), "status": "deleted", "pages": None, "transcribed": False,
                  "transcribe_cost_usd": None} for rel in sorted(indexed) if rel not in on_disk]
        return {"documents": rows, "chunks": len(kb.chunks), "ingest": ingest_state, "ocr": kb.pdf_ocr is not None,
                "needs_ingest": any(r["status"] not in ("indexed", "empty", "error") for r in rows),
                "gdrive": None if drive is None else {
                    "folder": drive.folder, "dir": str(drive.dest.relative_to(docs_dir)),
                    "last_sync": gdrive.last_sync(drive.dest)}}

    @router.post("/documents")
    async def upload(files: list[UploadFile], overwrite: bool = False) -> dict[str, Any]:
        """Save uploaded files. A name that already exists is returned in `conflicts` and left alone,
        unless overwrite=true."""
        saved, rejected, conflicts = [], [], []
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
            if target.exists() and not overwrite:
                conflicts.append(name)
                continue
            tmp = target.with_name(f".{target.name}.upload")
            tmp.write_bytes(data)
            tmp.replace(target)  # atomic: never leave a half-written document
            saved.append(name)
        ingest = job.request("upload") if saved else None
        return {"saved": saved, "rejected": rejected, "conflicts": conflicts, "ingest": ingest}

    def existing_pdf(name: str) -> Path:
        path = resolve(name)
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise HTTPException(404, "PDF not found")
        return path

    # Registered before DELETE /documents/{name:path}, which would otherwise also match these paths.
    @router.post("/documents/{name:path}/transcribe")
    def transcribe(name: str) -> dict[str, Any]:
        """Send one PDF to Claude for transcription, then re-index it. Runs in the background."""
        path = existing_pdf(name)

        def run(progress: Progress) -> None:
            progress("transcribing", 0, 1)
            if not kb.transcribe(path, transcriber()):
                raise RuntimeError("Claude returned no text for this file - see the server log for details")

        return job.start(reason="transcribe", force=(name,), before=run, file=name)

    @router.delete("/documents/{name:path}/transcription")
    def clear_transcription(name: str) -> dict[str, Any]:
        """Go back to the PDF's own text layer, then re-index it."""
        path = existing_pdf(name)
        if not kb.has_transcription(path):
            raise HTTPException(404, "this file has no transcription")
        if job.running:
            raise HTTPException(409, "indexing is already running - try again when it finishes")
        kb.clear_transcription(path)
        return job.start(reason="untranscribe", force=(name,), file=name)

    @router.delete("/documents/{name:path}")
    def delete(name: str) -> dict[str, Any]:
        path = resolve(name)
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            raise HTTPException(404, "document not found")
        # Delete its transcription now, not at the next ingest (which may wait behind a running one),
        # unless another file in the folder has identical contents and still uses it.
        digest = kb.indexed_files().get(name)
        if not any(d == digest for other, d in kb.indexed_files().items() if other != name):
            kb.clear_transcription(path)
        path.unlink()
        return {"deleted": name, "ingest": job.request("delete")}

    @router.get("/documents/{name:path}/chunks")
    def document_chunks(name: str) -> dict[str, Any]:
        chunks = [c for c in kb.chunks if c.source == name]
        if not chunks:
            raise HTTPException(404, "no indexed chunks for this document")
        return {"name": name, "doc_note": next((c.doc_note for c in chunks if c.doc_note), ""),
                "chunks": [{"index": c.index, "section": c.section, "text": c.text} for c in chunks]}

    @router.post("/gdrive/sync")
    def gdrive_sync() -> dict[str, Any]:
        """Mirror the configured Drive folder into the documents folder, then index what changed."""
        drive = drive_sync()
        if drive is None:
            raise HTTPException(404, "Google Drive isn't set up - connect it and choose a folder on the Documents screen")

        def run(progress: Progress) -> dict[str, Any]:
            report = gdrive.sync_folder(drive.session(), drive.folder, drive.dest, on_progress=progress)
            return {"sync": asdict(report)}

        return job.start(reason="gdrive", before=run)

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
