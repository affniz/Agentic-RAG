"""Document ingestion and hybrid (BM25 + vector) retrieval with optional cross-encoder reranking."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from pypdf.errors import DependencyError, FileNotDecryptedError
from rank_bm25 import BM25Plus

SUPPORTED = {".md", ".txt", ".pdf"}
COLLECTION = "docs"
INDEX_VERSION = 4  # bump when the chunk format changes; forces a full rebuild
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
SEARCH_MODES = ("hybrid", "bm25", "vector")


@dataclass
class Chunk:
    id: str  # content-derived (see _content_id): unchanged text keeps its id, so its vector is reused
    source: str  # path relative to the docs folder
    index: int  # position of the chunk within its document
    text: str
    section: str = ""  # heading path for Markdown, e.g. "Vacation Policy > Carry-over"
    doc_note: str = ""  # the document's intro (status, effective dates, scope), shown with every chunk

    @property
    def indexed_text(self) -> str:
        """What gets embedded / keyword-indexed: section path + body."""
        return f"{self.section}\n{self.text}" if self.section else self.text

    def label(self) -> str:
        return f"[{self.source}#{self.index}]" + (f" ({self.section})" if self.section else "")

    def render(self) -> str:
        """How a chunk is shown to the model: citation label, document context, text."""
        note = f"\n(About this document: {self.doc_note})" if self.doc_note else ""
        return f"{self.label()}{note}\n{self.text}"


@dataclass
class IngestReport:
    added: list[str]
    updated: list[str]
    removed: list[str]
    unchanged: int
    chunks: int
    empty: list[str] = field(default_factory=list)  # files that yielded no text (e.g. scanned PDFs)
    failed: dict[str, str] = field(default_factory=dict)  # files that couldn't be read: {name: reason}
    embedded: int = 0  # chunks whose text was new and had to be embedded; the rest reused their vectors


# ---------------------------------------------------------------- loading & chunking


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _read_error(path: Path, e: Exception) -> str:
    """A short, actionable reason a file couldn't be read."""
    if isinstance(e, DependencyError):
        return "This PDF is encrypted. Install the 'cryptography' package (pip install -r requirements.txt) and re-index."
    if isinstance(e, FileNotDecryptedError):
        return "This PDF is password-protected. Remove the password (e.g. print it to a new PDF) and upload it again."
    return f"Couldn't read this file: {type(e).__name__}: {e}"


def _lacks_text_layer(path: Path, text: str) -> bool:
    """A PDF with almost no extractable text is most likely scanned images of pages."""
    pages = max(1, len(PdfReader(path).pages))
    return len(text.strip()) < 20 * pages


def _sections(text: str, markdown: bool) -> list[tuple[str, str]]:
    """Split text into (heading path, body) pairs. Non-Markdown text is one section."""
    if not markdown:
        return [("", text)]
    out: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def flush() -> None:
        if "".join(body).strip():
            out.append((" > ".join(h for _, h in stack), "\n".join(body)))
        body.clear()

    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if m:
            flush()
            level = len(m.group(1))
            stack = [(lvl, h) for lvl, h in stack if lvl < level] + [(level, m.group(2))]
        else:
            body.append(line)
    flush()
    return out


def _chunk(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split on paragraphs, packing them into chunks of at most `size` chars with overlap."""
    pieces: list[str] = []
    for p in (p.strip() for p in re.split(r"\n\s*\n", text)):
        while len(p) > size:
            pieces.append(p[:size])
            p = p[size - overlap :]
        if p:
            pieces.append(p)

    chunks: list[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + 2 + len(p) > size:
            chunks.append(cur)
            tail = cur[-overlap:]
            cur = f"{tail}\n\n{p}" if len(tail) + 2 + len(p) <= size else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def _doc_note(sections: list[tuple[str, str]], max_chars: int = 300) -> tuple[str, str]:
    """The document's intro: text before the first subsection (Markdown) or its first paragraph.

    Returns (section the intro came from, note). Chunks deep inside a document otherwise lose
    facts stated once at the top, such as "Effective 2026-03-01" or "Status: SUPERSEDED".
    """
    if not sections:
        return "", ""
    section, body = sections[0]
    if " > " in section:  # document starts straight with a subsection: no intro
        return "", ""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not section and paras:  # plain text: the first paragraph is usually a title
        paras = paras[:1]
    note = " ".join(" ".join(paras).split())
    return section, (note[: max_chars - 3] + "...") if len(note) > max_chars else note


def _content_id(source: str, indexed_text: str, occurrence: int) -> str:
    """Vector id for a chunk: a hash of its document and of the text that gets embedded.

    Ids don't depend on position, so when a document is edited, chunks whose text didn't change
    keep their id (and their vector), even if an earlier insertion shifted their index.
    `occurrence` tells identical chunks within one document apart.
    """
    digest = hashlib.sha256(f"{source}\0{indexed_text}".encode()).hexdigest()[:24]
    return digest if occurrence == 0 else f"{digest}-{occurrence}"


def chunk_document(source: str, text: str, markdown: bool) -> list[Chunk]:
    sections = _sections(text, markdown)
    _, note = _doc_note(sections)
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for section, body in sections:
        for piece in _chunk(body):
            i = len(chunks)
            # The document's first chunk contains the intro itself.
            chunk = Chunk("", source, i, piece, section, "" if i == 0 else note)
            n = seen.get(chunk.indexed_text, 0)
            seen[chunk.indexed_text] = n + 1
            chunk.id = _content_id(source, chunk.indexed_text, n)
            chunks.append(chunk)
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _write_atomic(path: Path, text: str) -> None:
    """Readers (and a crash mid-write) never see a half-written file."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- reranker


class Reranker:
    """Local cross-encoder (ONNX via fastembed). Loads lazily; disables itself if unavailable."""

    def __init__(self, model: str = RERANK_MODEL):
        self.model_name = model
        self._model = None
        self.available = True

    def _load(self) -> bool:
        if self._model is None and self.available:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(self.model_name)
            except Exception as e:  # missing package, no network for the first download, ...
                print(f"[rerank] disabled - could not load {self.model_name}: {e}", file=sys.stderr)
                self.available = False
        return self._model is not None

    def warm(self) -> bool:
        """Load (and on first use, download) the model now rather than mid-search."""
        return self._load()

    def score(self, query: str, docs: list[str]) -> list[float] | None:
        if not docs or not self._load():
            return None
        return [float(s) for s in self._model.rerank(query, docs)]


# ---------------------------------------------------------------- store


@dataclass(frozen=True)
class _Index:
    """Everything search reads, swapped as one object so a search never sees a half-updated index."""

    chunks: list[Chunk]
    by_id: dict[str, Chunk]
    tokens: list[set[str]]
    bm25: BM25Plus | None

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "_Index":
        tokenized = [_tokenize(c.indexed_text) for c in chunks]
        # BM25Plus keeps IDF positive; BM25Okapi zeroes it out on tiny corpora.
        return cls(chunks, {c.id: c for c in chunks}, [set(t) for t in tokenized],
                   BM25Plus(tokenized) if chunks else None)


ProgressFn = Callable[[str, int, int], None]  # (stage, done, total)


class KnowledgeBase:
    def __init__(
        self,
        index_dir: str | Path = ".rag_index",
        rerank: bool = True,
        reranker: Reranker | None = None,
        pdf_ocr: Callable[[Path], str] | None = None,
    ):
        """pdf_ocr: optional fallback that returns text for PDFs without a text layer (see rag/ocr.py)."""
        self.index_dir = Path(index_dir)
        self.pdf_ocr = pdf_ocr
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))
        # all-MiniLM-L6-v2 run locally via ONNX - no API key, no torch.
        self._embed = embedding_functions.DefaultEmbeddingFunction()
        self._col = self._get_collection()
        self._chunks_file = self.index_dir / "chunks.json"
        self._manifest_file = self.index_dir / "manifest.json"
        self.reranker = reranker or (Reranker() if rerank else None)
        self._index = _Index.build([])
        self._ingest_lock = threading.Lock()
        self._load()

    @property
    def chunks(self) -> list[Chunk]:
        return self._index.chunks

    def _get_collection(self):
        return self._client.get_or_create_collection(
            COLLECTION, embedding_function=self._embed, metadata={"hnsw:space": "cosine"}
        )

    # -- persistence

    def _manifest(self) -> dict:
        if self._manifest_file.exists():
            return json.loads(self._manifest_file.read_text())
        return {}

    def _load(self) -> None:
        if self._chunks_file.exists() and self._manifest().get("version") == INDEX_VERSION:
            self._index = _Index.build([Chunk(**c) for c in json.loads(self._chunks_file.read_text())])

    def indexed_files(self) -> dict[str, str]:
        """{relative path: sha256} of the files the index was last built from."""
        manifest = self._manifest()
        return manifest.get("files", {}) if manifest.get("version") == INDEX_VERSION else {}

    def index_errors(self) -> dict[str, str]:
        """{relative path: reason} for files the last ingest couldn't read."""
        manifest = self._manifest()
        return manifest.get("errors", {}) if manifest.get("version") == INDEX_VERSION else {}

    def indexed_times(self) -> dict[str, float]:
        """{relative path: unix time} each file was last (re-)read into the index."""
        manifest = self._manifest()
        return manifest.get("indexed_at", {}) if manifest.get("version") == INDEX_VERSION else {}

    def ingest(
        self,
        docs_dir: str | Path,
        rebuild: bool = False,
        on_progress: ProgressFn | None = None,
        force: list[str] | tuple[str, ...] = (),
    ) -> IngestReport:
        """Sync the index with `docs_dir`: re-read new/changed files, drop deleted ones.

        Updates are incremental per chunk: a changed file is re-chunked, and only chunks whose text
        is new get embedded; unchanged chunks keep their vectors (see _content_id).
        force: files (relative paths) to re-read even if unchanged, e.g. after transcribing one.
        Searches keep working during an ingest and see the old index until the new one is swapped in.
        """
        with self._ingest_lock:
            return self._ingest(Path(docs_dir), rebuild, on_progress or (lambda *_: None), set(force))

    def _ingest(self, docs_dir: Path, rebuild: bool, progress: ProgressFn, force: set[str]) -> IngestReport:
        manifest = self._manifest()
        old_chunks = list(self.chunks)
        if rebuild or manifest.get("version") != INDEX_VERSION:
            self._client.delete_collection(COLLECTION)
            self._col = self._get_collection()
            old_chunks, manifest = [], {}
        old_hashes: dict[str, str] = manifest.get("files", {})

        current: dict[str, Path] = {
            str(p.relative_to(docs_dir)): p
            for p in sorted(docs_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED
        }
        new_hashes = {rel: _file_hash(p) for rel, p in current.items()}
        self._prune_transcriptions(set(new_hashes.values()))
        added = [r for r in current if r not in old_hashes]
        updated = [r for r in current if r in old_hashes and old_hashes[r] != new_hashes[r]]
        removed = [r for r in old_hashes if r not in current]
        forced = sorted(r for r in force if r in current and r not in added and r not in updated)

        # PDFs that produced no chunks (e.g. scanned, indexed before OCR was enabled) get another
        # try once OCR is available. Based on the index itself, so it also covers older indexes.
        with_chunks = {c.source for c in old_chunks}
        retry = [r for r in current if r not in with_chunks and r not in added and r not in updated
                 and r not in forced and self.pdf_ocr is not None and r.lower().endswith(".pdf")]
        to_read = added + updated + forced + retry

        # 1. Read and chunk. Nothing is removed yet: a file that fails to read keeps its old chunks.
        fresh: list[Chunk] = []
        reread: set[str] = set()
        empty: list[str] = []
        failed: dict[str, str] = {}
        for n, rel in enumerate(to_read):
            progress("reading", n, len(to_read))
            path = current[rel]
            try:
                text, markdown = self._extract(path, new_hashes[rel])
            except Exception as e:  # one unreadable file must not stop the rest from being indexed
                failed[rel] = _read_error(path, e)
                print(f"[ingest] skipped {rel}: {failed[rel]}", file=sys.stderr)
                continue
            doc_chunks = chunk_document(rel, text, markdown=markdown)
            if not doc_chunks:
                empty.append(rel)
            fresh += doc_chunks
            reread.add(rel)

        # 2. Embed only chunks the collection doesn't already hold. Ids are content hashes, so an
        # unchanged chunk of an edited file (or one left by an interrupted ingest) is reused.
        stored = set(self._col.get(include=[])["ids"])
        to_embed = [c for c in fresh if c.id not in stored]
        for start in range(0, len(to_embed), 64):
            progress("embedding", start, len(to_embed))
            batch = to_embed[start : start + 64]
            self._col.add(
                ids=[c.id for c in batch],
                documents=[c.indexed_text for c in batch],
                metadatas=[{"source": c.source} for c in batch],
            )
        progress("embedding", len(to_embed), len(to_embed))

        # 3. Swap in the new chunk list. Old chunks go only for files that were re-read or removed.
        gone = reread | set(removed)
        chunks = sorted([c for c in old_chunks if c.source not in gone] + fresh, key=lambda c: (c.source, c.index))
        _write_atomic(self._chunks_file, json.dumps([asdict(c) for c in chunks]))
        # A file that failed keeps the hash its current chunks came from (none if it was never read),
        # so it still differs from the file on disk and is retried on every ingest until it reads.
        for rel in failed:
            if rel in old_hashes:
                new_hashes[rel] = old_hashes[rel]
            else:
                new_hashes.pop(rel)
        now = time.time()
        old_times: dict[str, float] = manifest.get("indexed_at", {})
        times = {rel: now if rel in reread or rel not in old_times else old_times[rel] for rel in new_hashes}
        _write_atomic(self._manifest_file, json.dumps(
            {"version": INDEX_VERSION, "files": new_hashes, "indexed_at": times, "errors": failed}))
        self._index = _Index.build(chunks)  # atomic swap

        # 4. Only now delete vectors nothing refers to any more (old versions of edited chunks,
        # removed files, leftovers of an interrupted ingest), so vector search never has a gap.
        live = {c.id for c in chunks}
        orphans = sorted((stored | {c.id for c in to_embed}) - live)
        for start in range(0, len(orphans), 500):
            self._col.delete(ids=orphans[start : start + 500])

        reprocessed = [r for r in updated + forced if r not in failed]
        return IngestReport([r for r in added if r not in failed], reprocessed, removed,
                            len(current) - len(added) - len(reprocessed), len(chunks), empty, failed,
                            embedded=len(to_embed))

    # -- transcriptions (PDF text written by Claude, see rag/ocr.py)

    def _transcription_file(self, digest: str) -> Path:
        return self.index_dir / "ocr" / f"{digest}.md"

    def has_transcription(self, path: Path) -> bool:
        return self._transcription_file(_file_hash(path)).exists()

    def transcribe(self, path: Path, ocr: Callable[[Path], str]) -> bool:
        """Transcribe one PDF and save the result; it is used instead of the PDF's own text from the
        next ingest of that file on. Returns False if no text came back."""
        text = ocr(path)
        if not text.strip():
            return False
        cache = self._transcription_file(_file_hash(path))
        cache.parent.mkdir(exist_ok=True)
        _write_atomic(cache, text)
        return True

    def clear_transcription(self, path: Path) -> bool:
        """Go back to the PDF's own text layer from the next ingest of that file on."""
        cache = self._transcription_file(_file_hash(path))
        if not cache.exists():
            return False
        cache.unlink()
        return True

    def _prune_transcriptions(self, in_use: set[str]) -> None:
        """Delete saved transcriptions whose file is gone (deleted or replaced by a new version)."""
        for cache in (self.index_dir / "ocr").glob("*.md"):
            if cache.stem not in in_use:
                cache.unlink(missing_ok=True)

    def _extract(self, path: Path, digest: str) -> tuple[str, bool]:
        """(text, is_markdown) for one file.

        A saved transcription always wins for a PDF. Without one, a PDF with no text layer is
        transcribed automatically only if `pdf_ocr` is configured.
        """
        if path.suffix.lower() != ".pdf":
            return _read(path), path.suffix.lower() == ".md"
        cache = self._transcription_file(digest)
        if cache.exists():
            return cache.read_text(), True
        text = _read(path)
        if not _lacks_text_layer(path, text) or self.pdf_ocr is None:
            return text, False
        new = self.pdf_ocr(path)
        if new.strip():
            cache.parent.mkdir(exist_ok=True)
            _write_atomic(cache, new)
        return new, True

    # -- retrieval

    def search(
        self,
        query: str,
        k: int = 5,
        source: str | None = None,
        mode: str = "hybrid",
        rerank: bool | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve the top-k chunks.

        mode: "bm25", "vector", or "hybrid" (both, fused with Reciprocal Rank Fusion).
        rerank: re-score the candidate pool with the cross-encoder (default: on if configured).
        """
        if mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {SEARCH_MODES}")
        idx = self._index  # one consistent snapshot for the whole search
        if not idx.chunks:
            return []
        use_rerank = (self.reranker is not None) if rerank is None else (rerank and self.reranker is not None)
        pool = max(k * 4, 20)
        rankings: list[list[str]] = []

        if mode in ("bm25", "hybrid"):
            q_tokens = _tokenize(query)
            scores = idx.bm25.get_scores(q_tokens)
            matching = (i for i in range(len(idx.chunks)) if idx.tokens[i].intersection(q_tokens))
            order = sorted(matching, key=lambda i: -scores[i])
            rankings.append(
                [idx.chunks[i].id for i in order if source is None or idx.chunks[i].source == source][:pool]
            )
        if mode in ("vector", "hybrid"):
            try:
                res = self._col.query(
                    query_texts=[query],
                    n_results=min(pool, len(idx.chunks)),
                    where={"source": source} if source else None,
                )
                rankings.append(res["ids"][0])
            except Exception as e:  # e.g. the collection is being rebuilt; keyword results still work
                if mode == "vector":
                    raise
                print(f"[search] vector search failed, using keyword results only: {e}", file=sys.stderr)

        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
        candidates = [(idx.by_id[cid], s) for cid, s in sorted(fused.items(), key=lambda kv: -kv[1]) if cid in idx.by_id]

        if use_rerank and len(candidates) > 1:
            scores = self.reranker.score(query, [c.indexed_text for c, _ in candidates[:pool]])
            if scores is not None:
                candidates = sorted(zip([c for c, _ in candidates[:pool]], scores), key=lambda cs: -cs[1])
        return candidates[:k]

    def get_chunks(self, source: str, start: int, end: int) -> list[Chunk]:
        return [c for c in self.chunks if c.source == source and start <= c.index <= end]

    def sources(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.source] = out.get(c.source, 0) + 1
        return out
