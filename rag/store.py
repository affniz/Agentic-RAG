"""Document ingestion and hybrid (BM25 + vector) retrieval with optional cross-encoder reranking."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from rank_bm25 import BM25Plus

SUPPORTED = {".md", ".txt", ".pdf"}
COLLECTION = "docs"
INDEX_VERSION = 3  # bump when the chunk format changes; forces a full rebuild
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
SEARCH_MODES = ("hybrid", "bm25", "vector")


@dataclass
class Chunk:
    id: str
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


# ---------------------------------------------------------------- loading & chunking


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="replace")


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


def chunk_document(source: str, text: str, markdown: bool) -> list[Chunk]:
    sections = _sections(text, markdown)
    _, note = _doc_note(sections)
    chunks: list[Chunk] = []
    for section, body in sections:
        for piece in _chunk(body):
            i = len(chunks)
            # The document's first chunk contains the intro itself.
            cid = hashlib.sha1(f"{source}:{i}".encode()).hexdigest()[:16]
            chunks.append(Chunk(cid, source, i, piece, section, "" if i == 0 else note))
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


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


class KnowledgeBase:
    def __init__(self, index_dir: str | Path = ".rag_index", rerank: bool = True, reranker: Reranker | None = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))
        # all-MiniLM-L6-v2 run locally via ONNX - no API key, no torch.
        self._embed = embedding_functions.DefaultEmbeddingFunction()
        self._col = self._get_collection()
        self._chunks_file = self.index_dir / "chunks.json"
        self._manifest_file = self.index_dir / "manifest.json"
        self.reranker = reranker or (Reranker() if rerank else None)
        self.chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._bm25: BM25Plus | None = None
        self._tokens: list[set[str]] = []
        self._load()

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
            self.chunks = [Chunk(**c) for c in json.loads(self._chunks_file.read_text())]
        self._reindex()

    def _reindex(self) -> None:
        self._by_id = {c.id: c for c in self.chunks}
        tokenized = [_tokenize(c.indexed_text) for c in self.chunks]
        self._tokens = [set(t) for t in tokenized]
        # BM25Plus keeps IDF positive; BM25Okapi zeroes it out on tiny corpora.
        self._bm25 = BM25Plus(tokenized) if self.chunks else None

    def ingest(self, docs_dir: str | Path, rebuild: bool = False) -> IngestReport:
        """Sync the index with `docs_dir`: embed new/changed files, drop deleted ones."""
        docs_dir = Path(docs_dir)
        manifest = self._manifest()
        if rebuild or manifest.get("version") != INDEX_VERSION:
            self._client.delete_collection(COLLECTION)
            self._col = self._get_collection()
            self.chunks, manifest = [], {}
        old_hashes: dict[str, str] = manifest.get("files", {})

        current: dict[str, Path] = {
            str(p.relative_to(docs_dir)): p
            for p in sorted(docs_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED
        }
        new_hashes = {rel: _file_hash(p) for rel, p in current.items()}
        added = [r for r in current if r not in old_hashes]
        updated = [r for r in current if r in old_hashes and old_hashes[r] != new_hashes[r]]
        removed = [r for r in old_hashes if r not in current]
        stale = set(updated) | set(removed)

        if stale:
            self._col.delete(where={"source": {"$in": sorted(stale)}})
            self.chunks = [c for c in self.chunks if c.source not in stale]

        fresh: list[Chunk] = []
        for rel in added + updated:
            path = current[rel]
            fresh += chunk_document(rel, _read(path), markdown=path.suffix.lower() == ".md")
        for start in range(0, len(fresh), 256):
            batch = fresh[start : start + 256]
            self._col.add(
                ids=[c.id for c in batch],
                documents=[c.indexed_text for c in batch],
                metadatas=[{"source": c.source, "index": c.index} for c in batch],
            )

        self.chunks = sorted(self.chunks + fresh, key=lambda c: (c.source, c.index))
        self._chunks_file.write_text(json.dumps([asdict(c) for c in self.chunks]))
        self._manifest_file.write_text(json.dumps({"version": INDEX_VERSION, "files": new_hashes}))
        self._reindex()
        return IngestReport(added, updated, removed, len(current) - len(added) - len(updated), len(self.chunks))

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
        if not self.chunks:
            return []
        use_rerank = (self.reranker is not None) if rerank is None else (rerank and self.reranker is not None)
        pool = max(k * 4, 20)
        rankings: list[list[str]] = []

        if mode in ("bm25", "hybrid"):
            q_tokens = _tokenize(query)
            scores = self._bm25.get_scores(q_tokens)
            matching = (i for i in range(len(self.chunks)) if self._tokens[i].intersection(q_tokens))
            order = sorted(matching, key=lambda i: -scores[i])
            rankings.append(
                [self.chunks[i].id for i in order if source is None or self.chunks[i].source == source][:pool]
            )
        if mode in ("vector", "hybrid"):
            res = self._col.query(
                query_texts=[query],
                n_results=min(pool, len(self.chunks)),
                where={"source": source} if source else None,
            )
            rankings.append(res["ids"][0])

        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
        candidates = [(self._by_id[cid], s) for cid, s in sorted(fused.items(), key=lambda kv: -kv[1]) if cid in self._by_id]

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
