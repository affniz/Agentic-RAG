"""Document ingestion and hybrid (BM25 + vector) retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

SUPPORTED = {".md", ".txt", ".pdf"}
COLLECTION = "docs"


@dataclass
class Chunk:
    id: str
    source: str  # path relative to the docs folder
    index: int  # position of the chunk within its document
    text: str


# ---------------------------------------------------------------- loading


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split on paragraphs, packing them into ~`size`-char chunks with overlap."""
    # Break paragraphs that alone exceed the chunk size into overlapping pieces.
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
            # Carry an overlap tail into the next chunk only when it still fits.
            cur = f"{tail}\n\n{p}" if len(tail) + 2 + len(p) <= size else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------- store


class KnowledgeBase:
    def __init__(self, index_dir: str | Path = ".rag_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))
        # all-MiniLM-L6-v2 run locally via ONNX - no API key, no torch.
        self._embed = embedding_functions.DefaultEmbeddingFunction()
        self._col = self._client.get_or_create_collection(
            COLLECTION, embedding_function=self._embed, metadata={"hnsw:space": "cosine"}
        )
        self._chunks_file = self.index_dir / "chunks.json"
        self.chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._load()

    # -- persistence

    def _load(self) -> None:
        if self._chunks_file.exists():
            self.chunks = [Chunk(**c) for c in json.loads(self._chunks_file.read_text())]
        self._reindex()

    def _reindex(self) -> None:
        self._by_id = {c.id: c for c in self.chunks}
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks]) if self.chunks else None

    def ingest(self, docs_dir: str | Path) -> int:
        """(Re)build the index from every supported file under `docs_dir`."""
        docs_dir = Path(docs_dir)
        chunks: list[Chunk] = []
        for path in sorted(docs_dir.rglob("*")):
            if path.suffix.lower() not in SUPPORTED or not path.is_file():
                continue
            rel = str(path.relative_to(docs_dir))
            for i, text in enumerate(_chunk(_read(path))):
                cid = hashlib.sha1(f"{rel}:{i}".encode()).hexdigest()[:16]
                chunks.append(Chunk(cid, rel, i, text))

        self._client.delete_collection(COLLECTION)
        self._col = self._client.create_collection(
            COLLECTION, embedding_function=self._embed, metadata={"hnsw:space": "cosine"}
        )
        for start in range(0, len(chunks), 256):
            batch = chunks[start : start + 256]
            self._col.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{"source": c.source, "index": c.index} for c in batch],
            )
        self.chunks = chunks
        self._chunks_file.write_text(json.dumps([asdict(c) for c in chunks]))
        self._reindex()
        return len(chunks)

    # -- retrieval

    def search(self, query: str, k: int = 5, source: str | None = None) -> list[tuple[Chunk, float]]:
        """Hybrid search: BM25 + dense vectors fused with Reciprocal Rank Fusion."""
        if not self.chunks:
            return []
        pool = max(k * 4, 20)
        allowed = None if source is None else {c.id for c in self.chunks if c.source == source}

        # Keyword ranking
        scores = self._bm25.get_scores(_tokenize(query))
        kw = sorted(
            (i for i in range(len(self.chunks)) if scores[i] > 0),
            key=lambda i: -scores[i],
        )
        kw_ids = [self.chunks[i].id for i in kw if allowed is None or self.chunks[i].id in allowed][:pool]

        # Semantic ranking
        res = self._col.query(
            query_texts=[query],
            n_results=min(pool, len(self.chunks)),
            where={"source": source} if source else None,
        )
        vec_ids = res["ids"][0]

        # Reciprocal Rank Fusion
        fused: dict[str, float] = {}
        for ranking in (kw_ids, vec_ids):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(self._by_id[cid], score) for cid, score in best if cid in self._by_id]

    def get_chunks(self, source: str, start: int, end: int) -> list[Chunk]:
        return [c for c in self.chunks if c.source == source and start <= c.index <= end]

    def sources(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.source] = out.get(c.source, 0) + 1
        return out
