# Agentic RAG with Claude

A retrieval-augmented generation system in which **Claude drives retrieval itself**: it decides
what to search, splits compound questions into several searches, reformulates queries that miss,
reads surrounding chunks for context, and answers with inline citations like `[policy.md#2]`.

## How it works

```
question ─► Claude (claude-sonnet-5-5) ──tool calls──► list_documents
               ▲                                       search_knowledge_base
               │                                         BM25 + vectors ─► RRF fusion ─► cross-encoder rerank
               └──────────── tool results ◄─────────── read_chunks (neighbouring context)
            ... loops until it has enough evidence ...
            ─► cited answer  (+ token usage and cost)
```

| File | Role |
|---|---|
| `rag/store.py` | Loads `.md` / `.txt` / `.pdf`; heading-aware chunking (each chunk knows its Markdown section path); incremental indexing into ChromaDB (local all-MiniLM-L6-v2 ONNX embeddings) + BM25; hybrid search with Reciprocal Rank Fusion; optional local cross-encoder reranking |
| `rag/agent.py` | The agent loop: strict tool schemas, parallel tool execution, multi-turn memory, step limit, refusal / `max_tokens` handling, event callbacks, per-question cost tracking |
| `rag/evaluate.py` | Retrieval metrics (recall@k, hit@k, MRR per search mode) and end-to-end answer grading with a Claude Haiku judge |
| `evals/questions.jsonl` | 20 questions over the sample docs: evidence snippets, expected sources, reference answers (incl. one unanswerable) |
| `main.py` | CLI |
| `tests/` | pytest suite (no API key needed) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

The first run downloads two small local models once: the embedding model (~80 MB) and the
reranker (`Xenova/ms-marco-MiniLM-L-6-v2`, from Hugging Face). If the reranker can't be
downloaded, search keeps working without it (or pass `--no-rerank`).

## Usage

```bash
python main.py ingest docs                 # sync the index; only new/changed files are re-embedded
python main.py ingest docs --rebuild       # re-embed everything
python main.py search "carry over"         # test retrieval alone, no API call (--mode bm25|vector|hybrid)
python main.py ask "How many vacation days do I get after 6 years, and how many can I carry over?"
python main.py chat                        # multi-turn; /reset, /cost, /quit
```

`ask` / `chat` take `--model` (`claude-sonnet-5-5` default, `claude-opus-5-5`, `claude-haiku-5-5`),
`--effort` (`low` … `max`, default `medium`) and `-q` to hide the tool trace and cost line.

## Evaluation

```bash
python main.py eval                        # retrieval metrics for bm25 / vector / hybrid / hybrid+rerank - free
python main.py eval --answers              # also runs the agent on every question and grades the answers
python main.py eval --answers --limit 5    # cheaper trial run
```

Answer grading uses `claude-haiku-5-5` as a judge against each reference answer and checks that the
answer cites the expected document. A full run of the 20 questions costs roughly $0.50–1 on Sonnet
(estimate). Results are saved to `evals/results/`. **Replace the questions with ones about your own
documents** - the sample corpus is small enough that every search mode does well on it.

## Tests

```bash
pip install -r requirements-dev.txt && pytest -q
```

## Settings worth knowing

- **Model** – `MODEL` in `rag/agent.py` or `--model`. Sonnet 5.5 is ~$0.02–0.05 per typical question; Opus 5.5 is 2× that, Haiku 5.5 ~1/20.
- **Effort** – `medium` by default; `high` helps on hard multi-hop questions.
- **Refusal fallback** – requests use `fallbacks="default"` (beta `server-side-fallback-2026-07-01`): if a safety classifier declines, the API reruns the request on a fallback model. Remove those two arguments in `rag/agent.py` to turn it off.
- **Prompt caching** – the conversation prefix is cached automatically, so the agent's repeated calls within a question cost less once the prompt is long enough to be cached.
- **Step limit** – `max_steps=12` model calls per question; after that it is told to answer with what it has.
