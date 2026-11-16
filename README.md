# Agentic RAG with Claude

A retrieval-augmented generation system in which **Claude drives retrieval itself**: it decides
what to search, splits compound questions into several searches, reformulates queries that miss,
reads surrounding chunks for context, and answers with inline citations like `[policy.md#2]`.

## How it works

```
question ─► Claude (claude-haiku-5-5) ──tool calls──► list_documents
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
| `rag/evaluate.py` | Retrieval metrics, and answer grading that compares the agent with the one-shot baseline per question category |
| `rag/baseline.py` | One-shot RAG baseline (retrieve once, answer once) for comparison |
| `evals/basic`, `evals/hard` | Eval sets: questions with evidence snippets, expected sources and reference answers, plus the hard set's own documents |
| `main.py` | CLI |
| `server/app.py` | FastAPI backend for the web UI: streaming chat over Server-Sent Events, conversations, chunk lookup for citations, settings, feedback |
| `server/documents.py`, `server/evals.py`, `server/feedback.py` | Documents API; eval runner (background jobs, cost estimate, run history); settings and feedback storage and export |
| `frontend/` | React + Vite + Tailwind chat UI |
| `rag/fake_llm.py` | Stand-in Claude client for tests and for running the UI with no API key (demo mode) |
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

`ask` / `chat` take `--model` (`claude-haiku-5-5` default, `claude-sonnet-5-5`, `claude-opus-5-5`),
`--effort` (`low` … `max`, default `medium`) and `-q` to hide the tool trace and cost line.

## Web UI

![Chat UI in dark mode: the agent runs three searches, answers with citations, and the cited chunk is open in the source panel](assets/screenshot.png)

Four screens, all in the sidebar. Works on mobile and in dark mode.

**Chat** - streaming answers; a collapsible trace of the agent's steps ("Searched 'carry over' → 3 results",
"Read benefits.md #3-5"); numbered citation chips that open the cited chunk with its section path and
neighbours in a side panel; estimated cost, API calls and time under each answer; 👍/👎 buttons (a 👎
asks what the answer should have been); a conversation list with **New chat** and **Reset** (deletes all
conversations, keeps feedback).

The **Documents** screen (sidebar, or `/#/documents`) manages the knowledge base without the CLI:
drag-and-drop upload of `.md` / `.txt` / `.pdf` (indexed automatically), a status per file
(indexed / changed on disk / not indexed / deleted), re-index or full rebuild with live progress,
delete, a viewer showing each document's chunks and the note shown to the model with them, and a
**search tester** that runs the agent's search tool directly (hybrid, keyword or semantic, with or
without reranking) so you can debug retrieval without spending tokens.

```bash
cd frontend && npm install && npm run build && cd ..   # once, and after frontend changes
python -m uvicorn server.app:app --port 8000           # then open http://localhost:8000
```

The **Evals** screen (`/#/evals`) runs the eval sets from the browser: the retrieval eval (free) or the
answer eval, which first shows an estimated cost (from your last run with the same model, or a rough
token estimate) and runs only after you confirm, with live progress, spend so far and a Stop button.
A chart plots accuracy (or retrieval recall) across past runs of the selected set, so you can see whether
a change helped; click a run for per-question verdicts, the judge's reasoning and the full answer.
**Feedback from chat** collects the 👍/👎 answers; **Export as eval set** writes them to `evals/feedback/`
(a 👍 answer becomes the reference answer; a 👎 needs its correction to become one). Review that file
before relying on it: a 👍 only means the answer looked right.

The **Settings** screen (`/#/settings`) sets the model, effort, results per search (or let the model
choose), reranking and the step limit. They apply to chat from the next question and to answer evals,
so an eval measures what you get in chat. The answer eval box also has its own Model and Effort
dropdowns, which start at your settings; changing them affects that eval run only, so you can compare
models without touching chat. Each run records the settings it used.

- **Demo mode:** `RAG_FAKE_LLM=1 python -m uvicorn server.app:app` runs the whole UI with real search but a
  fake model: no API key, no cost. Handy for UI work.
- **Frontend dev with hot reload:** run the server as above, then `cd frontend && npm run dev` and open
  http://localhost:5173 (API calls are proxied to :8000).
- **Scanned PDFs** (images of pages, no text layer) can't be read by the text extractor; they show as "No text found". Click **Transcribe** on the file (or **Transcribe with Claude** in the document viewer, which also works for PDFs whose text layer is junk): after a confirmation showing what's sent and the estimated cost, Claude Haiku transcribes the page images and the file is re-indexed. Each file is sent once (the transcription is cached in the index folder), and "Use the PDF's own text instead" undoes it. To transcribe text-less PDFs automatically - e.g. a large folder - start the server with `RAG_OCR=1` or run `python main.py ingest --ocr`.
- Other settings: `RAG_DOCS` (documents folder, default `./docs`), `RAG_INDEX` (index directory), `RAG_RERANK=0` (turn off the reranker), `RAG_DATA` (settings, feedback and eval-set indexes, default `./.rag_data`), `RAG_EVALS` (eval sets and results, default `./evals`).
- Conversations are kept in memory for now and are lost when the server restarts; settings and feedback are saved in `RAG_DATA`.
- In demo mode the eval dashboard works too, with a keyword-matching stand-in judge: free, but the scores mean nothing.

## Evaluation

Two eval sets ship with the repo, each a folder with `questions.jsonl`, `config.json` and its own documents:

| Set | What it tests |
|---|---|
| `evals/basic` (20 questions) | Single-fact lookups over the sample Acme docs. A smoke test - everything should score ~100%. |
| `evals/past-dates` (7 questions) | Held-out check written after the agent missed a past-date question: answers come from superseded or not-yet-amended rules. Used to verify the fix, not to tune it. |
| `evals/hard` (40 questions) | A fictional company (Halcyon Robotics, 19 docs) built so that **one search with the question text is not enough**: multi-step lookups (person → manager → office → cost tier → hotel cap), policies that were superseded, amended or overridden by a decision log, country exceptions, lists spread across sections, false premises, unanswerable questions, and simple control questions. |

```bash
python main.py eval --set evals/hard                         # retrieval metrics only - free
python main.py eval --set evals/hard --answers --systems agentic,naive --model claude-haiku-5-5
python main.py eval --set evals/hard --answers --systems agentic,naive              # Sonnet
python main.py eval --set evals/hard --answers --systems naive --naive-k 15        # more context, still one shot
```

`--systems agentic,naive` compares the agent with a **one-shot RAG baseline** (`rag/baseline.py`):
retrieve the top-k chunks for the question, answer in a single call. Both use the same retriever,
reranker, model, effort and answering rules, so the difference is the agent loop itself. The report
breaks accuracy down by question category and shows **evidence retrieved** - how much of the text
needed to answer each question the system actually saw.

### Results on `evals/hard` (measured, 40 questions)

| System | Correct | Multi-step (12) | Evidence seen | API calls | Time | Cost / question |
|---|---|---|---|---|---|---|
| Haiku 5.5, agentic | 95% | 100% | 99% | 2.9 | 7.2s | $0.0009 |
| Sonnet 5.5, one-shot k=5 | 70% | 33% | 84% | 1.0 | 3.3s | $0.0044 |
| Sonnet 5.5, one-shot k=15 | 78% | 42% | 90% | 1.0 | 7.8s | $0.0070 |
| Sonnet 5.5, agentic | 98% | 100% | 99% | 2.6 | 7.0s | $0.0138 |
| **With document notes:** | | | | | | |
| Haiku 5.5, one-shot k=5 | 72% | 33% | 84% | 1.0 | 3.2s | $0.0003 |
| Haiku 5.5, agentic | **100%** | 100% | 99% | 3.0 | 7.4s | $0.0010 |

The agent loop is worth more than the bigger model: Haiku with the loop beats Sonnet without it
by 25 points at a fifth of the cost. More context doesn't close the gap - the missing facts in
multi-step questions aren't in the top 15 results for the question text either. Simple, false-premise
and unanswerable questions score 100% for every system, so the agent costs accuracy nowhere.
One question (`tmp-london-jan`, a past date under a superseded policy) failed for every agentic run.
The cause was chunking, not the model: the agent retrieved each policy's hotel table, but the
sentence saying when each policy applied sat in a different chunk, so it saw two undated tables and
picked the newer one. Every chunk now carries its document's intro (status, effective dates, scope) -
see `Chunk.render()`. That took date-dependent questions from 83% to 100% for *both* the agent and the one-shot baseline (the bottom two rows), while multi-step stayed at 100% vs 33%. A search rule for past dates was also tried first (`agentic`, versus
`agentic-v1` without it): it improved evidence retrieval (83% → 95% on `evals/past-dates`) and
citations, but not accuracy (86% for both).

Every run is saved to `evals/results/` (retrieval runs too), which is what the dashboard's chart reads.
Answers are graded by a `claude-haiku-5-5` judge against reference answers (`--judge-model` to change),
and results are saved to `evals/results/`. Other flags: `--category multi_hop`, `--limit N`, `--workers N`, and `--repeats N` - run everything N times and report the mean, the min-max range across runs and the questions whose verdict changed. On these sets one run flips about one answer, so use `--repeats 3` before trusting a difference of a question or two.

Rough cost for the hard set (estimates): Haiku, both systems ≈ $0.15; Sonnet, both systems ≈ $1-2.

## Tests

```bash
pip install -r requirements-dev.txt && pytest -q   # 36 tests, no API key needed
```

## Settings worth knowing

- **Model** – `MODEL` in `rag/agent.py` or `--model`. Default `claude-haiku-5-5`: on `evals/hard` it scored 95% against Sonnet 5.5's 98% (one question apart) at about $0.001 per question against $0.014. Use Sonnet or Opus for the extra margin.
- **Effort** – `medium` by default; `high` helps on hard multi-hop questions.
- **Refusal fallback** – requests use `fallbacks="default"` (beta `server-side-fallback-2026-07-01`): if a safety classifier declines, the API reruns the request on a fallback model. Remove those two arguments in `rag/agent.py` to turn it off.
- **Prompt caching** – the conversation prefix is cached automatically, so the agent's repeated calls within a question cost less once the prompt is long enough to be cached.
- **Step limit** – `max_steps=12` model calls per question; after that it is told to answer with what it has.
