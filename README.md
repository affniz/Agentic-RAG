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
| `rag/store.py` | Loads `.md` / `.txt` / `.pdf`; heading-aware chunking (each chunk knows its Markdown section path); incremental per-chunk indexing (content-hashed chunk ids, so an edit only re-embeds the chunks it changed) into ChromaDB (local all-MiniLM-L6-v2 ONNX embeddings) + BM25; hybrid search with Reciprocal Rank Fusion; optional local cross-encoder reranking |
| `rag/agent.py` | The agent loop: strict tool schemas, parallel tool execution (one step's tool calls run concurrently), per-call timing, multi-turn memory, step limit, refusal / `max_tokens` handling, event callbacks, per-question cost tracking |
| `rag/evaluate.py` | Retrieval metrics, and answer grading that compares the agent with the one-shot baseline per question category |
| `rag/gdrive.py` | Google Drive connector: mirrors a Drive folder into `docs/gdrive/` (Google Docs as Markdown, Slides as text, PDFs / `.md` / `.txt` as they are), so the normal ingest indexes it |
| `rag/baseline.py` | One-shot RAG baseline (retrieve once, answer once) for comparison |
| `rag/models.py` | Model prices (incl. Haiku's long-prompt tier) and per-model request options such as the refusal fallback |
| `evals/basic`, `evals/hard`, `evals/holdout` | Eval sets: questions with evidence snippets, expected sources and reference answers, plus the hard set's own documents |
| `main.py` | CLI |
| `server/app.py` | FastAPI backend for the web UI: streaming chat over Server-Sent Events, conversations (saved to disk by `server/conversations.py`), chunk lookup for citations, settings, feedback; `server/security.py` limits it to localhost and `server/auth.py` adds an optional password login |
| `server/documents.py`, `server/evals.py`, `server/feedback.py` | Documents API; eval runner (background jobs, cost estimate, run history); settings and feedback storage and export |
| `frontend/` | React + Vite + Tailwind chat UI |
| `rag/fake_llm.py` | Stand-in Claude client for tests and for running the UI with no API key (demo mode) |
| `tests/` | pytest suite (no API key needed) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -c requirements.lock   # exact, tested versions (Python 3.14); drop -c for the latest
export ANTHROPIC_API_KEY=sk-ant-...
```

The first run downloads two small local models once: the embedding model (~80 MB) and the
reranker (`Xenova/ms-marco-MiniLM-L-6-v2`, from Hugging Face). If the reranker can't be
downloaded, search keeps working without it (or pass `--no-rerank`).

## Usage

```bash
python main.py ingest docs                 # sync the index; only new or edited chunks are re-embedded
python main.py ingest docs --rebuild       # re-embed everything
python main.py search "carry over"         # test retrieval alone, no API call (--mode bm25|vector|hybrid)
python main.py ask "How many vacation days do I get after 6 years, and how many can I carry over?"
python main.py chat                        # multi-turn; /reset, /cost, /quit
```

`ask` / `chat` take `--model` (`claude-haiku-5-5` default, `claude-sonnet-5-5`, `claude-opus-5-5`),
`--effort` (`low` … `max`, default `medium`) and `-q` to hide the tool trace and cost line.

## Google Drive

Documents can come from a Google Drive folder (and its subfolders). The connector mirrors the folder into
`docs/gdrive/`, then the normal ingest indexes it, so only new or changed files are downloaded and re-embedded.

- **Google Docs** are exported as Markdown (headings become section paths), **Google Slides** as plain text,
  and **PDFs, `.md` and `.txt`** files are downloaded as they are. Sheets, images, Office files and shortcuts
  are skipped and listed in the sync report. Files over 50 MB are skipped.
- Deleting, trashing or moving a file out of the folder in Drive removes it locally on the next sync; a
  rename replaces the old copy. Only files the connector downloaded are ever deleted (tracked in
  `docs/gdrive/.gdrive.json`). A file that fails to download keeps its previous copy and is retried next time.
- Access is read-only (`drive.readonly` scope).

**Credentials.** Either a service account (best for the server and Docker):

1. In Google Cloud, enable the Google Drive API and create a service account with a JSON key.
2. Share the Drive folder with the service account's `client_email` (Viewer is enough).

or your own account, through Application Default Credentials:

```bash
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/drive.readonly,https://www.googleapis.com/auth/cloud-platform
```

**Sync from the CLI** (the folder is its id or its URL from the browser):

```bash
python main.py gdrive "https://drive.google.com/drive/folders/1AbC..." --credentials key.json   # sync, then ingest docs/
```

`--no-ingest` only downloads; `--dir` changes the subfolder of `docs/`.

**Sync from the web UI:** start the server with `RAG_GDRIVE_FOLDER` (id or URL) and, for a service account,
`RAG_GDRIVE_CREDENTIALS=key.json`. The Documents screen then has a **Sync Google Drive** button that downloads
the changes, indexes them and shows what was downloaded, removed or skipped. Synced files appear in the file
list under `gdrive/`. `RAG_GDRIVE_DIR` changes the subfolder (it must be inside `RAG_DOCS`).

## Web UI

![Chat UI in dark mode: the agent runs three searches, answers with citations, and the cited chunk is open in the source panel](assets/screenshot.png)

Four screens, all in the sidebar. Works on mobile and in dark mode.

**Chat** - streaming answers; a collapsible trace of the agent's steps ("Searched 'carry over' → 3 results",
with how long each model call and tool took, how much of a model call was spent waiting for the API to start answering,
and any automatic retries after rate limits or overloads,
"Read benefits.md #3-5"); numbered citation chips that open the cited chunk with its section path and
neighbours in a side panel; estimated cost, API calls and time under each answer; 👍/👎 buttons (a 👎
asks what the answer should have been); a conversation list with **New chat** and **Reset** (deletes all
conversations, keeps feedback).

The **Documents** screen (sidebar, or `/#/documents`) manages the knowledge base without the CLI:
drag-and-drop upload of `.md` / `.txt` / `.pdf` (indexed automatically; uploading a file whose name already exists asks before replacing it, and files added or deleted while indexing runs are picked up by one more pass), a status per file
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
- **Scanned PDFs** (images of pages, no text layer) can't be read by the text extractor; they show as "No text found". Click **Transcribe** on the file (or **Transcribe with Claude** in the document viewer, which also works for PDFs whose text layer is junk): after a confirmation showing what's sent and the estimated cost, Claude Haiku transcribes the page images and the file is re-indexed. Each file is sent once (the transcription is cached in the index folder; deleting the file, or replacing it with a new version, deletes its transcription), and "Use the PDF's own text instead" undoes it. To transcribe text-less PDFs automatically - e.g. a large folder - start the server with `RAG_OCR=1` or run `python main.py ingest --ocr`.
- **Encrypted PDFs** that open without a password (common for official documents) are read normally. A PDF that needs a password, or any file that can't be read, is skipped with the reason shown in the Documents table; the rest of the folder is still indexed.
- Other settings: `RAG_DOCS` (documents folder, default `./docs`), `RAG_INDEX` (index directory), `RAG_RERANK=0` (turn off the reranker), `RAG_DATA` (settings, feedback and eval-set indexes, default `./.rag_data`), `RAG_EVALS` (eval sets and results, default `./evals`), `RAG_GDRIVE_FOLDER` / `RAG_GDRIVE_CREDENTIALS` / `RAG_GDRIVE_DIR` (see *Google Drive*).
- Conversations, settings and feedback are saved in `RAG_DATA` (`conversations/<id>.json`), so chats survive a restart. Run a single uvicorn worker per `RAG_DATA` folder, or the processes overwrite each other's files.
- **Local only:** the server answers only requests addressed to `localhost` / `127.0.0.1` / `[::1]` / `*.localhost`, and refuses writes from other sites (cross-site `Origin` or `Sec-Fetch-Site`), so a web page you visit can't upload, delete or start paid evals. This protects against browsers, not the network: a client that can reach the port can send any `Host` header. Without a password there is no login, so keep the port on loopback (`uvicorn` binds to 127.0.0.1 by default; for Docker see below). `RAG_ALLOWED_HOSTS` adds hostnames (comma-separated). See `server/security.py`.
- **Login:** set `RAG_PASSWORD` to require a password: the UI shows a sign-in screen, every API route except the login itself answers 401 without a session, and the sidebar gets **Log out**. With a password set, any hostname is accepted (so you can open it from another device), and writes from other sites are still refused. Sessions are signed cookies (HttpOnly, SameSite=Strict, 30 days) keyed by a secret in `RAG_DATA/session_key`, so they survive restarts. Changing the password, or deleting that file, logs every browser out. After 10 wrong passwords from one address, it must wait 15 minutes. Over plain HTTP the password crosses the network unencrypted, so for anything beyond your own LAN put a TLS proxy in front and run uvicorn with `--proxy-headers` (the cookie is then marked Secure). See `server/auth.py`.
- In demo mode the eval dashboard works too, with a keyword-matching stand-in judge: free, but the scores mean nothing.

## Docker

One image serves the web UI and the API. The embedding model (and, by default, the reranker) are
baked in, so it starts quickly and indexes without network access.

```bash
docker build -t agentic-rag .
docker run -p 127.0.0.1:8000:8000 -e ANTHROPIC_API_KEY -v rag-data:/data agentic-rag   # http://localhost:8000
```

- **Keep the `127.0.0.1:` in `-p`.** Inside the container the server listens on all interfaces, so a plain `-p 8000:8000`
  publishes it on every network interface of your machine. The Host check (see *Local only*) does not stop that:
  any device on your network can send `Host: localhost` and use the API with your key - chat, paid evals, uploads
  and deletes. Publish on loopback only unless you set a password (below) or put an authenticating proxy in front.
- **Password login:** `docker run -p 8000:8000 -e RAG_PASSWORD='a long random passphrase' -e ANTHROPIC_API_KEY -v rag-data:/data agentic-rag`
  can be published on your network: everything but the sign-in screen needs the password (see *Login* above).
- **Demo mode** (no API key, real search, fake model): `docker run -p 127.0.0.1:8000:8000 -e RAG_FAKE_LLM=1 agentic-rag`.
- **Your documents:** mount them over the samples with `-v /path/to/docs:/app/docs`, then click **Re-index** on the Documents screen.
- **Google Drive:** `-e RAG_GDRIVE_FOLDER=<folder id or URL> -e RAG_GDRIVE_CREDENTIALS=/secrets/key.json -v /path/to/key.json:/secrets/key.json:ro`,
  then click **Sync Google Drive** (see *Google Drive* above). Mount a `docs` volume too if the synced files should survive a container rebuild.
- **What persists:** the index, settings, feedback and chat conversations live in the `/data` volume. Eval results are written inside the container.
- **Build without the reranker** (e.g. where Hugging Face is unreachable): `--build-arg BAKE_RERANKER=0`; it is then downloaded on first use, and search works without it if that fails.
- **Other hostnames:** without a password the server only answers requests addressed to localhost (see *Local only* above). To open it under another name, e.g. on a home server, set `RAG_PASSWORD` (any hostname then works), or add `-e RAG_ALLOWED_HOSTS=myhost` - without a password anyone who can reach the port can use your API key.
- `scripts/docker_smoke_test.sh [image]` checks a built image end to end in demo mode, then again with `RAG_PASSWORD` set (health check, API locked until login).

## CI

`.github/workflows/ci.yml` runs on every pull request and push to `main`:

1. **Python tests:** `pytest`; no API key needed.
2. **Frontend:** type-check and production build.
3. **Docker image:** builds the image and runs the smoke test. On pushes to `main` it also publishes
   `ghcr.io/<owner>/<repo>:latest` and `:sha-<commit>` to the GitHub Container Registry, for both
   `linux/amd64` and `linux/arm64` (Apple Silicon, ARM servers); the arm64 image is built under QEMU.

## Evaluation

Four eval sets ship with the repo, each a folder with `questions.jsonl`, `config.json` and its own documents:

| Set | What it tests |
|---|---|
| `evals/basic` (20 questions) | Single-fact lookups over the sample Acme docs. A smoke test - everything should score ~100%. |
| `evals/past-dates` (7 questions) | Held-out check written after the agent missed a past-date question: answers come from superseded or not-yet-amended rules. Used to verify the fix, not to tune it. |
| `evals/holdout` (20 questions) | A second fictional company (Quillfeather Logistics, 8 docs) with the hard set's kinds of questions: multi-step lookups, bulletins that override policies, superseded rates, recalls, aggregation, a false premise and an unanswerable question. Written after the agent was tuned on `evals/hard` and **never used for tuning** - run it to check that a gain on the hard set generalises. |
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
Answers are graded by a `claude-sonnet-5-5` judge against reference answers (`--judge-model` to change), so the
default Haiku agent isn't graded by itself; a run whose judge is the model being evaluated prints a warning and is
flagged in the dashboard. The results table above was graded by a Haiku judge, so compare new runs with each other,
not with it. Other flags: `--category multi_hop`, `--limit N`, `--workers N`, and `--repeats N` - run everything N times and report the mean, the min-max range across runs and the questions whose verdict changed. On these sets one run flips about one answer, so use `--repeats 3` before trusting a difference of a question or two.
A question that fails with an API error (a rate limit, an overload) is saved with the verdict `error` and left out of the
accuracy numbers, which report how many there were; an invalid API key stops the run at once.

Rough cost for the hard set (estimates): Haiku, both systems ≈ $0.15; Sonnet, both systems ≈ $1-2.

## Tests

```bash
pip install -r requirements-dev.txt && pytest -q   # no API key needed
```

## Settings worth knowing

- **Model** – `MODEL` in `rag/agent.py` or `--model`. Default `claude-haiku-5-5`: on `evals/hard` it scored 95% against Sonnet 5.5's 98% (one question apart) at about $0.001 per question against $0.014. Use Sonnet or Opus for the extra margin.
- **Effort** – `medium` by default; `high` helps on hard multi-hop questions.
- **Refusal fallback** – Sonnet 5.5 and Opus 5.5 requests use `fallbacks="default"` (beta `server-side-fallback-2026-07-01`): if a safety classifier declines, the API reruns the request on a fallback model. Haiku 5.5 has no server-side fallback, so its requests don't send it and a refusal is shown as such. See `request_options` in `rag/models.py`.
- **Long chats** – the agent uses server-side context editing (beta `context-management-2025-06-27`): once a request's prompt passes ~30K tokens, the API clears the oldest search results and keeps the 6 most recent tool calls, so follow-up questions don't resend every earlier search. The local history stays append-only. Tune `CONTEXT_MANAGEMENT` in `rag/agent.py`.
- **Prices** – list prices in `rag/models.py` (checked on the date in `PRICES_CHECKED`); each API call is priced separately, so Haiku calls with prompts over 100K tokens use the higher rate.
- **Prompt caching** – the conversation prefix is cached automatically, so the agent's repeated calls within a question cost less once the prompt is long enough to be cached.
- **Step limit** – `max_steps=12` model calls per question; after that it is told to answer with what it has.
- **Cost limit** – in chat, once one question has cost $0.50 (Settings screen; `--max-cost` for `ask` / `chat`, off by
  default there) the agent stops searching and answers with what it has. The answer is marked as stopped early.
- **Citations** – a citation chip turns amber when the model cited a chunk it was never shown, and answer evals
  report the share of answers that did (`cited unseen chunks`).
