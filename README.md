# Agentic RAG with Claude

A retrieval-augmented generation system in which **Claude drives retrieval itself**: it decides
what to search, splits compound questions into several searches, reformulates queries that miss,
reads surrounding chunks for context, and answers with inline citations like `[policy.md#2]`.

## How it works

```
question ─► Claude (claude-sonnet-5-5) ──tool calls──► list_documents
               ▲                                     search_knowledge_base  (BM25 + vectors, RRF fusion)
               └────────── tool results ◄─────────── read_chunks            (neighbouring context)
            ... loops until it has enough evidence ...
            ─► cited answer
```

| File | Role |
|---|---|
| `rag/store.py` | Loads `.md` / `.txt` / `.pdf`, chunks by paragraph (~800 chars, 150 overlap), indexes into ChromaDB (local all-MiniLM-L6-v2 ONNX embeddings) + BM25, and does hybrid search with Reciprocal Rank Fusion |
| `rag/agent.py` | The agent loop: tool definitions (`strict: true`), parallel tool execution, multi-turn memory, step limit, refusal and `max_tokens` handling |
| `main.py` | CLI |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

The first ingest downloads the ~80 MB embedding model once. No other API keys are needed.

## Usage

```bash
python main.py ingest docs            # index a folder (re-run after changing documents)
python main.py search "carry over"    # test retrieval alone, no API call
python main.py ask "How many vacation days do I get after 6 years, and how many can I carry over?"
python main.py chat                   # multi-turn; /reset clears history, /quit exits
```

Add `-q` to `ask` / `chat` to hide the tool-call trace.

## Settings worth knowing

- **Model** – `MODEL` in `rag/agent.py` (default `claude-sonnet-5-5`, ~$0.02–0.05 per typical question). Use `claude-opus-5-5` for harder multi-hop questions or `claude-haiku-5-5` for the lowest cost.
- **Effort** – `RagAgent(effort="medium")`. Raise it to `"high"` for harder multi-hop questions, lower to `"low"` for speed.
- **Refusal fallback** – requests use `fallbacks="default"` (beta `server-side-fallback-2026-07-01`): if a safety classifier declines, the API reruns the request on a fallback model. Remove those two arguments in `rag/agent.py` to turn it off.
- **Prompt caching** – the conversation prefix is cached automatically, so the agent's repeated calls within one question cost less.
- **Step limit** – `max_steps=12` model calls per question; after that it is told to answer with what it has.
