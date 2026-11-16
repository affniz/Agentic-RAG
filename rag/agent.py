"""The agent: Claude decides when and how to search, reads more context, and cites sources."""

from __future__ import annotations

import json
from typing import Any, Callable

import anthropic

from .store import KnowledgeBase

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You are a research assistant that answers questions using a private knowledge base.
You have tools to list the documents, search them, and read chunks in full.

How to work:
- Search before answering anything that depends on the knowledge base. Break compound \
questions into separate searches (in parallel when they are independent), and rephrase \
or use synonyms when a search misses.
- When a hit looks relevant but cut off, read the neighbouring chunks of that document.
- Stop searching once you have enough evidence; don't repeat searches that already worked.
- Answer only from what you retrieved. If the knowledge base doesn't cover something, \
say so plainly rather than guessing.
- Cite every factual claim inline as [source#chunk], e.g. [policy.md#2].
- Keep answers concise and direct."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_documents",
        "description": "List every document in the knowledge base with its number of chunks.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Hybrid keyword + semantic search over the knowledge base. Returns the best-matching "
            "chunks, each labelled with its citation id [source#chunk]. Use focused queries; run "
            "several searches for multi-part questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for (natural language or keywords)."},
                "k": {"type": "integer", "description": "Number of chunks to return, 1-10. Use 5 if unsure."},
                "source": {
                    "type": ["string", "null"],
                    "description": "Restrict to one document name from list_documents, or null for all.",
                },
            },
            "required": ["query", "k", "source"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "read_chunks",
        "description": (
            "Read a contiguous range of chunks from one document, e.g. to see the context around "
            "a search hit. At most 8 chunks per call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Document name exactly as shown in citations."},
                "start": {"type": "integer", "description": "First chunk index (inclusive)."},
                "end": {"type": "integer", "description": "Last chunk index (inclusive)."},
            },
            "required": ["source", "start", "end"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def build_handlers(kb: KnowledgeBase) -> dict[str, Callable[..., str]]:
    def list_documents() -> str:
        src = kb.sources()
        if not src:
            return "The knowledge base is empty. Run `python main.py ingest` first."
        return "\n".join(f"{name} ({n} chunks)" for name, n in sorted(src.items()))

    def search_knowledge_base(query: str, k: int = 5, source: str | None = None) -> str:
        hits = kb.search(query, k=max(1, min(int(k), 10)), source=source or None)
        if not hits:
            return "No results."
        return "\n\n".join(f"[{c.source}#{c.index}] (score {s:.3f})\n{c.text}" for c, s in hits)

    def read_chunks(source: str, start: int, end: int) -> str:
        end = min(end, start + 7)
        chunks = kb.get_chunks(source, start, end)
        if not chunks:
            return f"No chunks {start}-{end} found for '{source}'."
        return "\n\n".join(f"[{c.source}#{c.index}]\n{c.text}" for c in chunks)

    return {
        "list_documents": list_documents,
        "search_knowledge_base": search_knowledge_base,
        "read_chunks": read_chunks,
    }


class RagAgent:
    def __init__(self, kb: KnowledgeBase, verbose: bool = True, max_steps: int = 12, effort: str = "medium"):
        self.client = anthropic.Anthropic()
        self.handlers = build_handlers(kb)
        self.verbose = verbose
        self.max_steps = max_steps
        self.effort = effort
        self.messages: list[dict[str, Any]] = []  # full history, kept across questions

    def _run_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        handler = self.handlers.get(name)
        if handler is None:
            return f"Unknown tool: {name}", True
        try:
            return handler(**args), False
        except Exception as e:  # report tool failures to the model instead of crashing
            return f"Tool error: {type(e).__name__}: {e}", True

    def ask(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})

        for _ in range(self.max_steps):
            response = self.client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.messages,
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},  # cache the growing conversation prefix
                # If a safety classifier declines, the API retries on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            # Append the full content (thinking + tool_use blocks), never just the text.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                return "(The model declined to answer this request.)"
            if response.stop_reason == "max_tokens":
                return _text(response) + "\n\n(Answer truncated: hit max_tokens.)"
            if response.stop_reason != "tool_use":
                return _text(response) or "(no text in response)"

            # Run every tool call from this turn; return all results in ONE user message.
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                args = block.input if isinstance(block.input, dict) else json.loads(block.input)
                if self.verbose:
                    shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    print(f"  -> {block.name}({shown})")
                output, is_error = self._run_tool(block.name, args)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error}
                )
            self.messages.append({"role": "user", "content": results})

        # Out of steps: ask for a final answer with what it has (no more tools).
        self.messages.append(
            {"role": "user", "content": "Stop searching and answer now with the evidence you have."}
        )
        response = self.client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "none"},
            messages=self.messages,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.messages.append({"role": "assistant", "content": response.content})
        return _text(response) or "(no text in response)"

    def reset(self) -> None:
        self.messages = []


def _text(response: Any) -> str:
    return "".join(b.text for b in response.content if b.type == "text").strip()
