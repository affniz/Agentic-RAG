"""The agent: Claude decides when and how to search, reads more context, and cites sources."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from .store import KnowledgeBase

MODEL = "claude-sonnet-5-5"

# USD per million tokens: (input, output, cache read, cache write). List prices; used for
# the per-question cost estimate only. If a refusal fallback serves a turn, the real price
# is that of the fallback model.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}

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
            "Hybrid keyword + semantic search over the knowledge base, reranked for relevance. "
            "Returns the best-matching chunks, each labelled with its citation id [source#chunk] "
            "and section heading. Use focused queries; run several searches for multi-part questions."
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
        return "\n\n".join(f"{c.label()}\n{c.text}" for c, _ in hits)

    def read_chunks(source: str, start: int, end: int) -> str:
        end = min(end, start + 7)
        chunks = kb.get_chunks(source, start, end)
        if not chunks:
            return f"No chunks {start}-{end} found for '{source}'."
        return "\n\n".join(f"{c.label()}\n{c.text}" for c in chunks)

    return {
        "list_documents": list_documents,
        "search_knowledge_base": search_knowledge_base,
        "read_chunks": read_chunks,
    }


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    api_calls: int = 0

    def add(self, u: Any) -> None:
        self.input_tokens += u.input_tokens or 0
        self.output_tokens += u.output_tokens or 0
        self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.api_calls += 1

    def cost(self, model: str) -> float | None:
        p = PRICES.get(model)
        if p is None:
            return None
        return (
            self.input_tokens * p[0]
            + self.output_tokens * p[1]
            + self.cache_read_tokens * p[2]
            + self.cache_write_tokens * p[3]
        ) / 1_000_000


@dataclass
class Answer:
    text: str
    usage: Usage
    cost_usd: float | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop: str = "end_turn"  # end_turn | refusal | max_tokens | step_limit


Event = dict[str, Any]


class RagAgent:
    """Runs the tool loop. Pass `on_event` to observe progress (used by the CLI and the future UI).

    Events: {"type": "tool_call", "name", "args"}, {"type": "tool_result", "name", "is_error", "chars"},
            {"type": "answer", "text"}, {"type": "usage", "usage", "cost_usd"}
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        model: str = MODEL,
        effort: str = "medium",
        max_steps: int = 12,
        on_event: Callable[[Event], None] | None = None,
        client: Any = None,
    ):
        self.client = client or anthropic.Anthropic()
        self.handlers = build_handlers(kb)
        self.model = model
        self.effort = effort
        self.max_steps = max_steps
        self.on_event = on_event or (lambda e: None)
        self.messages: list[dict[str, Any]] = []  # full history, kept across questions
        self.total = Usage()

    def _create(self, **extra: Any) -> Any:
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=self.messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # cache the growing conversation prefix
            # If a safety classifier declines, the API retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **extra,
        )

    def _run_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        handler = self.handlers.get(name)
        if handler is None:
            return f"Unknown tool: {name}", True
        try:
            return handler(**args), False
        except Exception as e:  # report tool failures to the model instead of crashing
            return f"Tool error: {type(e).__name__}: {e}", True

    def ask(self, question: str) -> Answer:
        self.messages.append({"role": "user", "content": question})
        usage = Usage()
        calls: list[dict[str, Any]] = []

        def finish(text: str, stop: str) -> Answer:
            self.total.input_tokens += usage.input_tokens
            self.total.output_tokens += usage.output_tokens
            self.total.cache_read_tokens += usage.cache_read_tokens
            self.total.cache_write_tokens += usage.cache_write_tokens
            self.total.api_calls += usage.api_calls
            ans = Answer(text, usage, usage.cost(self.model), calls, stop)
            self.on_event({"type": "answer", "text": text})
            self.on_event({"type": "usage", "usage": usage, "cost_usd": ans.cost_usd})
            return ans

        for _ in range(self.max_steps):
            response = self._create()
            usage.add(response.usage)
            # Append the full content (thinking + tool_use blocks), never just the text.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                return finish("(The model declined to answer this request.)", "refusal")
            if response.stop_reason == "max_tokens":
                return finish(_text(response) + "\n\n(Answer truncated: hit max_tokens.)", "max_tokens")
            if response.stop_reason != "tool_use":
                return finish(_text(response) or "(no text in response)", "end_turn")

            # Run every tool call from this turn; return all results in ONE user message.
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                args = block.input if isinstance(block.input, dict) else json.loads(block.input)
                calls.append({"name": block.name, "args": args})
                self.on_event({"type": "tool_call", "name": block.name, "args": args})
                output, is_error = self._run_tool(block.name, args)
                self.on_event({"type": "tool_result", "name": block.name, "is_error": is_error, "chars": len(output)})
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error}
                )
            self.messages.append({"role": "user", "content": results})

        # Out of steps: ask for a final answer with what it has (no more tools).
        self.messages.append({"role": "user", "content": "Stop searching and answer now with the evidence you have."})
        response = self._create(tool_choice={"type": "none"})
        usage.add(response.usage)
        self.messages.append({"role": "assistant", "content": response.content})
        return finish(_text(response) or "(no text in response)", "step_limit")

    def reset(self) -> None:
        self.messages = []


def _text(response: Any) -> str:
    return "".join(b.text for b in response.content if b.type == "text").strip()
