"""The agent: Claude decides when and how to search, reads more context, and cites sources."""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from .models import PRICES, call_cost, request_options
from .store import KnowledgeBase

_ = PRICES  # re-exported: other modules import PRICES from here

# Haiku 5.5 matched Sonnet 5.5 within one question on evals/hard (95% vs 98%) at ~1/16 the cost.
MODEL = "claude-haiku-5-5"

# Server-side context editing: once a request's prompt passes `trigger`, the API clears the
# oldest tool results (search output) and keeps the most recent `keep` tool uses. This stops a
# long chat from resending every earlier question's search results. It runs on the server, so
# our own `messages` stay append-only, which the API requires for replaying thinking blocks.
CONTEXT_EDITING_BETA = "context-management-2025-06-27"
CONTEXT_MANAGEMENT: dict[str, Any] = {
    "edits": [{
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": 30_000},
        "keep": {"type": "tool_uses", "value": 6},
        # Each clear resets the prompt cache, so only clear when it saves a worthwhile amount.
        "clear_at_least": {"type": "input_tokens", "value": 10_000},
    }]
}

CITATION_RE = re.compile(r"\[([^\[\]#\n]+)#(\d+)\]")

# Rules for answering from retrieved text. Shared with the non-agentic baseline
# (rag/baseline.py) so evals compare retrieval strategies, not prompts.
ANSWER_RULES = """\
- Answer only from the retrieved text. If the knowledge base doesn't cover something, say so \
plainly rather than guessing. If the question rests on a false premise, say so.
- Documents can be superseded, amended or overridden (newer policy versions, amendments, \
decision logs, country or team exceptions). Use the rule in force on the date the question \
is about, and mention when an exception or newer rule changes the answer.
- Cite every factual claim inline as [source#chunk], e.g. [policy.md#2].
- Keep answers concise and direct."""

AGENT_INSTRUCTIONS_V1 = """\
You are a research assistant that answers questions using a private knowledge base.
You have tools to list the documents, search them, and read chunks in full.

How to work:
- Search before answering anything that depends on the knowledge base. Break compound \
questions into separate searches (in parallel when they are independent), and rephrase \
or use synonyms when a search misses.
- When an answer depends on another fact (a person's office, a team's director, a date's \
rotation), look that fact up too instead of assuming it.
- When a hit looks relevant but cut off, or a list may continue, read the neighbouring chunks.
- Stop searching once you have enough evidence; don't repeat searches that already worked."""

# v2: when a question is about a specific (especially past) date, find the version of the rule
# in force then. v1 tended to stop at the current policy. Compare with --systems agentic-v1,agentic.
AGENT_INSTRUCTIONS = AGENT_INSTRUCTIONS_V1 + """
- When a question is about a specific date, especially a past one, find out which version of \
a rule was in force on that date: look for earlier versions, effective dates, amendments and \
decisions, rather than stopping at the current version."""


def system_prompt(instructions: str, today: str) -> str:
    return f"{instructions}\n\nWhen answering:\n{ANSWER_RULES}\n\nToday's date: {today}."


SYSTEM_PROMPT = system_prompt(AGENT_INSTRUCTIONS, "YYYY-MM-DD")  # for reference; see RagAgent

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


@dataclass
class SearchOptions:
    """Overrides for the search tool, set from the UI. None = leave the choice to the model / the KB default."""

    k: int | None = None  # results per search; replaces the k the model asks for
    rerank: bool | None = None


def build_handlers(kb: KnowledgeBase, seen: list[str] | None = None,
                   options: SearchOptions | None = None) -> dict[str, Callable[..., str]]:
    """Tool implementations. Every chunk returned to the model is recorded in `seen` as 'source#index'."""
    seen = seen if seen is not None else []
    options = options or SearchOptions()

    def note(chunks: Any) -> None:
        seen.extend(f"{c.source}#{c.index}" for c in chunks)

    def list_documents() -> str:
        src = kb.sources()
        if not src:
            return "The knowledge base is empty. Run `python main.py ingest` first."
        return "\n".join(f"{name} ({n} chunks)" for name, n in sorted(src.items()))

    def search_knowledge_base(query: str, k: int = 5, source: str | None = None) -> str:
        k = options.k if options.k is not None else k
        hits = kb.search(query, k=max(1, min(int(k), 10)), source=source or None, rerank=options.rerank)
        if not hits:
            return "No results."
        note(c for c, _ in hits)
        return "\n\n".join(c.render() for c, _ in hits)

    def read_chunks(source: str, start: int, end: int) -> str:
        end = min(end, start + 7)
        chunks = kb.get_chunks(source, start, end)
        if not chunks:
            return f"No chunks {start}-{end} found for '{source}'."
        note(chunks)
        return "\n\n".join(c.render() for c in chunks)

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
    cost_usd: float | None = 0.0  # summed per call, since a call's price can depend on its prompt size

    def add(self, u: Any, model: str) -> None:
        """Count one API call's usage and price it at `model`'s rates (None if unknown)."""
        i, o = u.input_tokens or 0, u.output_tokens or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        self.input_tokens += i
        self.output_tokens += o
        self.cache_read_tokens += cr
        self.cache_write_tokens += cw
        self.api_calls += 1
        c = call_cost(model, i, o, cr, cw)
        self.cost_usd = None if c is None or self.cost_usd is None else self.cost_usd + c

    def merge(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.api_calls += other.api_calls
        self.cost_usd = None if other.cost_usd is None or self.cost_usd is None else self.cost_usd + other.cost_usd

    def cost(self, model: str | None = None) -> float | None:
        """Total cost in USD (list prices). `model` is accepted for compatibility and ignored."""
        return self.cost_usd


@dataclass
class Answer:
    text: str
    usage: Usage
    cost_usd: float | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop: str = "end_turn"  # end_turn | refusal | max_tokens | step_limit
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)  # chunks shown to the model, 'source#index'


Event = dict[str, Any]


class RagAgent:
    """Runs the tool loop. Pass `on_event` to observe progress (used by the CLI and the future UI).

    Events: {"type": "text_delta", "text"}, {"type": "tool_call", "id", "name", "args"},
            {"type": "tool_result", "id", "name", "is_error", "chars", "results"},
            {"type": "answer", "text", "stop", "citations"}, {"type": "usage", "usage", "cost_usd"}
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        model: str = MODEL,
        effort: str = "medium",
        max_steps: int = 12,
        on_event: Callable[[Event], None] | None = None,
        client: Any = None,
        today: str | None = None,
        instructions: str = AGENT_INSTRUCTIONS,
        search: SearchOptions | None = None,
    ):
        self.client = client or anthropic.Anthropic()
        self.kb = kb
        self._seen: list[str] = []
        self.search = search or SearchOptions()  # mutable: changes apply to the next search
        self.handlers = build_handlers(kb, self._seen, self.search)
        self.system = system_prompt(instructions, today or datetime.date.today().isoformat())
        self.model = model
        self.effort = effort
        self.max_steps = max_steps
        self.on_event = on_event or (lambda e: None)
        self.messages: list[dict[str, Any]] = []  # full history, kept across questions
        self.total = Usage()

    def _create(self, **extra: Any) -> Any:
        """One model call, streamed: text deltas go out as events, the full message is returned."""
        with self.client.beta.messages.stream(**self._params(**extra)) as stream:
            for event in stream:
                if event.type == "text":
                    self.on_event({"type": "text_delta", "text": event.text})
            return stream.get_final_message()

    def _params(self, **extra: Any) -> dict[str, Any]:
        return dict(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            tools=TOOLS,
            messages=self.messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # cache the growing conversation prefix
            context_management=CONTEXT_MANAGEMENT,
            # Context editing beta, plus a refusal fallback on models that have one.
            **request_options(self.model, [CONTEXT_EDITING_BETA]),
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
        self._seen.clear()

        def finish(text: str, stop: str) -> Answer:
            self.total.merge(usage)
            ans = Answer(text, usage, usage.cost_usd, calls, stop, self._citations(text),
                         list(dict.fromkeys(self._seen)))
            self.on_event({"type": "answer", "text": text, "stop": stop, "citations": ans.citations})
            self.on_event({"type": "usage", "usage": usage, "cost_usd": ans.cost_usd})
            return ans

        for _ in range(self.max_steps):
            response = self._create()
            usage.add(response.usage, _served_by(response, self.model))
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
                self.on_event({"type": "tool_call", "id": block.id, "name": block.name, "args": args})
                seen_before = len(self._seen)
                output, is_error = self._run_tool(block.name, args)
                self.on_event({"type": "tool_result", "id": block.id, "name": block.name, "is_error": is_error,
                               "chars": len(output), "results": len(self._seen) - seen_before})
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error}
                )
            self.messages.append({"role": "user", "content": results})

        # Out of steps: ask for a final answer with what it has (no more tools).
        self.messages.append({"role": "user", "content": "Stop searching and answer now with the evidence you have."})
        response = self._create(tool_choice={"type": "none"})
        usage.add(response.usage, _served_by(response, self.model))
        self.messages.append({"role": "assistant", "content": response.content})
        return finish(_text(response) or "(no text in response)", "step_limit")

    def _citations(self, text: str) -> list[dict[str, Any]]:
        """Resolve [source#n] markers in the answer to the chunks they point at."""
        out, seen = [], set()
        for source, idx in CITATION_RE.findall(text):
            key = (source, int(idx))
            if key in seen:
                continue
            seen.add(key)
            chunk = next(iter(self.kb.get_chunks(source, key[1], key[1])), None)
            out.append({"source": source, "index": key[1], "section": chunk.section if chunk else None,
                        "found": chunk is not None})
        return out

    def reset(self) -> None:
        self.messages = []


def _text(response: Any) -> str:
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _served_by(response: Any, requested: str) -> str:
    """The model that produced a response: the fallback model if a refusal fallback served it."""
    model = getattr(response, "model", None)
    return model if isinstance(model, str) and model else requested
