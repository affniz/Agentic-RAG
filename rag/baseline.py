"""Non-agentic baseline: retrieve once with the question, then answer in a single call.

This is "classic" RAG. It uses the same retriever (hybrid search + reranking), model and
answering rules as the agent, so comparing the two isolates what the agent loop adds:
deciding what to search for, following up on intermediate facts, and reading more context.
"""

from __future__ import annotations

import datetime
from typing import Any

import anthropic

from .agent import ANSWER_RULES, MODEL, Answer, Usage, system_prompt
from .store import KnowledgeBase

BASELINE_INSTRUCTIONS = """\
You are a research assistant that answers questions using excerpts retrieved from a private \
knowledge base. The excerpts are in <context>, each labelled with its citation id."""


class NaiveRag:
    def __init__(
        self,
        kb: KnowledgeBase,
        model: str = MODEL,
        effort: str = "medium",
        k: int = 5,
        client: Any = None,
        today: str | None = None,
    ):
        self.kb = kb
        self.model = model
        self.effort = effort
        self.k = k
        self.client = client or anthropic.Anthropic()
        self.system = system_prompt(BASELINE_INSTRUCTIONS, today or datetime.date.today().isoformat())

    def ask(self, question: str) -> Answer:
        hits = self.kb.search(question, k=self.k)
        context = "\n\n".join(f"{c.label()}\n{c.text}" for c, _ in hits) or "(no results)"
        messages = [{"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"}]
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            messages=messages,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            response = stream.get_final_message()

        usage = Usage()
        usage.add(response.usage)
        if response.stop_reason == "refusal":
            text, stop = "(The model declined to answer this request.)", "refusal"
        else:
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            stop = "max_tokens" if response.stop_reason == "max_tokens" else "end_turn"
        return Answer(
            text=text,
            usage=usage,
            cost_usd=usage.cost(self.model),
            tool_calls=[],
            stop=stop,
            retrieved=[f"{c.source}#{c.index}" for c, _ in hits],
        )
