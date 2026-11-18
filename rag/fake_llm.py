"""Stand-ins for the Anthropic client, for tests and for UI work without an API key.

`ScriptedClient` replays a fixed list of responses. `DemoClient` behaves like a tiny agent:
it searches the knowledge base with the user's question, then streams back the top hits
with citations. Enable it in the web app with RAG_FAKE_LLM=1. It does not call any API.
"""

from __future__ import annotations

import re
import time
from types import SimpleNamespace as NS
from typing import Any


def _usage(i: int = 0, o: int = 0) -> NS:
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def message(stop_reason: str, content: list, usage: NS | None = None) -> NS:
    return NS(stop_reason=stop_reason, content=content, usage=usage or _usage(100, 50))


def text_block(t: str) -> NS:
    return NS(type="text", text=t)


def tool_block(id_: str, name: str, **inp: Any) -> NS:
    return NS(type="tool_use", id=id_, name=name, input=inp)


class _Stream:
    """Mimics the SDK's MessageStream: iterate for events, then get_final_message()."""

    def __init__(self, msg: NS, delay: float = 0.0):
        self._msg, self._delay = msg, delay

    def __enter__(self) -> "_Stream":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def __iter__(self):
        for block in self._msg.content:
            if block.type == "text":
                for piece in re.findall(r"\S+\s*", block.text):
                    if self._delay:
                        time.sleep(self._delay)
                    yield NS(type="text", text=piece)

    def get_final_message(self) -> NS:
        return self._msg


class ScriptedClient:
    """Returns scripted responses in order and records each request's params."""

    def __init__(self, responses: list[NS]):
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kw: Any) -> _Stream:
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return _Stream(self.responses.pop(0))


class DemoClient:
    def __init__(self, delay: float = 0.03):
        self.delay = delay
        self.beta = NS(messages=NS(stream=self._stream, parse=self._parse))
        self.messages = NS(parse=self._parse)  # the eval judge (beta endpoint when it uses a refusal fallback)

    def _parse(self, **kw: Any) -> NS:
        """Demo judge: grades by how many of the reference's key terms appear in the answer."""
        from .evaluate import Verdict

        prompt = kw["messages"][0]["content"]
        ref = re.search(r"Reference answer: (.*)", prompt)
        ans = re.search(r"<answer>\n(.*)\n</answer>", prompt, flags=re.S)
        terms = {t.lower() for t in re.findall(r"[A-Za-z0-9$%.]{4,}", ref.group(1) if ref else "")}
        text = (ans.group(1) if ans else "").lower()
        share = sum(t in text for t in terms) / len(terms) if terms else 0.0
        verdict = "correct" if share >= 0.6 else "partially_correct" if share >= 0.3 else "incorrect"
        reasoning = f"Demo judge (no model call): {share:.0%} of the reference's key terms appear in the answer."
        return NS(parsed_output=Verdict(reasoning=reasoning, verdict=verdict), stop_reason="end_turn",
                  usage=_usage(0, 0))

    def _stream(self, **kw: Any) -> _Stream:
        last = kw["messages"][-1]
        if "tools" not in kw:  # one-shot baseline: the retrieved excerpts are in the prompt
            ctx = re.search(r"<context>\n(.*)\n</context>", last["content"], flags=re.S)
            msg = message("end_turn", [text_block(_compose(ctx.group(1) if ctx else ""))], _usage(0, 0))
        elif isinstance(last["content"], str):  # a new question -> search for it
            msg = message("tool_use", [tool_block(f"demo_{time.time_ns()}", "search_knowledge_base",
                                                  query=last["content"], k=3, source=None)], _usage(0, 0))
        else:  # tool results -> answer from them
            result = next((b["content"] for b in last["content"] if b.get("type") == "tool_result"), "")
            msg = message("end_turn", [text_block(_compose(result))], _usage(0, 0))
        time.sleep(self.delay * 10)
        return _Stream(msg, self.delay)


def _compose(result: str) -> str:
    hits = re.findall(r"^(\[[^\]]+#\d+\])[^\n]*\n(.+?)(?=\n\n\[|\Z)", result, flags=re.S | re.M)
    if not hits:
        return "I couldn't find anything about that in the knowledge base."
    lines = ["**Demo mode** - no model is called; these are the top search hits.\n"]
    for label, body in hits[:2]:
        body = re.sub(r"^\(About this document:.*?\)\n", "", body)
        sentence = " ".join(body.split())[:220]
        lines.append(f"- {sentence} {label}")
    return "\n".join(lines)



def demo_transcriber(path: Any) -> str:
    """Demo-mode stand-in for rag.ocr.claude_pdf_transcriber: no API call."""
    return (f"# Demo transcription of {path.name}\n\nDemo mode: no model was called. With an API key, Claude "
            "would transcribe the text of this PDF here.")


usage = _usage
