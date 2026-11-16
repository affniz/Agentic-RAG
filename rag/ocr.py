"""Text extraction for scanned PDFs (no text layer) using Claude's PDF understanding.

Opt-in: it sends the whole PDF to the Anthropic API. Enable with `ingest --ocr` or RAG_OCR=1.
The store caches each transcription by file hash, so a file is only sent once.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Callable

import anthropic

OCR_MODEL = "claude-haiku-5-5"
MAX_PDF_BYTES = 32 * 1024 * 1024  # API request limit for inline PDFs

PROMPT = """\
Transcribe all text in this document as Markdown, in reading order.
- Start with a single "# " heading naming what the document is (e.g. "# Invoice - Acme Ltd, \
March 2026"), using only information in the document.
- Keep tables as Markdown tables, with every row and column.
- Keep names, numbers, dates and codes exactly as written. Mark anything illegible as [illegible].
Output only the transcription, with no commentary."""


def claude_pdf_transcriber(model: str = OCR_MODEL, client: Any = None) -> Callable[[Path], str]:
    client = client or anthropic.Anthropic()

    def transcribe(path: Path) -> str:
        data = path.read_bytes()
        if len(data) > MAX_PDF_BYTES:
            print(f"[ocr] {path.name}: larger than 32 MB, skipped", file=sys.stderr)
            return ""
        with client.messages.stream(
            model=model,
            max_tokens=32000,
            output_config={"effort": "low"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                    "data": base64.standard_b64encode(data).decode("ascii")}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            print(f"[ocr] {path.name}: the model declined to transcribe this file", file=sys.stderr)
            return ""
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        if msg.stop_reason == "max_tokens":
            print(f"[ocr] {path.name}: transcription truncated (document too long)", file=sys.stderr)
        return text

    return transcribe
