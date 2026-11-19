"""Text extraction for scanned PDFs (no text layer) using Claude's PDF understanding.

Opt-in: it sends the PDF to the Anthropic API. Enable with `ingest --ocr` or RAG_OCR=1.
The store caches each transcription by file hash, so a file is only sent once.

Long PDFs are sent in batches of pages, so no single request hits the output-token limit, the
per-request page limit or the request size limit. A part that still can't be transcribed in full
leaves a visible "[Transcription incomplete: ...]" note in the text, so the gap shows in the
document viewer and the model sees it too, instead of the document silently ending early.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import anthropic
from pypdf import PdfReader, PdfWriter

from .models import request_options

OCR_MODEL = "claude-haiku-5-5"
PAGES_PER_REQUEST = 20
MAX_REQUEST_BYTES = 32 * 1024 * 1024  # API request size limit
# Base64 makes the PDF 4/3 larger; leave room for the rest of the request.
MAX_PDF_BYTES = (MAX_REQUEST_BYTES - 1024 * 1024) * 3 // 4

RULES = """\
- Keep tables as Markdown tables, with every row and column.
- Keep names, numbers, dates and codes exactly as written. Mark anything illegible as [illegible].
Output only the transcription, with no commentary."""

PROMPT = """\
Transcribe all text in this document as Markdown, in reading order.
- Start with a single "# " heading naming what the document is (e.g. "# Invoice - Acme Ltd, \
March 2026"), using only information in the document.
""" + RULES

PROMPT_CONTINUED = """\
These are pages {first}-{last} of a {total}-page document whose earlier pages were transcribed \
separately. Transcribe all text on them as Markdown, in reading order, continuing that \
transcription: don't add a title heading for the document.
""" + RULES


def _incomplete(first: int, last: int, why: str) -> str:
    pages = f"page {first}" if first == last else f"pages {first}-{last}"
    return f"[Transcription incomplete: {pages} {why}.]"


def _subset(reader: PdfReader, start: int, end: int) -> bytes:
    """Pages start..end-1 (0-based) as a PDF of their own."""
    w = PdfWriter()
    for i in range(start, end):
        w.add_page(reader.pages[i])
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def _parts(data: bytes) -> Iterator[tuple[int, int, int, bytes | None]]:
    """(first page, last page, total pages, PDF bytes or None if a single page is too large),
    1-based and inclusive: the whole file if it fits in one request, else batches of pages,
    halved further while a batch is too large to send."""
    reader = PdfReader(io.BytesIO(data))
    total = len(reader.pages)
    if total <= PAGES_PER_REQUEST and len(data) <= MAX_PDF_BYTES:
        yield 1, max(total, 1), total, data
        return

    def split(start: int, end: int) -> Iterator[tuple[int, int, int, bytes | None]]:
        blob = _subset(reader, start, end)
        if len(blob) <= MAX_PDF_BYTES:
            yield start + 1, end, total, blob
        elif end - start == 1:
            yield start + 1, end, total, None
        else:
            mid = (start + end) // 2
            yield from split(start, mid)
            yield from split(mid, end)

    for start in range(0, total, PAGES_PER_REQUEST):
        yield from split(start, min(start + PAGES_PER_REQUEST, total))


def claude_pdf_transcriber(model: str = OCR_MODEL, client: Any = None) -> Callable[[Path], str]:
    client = client or anthropic.Anthropic()
    opts = request_options(model)  # a refusal fallback needs the beta endpoint
    api = client.beta.messages if opts else client.messages

    def ask(blob: bytes, prompt: str) -> Any:
        with api.stream(
            model=model,
            max_tokens=32000,
            output_config={"effort": "low"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                    "data": base64.standard_b64encode(blob).decode("ascii")}},
                    {"type": "text", "text": prompt},
                ],
            }],
            **opts,
        ) as stream:
            return stream.get_final_message()

    def transcribe(path: Path) -> str:
        texts: list[str] = []
        notes = 0
        got_text = False
        for first, last, total, blob in _parts(path.read_bytes()):
            if blob is None:
                texts.append(_incomplete(first, last, "skipped: too large to send in one request"))
                notes += 1
                continue
            prompt = PROMPT if first == 1 else PROMPT_CONTINUED.format(first=first, last=last, total=total)
            msg = ask(blob, prompt)
            if msg.stop_reason == "refusal":
                texts.append(_incomplete(first, last, "were not transcribed: the model declined"))
                notes += 1
                continue
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
            got_text = got_text or bool(text)
            if msg.stop_reason == "max_tokens":
                text += "\n\n" + _incomplete(first, last, "were cut off: too much text for one request")
                notes += 1
            texts.append(text)
        if notes:
            print(f"[ocr] {path.name}: {notes} part(s) incomplete - see the notes in the transcription",
                  file=sys.stderr)
        if not got_text:  # nothing at all was transcribed
            return ""
        return "\n\n".join(t for t in texts if t)

    return transcribe
