import base64
import io
from types import SimpleNamespace as NS

from pypdf import PdfReader, PdfWriter

import rag.ocr as ocr
from rag.fake_llm import _Stream, message, text_block


def _pdf(path, pages):
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=595, height=842)
    with open(path, "wb") as f:
        w.write(f)


class FakeClient:
    """Answers each transcription request with 'PART n', or with the scripted stop reasons."""

    def __init__(self, stops=()):
        self.stops, self.requests = list(stops), []
        self.messages = NS(stream=self._stream)

    def _stream(self, **kw):
        self.requests.append(kw)
        stop = self.stops.pop(0) if self.stops else "end_turn"
        return _Stream(message(stop, [text_block(f"PART {len(self.requests)}")]))


def _pages(req):
    data = base64.standard_b64decode(req["messages"][0]["content"][0]["source"]["data"])
    return len(PdfReader(io.BytesIO(data)).pages)


def _prompt(req):
    return req["messages"][0]["content"][1]["text"]


def test_short_pdf_is_sent_whole_in_one_request(tmp_path):
    _pdf(tmp_path / "a.pdf", 3)
    client = FakeClient()
    assert ocr.claude_pdf_transcriber(client=client)(tmp_path / "a.pdf") == "PART 1"
    assert len(client.requests) == 1 and _pages(client.requests[0]) == 3
    assert _prompt(client.requests[0]) == ocr.PROMPT


def test_long_pdf_is_sent_in_page_batches(tmp_path):
    _pdf(tmp_path / "long.pdf", 45)
    client = FakeClient()
    text = ocr.claude_pdf_transcriber(client=client)(tmp_path / "long.pdf")
    assert text == "PART 1\n\nPART 2\n\nPART 3"
    assert [_pages(r) for r in client.requests] == [20, 20, 5]
    assert _prompt(client.requests[0]) == ocr.PROMPT  # only the first part gets a title heading
    assert "pages 21-40 of a 45-page document" in _prompt(client.requests[1])


def test_cut_off_or_declined_parts_leave_a_visible_note(tmp_path):
    _pdf(tmp_path / "long.pdf", 45)
    text = ocr.claude_pdf_transcriber(client=FakeClient(["end_turn", "max_tokens", "refusal"]))(tmp_path / "long.pdf")
    assert text.startswith("PART 1\n\nPART 2\n\n[Transcription incomplete: pages 21-40 were cut off")
    assert text.endswith("[Transcription incomplete: pages 41-45 were not transcribed: the model declined.]")


def test_nothing_transcribed_returns_empty(tmp_path):
    _pdf(tmp_path / "a.pdf", 2)
    assert ocr.claude_pdf_transcriber(client=FakeClient(["refusal"]))(tmp_path / "a.pdf") == ""


def test_batches_too_large_to_send_are_split_further(tmp_path, monkeypatch):
    _pdf(tmp_path / "a.pdf", 4)
    one_page = len(ocr._subset(PdfReader(tmp_path / "a.pdf"), 0, 1))
    monkeypatch.setattr(ocr, "MAX_PDF_BYTES", one_page + 50)  # room for one page per request
    client = FakeClient()
    ocr.claude_pdf_transcriber(client=client)(tmp_path / "a.pdf")
    assert [_pages(r) for r in client.requests] == [1, 1, 1, 1]

    monkeypatch.setattr(ocr, "MAX_PDF_BYTES", 10)  # not even one page fits
    text = ocr.claude_pdf_transcriber(client=FakeClient())(tmp_path / "a.pdf")
    assert text == "" and "skipped" in ocr._incomplete(1, 1, "skipped: too large to send in one request")
