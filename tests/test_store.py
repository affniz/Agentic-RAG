from rag.store import KnowledgeBase, _chunk, _sections, chunk_document


def test_chunks_respect_size_and_keep_all_text():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20)) + "\n\n" + "x" * 2000
    chunks = _chunk(text, size=800, overlap=150)
    assert max(map(len, chunks)) <= 800
    joined = "".join(chunks)
    assert all(f"Paragraph {i} " in joined for i in range(20))


def test_markdown_sections_track_heading_path():
    md = "# Title\nintro\n## A\nalpha\n### A1\ndeep\n## B\nbeta\n```\n# not a heading\n```\n"
    secs = dict((body.strip().splitlines()[0], path) for path, body in _sections(md, markdown=True))
    assert secs["intro"] == "Title"
    assert secs["alpha"] == "Title > A"
    assert secs["deep"] == "Title > A > A1"
    assert secs["beta"] == "Title > B"  # A1 popped, code-fence line not treated as heading


def test_chunk_document_labels():
    chunks = chunk_document("p.md", "# Policy\n## Carry-over\nUp to 5 days.", markdown=True)
    assert chunks[0].section == "Policy > Carry-over"
    assert chunks[0].label() == "[p.md#0] (Policy > Carry-over)"
    assert chunks[0].indexed_text.startswith("Policy > Carry-over\n")


def make_docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "vacation.md").write_text("# Vacation\n## Carry-over\nUp to 5 unused days carry over until March 31.")
    (d / "travel.txt").write_text("Hotel cap is $350 per night in London.")
    return d


def test_incremental_ingest(tmp_path):
    docs = make_docs(tmp_path)
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    r = kb.ingest(docs)
    assert (len(r.added), r.unchanged) == (2, 0)

    r = kb.ingest(docs)
    assert (len(r.added), len(r.updated), r.unchanged) == (0, 0, 2)

    (docs / "travel.txt").write_text("Hotel cap is $400 per night in London.")
    (docs / "vacation.md").unlink()
    r = kb.ingest(docs)
    assert r.updated == ["travel.txt"] and r.removed == ["vacation.md"]
    assert set(kb.sources()) == {"travel.txt"}
    assert "$400" in kb.search("london hotel", k=1)[0][0].text

    # Reloading from disk gives the same state.
    kb2 = KnowledgeBase(tmp_path / "idx", rerank=False)
    assert [c.id for c in kb2.chunks] == [c.id for c in kb.chunks]


def test_search_modes_and_source_filter(tmp_path):
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(make_docs(tmp_path))
    for mode in ("bm25", "vector", "hybrid"):
        top = kb.search("how many vacation days carry over", k=1, mode=mode)[0][0]
        assert top.source == "vacation.md", mode
    only = kb.search("vacation days", k=5, source="travel.txt")
    assert {c.source for c, _ in only} == {"travel.txt"}


class FakeReranker:
    available = True

    def score(self, query, docs):
        return [1.0 if "London" in d else 0.0 for d in docs]


def test_reranker_reorders(tmp_path):
    kb = KnowledgeBase(tmp_path / "idx", reranker=FakeReranker())
    kb.ingest(make_docs(tmp_path))
    top = kb.search("vacation carry over", k=1)[0][0]
    assert top.source == "travel.txt"  # fake reranker prefers the London chunk
    assert kb.search("vacation carry over", k=1, rerank=False)[0][0].source == "vacation.md"


def test_document_intro_is_attached_to_every_chunk():
    md = "# Travel Policy 2025\nStatus: SUPERSEDED. In force until 2026-02-28.\n\n## Hotels\nTier 1: $320\n## Flights\nEconomy."
    chunks = chunk_document("t.md", md, markdown=True)
    intro, hotels, flights = chunks
    assert intro.doc_note == ""  # the intro chunk already contains it
    assert hotels.doc_note == flights.doc_note == "Status: SUPERSEDED. In force until 2026-02-28."
    assert hotels.render().splitlines()[1] == "(About this document: Status: SUPERSEDED. In force until 2026-02-28.)"
    assert "SUPERSEDED" not in hotels.indexed_text  # display only; retrieval is unchanged

    # No intro (title straight into a subsection) -> no note; plain text uses its first paragraph.
    assert all(c.doc_note == "" for c in chunk_document("r.md", "# Notes\n## v1\nx\n## v2\ny", markdown=True))
    txt = chunk_document("p.txt", "Expense Policy\n\n" + "\n\n".join("rule " * 40 for _ in range(6)), markdown=False)
    assert txt[0].doc_note == "" and txt[-1].doc_note == "Expense Policy"


def test_manifest_readable_during_ingest(tmp_path):
    """Regression: the documents API read manifest.json while ingest was rewriting it."""
    import threading

    docs = make_docs(tmp_path)
    for i in range(30):
        (docs / f"extra{i}.md").write_text(f"# Doc {i}\n" + "word " * 200)
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(docs)
    errors, stop = [], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                kb.indexed_files()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    t = threading.Thread(target=reader)
    t.start()
    for _ in range(3):
        kb.ingest(docs, rebuild=True)
    stop.set()
    t.join()
    assert errors == []


def _blank_pdf(path, pages=1):
    from pypdf import PdfWriter

    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=595, height=842)
    with open(path, "wb") as f:
        w.write(f)


def test_scanned_pdf_reported_empty_then_transcribed_once(tmp_path):
    docs = make_docs(tmp_path)
    _blank_pdf(docs / "scan.pdf", pages=2)

    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    r = kb.ingest(docs)
    assert r.empty == ["scan.pdf"] and "scan.pdf" not in kb.sources()

    calls = []

    def fake_ocr(path):
        calls.append(path.name)
        return "# Report Card 2020\n\n## Marks\n| Subject | Marks |\n|---|---|\n| Maths | 95 |"

    kb = KnowledgeBase(tmp_path / "idx", rerank=False, pdf_ocr=fake_ocr)
    r = kb.ingest(docs)  # file unchanged, but it was empty and OCR is now available -> retried
    assert r.empty == [] and calls == ["scan.pdf"]
    hit = kb.search("maths marks", k=1)[0][0]
    assert hit.source == "scan.pdf" and hit.section == "Report Card 2020 > Marks"

    kb.ingest(docs, rebuild=True)  # cached by file hash: not sent again
    assert calls == ["scan.pdf"] and "scan.pdf" in kb.sources()
    assert kb.ingest(docs).empty == [] and calls == ["scan.pdf"]  # and not retried once it has text


def test_scanned_pdf_from_older_index_is_retried(tmp_path):
    """Regression: a PDF indexed (with 0 chunks) before this version must still be transcribed."""
    import json

    docs = make_docs(tmp_path)
    _blank_pdf(docs / "scan.pdf")
    KnowledgeBase(tmp_path / "idx", rerank=False).ingest(docs)
    manifest = tmp_path / "idx" / "manifest.json"
    m = json.loads(manifest.read_text())
    m.pop("empty", None)  # what an older version wrote
    manifest.write_text(json.dumps(m))

    kb = KnowledgeBase(tmp_path / "idx", rerank=False, pdf_ocr=lambda p: "# Scan\nMaths 95")
    assert kb.ingest(docs).empty == [] and "scan.pdf" in kb.sources()


def test_manual_transcribe_and_undo(tmp_path):
    """Per-file transcription overrides a PDF's own text (even a junk text layer); clearing undoes it."""
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, StreamObject, DictionaryObject

    docs = make_docs(tmp_path)
    # A PDF whose text layer is only junk long enough to pass the "has text" check.
    w = PdfWriter()
    page = w.add_blank_page(width=595, height=842)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})})
    content = StreamObject()
    content.set_data(b"BT /F1 12 Tf 72 720 Td (SCANNER WATERMARK XJ-2000 PAGE ONE OF ONE) Tj ET")
    page[NameObject("/Contents")] = w._add_object(content)
    with open(docs / "card.pdf", "wb") as f:
        w.write(f)

    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(docs)
    assert "WATERMARK" in kb.get_chunks("card.pdf", 0, 0)[0].text  # junk layer used, not auto-transcribed

    assert kb.transcribe(docs / "card.pdf", lambda p: "# Report Card\nMaths 95") and kb.has_transcription(docs / "card.pdf")
    r = kb.ingest(docs, force=["card.pdf"])
    assert r.updated == ["card.pdf"] and kb.get_chunks("card.pdf", 0, 0)[0].text == "Maths 95"
    assert kb.ingest(docs).updated == []  # stays transcribed without forcing

    assert kb.clear_transcription(docs / "card.pdf")
    kb.ingest(docs, force=["card.pdf"])
    assert "WATERMARK" in kb.get_chunks("card.pdf", 0, 0)[0].text
    assert not kb.transcribe(docs / "card.pdf", lambda p: "  ")  # nothing returned -> nothing saved
