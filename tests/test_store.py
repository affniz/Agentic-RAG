import pytest

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


def test_indexed_times_only_change_for_reread_files(tmp_path):
    import time

    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# A\nalpha")
    (d / "b.md").write_text("# B\nbeta")
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(d)
    first = kb.indexed_times()
    assert set(first) == {"a.md", "b.md"}
    time.sleep(0.01)
    (d / "b.md").write_text("# B\nbeta changed")
    kb.ingest(d)
    second = kb.indexed_times()
    assert second["a.md"] == first["a.md"] and second["b.md"] > first["b.md"]


def _text_pdf(path, text, user_password=None):
    """A one-page PDF with real text, AES-256 encrypted when user_password is given ("" = opens without one)."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    w = PdfWriter()
    page = w.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = w._add_object(stream)
    if user_password is not None:
        w.encrypt(user_password=user_password, owner_password="owner", algorithm="AES-256")
    with open(path, "wb") as f:
        w.write(f)


def test_aes_encrypted_pdf_without_password_is_indexed(tmp_path):
    docs = make_docs(tmp_path)
    _text_pdf(docs / "card.pdf", "Exam centre is Hall B, report by nine in the morning.", user_password="")
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    r = kb.ingest(docs)
    assert r.failed == {} and "card.pdf" in r.added
    assert "Hall B" in kb.get_chunks("card.pdf", 0, 0)[0].text


def test_unreadable_file_is_skipped_not_fatal(tmp_path):
    docs = make_docs(tmp_path)
    _text_pdf(docs / "locked.pdf", "Secret contents that need a password to open.", user_password="hunter2")
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    r = kb.ingest(docs)
    assert set(r.failed) == {"locked.pdf"} and "password-protected" in r.failed["locked.pdf"]
    assert "locked.pdf" not in r.added and "locked.pdf" not in kb.indexed_files()
    assert kb.sources() and "locked.pdf" not in kb.sources()  # everything else was indexed
    assert kb.index_errors() == r.failed
    assert set(kb.ingest(docs).failed) == {"locked.pdf"}  # retried, still failing
    (docs / "locked.pdf").unlink()
    assert kb.ingest(docs).failed == {} and kb.index_errors() == {}


def test_missing_crypto_dependency_says_what_to_install(tmp_path, monkeypatch):
    from pypdf.errors import DependencyError

    import rag.store as store

    docs = make_docs(tmp_path)
    _text_pdf(docs / "card.pdf", "Exam centre is Hall B, report by nine in the morning.", user_password="")

    def fail(path):
        raise DependencyError("cryptography>=3.1 is required for AES algorithm")

    monkeypatch.setattr(store, "_read", fail)
    r = KnowledgeBase(tmp_path / "idx", rerank=False).ingest(docs)
    assert "cryptography" in r.failed["card.pdf"] and "pip install" in r.failed["card.pdf"]


def test_ingest_prunes_transcriptions_of_removed_or_replaced_files(tmp_path):
    docs = make_docs(tmp_path)
    _blank_pdf(docs / "a.pdf")
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    assert kb.transcribe(docs / "a.pdf", lambda p: "# A\nTranscribed text of a.")
    kb.ingest(docs)
    assert kb.has_transcription(docs / "a.pdf")

    _blank_pdf(docs / "a.pdf", pages=2)  # replaced by a different version: the old text no longer applies
    kb.ingest(docs)
    assert not list((tmp_path / "idx" / "ocr").glob("*.md"))

    assert kb.transcribe(docs / "a.pdf", lambda p: "# A\nTranscribed text of a.")
    (docs / "a.pdf").unlink()  # removed outside the app
    kb.ingest(docs)
    assert not list((tmp_path / "idx" / "ocr").glob("*.md"))


def test_long_paragraph_splits_at_sentence_ends_without_cutting_words():
    from rag.store import _chunk

    sentences = [f"Sentence {i} has " + " ".join(f"w{i:02d}x{j}" for j in range(12)) + "." for i in range(40)]
    words = set(" ".join(sentences).split())
    chunks = _chunk(" ".join(sentences), size=300, overlap=80)
    assert len(chunks) > 5
    for c in chunks:
        assert len(c) <= 300
        assert set(c.split()) <= words  # every token is a whole word from the source
        assert c.endswith(".")  # pieces end at a sentence end
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)  # nothing lost


def test_long_table_splits_at_row_boundaries():
    from rag.store import _chunk

    rows = [f"| row {i} | value {i * 7} |" for i in range(60)]
    chunks = _chunk("\n".join(rows), size=200, overlap=50)
    for c in chunks:
        assert all(line in rows for line in c.split("\n"))  # whole rows only
    assert {r for c in chunks for r in c.split("\n")} == set(rows)


def test_overlong_word_is_the_only_thing_cut():
    from rag.store import _chunk

    chunks = _chunk("a" * 250 + " tail words.", size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks) and chunks[-1].endswith("tail words.")


def test_get_chunks_and_sources_use_the_per_document_index(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# A\n" + "\n\n".join(f"## S{i}\nPara {i} text." for i in range(5)))
    (d / "b.txt").write_text("Only one paragraph.")
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(d)
    assert kb.sources() == {"a.md": 5, "b.txt": 1}
    assert [c.index for c in kb.get_chunks("a.md", 1, 3)] == [1, 2, 3]
    assert [c.index for c in kb.get_chunks("a.md", 3, 99)] == [3, 4]
    assert kb.get_chunks("a.md", 4, 2) == [] and kb.get_chunks("nope.md", 0, 5) == []
    assert all(c.source == "a.md" for c in kb.get_chunks("a.md", 0, 4))


# ---------------------------------------------------------------- incremental, per-chunk updates


def _sectioned(n: int, edit: str = "") -> str:
    parts = ["# Leave Policy\nApplies to all employees."]
    for i in range(n):
        parts.append(f"## Section {i}\n" + f"Rule {i}: " + "standard wording " * 20 + (edit if i == 3 else ""))
    return "\n\n".join(parts)


def test_chunk_ids_follow_content_not_position():
    a = chunk_document("p.md", _sectioned(6), markdown=True)
    # Insert a new section at the top: every later chunk shifts index but keeps its id.
    b = chunk_document("p.md", "# Leave Policy\nApplies to all employees.\n\n## New\nBrand new rule.\n\n"
                       + _sectioned(6).split("\n\n", 1)[1], markdown=True)
    assert [c.index for c in b if c.id in {x.id for x in a}] != [c.index for c in a]
    assert len({c.id for c in b} - {c.id for c in a}) == 1
    # Same text in two places, or in two documents, still gets distinct ids.
    dup = chunk_document("d.md", "# T\n## A\nsame\n## B\nx\n## A\nsame", markdown=True)
    assert len({c.id for c in dup}) == len(dup)
    assert chunk_document("x.md", "same", False)[0].id != chunk_document("y.md", "same", False)[0].id


def test_update_embeds_only_changed_chunks(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave_policy.md").write_text(_sectioned(8))
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    r = kb.ingest(docs)
    total = len(kb.chunks)
    assert r.embedded == total > 5

    (docs / "leave_policy.md").write_text(_sectioned(8, edit=" Carry-over is now 10 days."))
    r = kb.ingest(docs)
    assert r.updated == ["leave_policy.md"] and r.embedded == 1
    assert "10 days" in kb.search("carry-over days", k=1, mode="vector")[0][0].text
    # The old version of the edited chunk is gone from the vector store; nothing else is left over.
    assert sorted(kb._col.get(include=[])["ids"]) == sorted(c.id for c in kb.chunks)

    r = kb.ingest(docs)
    assert r.embedded == 0 and r.unchanged == 1

    (docs / "leave_policy.md").unlink()
    kb.ingest(docs)
    assert kb._col.get(include=[])["ids"] == [] and kb.chunks == []


def test_failed_update_keeps_previous_version_and_retries(tmp_path, monkeypatch):
    import rag.store as store

    docs = make_docs(tmp_path)
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(docs)
    old_hash = kb.indexed_files()["travel.txt"]

    (docs / "travel.txt").write_text("Hotel cap is $400 per night in London.")
    real_read = store._read
    monkeypatch.setattr(store, "_read", lambda p: (_ for _ in ()).throw(OSError("disk error"))
                        if p.name == "travel.txt" else real_read(p))
    r = kb.ingest(docs)
    assert set(r.failed) == {"travel.txt"} and r.updated == []
    # The previous version is still fully searchable, by keywords and by vectors.
    assert "$350" in kb.search("london hotel", k=1, mode="bm25")[0][0].text
    assert "$350" in kb.search("london hotel", k=1, mode="vector")[0][0].text
    assert kb.indexed_files()["travel.txt"] == old_hash  # still differs from disk -> retried
    assert set(kb.ingest(docs).failed) == {"travel.txt"}

    monkeypatch.setattr(store, "_read", real_read)
    r = kb.ingest(docs)
    assert r.failed == {} and r.updated == ["travel.txt"]
    assert "$400" in kb.search("london hotel", k=1)[0][0].text
    assert len(kb.get_chunks("travel.txt", 0, 99)) == 1


def test_vector_search_has_no_gap_while_a_file_is_updated(tmp_path):
    docs = make_docs(tmp_path)
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(docs)
    (docs / "travel.txt").write_text("Hotel cap is $400 per night in London.")
    seen = []

    def progress(stage, done, total):  # runs mid-ingest: while reading, and after the new vectors are added
        if stage == "reading" or (stage == "embedding" and done == total):
            hits = kb.search("london hotel", k=1, mode="vector", source="travel.txt")
            seen.append(hits[0][0].text if hits else None)

    kb.ingest(docs, on_progress=progress)
    assert len(seen) == 2 and all(t and "$350" in t for t in seen)  # old version served until the swap
    assert "$400" in kb.search("london hotel", k=1, mode="vector")[0][0].text


def test_vectors_left_by_an_interrupted_ingest_are_reused_or_cleaned(tmp_path):
    docs = make_docs(tmp_path)
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(docs)
    (docs / "travel.txt").write_text("Hotel cap is $400 per night in London.")
    new_id = chunk_document("travel.txt", (docs / "travel.txt").read_text(), False)[0].id
    # Simulate a crash after embedding but before the swap, plus a stray vector.
    kb._col.add(ids=[new_id, "stray"], documents=["Hotel cap is $400 per night in London.", "junk"],
                metadatas=[{"source": "travel.txt"}, {"source": "gone.txt"}])

    r = KnowledgeBase(tmp_path / "idx", rerank=False).ingest(docs)
    assert r.embedded == 0  # the vector from the interrupted run is reused
    assert sorted(kb._col.get(include=[])["ids"]) == sorted(c.id for c in KnowledgeBase(tmp_path / "idx", rerank=False).chunks)


def test_missing_docs_folder_changes_nothing(tmp_path):
    # Before the guard, a typo in the folder name looked like "every file was deleted".
    docs = make_docs(tmp_path)
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(docs)
    before = (len(kb.chunks), kb.indexed_files())
    with pytest.raises(FileNotFoundError):
        kb.ingest(tmp_path / "doc")
    assert (len(kb.chunks), kb.indexed_files()) == before
    assert len(KnowledgeBase(tmp_path / "idx", rerank=False).chunks) == before[0]


def test_vectors_lost_by_an_interrupted_rebuild_are_restored(tmp_path, monkeypatch):
    docs = make_docs(tmp_path)
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(docs)
    n = len(kb.chunks)

    class NoAdd:  # the rebuild empties the collection, then dies while embedding
        def __init__(self, col):
            self._col = col

        def __getattr__(self, name):
            return getattr(self._col, name)

        def add(self, **_):
            raise RuntimeError("killed mid-rebuild")

    real = kb._get_collection
    monkeypatch.setattr(kb, "_get_collection", lambda: NoAdd(real()))
    with pytest.raises(RuntimeError):
        kb.ingest(docs, rebuild=True)
    assert kb._col.count() == 0

    # After a restart nothing on disk changed, so every file looks unchanged - but its vectors are gone.
    kb2 = KnowledgeBase(tmp_path / "idx", rerank=False)
    assert len(kb2.chunks) == n
    r = kb2.ingest(docs)
    assert r.embedded == n and r.added == r.updated == []
    assert kb2._col.count() == n
    assert kb2.search("london hotel", k=1, mode="vector")[0][0].source == "travel.txt"


def test_keyword_search_handles_accents_and_scripts_without_spaces(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "de.md").write_text("# Reise\nDie Geschäftsreise nach München wird erstattet.")
    (d / "ja.md").write_text("# 休暇\n有給休暇は年間20日です。")
    (d / "en.md").write_text("# Travel\nHotel cap is $350 per night in London.")
    kb = KnowledgeBase(tmp_path / "idx", rerank=False)
    kb.ingest(d)
    assert kb.search("geschaftsreise munchen", k=1, mode="bm25")[0][0].source == "de.md"  # accents folded
    assert kb.search("Geschäftsreise", k=1, mode="bm25")[0][0].source == "de.md"
    assert kb.search("有給休暇", k=1, mode="bm25")[0][0].source == "ja.md"
    assert kb.search("London hotel", k=1, mode="bm25")[0][0].source == "en.md"


def test_ingest_by_another_instance_is_not_undone(tmp_path):
    # Two KnowledgeBases on one index folder: the server and the CLI, or the eval runner's two
    # instances of a set. Re-indexing with the one whose in-memory chunks are older must not
    # bring the old text back.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Policy\n\nThe hotel cap is 100 dollars.\n")
    first = KnowledgeBase(tmp_path / "idx", rerank=False)
    first.ingest(docs)
    second = KnowledgeBase(tmp_path / "idx", rerank=False)
    (docs / "a.md").write_text("# Policy\n\nThe hotel cap is 250 dollars.\n")
    second.ingest(docs)
    r = first.ingest(docs)
    assert r.updated == [] and r.embedded == 0
    for kb in (first, KnowledgeBase(tmp_path / "idx", rerank=False)):
        assert [c.text for c in kb.chunks] == ["The hotel cap is 250 dollars."]
        assert [c.text for c, _ in kb.search("hotel cap", k=1, mode="vector")] == ["The hotel cap is 250 dollars."]
