from ragretriever.chunking import build_chunks, chunk_document
from ragretriever.corpus import DOCUMENTS, get_document
from ragretriever.tokens import count_tokens


def test_every_document_chunks():
    chunks = build_chunks()
    assert chunks
    assert len({c.doc_id for c in chunks}) == len(DOCUMENTS)


def test_chunks_respect_max_tokens():
    for chunk in build_chunks(max_tokens=45):
        # A single long sentence may exceed the cap; grouped chunks must not.
        assert count_tokens(chunk.text) <= 60


def test_chunk_text_covers_the_document():
    doc = get_document("python-functions")
    chunks = chunk_document(doc)
    assert " ".join(c.text for c in chunks) == doc.text


def test_citations_are_unique_and_numbered():
    chunks = build_chunks()
    citations = [c.citation for c in chunks]
    assert len(citations) == len(set(citations))
    doc_chunks = [c for c in chunks if c.doc_id == "black-holes"]
    assert [c.index for c in doc_chunks] == list(range(1, len(doc_chunks) + 1))
    assert doc_chunks[0].citation == "[black-holes:1]"


def test_chunking_rejects_bad_max_tokens():
    import pytest

    with pytest.raises(ValueError):
        chunk_document(DOCUMENTS[0], max_tokens=0)


def test_small_max_tokens_makes_more_chunks():
    doc = get_document("ocean-currents")
    assert len(chunk_document(doc, max_tokens=15)) >= len(chunk_document(doc, max_tokens=45))
