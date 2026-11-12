import pytest

from ragretriever.assemble import assemble, compare
from ragretriever.chunking import Chunk
from ragretriever.retrieval import Retriever, ScoredChunk
from ragretriever.tokens import count_tokens


def _scored(doc_id, index, text, score=1.0):
    return ScoredChunk(Chunk(doc_id, "Title", index, text), score)


def test_budget_is_never_exceeded():
    retriever = Retriever()
    for budget in (30, 60, 120, 200):
        assembled, _ = compare("How does photosynthesis make food for plants?", budget=budget, retriever=retriever)
        assert assembled.tokens <= budget


def test_tiny_budget_still_fits_or_is_empty():
    assembled, _ = compare("photosynthesis sunlight", budget=10)
    assert assembled.tokens <= 10


def test_assemble_rejects_bad_budget():
    with pytest.raises(ValueError):
        assemble([], budget=0)


def test_near_duplicate_chunk_is_dropped():
    text = "Chlorophyll in the leaves captures sunlight and gives plants their green colour. The plant uses energy to grow."
    ranked = [
        _scored("doc-a", 1, text, score=5.0),
        _scored("doc-b", 1, text, score=4.0),
        _scored("doc-c", 1, "Bees pollinate flowers as they collect nectar from blooms.", score=3.0),
    ]
    result = assemble(ranked, budget=200)
    assert [c.citation for c in result.chunks] == ["[doc-a:1]", "[doc-c:1]"]
    assert result.dropped_duplicates == ["[doc-b:1]"]


def test_truncation_keeps_citation_and_fits():
    long_text = (
        "Photosynthesis is the process by which green plants make food using sunlight, water and carbon dioxide. "
        "Chlorophyll in the leaves captures sunlight and gives plants their green colour. "
        "The plant uses that captured energy to turn water and carbon dioxide into glucose, a sugar it can use for growth. "
        "Oxygen is released as a byproduct of photosynthesis, which is why forests help refresh the air we breathe."
    )
    ranked = [_scored("photosynthesis-basics", 1, long_text, score=5.0)]
    result = assemble(ranked, budget=40)
    assert result.tokens <= 40
    assert result.chunks
    assert result.chunks[0].citation == "[photosynthesis-basics:1]"
    assert "[truncated]" in result.chunks[0].text
    assert result.truncated == ["[photosynthesis-basics:1]"]


def test_compare_reports_real_savings():
    _, comparison = compare("How does photosynthesis make food for plants?", budget=120)
    assert comparison.budgeted_tokens <= comparison.naive_tokens
    assert comparison.tokens_saved == comparison.naive_tokens - comparison.budgeted_tokens
    assert comparison.within_budget
    assert comparison.budgeted_citations
    assert all(c in comparison.naive_citations for c in comparison.budgeted_citations if "[truncated]" not in c)


def test_cited_text_includes_citation():
    chunk = Chunk("doc-a", "Title", 2, "Some text here.")
    assert chunk.cited_text().startswith("[doc-a:2] ")
    assert count_tokens(chunk.cited_text()) == count_tokens("[doc-a:2]") + count_tokens("Some text here.")
