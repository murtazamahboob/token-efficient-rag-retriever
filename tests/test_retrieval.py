import pytest

from ragretriever.retrieval import Retriever


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


def test_photosynthesis_query_ranks_photosynthesis_first(retriever):
    ranked = retriever.retrieve("How does photosynthesis make food for plants?")
    assert ranked
    assert ranked[0].chunk.doc_id.startswith("photosynthesis")


def test_python_query_ranks_python_first(retriever):
    ranked = retriever.retrieve("How do Python functions return values?")
    assert ranked
    assert ranked[0].chunk.doc_id.startswith("python")


def test_scores_descend(retriever):
    ranked = retriever.retrieve("black hole gravity light escape", top_k=5)
    scores = [c.score for c in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_unknown_query_returns_nothing(retriever):
    assert retriever.retrieve("xyzqq wobbleflop zanzibar") == []


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("") == []


def test_top_k_is_respected(retriever):
    assert len(retriever.retrieve("plants sunlight water", top_k=3)) <= 3


def test_retrieval_is_deterministic(retriever):
    q = "How do vaccines train the immune system?"
    first = [c.chunk.citation for c in retriever.retrieve(q)]
    second = [c.chunk.citation for c in retriever.retrieve(q)]
    assert first == second


def test_top_k_rejects_zero():
    with pytest.raises(ValueError):
        Retriever().retrieve("plants", top_k=0)
