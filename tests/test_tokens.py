from ragretriever.tokens import content_words, count_tokens, jaccard, split_sentences


def test_count_tokens_words_and_punctuation():
    assert count_tokens("Hello, world!") == 4
    assert count_tokens("") == 0
    assert count_tokens("one two three") == 3


def test_count_tokens_is_deterministic():
    text = "Photosynthesis uses sunlight, water and carbon dioxide."
    assert count_tokens(text) == count_tokens(text)


def test_split_sentences_keeps_punctuation():
    parts = split_sentences("First sentence. Second one! Third?")
    assert parts == ["First sentence.", "Second one!", "Third?"]


def test_content_words_drops_stopwords_and_short_words():
    words = content_words("The cat sat on the mat and ate fish")
    assert "cat" in words
    assert "the" not in words
    assert "on" not in words


def test_jaccard_identical_and_disjoint():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 0.5
