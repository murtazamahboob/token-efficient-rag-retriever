"""Deterministic token counting and text helpers.

A token here is a word or a single punctuation mark, counted with one
regex everywhere in the project, so every budget, saving and benchmark
number is reproducible and internally consistent.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_WORD_RE = re.compile(r"[a-z]+")

STOPWORDS = frozenset(
    "a an the is are was were be been being to of in on at for with by and or as it its this that these those from into over under after before between through during each other some any no not can could will would should may might do does did have has had i you he she we they them his her their our your my me him us what when where which who whom how why if then than so such very really just also".split()
)


def count_tokens(text: str) -> int:
    """Count tokens in *text* (words + punctuation marks)."""
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))


def content_words(text: str) -> set[str]:
    """Lowercase alphabetic words of length > 2, minus stopwords."""
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, keeping the trailing punctuation."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return parts


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two word sets (1.0 when both empty)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
