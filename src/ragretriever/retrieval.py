"""BM25 retrieval over the chunked corpus — offline, deterministic.

Classic BM25 (k1=1.5, b=0.75) over lowercased word tokens. Ties break by
corpus order, so the same query always returns the same ranking.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .chunking import Chunk, build_chunks
from .tokens import count_tokens

_WORD_RE = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float

    @property
    def tokens(self) -> int:
        return self.chunk.tokens


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks if chunks is not None else build_chunks()
        self.k1 = k1
        self.b = b
        self._doc_freq: dict[str, int] = {}
        self._lengths: list[int] = []
        for chunk in self.chunks:
            terms = set(_terms(chunk.text))
            for term in terms:
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
            self._lengths.append(count_tokens(chunk.text))
        self._avg_len = sum(self._lengths) / max(1, len(self._lengths))
        self._n = len(self.chunks)

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def score(self, query: str, chunk: Chunk) -> float:
        terms = _terms(query)
        if not terms:
            return 0.0
        freq: dict[str, int] = {}
        for term in _terms(chunk.text):
            freq[term] = freq.get(term, 0) + 1
        length = count_tokens(chunk.text)
        total = 0.0
        for term in set(terms):
            tf = freq.get(term, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * length / self._avg_len)
            total += self._idf(term) * tf * (self.k1 + 1) / denom
        return total

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        scored = [
            (self.score(query, chunk), i, chunk)
            for i, chunk in enumerate(self.chunks)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            ScoredChunk(chunk, score)
            for score, _, chunk in scored[:top_k]
            if score > 0
        ]
