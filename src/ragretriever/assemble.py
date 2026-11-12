"""Token-budgeted context assembly.

Takes ranked chunks and fits them under a hard token budget:
1. drop near-duplicate chunks (content-word Jaccard >= threshold),
2. take chunks in score order while they fit,
3. if a chunk does not fit but enough budget remains, truncate it at a
   sentence boundary, keeping its citation so the source is never lost.

The budget is a guarantee: ``assembled.tokens <= budget`` always holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import Chunk
from .retrieval import Retriever, ScoredChunk
from .tokens import content_words, count_tokens, jaccard, split_sentences


@dataclass
class AssembledContext:
    query: str
    budget: int
    chunks: list[Chunk] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)  # citations truncated
    dropped_duplicates: list[str] = field(default_factory=list)
    skipped_over_budget: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(count_tokens(c.cited_text()) for c in self.chunks)

    @property
    def text(self) -> str:
        return "\n".join(c.cited_text() for c in self.chunks)

    @property
    def citations(self) -> list[str]:
        return [c.citation for c in self.chunks]


def _truncate_to_fit(chunk: Chunk, remaining: int) -> Chunk | None:
    """Truncate *chunk* to whole sentences fitting *remaining* tokens.

    Returns None when not even one sentence plus the citation fits.
    The citation is always kept; the chunk keeps its original index.
    """
    citation_tokens = count_tokens(chunk.citation)
    budget_for_text = remaining - citation_tokens - count_tokens("[truncated]")
    if budget_for_text < 8:
        return None
    kept: list[str] = []
    used = 0
    for sentence in split_sentences(chunk.text):
        n = count_tokens(sentence)
        if kept and used + n > budget_for_text:
            break
        if used + n > budget_for_text:
            break
        kept.append(sentence)
        used += n
    if not kept:
        return None
    text = " ".join(kept) + " [truncated]"
    return Chunk(chunk.doc_id, chunk.title, chunk.index, text)


def assemble(
    ranked: list[ScoredChunk],
    budget: int,
    dedup_threshold: float = 0.7,
) -> AssembledContext:
    if budget < 1:
        raise ValueError("budget must be at least 1")
    result = AssembledContext(query="", budget=budget)
    seen_word_sets: list[set[str]] = []
    remaining = budget
    for scored in ranked:
        chunk = scored.chunk
        words = content_words(chunk.text)
        if any(jaccard(words, seen) >= dedup_threshold for seen in seen_word_sets):
            result.dropped_duplicates.append(chunk.citation)
            continue
        cited_tokens = count_tokens(chunk.cited_text())
        if cited_tokens <= remaining:
            result.chunks.append(chunk)
            seen_word_sets.append(words)
            remaining -= cited_tokens
        elif remaining >= 25:
            cut = _truncate_to_fit(chunk, remaining)
            if cut is None:
                result.skipped_over_budget.append(chunk.citation)
                continue
            result.chunks.append(cut)
            result.truncated.append(chunk.citation)
            seen_word_sets.append(words)
            remaining -= count_tokens(cut.cited_text())
        else:
            result.skipped_over_budget.append(chunk.citation)
    return result


@dataclass
class Comparison:
    query: str
    budget: int
    naive_tokens: int
    budgeted_tokens: int
    tokens_saved: int
    pct_saved: float
    naive_citations: list[str]
    budgeted_citations: list[str]

    @property
    def within_budget(self) -> bool:
        return self.budgeted_tokens <= self.budget


def compare(query: str, budget: int = 120, top_k: int = 5, retriever: Retriever | None = None) -> tuple[AssembledContext, Comparison]:
    """Assemble a budgeted context and compare it with naive top-k."""
    retriever = retriever or Retriever()
    ranked = retriever.retrieve(query, top_k=top_k)
    naive_tokens = sum(count_tokens(c.chunk.cited_text()) for c in ranked)
    assembled = assemble(ranked, budget=budget)
    assembled.query = query
    comparison = Comparison(
        query=query,
        budget=budget,
        naive_tokens=naive_tokens,
        budgeted_tokens=assembled.tokens,
        tokens_saved=naive_tokens - assembled.tokens,
        pct_saved=round(100 * (naive_tokens - assembled.tokens) / naive_tokens, 1) if naive_tokens else 0.0,
        naive_citations=[c.chunk.citation for c in ranked],
        budgeted_citations=assembled.citations,
    )
    return assembled, comparison
