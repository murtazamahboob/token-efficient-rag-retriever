"""Split documents into citable chunks.

Chunks are groups of whole sentences, at most ``max_tokens`` each, so a
chunk never cuts a sentence in half and every chunk carries a stable
citation of the form ``[doc-id:chunk-number]``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .corpus import DOCUMENTS, Document
from .tokens import count_tokens, split_sentences


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    index: int  # 1-based chunk number within its document
    text: str

    @property
    def citation(self) -> str:
        return f"[{self.doc_id}:{self.index}]"

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)

    def cited_text(self) -> str:
        return f"{self.citation} {self.text}"


def chunk_document(doc: Document, max_tokens: int = 45) -> list[Chunk]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in split_sentences(doc.text):
        n = count_tokens(sentence)
        if current and current_tokens + n > max_tokens:
            chunks.append(Chunk(doc.doc_id, doc.title, len(chunks) + 1, " ".join(current)))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += n
    if current:
        chunks.append(Chunk(doc.doc_id, doc.title, len(chunks) + 1, " ".join(current)))
    return chunks


def build_chunks(max_tokens: int = 45) -> list[Chunk]:
    """Chunk every bundled document, in corpus order."""
    out: list[Chunk] = []
    for doc in DOCUMENTS:
        out.extend(chunk_document(doc, max_tokens=max_tokens))
    return out
