"""Token-Efficient RAG Retriever — retrieve only the context an answer needs.

Offline BM25 retrieval with token-budgeted context assembly, semantic
deduplication and citation-preserving truncation.
"""

from .assemble import AssembledContext, Comparison, assemble, compare
from .benchmark import BenchmarkResult, run_benchmark
from .chunking import Chunk, build_chunks, chunk_document
from .corpus import DOCUMENTS, Document
from .retrieval import Retriever, ScoredChunk
from .tokens import count_tokens

__version__ = "1.0.0"
__all__ = [
    "AssembledContext",
    "BenchmarkResult",
    "Chunk",
    "Comparison",
    "DOCUMENTS",
    "Document",
    "Retriever",
    "ScoredChunk",
    "assemble",
    "build_chunks",
    "chunk_document",
    "compare",
    "count_tokens",
    "run_benchmark",
]
