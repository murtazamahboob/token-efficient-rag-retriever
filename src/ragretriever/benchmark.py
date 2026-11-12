"""Benchmark: budgeted retrieval vs naive top-k retrieval.

Six questions with expected keywords. For each one, compare the naive
top-5 context with the token-budgeted context: tokens used, tokens
saved, and keyword recall (how many expected keywords survive in the
assembled context). A good budgeted retriever saves a lot of tokens
while keeping recall close to naive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .assemble import compare
from .retrieval import Retriever
from .tokens import content_words

BENCHMARK_QUERIES: list[dict] = [
    {
        "query": "How does photosynthesis make food for plants?",
        "keywords": ["sunlight", "chlorophyll", "carbon", "glucose", "oxygen"],
    },
    {
        "query": "How do Python functions return values?",
        "keywords": ["def", "return", "parameters", "arguments", "call"],
    },
    {
        "query": "What happens near a black hole event horizon?",
        "keywords": ["gravity", "light", "escape", "singularity", "time"],
    },
    {
        "query": "How do vaccines train the immune system?",
        "keywords": ["immune", "antibodies", "memory", "pathogen", "protection"],
    },
    {
        "query": "Why do ocean currents move heat around the planet?",
        "keywords": ["wind", "temperature", "density", "currents", "heat"],
    },
    {
        "query": "How do neural networks learn from data?",
        "keywords": ["weights", "training", "gradient", "error", "layers"],
    },
]


@dataclass
class BenchmarkRow:
    query: str
    naive_tokens: int
    budgeted_tokens: int
    pct_saved: float
    naive_recall: float
    budgeted_recall: float


@dataclass
class BenchmarkResult:
    rows: list[BenchmarkRow] = field(default_factory=list)
    budget: int = 120

    @property
    def total_naive(self) -> int:
        return sum(r.naive_tokens for r in self.rows)

    @property
    def total_budgeted(self) -> int:
        return sum(r.budgeted_tokens for r in self.rows)

    @property
    def mean_pct_saved(self) -> float:
        return round(sum(r.pct_saved for r in self.rows) / len(self.rows), 1) if self.rows else 0.0

    @property
    def mean_budgeted_recall(self) -> float:
        return round(sum(r.budgeted_recall for r in self.rows) / len(self.rows), 3) if self.rows else 0.0

    @property
    def mean_naive_recall(self) -> float:
        return round(sum(r.naive_recall for r in self.rows) / len(self.rows), 3) if self.rows else 0.0

    def report(self) -> str:
        lines = [
            "# Token-Efficient RAG — Benchmark",
            "",
            f"Budget: {self.budget} tokens per question. Naive: top-5 chunks, no budget.",
            "",
            "| Question | Naive | Budgeted | Saved | Naive recall | Budgeted recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in self.rows:
            lines.append(
                f"| {row.query} | {row.naive_tokens} | {row.budgeted_tokens} | "
                f"{row.pct_saved}% | {row.naive_recall:.2f} | {row.budgeted_recall:.2f} |"
            )
        lines += [
            "",
            f"**Total: {self.total_naive} -> {self.total_budgeted} tokens "
            f"({self.total_naive - self.total_budgeted} saved, {self.mean_pct_saved}% mean). "
            f"Mean keyword recall: naive {self.mean_naive_recall}, budgeted {self.mean_budgeted_recall}.**",
            "",
        ]
        return "\n".join(lines)


def _recall(text: str, keywords: list[str]) -> float:
    words = content_words(text) | {w.lower() for w in text.lower().split()}
    # keywords are single words; match them as substrings of the raw text
    # as well, so "carbon" matches "carbon dioxide" and "def" matches "def".
    lowered = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lowered)
    return round(hits / len(keywords), 3) if keywords else 0.0


def run_benchmark(budget: int = 120, top_k: int = 5) -> BenchmarkResult:
    retriever = Retriever()
    result = BenchmarkResult(budget=budget)
    for item in BENCHMARK_QUERIES:
        assembled, comparison = compare(item["query"], budget=budget, top_k=top_k, retriever=retriever)
        naive_text = "\n".join(
            c.chunk.cited_text() for c in retriever.retrieve(item["query"], top_k=top_k)
        )
        result.rows.append(
            BenchmarkRow(
                query=item["query"],
                naive_tokens=comparison.naive_tokens,
                budgeted_tokens=comparison.budgeted_tokens,
                pct_saved=comparison.pct_saved,
                naive_recall=_recall(naive_text, item["keywords"]),
                budgeted_recall=_recall(assembled.text, item["keywords"]),
            )
        )
    return result
