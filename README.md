# Token-Efficient RAG Retriever — Retrieve Only the Context an Answer Needs

**Offline BM25 retrieval with a hard token budget, semantic deduplication and citation-preserving truncation — no API key, deterministic.**

Naive RAG dumps the top-k chunks into the prompt and pays for every token, including restated passages and long tails the answer never uses. This retriever spends tokens like a budget: it ranks chunks with BM25, drops near-duplicates, fits the rest under a token budget, and truncates only at sentence boundaries — always keeping the citation, so you can still trace every fact to its source.

## Results (measured, not claimed)

Benchmark on 6 questions (photosynthesis, Python functions, black holes, vaccines, ocean currents, neural networks), budget 120 tokens per question, naive = top-5 chunks with no budget:

| Question | Naive | Budgeted | Saved | Naive recall | Budgeted recall |
|---|---:|---:|---:|---:|---:|
| How does photosynthesis make food for plants? | 217 | 114 | 47.5% | 1.00 | 0.80 |
| How do Python functions return values? | 216 | 116 | 46.3% | 1.00 | 1.00 |
| What happens near a black hole event horizon? | 183 | 97 | 47.0% | 1.00 | 1.00 |
| How do vaccines train the immune system? | 219 | 117 | 46.6% | 0.60 | 0.60 |
| Why do ocean currents move heat around the planet? | 217 | 116 | 46.5% | 1.00 | 1.00 |
| How do neural networks learn from data? | 194 | 120 | 38.1% | 1.00 | 1.00 |
| **Total** | **1246** | **680** | **566 tokens saved, 45.3% mean** | **0.933** | **0.900** |

Meaning retained: **mean keyword recall 0.900 budgeted vs 0.933 naive** — the budgeted context keeps 96% of the naive recall while using 45% fewer tokens. The budget is a guarantee: every assembled context is at or under the budget, verified across all benchmark questions and budgets (30–200 tokens).

Worked example: "How does photosynthesis make food for plants?" goes from **217 → 114 tokens (47.5% saved)**, with one chunk truncated at a sentence boundary and its citation `[photosynthesis-basics:2]` kept.

## Features

- **BM25 retrieval, built from scratch** — classic BM25 (k1=1.5, b=0.75) over a bundled 12-document knowledge base, chunked into whole sentences (max 45 tokens per chunk). Deterministic ranking; ties break by corpus order.
- **Token-budgeted assembly** — take ranked chunks in score order while they fit a hard budget; the assembled context never exceeds it.
- **Semantic deduplication** — chunks that restate an already-selected chunk (content-word Jaccard ≥ 0.7) are dropped, so repeated passages are paid for once. The bundled corpus deliberately contains a verbatim two-sentence passage shared by two photosynthesis documents, and the deduplicator catches it.
- **Citation-preserving truncation** — a chunk that does not fit is cut at a sentence boundary and marked `[truncated]`, with its `[doc-id:chunk]` citation intact. Sources are never silently lost.
- **Benchmark** — naive vs budgeted tokens and keyword recall per question, with a Markdown report and JSON summary as real artifacts.
- **CLI** — `search`, `ask`, `benchmark` and `list` subcommands, with `--json` output for scripting.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Rank chunks for a question
ragretriever search "How do vaccines train the immune system?"

# Assemble a context that fits 120 tokens (JSON shows savings + citations)
ragretriever ask --budget 120 --json "How does photosynthesis make food for plants?"

# Run the benchmark and write outputs/report.md
ragretriever benchmark

# Run the full demo (worked example + benchmark)
python examples/demo.py
```

## Python API

```python
from ragretriever import Retriever, compare, run_benchmark

retriever = Retriever()
ranked = retriever.retrieve("How do neural networks learn from data?")
ranked[0].chunk.citation   # '[neural-networks:1]'
ranked[0].score            # BM25 score

assembled, comparison = compare(
    "How does photosynthesis make food for plants?", budget=120
)
assembled.tokens           # 114 — always <= budget
comparison.pct_saved       # 47.5
assembled.citations        # ['[photosynthesis-basics:1]', ...]
assembled.dropped_duplicates  # citations dropped as restatements

bench = run_benchmark(budget=120)
bench.mean_pct_saved       # 45.3
bench.mean_budgeted_recall # 0.9
```

## How it works

1. **Chunk** — each document is split into whole sentences, grouped into chunks of at most 45 tokens. Every chunk gets a stable citation `[doc-id:chunk-number]`.
2. **Retrieve** — BM25 scores every chunk against the query; the top 5 are the naive context.
3. **Dedup** — a chunk whose content words overlap an already-selected chunk by Jaccard ≥ 0.7 is dropped as a restatement.
4. **Fit the budget** — chunks are taken in score order while they fit. A chunk that does not fit is truncated at a sentence boundary if at least 25 tokens of budget remain, and marked `[truncated]`; otherwise it is skipped.
5. **Measure** — the benchmark compares budgeted vs naive tokens and keyword recall, so the saving is always tied to what the answer still contains.

## Project structure

```
src/ragretriever/
  tokens.py      deterministic token counting, content words, Jaccard
  corpus.py      12 bundled factual documents
  chunking.py    sentence-group chunking with stable citations
  retrieval.py   BM25 retriever
  assemble.py    budgeted assembly, dedup, citation-preserving truncation
  benchmark.py   6-question benchmark (tokens + keyword recall)
  cli.py         search / ask / benchmark / list
examples/demo.py worked example + benchmark, writes outputs/
tests/           36 tests: tokens, chunking, retrieval, assembly, benchmark, CLI
```

## Why this is different from Token Squeezer

Token Squeezer (#6 in this series) compresses the prompt itself — it rewrites wording to use fewer tokens. This project attacks the other side of the same bill: retrieval. It decides *which* context is worth sending at all, so the prompt never carries duplicate or over-long passages in the first place. The two compose: retrieve efficiently, then squeeze what remains.
