"""Command-line interface: search, ask, benchmark, list."""
from __future__ import annotations

import argparse
import json
import sys

from .assemble import compare
from .benchmark import run_benchmark
from .chunking import build_chunks
from .corpus import DOCUMENTS
from .retrieval import Retriever
from .tokens import count_tokens


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragretriever",
        description="Token-efficient RAG: BM25 retrieval with a hard token budget.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="Show ranked chunks for a query.")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("ask", help="Assemble a budgeted context for a query.")
    p.add_argument("query")
    p.add_argument("--budget", type=int, default=120)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("benchmark", help="Run the 6-question benchmark.")
    p.add_argument("--budget", type=int, default=120)
    p.add_argument("--output", default=None, help="Write the Markdown report here.")

    sub.add_parser("list", help="List the bundled documents.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "list":
        for doc in DOCUMENTS:
            print(f"{doc.doc_id} — {doc.title} ({count_tokens(doc.text)} tokens)")
        return 0
    if args.command == "search":
        retriever = Retriever()
        ranked = retriever.retrieve(args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps([
                {"citation": c.chunk.citation, "score": round(c.score, 3),
                 "tokens": c.tokens, "text": c.chunk.text}
                for c in ranked
            ], indent=2))
        else:
            for c in ranked:
                print(f"{c.chunk.citation} score={c.score:.2f} tokens={c.tokens}")
                print(f"  {c.chunk.text}")
        return 0
    if args.command == "ask":
        assembled, comparison = compare(args.query, budget=args.budget, top_k=args.top_k)
        if args.json:
            print(json.dumps({
                "query": args.query, "budget": args.budget,
                "naive_tokens": comparison.naive_tokens,
                "budgeted_tokens": comparison.budgeted_tokens,
                "tokens_saved": comparison.tokens_saved,
                "pct_saved": comparison.pct_saved,
                "citations": assembled.citations,
                "context": assembled.text,
                "dropped_duplicates": assembled.dropped_duplicates,
                "truncated": assembled.truncated,
            }, indent=2))
        else:
            print(f"Naive top-{args.top_k}: {comparison.naive_tokens} tokens")
            print(f"Budgeted (budget {args.budget}): {comparison.budgeted_tokens} tokens "
                  f"— saved {comparison.tokens_saved} ({comparison.pct_saved}%)")
            if assembled.dropped_duplicates:
                print(f"Dropped duplicates: {', '.join(assembled.dropped_duplicates)}")
            if assembled.truncated:
                print(f"Truncated (citation kept): {', '.join(assembled.truncated)}")
            print()
            print(assembled.text)
        return 0
    if args.command == "benchmark":
        result = run_benchmark(budget=args.budget)
        print(result.report())
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result.report())
            print(f"Wrote {args.output}", file=sys.stderr)
        return 0
    return 2
