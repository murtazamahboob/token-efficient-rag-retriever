"""Demo: naive top-k retrieval vs token-budgeted retrieval.

Prints one worked question in full, then the 6-question benchmark
table. Writes outputs/report.md and outputs/summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragretriever import compare, run_benchmark  # noqa: E402

QUESTION = "How does photosynthesis make food for plants?"
BUDGET = 120


def main() -> None:
    assembled, comparison = compare(QUESTION, budget=BUDGET, top_k=5)
    print(f"Question: {QUESTION}")
    print(f"Naive top-5 context: {comparison.naive_tokens} tokens")
    print(f"Budgeted context (budget {BUDGET}): {comparison.budgeted_tokens} tokens")
    print(f"Saved: {comparison.tokens_saved} tokens ({comparison.pct_saved}%)")
    if assembled.dropped_duplicates:
        print(f"Dropped duplicates: {', '.join(assembled.dropped_duplicates)}")
    if assembled.truncated:
        print(f"Truncated (citation kept): {', '.join(assembled.truncated)}")
    print()
    print(assembled.text)
    print()
    print("=" * 70)
    print()
    result = run_benchmark(budget=BUDGET)
    print(result.report())

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "report.md").write_text(result.report(), encoding="utf-8")
    summary = {
        "budget": BUDGET,
        "total_naive_tokens": result.total_naive,
        "total_budgeted_tokens": result.total_budgeted,
        "tokens_saved": result.total_naive - result.total_budgeted,
        "mean_pct_saved": result.mean_pct_saved,
        "mean_naive_recall": result.mean_naive_recall,
        "mean_budgeted_recall": result.mean_budgeted_recall,
        "rows": [
            {
                "query": r.query,
                "naive_tokens": r.naive_tokens,
                "budgeted_tokens": r.budgeted_tokens,
                "pct_saved": r.pct_saved,
                "naive_recall": r.naive_recall,
                "budgeted_recall": r.budgeted_recall,
            }
            for r in result.rows
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote outputs/report.md and outputs/summary.json")


if __name__ == "__main__":
    main()
