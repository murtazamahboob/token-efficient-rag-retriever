from ragretriever.benchmark import BENCHMARK_QUERIES, run_benchmark


def test_benchmark_has_six_questions():
    assert len(BENCHMARK_QUERIES) == 6
    assert all(q["keywords"] for q in BENCHMARK_QUERIES)


def test_benchmark_saves_tokens_and_keeps_recall():
    result = run_benchmark(budget=120)
    assert len(result.rows) == 6
    assert result.total_budgeted < result.total_naive
    assert result.mean_pct_saved > 0
    assert all(r.budgeted_tokens <= 120 for r in result.rows)
    # Budgeted recall must stay close to naive recall.
    assert result.mean_budgeted_recall >= 0.8 * result.mean_naive_recall
    assert result.mean_budgeted_recall >= 0.6


def test_benchmark_report_contains_totals():
    result = run_benchmark(budget=120)
    report = result.report()
    assert str(result.total_naive) in report
    assert str(result.total_budgeted) in report
    assert "120" in report
