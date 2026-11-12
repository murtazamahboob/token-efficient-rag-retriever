import json

import pytest

from ragretriever.cli import main


def test_list_command(capsys):
    assert main(["list"]) == 0
    assert "photosynthesis-basics" in capsys.readouterr().out


def test_search_command(capsys):
    assert main(["search", "photosynthesis sunlight"]) == 0
    assert "photosynthesis" in capsys.readouterr().out


def test_search_json_is_valid(capsys):
    assert main(["search", "photosynthesis sunlight", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data and "citation" in data[0]


def test_ask_command_reports_savings(capsys):
    assert main(["ask", "How does photosynthesis make food for plants?", "--budget", "120"]) == 0
    out = capsys.readouterr().out
    assert "Budgeted" in out
    assert "[photosynthesis" in out


def test_ask_json_respects_budget(capsys):
    assert main(["ask", "How do vaccines train the immune system?", "--budget", "120", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["budgeted_tokens"] <= 120
    assert data["tokens_saved"] >= 0


def test_benchmark_command(capsys):
    assert main(["benchmark", "--budget", "120"]) == 0
    assert "Benchmark" in capsys.readouterr().out


def test_no_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main([])
