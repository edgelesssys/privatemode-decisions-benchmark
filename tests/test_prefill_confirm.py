"""bench.prefill_confirm on small fixture runs: published arms are compared
on the rows both answered."""

import argparse
import json

import pytest

from bench import prefill_confirm

ROWS = 80


def row(index, right, prompt_tokens=100):
    p = 0.8 if right else 0.2
    return {"index": index, "gold": "yes", "probabilities": {"yes": p, "no": 1 - p},
            "option_mass": 1.0, "prompt_tokens": prompt_tokens, "cached_tokens": 0,
            "latency_s": 0.2}


def write_runs(root):
    """B right on the first 40 rows, R-Q on all but every fourth."""
    for arm, right, tokens in (("B", lambda i: i < 40, 100), ("R-Q", lambda i: i % 4 != 0, 160)):
        for replicate in (1, 2):
            folder = root / f"{arm}-r{replicate}"
            folder.mkdir(parents=True)
            for name in ("alpha", "beta", "gamma"):
                (folder / f"{name}.jsonl").write_text("".join(
                    json.dumps(row(i, right(i), tokens)) + "\n" for i in range(ROWS)))


def write_published(root):
    """Jev answered only rows 0-49 of alpha and beta, all of them right."""
    for name in ("alpha", "beta"):
        folder = root / name
        folder.mkdir(parents=True)
        lines = [{"kind": "meta", "n": 1000, "perturb": "none"}] + [
            {"kind": "row", "arm": "jev", "index": i, "gold": "yes",
             "probabilities": {"yes": 0.9, "no": 0.1}} for i in range(50)]
        (folder / "x-r0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))


@pytest.fixture
def result(tmp_path):
    write_runs(tmp_path / "runs")
    write_published(tmp_path / "published")
    args = argparse.Namespace(runs=str(tmp_path / "runs"), variant="R-Q",
                              published=str(tmp_path / "published"), renamed=None, latency=None,
                              mmlu_extra="", jevbench=None, jevbench_runs=None)
    return prefill_confirm.report(args)


def test_ours_on_all_rows_and_against_jev_on_the_rows_both_answered(result):
    md, data = result
    alpha = data["per"]["alpha"]
    assert alpha["rows"] == ROWS and alpha["Jev rows"] == 50
    assert alpha["B"] == pytest.approx(40 / 80)              # all of our rows
    assert alpha["B@Jev"] == pytest.approx(40 / 50)          # Jev's rows only
    assert alpha["variant@Jev"] == pytest.approx(37 / 50)
    assert alpha["Jev"] == 1.0
    assert "Jev" not in data["per"]["gamma"]
    # The headline line compares on the shared rows: 100 of 160.
    assert "on the rows both answered (100 of 160)" in md
    assert data["products"]["Jev"]["datasets"] == 2
    assert "an arm answers (3 and 2: not the same sets)" in md


def test_wins_against_the_baseline_count_each_dataset(result):
    _, data = result
    assert data["wins"] == 3 and data["losses"] == 0
