"""The held-out test: the frozen tasks, and the pass/fail of its criteria."""

import json

import pytest

from bench import holdout, holdout_data
from bench.holdout import FAMILY, LABELS, verdict
from bench.holdout_data import FROZEN, TASKS


@pytest.mark.parametrize("name", sorted(TASKS))
def test_frozen_tasks_match_their_definitions(name):
    question, options = TASKS[name]
    frozen = json.loads((FROZEN / f"{name}.json").read_text())
    assert frozen["question"] == question and frozen["options"] == options
    examples = frozen["examples"]
    assert 0 < len(examples) <= 1000
    assert len({e["id"] for e in examples}) == len(examples)
    assert all(e["label"] in options for e in examples)
    assert frozen["counts"] == {o: sum(e["label"] == o for e in examples) for o in options}
    assert [e["index"] for e in examples] == list(range(len(examples)))


def test_every_task_has_a_family():
    assert set(FAMILY) == set(TASKS)


def test_build_refuses_to_replace_a_frozen_task(monkeypatch):
    monkeypatch.setattr("sys.argv", ["holdout_data", "build", "--only", "fin_topic"])
    monkeypatch.setitem(holdout_data.BUILDERS, "fin_topic", lambda: pytest.fail("fetched anyway"))
    with pytest.raises(SystemExit, match="frozen"):
        holdout_data.main()


def result(excess=0.05, coverage=0.9, violations=0, given=0.78, biased=0.80) -> dict:
    """One dataset's entry, as ``evaluate`` writes it, with the measured values."""
    fit = {"accuracy": biased, "draws": 50, "violations": violations}
    return {"zero_label": {"option formula (default)": {"excess_ece": excess},
                           "raw": {"accuracy": given}},
            "coverage_check": {"coverage": coverage},
            "labelled": {f"T + bias, {LABELS} labels": fit, "T + bias, all labels": fit}}


def passes(every=None, last=None) -> list[bool]:
    """The verdict on five tasks: ``every`` changes all of them, ``last`` one."""
    tasks = {f"t{i}": result(**(every or {})) for i in range(4)}
    tasks["t4"] = result(**((every or {}) | (last or {})))
    return [ok for *_, ok in verdict(tasks)]


def test_the_criteria_pass_on_values_inside_every_bound():
    assert all(passes())


@pytest.mark.parametrize("every, last, failing", [
    ({"excess": 0.07}, None, [0]),            # the mean crosses 0.06
    ({"excess": 0.03}, {"excess": 0.13}, [1]),  # one dataset above 0.12, the mean still fine
    ({"coverage": 0.85}, None, [2]),          # mean coverage below 0.87
    (None, {"violations": 30}, [3, 4]),       # 30 of 250 draws over the bound: 12%
    ({"biased": 0.77}, None, [5]),            # the bias loses accuracy on average
])
def test_each_criterion_fails_on_its_own(every, last, failing):
    result_ = passes(every, last)
    assert [i for i, ok in enumerate(result_) if not ok] == failing


def test_the_report_needs_a_recorded_served_model(tmp_path, monkeypatch):
    class Run:
        meta = {}
    monkeypatch.setattr(holdout.c, "load_run", lambda _: {name: Run() for name in TASKS})
    with pytest.raises(SystemExit, match="no served model"):
        holdout.report(tmp_path, tmp_path)
