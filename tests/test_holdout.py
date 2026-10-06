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


def test_texts_survive_export_and_fetch(tmp_path, monkeypatch):
    """The reproduce path on a small task: the released task through a
    file:// release, the other by id through its refetcher; a text edited
    since the freeze fails check and is left out of a run."""
    frozen, cache = tmp_path / "frozen", tmp_path / "cache"
    frozen.mkdir()
    texts = {"shipped": {"arxiv:1": "an abstract", "pmc:PMC2": "another abstract"},
             "refetched": {"github:o/r#1": "an issue", "github:o/r#2": "another issue"}}
    licences = {"arxiv:1": "http://creativecommons.org/licenses/by/4.0/", "pmc:PMC2": "cc by"}
    for name, rows in texts.items():
        (frozen / f"{name}.json").write_text(json.dumps({
            "task": name, "question": "Which?", "options": ["a", "b"],
            "examples": [{"index": i, "id": k, "label": "a", "sha256": holdout_data.sha(t),
                          **({"license": licences[k]} if k in licences else {})}
                         for i, (k, t) in enumerate(rows.items())]}))
    monkeypatch.setattr(holdout_data, "FROZEN", frozen)
    monkeypatch.setattr(holdout_data, "CACHE", cache)
    monkeypatch.setattr(holdout_data, "TASKS", {"shipped": ("Which?", ["a", "b"]),
                                                "refetched": ("Which?", ["a", "b"])})
    monkeypatch.setattr(holdout_data, "SHIPPED", ("shipped",))
    cache.mkdir()
    (cache / "shipped.jsonl").write_text("".join(
        json.dumps({"id": k, "text": t}) + "\n" for k, t in texts["shipped"].items()))
    release = tmp_path / "holdout-texts.jsonl"
    holdout_data.export(release)
    assert {json.loads(line)["license"] for line in release.read_text().splitlines()} == {
        "CC BY 4.0", "CC BY"}
    edited = dict(texts["refetched"], **{"github:o/r#2": "edited since"})
    monkeypatch.setattr(holdout_data, "REFETCH", {"refetched": lambda name, ids: edited})
    for path in cache.iterdir():
        path.unlink()
    holdout_data.fetch(release.as_uri())
    assert [t.state for t in holdout_data.load("shipped")] == list(texts["shipped"].values())
    with pytest.raises(SystemExit, match="changed"):
        holdout_data.load("refetched")
    assert [t.state for t in holdout_data.load("refetched", strict=False)] == ["an issue"]


def test_library_version_names_the_commit(monkeypatch, tmp_path):
    from bench import adapters

    class Dist:
        version = "0.1.0"

        def __init__(self, direct_url):
            self.direct_url = direct_url

        def read_text(self, name):
            return json.dumps(self.direct_url)

    def installed(direct_url):
        monkeypatch.setattr(adapters.metadata, "distribution", lambda _: Dist(direct_url))
        return adapters.library_version()

    assert installed({"url": "https://github.com/x/y", "vcs_info": {"commit_id": "abc"}}) == "0.1.0+abc"
    plain = tmp_path / "with space"                        # not a git checkout: version only
    plain.mkdir()
    assert installed({"url": plain.as_uri(), "dir_info": {"editable": True}}) == "0.1.0"


def test_refetching_issues_tells_deleted_from_failed(monkeypatch):
    """A 404 is a deleted issue, left out; a rate limit is retried and then
    raised, not taken for a deletion (a run would ask a silent subset)."""
    import subprocess

    def gh(path):
        if path.endswith("/1"):
            return {"title": "An issue", "body": "text"}
        stderr = "HTTP 404: Not Found" if path.endswith("/2") else "HTTP 403: API rate limit exceeded"
        raise subprocess.CalledProcessError(1, "gh", stderr=stderr)
    monkeypatch.setattr(holdout_data, "gh", gh)
    monkeypatch.setattr(holdout_data.time, "sleep", lambda s: None)
    assert set(holdout_data.refetch_github("github_issue", ["github:o/r#1", "github:o/r#2"])) == {
        "github:o/r#1"}
    with pytest.raises(RuntimeError, match="rate limit"):
        holdout_data.refetch_github("github_issue", ["github:o/r#3"])


def test_check_tolerates_some_drift_but_not_an_empty_cache(tmp_path, monkeypatch):
    frozen, cache = tmp_path / "frozen", tmp_path / "cache"
    frozen.mkdir()
    texts = {f"github:o/r#{i}": f"issue {i}" for i in range(20)}
    (frozen / "github_issue.json").write_text(json.dumps({
        "task": "github_issue", "question": "Which?", "options": ["a", "b"],
        "examples": [{"index": i, "id": k, "label": "a", "sha256": holdout_data.sha(t)}
                     for i, (k, t) in enumerate(texts.items())]}))
    monkeypatch.setattr(holdout_data, "FROZEN", frozen)
    monkeypatch.setattr(holdout_data, "CACHE", cache)
    with pytest.raises(SystemExit, match="no texts cached"):
        holdout_data.verify("github_issue")
    cache.mkdir()

    def cached(n_edited):
        (cache / "github_issue.jsonl").write_text("".join(
            json.dumps({"id": k, "text": "edited" if i < n_edited else t}) + "\n"
            for i, (k, t) in enumerate(texts.items())))
    cached(2)                                    # 18 of 20 match: at the floor
    assert holdout_data.verify("github_issue") == 18
    cached(3)                                    # 17 of 20: below it
    with pytest.raises(SystemExit, match="only 17 of 20"):
        holdout_data.verify("github_issue")
    (cache / "github_issue.jsonl").write_text("")   # a total refetch loss
    with pytest.raises(SystemExit, match="only 0 of 20"):
        holdout_data.verify("github_issue")


def test_the_share_of_gain_is_on_plain_ece_like_part_one():
    # ECE and excess ECE disagree here on purpose: on ECE the method covers
    # half of the task T's reduction, on excess ECE all of it.
    def task():
        return {"zero_label": {
            "raw": {"ece": 0.20, "excess_ece": 0.10},
            "method": {"ece": 0.15, "excess_ece": 0.00},
            "task T (calibration half)": {"ece": 0.10, "excess_ece": 0.00}}}
    assert holdout.share_of_gain({"a": task(), "b": task()}, "method") == pytest.approx(0.5)
