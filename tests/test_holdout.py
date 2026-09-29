"""The frozen held-out tasks are what the plan says they are."""

import json

from bench.holdout import FAMILY
from bench.holdout_data import FROZEN, TASKS


def test_frozen_tasks_match_their_definitions():
    for name, (question, options) in TASKS.items():
        path = FROZEN / f"{name}.json"
        if not path.exists():
            continue
        frozen = json.loads(path.read_text())
        assert frozen["question"] == question and frozen["options"] == options
        examples = frozen["examples"]
        assert 0 < len(examples) <= 1000
        assert len({e["id"] for e in examples}) == len(examples)
        assert all(e["label"] in options for e in examples)
        assert sum(frozen["counts"].values()) == len(examples)
        assert [e["index"] for e in examples] == list(range(len(examples)))


def test_every_task_has_a_family():
    assert set(FAMILY) == set(TASKS)
