"""The prompt arms of bench.prefill, without a model."""

import json

import pytest

from bench.prefill import PREAMBLE, PREFIX, Prefill, jevbench_tasks
from decisions import Choice


class Offline:
    """Stands in for the client; the arms only build payloads."""

    def post(self, path, payload):  # pragma: no cover - never called
        raise AssertionError("no requests in these tests")


def engine():
    return Prefill(Offline(), "glm-5.3-flash", temperature=1.0,
                   fillers={"dots": ". . .", "alpha": "a b c d", "words": "one two", "count": "1 2"},
                   generic="Let me read the state carefully.")


QUESTION = Choice({"yes": "it applies", "no": None}, instructions="Does it apply?")


def payload(arm, **kwargs):
    return engine().payload(arm, {"text": "state"}, QUESTION, [11, 12], [11, 12], 3, **kwargs)


@pytest.mark.parametrize("arm", ["B", "R-Q", "R-Qi", "R-QSQS", "R-full", "R-think", "F-dots",
                                 "F-alpha", "F-words", "F-scrambled", "F-count", "F-before", "G",
                                 "RQ-F", "RQ-mid"])
def test_every_arm_ends_in_the_prefill_and_keeps_the_mask(arm):
    body = payload(arm)
    assert body["messages"][1]["content"].endswith(PREFIX)
    assert body["allowed_token_ids"] == [11, 12]
    assert body["max_tokens"] == 1


def test_think_arms_fill_the_think_block_and_the_rest_leave_it_to_the_template():
    assert payload("B")["messages"][1]["content"] == PREFIX
    assert payload("F-dots")["messages"][1]["content"] == f"<think>. . .</think>{PREFIX}"
    assert payload("H-32", thought="hmm")["messages"][1]["content"] == f"<think>hmm</think>{PREFIX}"


def test_question_first_puts_the_question_and_options_before_the_state():
    user = payload("R-Q")["messages"][0]["content"]
    question = json.dumps(Prefill._question(QUESTION), ensure_ascii=False)
    assert user.startswith(PREAMBLE + question + "\n")
    assert user.count('"options"') == 2
    assert payload("R-Qi")["messages"][0]["content"].count('"options"') == 1


def test_question_first_matches_the_library():
    library = engine()
    if not hasattr(library, "question_first"):
        pytest.skip("installed privatemode-decisions has no question_first")
    ours = payload("R-Q")["messages"][0]["content"]
    assert library._content({"text": "state"}, QUESTION, question_first=True) == ours


def test_jevbench_items_map_as_its_own_adapter_does(tmp_path):
    public = tmp_path / "datasets" / "public"
    public.mkdir(parents=True)
    items = {
        "original": [{"state": "s", "labels": ["no", "yes"], "expected": "no",
                      "question": {"type": "noul", "instructions": "ok?",
                                   "criteria": {"true": "fine", "false": "not fine"}}}],
        "easy": [{"state": {"k": 1}, "labels": ["a", "b"], "expected": "b",
                  "question": {"type": "choice", "instructions": "which?",
                               "criteria": {"a": "first", "b": None}}}],
        "hard": [{"state": "s", "labels": ["0", "1", "2"], "expected": 2,
                  "question": {"type": "score", "instructions": "how bad?",
                               "criteria": ["none", "some", "a lot"]}}],
    }
    for tier, rows in items.items():
        (public / f"{tier}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    noul, choice, score = jevbench_tasks(tmp_path)
    assert noul.criteria == {"no": "not fine", "yes": "fine"} and noul.gold == "no"
    assert choice.criteria == {"a": "first", "b": None} and choice.state == {"k": 1}
    assert score.criteria == {"0": "none", "1": "some", "2": "a lot"} and score.gold == "2"
    assert [t.index for t in (noul, choice, score)] == [0, 1, 2]


def test_filler_combined_with_the_question_first():
    question = json.dumps(Prefill._question(QUESTION), ensure_ascii=False)
    with_think = payload("RQ-F")
    assert with_think["messages"][0]["content"] == payload("R-Q")["messages"][0]["content"]
    assert with_think["messages"][1]["content"] == f"<think>. . .</think>{PREFIX}"
    middle = payload("RQ-mid")["messages"][0]["content"]
    assert middle == (PREAMBLE + question + "\n" + json.dumps({"state": {"text": "state"}}) + "\n"
                      + ". . ." + "\n" + question)
