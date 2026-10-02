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
    """R-Q is the library's default prompt, byte for byte."""
    ours = payload("R-Q")["messages"][0]["content"]
    lead = Prefill._question_text(QUESTION) + "\n"
    assert engine()._content({"text": "state"}, QUESTION, lead) == ours


def test_the_baseline_is_the_state_first_prompt_of_the_suite():
    whole = json.dumps({"state": {"text": "state"}, **Prefill._question(QUESTION)}, ensure_ascii=False)
    assert payload("B")["messages"][0]["content"] == PREAMBLE + whole


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


def test_all_questions_first_matches_the_library():
    from bench.prefill import call_questions

    questions, _ = call_questions(QUESTION, 3)
    block = "".join(Prefill._question_text(q) + "\n" for q in questions)
    ours = payload("QA")["messages"][0]["content"]
    assert engine()._content({"text": "state"}, QUESTION, block) == ours


class FakeTokens:
    """One token per character: enough to size the padded thoughts."""

    def count(self, text):
        return len(text)

    def units_for(self, kind, tokens):
        return max(1, tokens // 2)


def test_padded_thoughts_restate_the_question_unless_it_already_came_first():
    padded = Prefill(Offline(), "glm-5.3-flash", temperature=1.0, fillers={"dots": ". ."},
                     generic=None, tokens=FakeTokens(), length=120)
    rf = padded.payload("RF", {"text": "state"}, QUESTION, [11, 12], [11, 12], 3)
    think = rf["messages"][1]["content"]
    assert think.startswith("<think>Let me think. The question: Does it apply?")
    assert think.endswith(f"Ok, now let me answer.</think>{PREFIX}")
    ff = padded.payload("RQ-FF", {"text": "state"}, QUESTION, [11, 12], [11, 12], 3)
    assert "The question:" not in ff["messages"][1]["content"]
    assert ff["messages"][0]["content"] == payload("R-Q")["messages"][0]["content"]


def test_unknown_arms_stop_the_run():
    from bench.prefill import check_arm

    assert check_arm("R-Q") == "R-Q" and check_arm("H-1024") == "H-1024"
    for typo in ("R-q", "H-", "H-0", "RQF"):
        with pytest.raises(SystemExit, match="unknown arm"):
            check_arm(typo)


def test_mcnemar_is_exact_and_two_sided():
    import numpy as np

    from bench.prefill_report import mcnemar

    same = np.array([True, False, True])
    assert mcnemar(same, same) == 1.0
    a, b = np.array([True] * 10 + [False] * 5), np.array([False] * 10 + [False] * 5)
    assert mcnemar(a, b) == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar(b, a) == mcnemar(a, b)


def test_the_run_identity_records_the_layout_and_library():
    from bench import run as run_module

    class Arm:
        name, model, temperature, optimize = "privatemode", "glm-5.3-flash", 1.0, "accuracy"

    class Other:
        name, model = "jev", "jev-latest"

    args = run_module.build_parser().parse_args(["--dataset", "sst2"])
    with_glm = run_module.identity(args, [Arm(), Other()])
    assert with_glm["optimize"] == "accuracy" and with_glm["library"]
    assert with_glm["prefill"] == PREFIX
    without = run_module.identity(args, [Other()])
    assert not {"optimize", "library", "prefill"} & set(without)
