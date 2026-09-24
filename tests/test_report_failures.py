"""An arm's own failure to answer counts against it; an outage does not."""

from bench.report import paired


def row(index, arm, **extra):
    return {"kind": "row", "arm": arm, "index": index, "gold": "a",
            "choice": "a", "correct": True, "probabilities": {"a": 1.0},
            "confidence": 1.0, "latency_s": 0.1, "input_tokens": 10,
            "output_tokens": 1, "cached_tokens": 0, **extra}


def test_budget_exhaustion_is_a_wrong_answer_not_a_dropped_example():
    raw = {
        "glm-cot": [row(0, "glm-cot"),
                    {"kind": "row", "arm": "glm-cot", "index": 1, "gold": "a",
                     "error": "RuntimeError: ran out of tokens while reasoning (900 characters of it)"}],
        "embed-nn": [row(0, "embed-nn"), row(1, "embed-nn")],
    }
    arms, rows = paired(raw, {"identity": {"cot_max_tokens": 4000}})
    assert len(rows["glm-cot"]) == 2 and len(rows["embed-nn"]) == 2
    failed = rows["glm-cot"][1]
    assert failed["correct"] is False and failed["latency_s"] is None
    assert failed["output_tokens"] == 4000


def test_an_outage_drops_the_example_for_every_arm():
    raw = {
        "glm-cot": [row(0, "glm-cot"),
                    {"kind": "row", "arm": "glm-cot", "index": 1, "gold": "a",
                     "error": "APIError: HTTP 502: bad gateway"}],
        "embed-nn": [row(0, "embed-nn"), row(1, "embed-nn")],
    }
    arms, rows = paired(raw, {})
    assert [r["index"] for r in rows["embed-nn"]] == [0]
