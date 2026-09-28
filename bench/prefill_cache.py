"""What asking the question first costs when several questions share a state.

    python -m bench.prefill_cache --out runs/cache.json [-n 40]

The benchmark asks one question per state, so it can't show the cost of
``question_first``: the state is no longer the prefix that the questions of
one call share, so vLLM's prefix cache can reuse only the preamble. This asks
five questions about each of ``-n`` states (ag_news articles by default;
``--dataset scotus`` for long ones, where sharing the state matters) with ``mode="staged"``,
one call at a time, with and without ``question_first`` (alternating, so
both see the same server conditions), and records the wall time of each
call and the prompt tokens served from the cache. Needs the library with
``question_first`` (privatemode-decisions, branch ``accuracy`` or later).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decisions import Choice, SystemOne
from decisions.client import set_max_in_flight

from .datasets import load

QUESTIONS = {
    "topic": Choice({"World": None, "Sports": None, "Business": None, "Sci/Tech": None},
                    instructions="Which topic is the article about?"),
    "tone": Choice({"positive": None, "neutral": None, "negative": None},
                   instructions="What is the tone of the article?"),
    "company": Choice({"yes": "a company is named", "no": "no company is named"},
                      instructions="Does the article name a company?"),
    "region": Choice({"Europe": None, "North America": None, "Asia": None, "elsewhere or unclear": None},
                     instructions="Where does the story take place?"),
    "urgency": Choice({"breaking": None, "developing": None, "background": None},
                      instructions="How time-sensitive is the story?"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("-n", type=int, default=40)
    parser.add_argument("--dataset", default="ag_news", help="where the states come from")
    args = parser.parse_args()
    set_max_in_flight(9)
    engine = SystemOne.from_env("glm-5.3-flash", temperature=1.0)
    rows = []
    tasks = load(args.dataset, 1000)
    for task in tasks[len(tasks) - args.n:]:     # rows the other runs used least
        for first in ((False, True) if task.index % 2 else (True, False)):
            response = engine.system_one(task.state, QUESTIONS, mode="staged", question_first=first)
            rows.append({"index": task.index, "question_first": first,
                         "wall_s": response.timings["wall"],
                         "input_tokens": response.usage.input_tokens,
                         "cached_tokens": response.usage.cached_tokens})
    summary = {}
    for first in (False, True):
        r = [x for x in rows if x["question_first"] == first]
        summary["question first" if first else "state first (default)"] = {
            "calls": len(r), "wall_p50_s": float(np.median([x["wall_s"] for x in r])),
            "wall_p95_s": float(np.percentile([x["wall_s"] for x in r], 95)),
            "input_tokens": float(np.mean([x["input_tokens"] for x in r])),
            "cached_share": float(np.sum([x["cached_tokens"] for x in r]) / np.sum([x["input_tokens"] for x in r]))}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
