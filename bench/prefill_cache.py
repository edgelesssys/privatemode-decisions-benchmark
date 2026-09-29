"""What the prompt layouts cost when several questions share a state.

    python -m bench.prefill_cache --out runs/cache.json [-n 40] [--dataset scotus]

The benchmark asks one question per state, so it can't show how the layouts
differ when a call has several: with the state first, the requests of a call
share the state; with each request's own question first, only the preamble;
with all of the call's questions first (the library's default), the
question block and the state. This asks five questions about each of ``-n``
states (ag_news articles by default; ``--dataset scotus`` for long ones)
with ``mode="staged"``, one call at a time, in all three layouts (rotating
which goes first), and records each call's wall time and the prompt tokens
served from the cache. Privatemode only caches prefixes of about 2,300
tokens, so the difference shows with long states.
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


LAYOUTS = ("state first", "own question first", "all questions first")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("-n", type=int, default=40)
    parser.add_argument("--dataset", default="ag_news", help="where the states come from")
    args = parser.parse_args()
    set_max_in_flight(9)
    engine = SystemOne.from_env("glm-5.3-flash", temperature=1.0)
    calls = {"state first": lambda s: engine.system_one(s, QUESTIONS, mode="staged", question_first=False),
             "own question first": lambda s: engine.system_one(s, QUESTIONS, mode="staged", question_first="own"),
             "all questions first": lambda s: engine.system_one(s, QUESTIONS, mode="staged", question_first=True)}
    rows = []
    tasks = load(args.dataset, 1000)
    for number, task in enumerate(tasks[len(tasks) - args.n:]):     # rows the other runs used least
        # Rotate which layout goes first, so none always follows a warm cache.
        order = LAYOUTS[number % 3:] + LAYOUTS[:number % 3]
        for layout in order:
            response = calls[layout](task.state)
            rows.append({"index": task.index, "layout": layout,
                         "wall_s": response.timings["wall"],
                         "input_tokens": response.usage.input_tokens,
                         "cached_tokens": response.usage.cached_tokens})
    summary = {}
    for layout in LAYOUTS:
        r = [x for x in rows if x["layout"] == layout]
        summary[layout] = {
            "calls": len(r), "wall_p50_s": float(np.median([x["wall_s"] for x in r])),
            "wall_p95_s": float(np.percentile([x["wall_s"] for x in r], 95)),
            "input_tokens": float(np.mean([x["input_tokens"] for x in r])),
            "cached_share": float(np.sum([x["cached_tokens"] for x in r]) / np.sum([x["input_tokens"] for x in r]))}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
