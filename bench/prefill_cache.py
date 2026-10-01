"""What the prompt layouts cost when several questions share a state.

    python -m bench.prefill_cache --out runs/cache-ag.json -n 40
    python -m bench.prefill_cache --out runs/cache-scotus.json -n 30 --dataset scotus
    python -m bench.prefill_cache --combine runs/cache-ag.json runs/cache-scotus.json \
        --out results/prefill/cache-multi-question.json

The benchmark asks one question per state, so it can't show how the layouts
differ when a call has several: with each request's own question first
(``optimize="accuracy"``, the library's default) the requests of a call
share only the preamble; with all of the call's questions first
(``"cost"``), the question block and the state. This asks five questions
about each of ``-n`` states (ag_news articles by default; ``--dataset
scotus`` for long ones) with ``mode="staged"``, one call at a time, in both
layouts (alternating which goes first), and records each call's wall time
and the prompt tokens served from the cache. Privatemode only caches
prefixes of about 2,300 tokens, so the difference shows with long states.
``--combine`` puts the summaries of several runs into one file, keyed by
what they measured.
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


LAYOUTS = ("accuracy", "cost")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("-n", type=int, default=40)
    parser.add_argument("--dataset", default="ag_news", help="where the states come from")
    parser.add_argument("--combine", nargs="+", help="runs of this tool to put into --out")
    args = parser.parse_args()
    if args.combine:
        combined = {}
        for path in args.combine:
            run = json.loads(Path(path).read_text())
            combined[f"{run['dataset']}, {run['calls']} calls"] = run["summary"]
        Path(args.out).write_text(json.dumps(combined, indent=1))
        return
    set_max_in_flight(9)
    engine = SystemOne.from_env("glm-5.3-flash", temperature=1.0)
    rows = []
    tasks = load(args.dataset, 1000)
    for number, task in enumerate(tasks[len(tasks) - args.n:]):     # rows the other runs used least
        # Alternate which layout goes first, so neither always follows a warm cache.
        order = LAYOUTS[number % 2:] + LAYOUTS[:number % 2]
        for layout in order:
            response = engine.system_one(task.state, QUESTIONS, mode="staged", optimize=layout)
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
    Path(args.out).write_text(json.dumps({"dataset": args.dataset, "calls": args.n,
                                          "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
