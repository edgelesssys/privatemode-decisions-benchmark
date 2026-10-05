"""Does off-option mass flag a question whose options don't fit?

    python -m bench.off_option --out off_option.json [-n 100]

The same inputs are asked twice: with their own dataset's question and
options, and with another dataset's question and options that don't apply
(news articles asked for a banking intent, banking requests asked for a news
topic). The model must pick an option either way; the question is whether the
probability it put on the options before the mask is lower when none fits.
At most four requests are in flight.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from decisions import Choice, SystemOne
from decisions.client import set_max_in_flight

from .calibration import auroc
from .datasets import load

#: (inputs from, question and options from)
PAIRS = (("ag_news", "banking77"), ("banking77", "ag_news"),
         ("sst2", "trec_coarse"), ("trec_coarse", "sst2"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--model", default=os.environ.get("DECISIONS_MODEL", "glm-5.3-flash"))
    args = parser.parse_args()
    set_max_in_flight(4)
    engine = SystemOne.from_env(args.model, temperature=1.0, max_workers=4)

    def ask(job):
        state, question = job
        answer = engine.system_one(state, {"q": question}, mode="sequential").answers["q"]
        return answer.option_mass, answer.probabilities[answer.choice]

    results = {}
    for source, target in PAIRS:
        inputs = load(source, args.n)
        own = load(source, 1)[0]
        other = load(target, 1)[0]
        fit_q = Choice(own.criteria, instructions=own.instructions)
        misfit_q = Choice(other.criteria, instructions=other.instructions)
        with ThreadPoolExecutor(4) as pool:
            fit = list(pool.map(ask, [(t.state, fit_q) for t in inputs]))
            misfit = list(pool.map(ask, [(t.state, misfit_q) for t in inputs]))
        mass = np.array([m for m, _ in fit] + [m for m, _ in misfit])
        conf = np.array([p for _, p in fit] + [p for _, p in misfit])
        is_misfit = np.array([False] * len(fit) + [True] * len(misfit))
        results[f"{source} asked as {target}"] = {
            "mass fit median": statistics.median(m for m, _ in fit),
            "mass misfit median": statistics.median(m for m, _ in misfit),
            "confidence fit median": statistics.median(p for _, p in fit),
            "confidence misfit median": statistics.median(p for _, p in misfit),
            "AUROC low mass flags misfit": auroc(-mass, is_misfit),
            "AUROC low confidence flags misfit": auroc(-conf, is_misfit),
        }
        print(f"{source} asked as {target}: " + ", ".join(
            f"{k} {v:.3f}" for k, v in results[f'{source} asked as {target}'].items()), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
