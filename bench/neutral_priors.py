"""What the model prefers with no content: priors for contextual calibration.

    python -m bench.neutral_priors --out priors.json [--only a,b] [--perturb rename]

For every text dataset, the same instruction and options are asked with
neutral content in place of the state (``N/A``, an empty string, ``[MASK]``,
and ``k. A.`` for German sets). Their raw distributions are averaged into one
prior per option set: what the model picks from the option names, order and
numbers alone. At most four requests are in flight.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decisions import Choice, SystemOne
from decisions.client import set_max_in_flight

from .datasets import load
from .specs import SPECS

NEUTRAL = ("N/A", "", "[MASK]")
NEUTRAL_DE = ("k. A.",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--perturb", default=None)
    parser.add_argument("--model", default=os.environ.get("DECISIONS_MODEL", "glm-5.3-flash"))
    args = parser.parse_args()
    set_max_in_flight(4)
    engine = SystemOne.from_env(args.model, temperature=1.0, max_workers=4)
    only = {n for n in args.only.split(",") if n}
    specs = [s for s in SPECS if s.state.get("kind") != "image" and (not only or s.name in only)]

    jobs = []
    for spec in specs:
        task = load(spec.name, 1, perturbation=args.perturb)[0]
        contents = NEUTRAL + (NEUTRAL_DE if spec.language == "de" else ())
        jobs += [(spec.name, task, content) for content in contents]

    def ask(job):
        name, task, content = job
        question = Choice(task.criteria, instructions=task.instructions)
        answer = engine.system_one(content, {"q": question}, mode="sequential").answers["q"]
        return name, [answer.probabilities[o] for o in task.criteria]

    priors: dict[str, list[list[float]]] = {}
    with ThreadPoolExecutor(4) as pool:
        for name, probabilities in pool.map(ask, jobs):
            priors.setdefault(name, []).append(probabilities)
    averaged = {name: [sum(col) / len(rows) for col in zip(*rows)] for name, rows in priors.items()}
    Path(args.out).write_text(json.dumps(averaged, indent=1))
    print(f"wrote {len(averaged)} priors from {len(jobs)} requests to {args.out}")


if __name__ == "__main__":
    main()
