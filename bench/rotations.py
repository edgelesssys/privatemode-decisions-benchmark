"""Every rotation's distribution, for measuring position bias (PriDe).

    python -m bench.rotations --out runs/rotations [-n 100] [--rotations 4]
        [--only boolq] [--perturb rename]

Each row is asked with its options in ``--rotations`` rotated orders (the
same orders the library's ``permutations=`` uses), and every order's raw
distribution is stored with the positions the options had, and the model
the endpoint reported. The library averages the orders away; this keeps
them, so the position prior can be estimated and checked. Each file starts
with the settings it was written with, and a resume with other settings is
refused. At most four requests are in flight.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decisions import Choice, SystemOne
from decisions.client import set_max_in_flight
from decisions.inference import PREFIX, rotations

from .datasets import load
from .specs import SPECS
from .timing import TimedClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--rotations", type=int, default=4)
    parser.add_argument("--only", default="")
    parser.add_argument("--perturb", default=None)
    parser.add_argument("--model", default=os.environ.get("DECISIONS_MODEL", "glm-5.3-flash"))
    args = parser.parse_args()
    set_max_in_flight(4)
    client = TimedClient(os.environ["DECISIONS_BASE_URL"],
                         os.environ.get("DECISIONS_API_KEY") or None)
    engine = SystemOne(client, args.model, temperature=1.0, max_workers=4)
    # What a file holds; a resume with other settings would mix them.
    settings = {"kind": "meta", "model": args.model, "n": args.n, "rotations": args.rotations,
                "perturb": args.perturb, "prefill": PREFIX}
    only = {n for n in args.only.split(",") if n}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for spec in SPECS:
        if spec.state.get("kind") == "image" or (only and spec.name not in only):
            continue
        path = out / f"{spec.name}.jsonl"
        done = set()
        if path.exists():
            with path.open() as lines:
                records = [json.loads(line) for line in lines]
            meta = next((r for r in records if r.get("kind") == "meta"), None)
            if meta is not None and meta != settings:
                raise SystemExit(f"{path} was written with {meta}, not {settings}; use another --out")
            done = {r["index"] for r in records if r.get("kind") != "meta"}
        else:
            path.write_text(json.dumps(settings) + "\n")
        tasks = [t for t in load(spec.name, args.n, perturbation=args.perturb) if t.index not in done]
        if not tasks:
            continue
        names = list(tasks[0].criteria)
        orders = rotations(len(names), args.rotations)

        def ask(item, names=names):
            task, order = item
            asked = Choice({names[i]: task.criteria[names[i]] for i in order},
                           instructions=task.instructions)
            try:
                answer = engine.system_one(task.state, {"q": asked}, mode="sequential").answers["q"]
            except Exception as failure:   # noqa: BLE001 - the row is retried on the next run
                return task, order, None, f"{type(failure).__name__}: {failure}"[:200]
            return task, order, answer, None

        jobs = [(task, order) for task in tasks for order in orders]
        by_task: dict[int, list] = {}
        with ThreadPoolExecutor(4) as pool, path.open("a") as handle:
            for task, order, answer, _ in pool.map(ask, jobs):
                entry = by_task.setdefault(task.index, [])
                entry.append(None if answer is None else {
                    # position p held option order[p]: store each option's position
                    "position": {names[i]: p for p, i in enumerate(order)},
                    "probabilities": answer.probabilities})
                if len(entry) == len(orders):
                    if all(e is not None for e in entry):
                        handle.write(json.dumps({"index": task.index, "gold": task.gold,
                                                 "options": names, "rotations": entry,
                                                 "served_model": client.last().get("served_model")})
                                     + "\n")
                        handle.flush()
                    del by_task[task.index]
        print(f"{spec.name}: {len(tasks)} rows x {len(orders)} rotations", flush=True)


if __name__ == "__main__":
    main()
