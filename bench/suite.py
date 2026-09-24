"""Run the whole registry, resumably, under a budget.

One command over every dataset that matches the filters, each replicate in
its own file, each file resumed rather than restarted. Interrupting the
suite and running it again costs only what had not been answered yet, which
is what makes a run of this size practical to babysit.

Two guards, both there because the failure they prevent is expensive and
silent:

*The budget.* The cost of a suite is estimated from each dataset's frozen
statistics before a single request goes out, and a suite over the estimate
refuses to start instead of discovering the problem on the invoice. The
estimator is fitted to the pilot's measured token counts and lands within 3%
there, which is the right accuracy for a guard and the wrong accuracy for a
result -- so it never appears in a report.

*The dataset order.* Sets run widest-options-first. The wide sets are the
expensive ones and the ones most likely to expose a problem with an arm, so
a suite that is going to fail should fail in its first ten minutes rather
than its last.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from decisions.client import set_max_in_flight

from . import run as run_module
from .datasets import frozen, has_images
from .pricing import estimate_eur
from .specs import BY_NAME, SPECS

ARM_MODELS = {"privatemode": "glm-5.3-flash", "jev": "jev-latest",
              "laya": "english", "glm-cot": "glm-5.3-flash",
              "embed-nn": "qwen3-embedding-4b"}


def select(args) -> list:
    chosen = list(SPECS)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        unknown = wanted - {spec.name for spec in chosen}
        if unknown:
            raise SystemExit(f"unknown datasets: {', '.join(sorted(unknown))}")
        chosen = [spec for spec in chosen if spec.name in wanted]
    if args.skip:
        unwanted = {name.strip() for name in args.skip.split(",")}
        chosen = [spec for spec in chosen if spec.name not in unwanted]
    if args.family:
        families = {f.strip() for f in args.family.split(",")}
        chosen = [spec for spec in chosen if spec.family in families]
    if args.language:
        languages = {l.strip() for l in args.language.split(",")}
        chosen = [spec for spec in chosen if spec.language in languages]
    if args.max_options:
        chosen = [spec for spec in chosen
                  if frozen(spec.name)["option_count"] <= args.max_options]
    # Widest first: the expensive, failure-prone sets go early.
    return sorted(chosen, key=lambda s: -frozen(s.name)["option_count"])


def forecast(specs, args) -> tuple[float, list[tuple[str, float]]]:
    arms = run_module.ARM_SETS[args.arms]
    rows = []
    for spec in specs:
        meta = frozen(spec.name)
        here = arms
        if has_images(spec):
            here = tuple(a for a in arms if a in ("privatemode", "glm-cot"))
        side = args.image_max_side if has_images(spec) else None
        chars = min(meta["state_chars"]["mean"], args.max_chars)
        total = sum(estimate_eur(arm, ARM_MODELS[arm],
                                 min(args.n, meta["rows"]),
                                 meta["option_count"], chars, side,
                                 args.eur_per_usd)
                    for arm in here) * args.replicates
        rows.append((spec.name, total))
    return sum(cost for _, cost in rows), rows


def main(argv: list[str] | None = None) -> int:
    parent = run_module.build_parser()
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[parent], conflict_handler="resolve")
    parser.add_argument("--dataset", help=argparse.SUPPRESS)  # per-set, not here
    parser.add_argument("--replicate", help=argparse.SUPPRESS)
    parser.add_argument("--only", help="comma-separated dataset names")
    parser.add_argument("--skip", help="comma-separated dataset names")
    parser.add_argument("--family", help="comma-separated families")
    parser.add_argument("--language", help="comma-separated languages")
    parser.add_argument("--max-options", type=int,
                        help="only sets with at most this many options")
    parser.add_argument("--replicates", type=int, default=2,
                        help="replicate runs per dataset (default: 2, which "
                             "is what the measured run-to-run variance needs)")
    parser.add_argument("--budget-eur", type=float, default=50.0,
                        help="refuse to start if the estimate exceeds this")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the forecast and stop")
    args = parser.parse_args(argv)

    run_module.load_env(Path(__file__).resolve().parent.parent / ".env")
    specs = select(args)
    if not specs:
        raise SystemExit("no dataset matched the filters")

    total, rows = forecast(specs, args)
    print(f"{len(specs)} datasets x {args.replicates} replicates x "
          f"{args.n} examples, arms {args.arms}", file=sys.stderr)
    for name, cost in rows:
        print(f"  {name:24} ~EUR {cost:7.3f}", file=sys.stderr)
    print(f"  {'estimated total':24} ~EUR {total:7.3f} "
          f"(budget {args.budget_eur})", file=sys.stderr)
    if total > args.budget_eur:
        raise SystemExit("estimate exceeds --budget-eur; raise it or narrow "
                         "the selection")
    if args.dry_run:
        return 0

    set_max_in_flight(max(2, 2 * args.concurrency * max(1, args.permutations)))
    started = time.perf_counter()
    paths, failures = [], []
    for spec in specs:
        for replicate in range(args.replicates):
            single = argparse.Namespace(**vars(args))
            single.dataset = spec.name
            single.replicate = replicate
            try:
                paths.append(run_module.execute(single))
            except SystemExit as error:
                print(f"! {spec.name} r{replicate}: {error}", file=sys.stderr)
                failures.append(spec.name)
            except Exception as error:
                print(f"! {spec.name} r{replicate}: {type(error).__name__}: "
                      f"{error}", file=sys.stderr)
                failures.append(spec.name)
    wall = time.perf_counter() - started

    index = Path(args.out) / "suite.json"
    index.write_text(json.dumps({
        "runs": [str(p) for p in paths], "failed": failures,
        "wall_s": wall, "arms": args.arms, "n": args.n,
        "replicates": args.replicates,
    }, indent=2) + "\n")
    print(f"\n{len(paths)} runs in {wall / 60:.1f} min; index at {index}",
          file=sys.stderr)
    if failures:
        print(f"failed: {', '.join(sorted(set(failures)))}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
