"""Scan a dataset once, freeze its option set, and validate it.

Run before any arm sees the data. The output, ``datasets/<name>.json``, is
checked in, so the option set and the validation numbers are reviewable
artefacts rather than something a run produced on the way past.

What is checked, and why each one has bitten somebody:

*The option vocabulary.* For a ``ClassLabel`` dataset the server states it.
For the mteb-style mirrors it does not, so the split is scanned until the
vocabulary stops growing -- and if it is still growing when the scan budget
runs out, that is recorded as ``stable: false`` rather than quietly
truncating the option set, which would make every later run wrong in a way
no metric would reveal.

*The majority-class share.* Chance level is ``1/options`` only for a
balanced set. A moderation corpus that is 92% clean makes 0.92 accuracy look
like a result; the majority share is what that number has to beat.

*State length.* Decides what the truncation budget costs on this set, and
which arms can see a whole example at all.

*Duplicate states.* Identical inputs with different labels put a ceiling on
accuracy that has nothing to do with the arms.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter

from . import hub
from .datasets import frozen_dir, gold_of, has_images, present, state_of
from .specs import BY_NAME, SPECS, Spec

#: Rows scanned to discover a vocabulary the server does not declare. A
#: split small enough is read whole, because the settling heuristic below
#: fails exactly where it is least visible: a split ordered by label hides
#: its last class past any fixed budget, and a short option set then looks
#: settled while it is simply incomplete. amazon_reviews_de found four of
#: five stars this way.
VOCAB_SCAN = 10000
#: ...and how many consecutive rows must add nothing for it to count as done.
VOCAB_SETTLED = 1500
#: Rows sampled for the length, balance and duplicate statistics.
STATS_SAMPLE = 2000


def discover_options(spec: Spec, features: list[dict]) -> tuple[list[str], bool]:
    kind = spec.gold["kind"]
    if kind == "bool":
        return ["true", "false"], True
    if kind == "class_label":
        names = hub.class_label_names(features, spec.gold["field"])
        if not names:
            raise RuntimeError(f"{spec.name}: {spec.gold['field']!r} is not a "
                               "ClassLabel; use a label_text gold instead")
        return [present(spec, name) for name in names], True
    total = hub.head(spec.hf, spec.config, spec.split)["num_rows_total"]
    budget = total if total <= VOCAB_SCAN else VOCAB_SCAN
    seen: dict[str, int] = {}
    since_new = scanned = 0
    for position, (_, row) in enumerate(hub.scan(spec.hf, spec.config,
                                                 spec.split, budget)):
        scanned = position + 1
        value = str(row[spec.gold["field"]])
        if value not in seen:
            seen[value] = position
            since_new = 0
        else:
            since_new += 1
        if since_new >= VOCAB_SETTLED:
            return sorted(seen), True
    # Having read the whole split, the vocabulary is complete by definition;
    # the settling rule only exists for splits too big to read.
    return [present(spec, value) for value in sorted(seen)], (
        scanned >= total or since_new >= VOCAB_SETTLED)


def curate(name: str) -> dict:
    spec = BY_NAME[name]
    head = hub.head(spec.hf, spec.config, spec.split)
    total = head["num_rows_total"]
    options, stable = discover_options(spec, head["features"])

    sample = sorted(random.Random(0).sample(range(total),
                                            min(STATS_SAMPLE, total)))
    rows = hub.pages(spec.hf, spec.config, spec.split, sample)
    lengths, golds, states, pixels = [], [], Counter(), []
    unknown = 0
    for index in sample:
        row = rows[index]
        state = state_of(spec, row)
        lengths.append(len(state))
        if has_images(spec):
            image = row[spec.state["field"]]
            pixels.append(max(image["width"], image["height"]))
        else:
            states[state] += 1
        gold = gold_of(spec, row, options)
        if gold not in options:
            unknown += 1
        golds.append(gold)

    lengths.sort()
    def q(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))]

    counts = Counter(golds)
    majority = max(counts.values()) / len(golds)
    return {
        "name": spec.name,
        "hf": spec.hf, "config": spec.config, "split": spec.split,
        "rows": total,
        "options": options,
        "option_count": len(options),
        "options_stable": stable,
        "chance_uniform": 1 / len(options),
        "majority_share": majority,
        "classes_in_sample": len(counts),
        "unknown_gold_in_sample": unknown,
        "state_chars": {"mean": sum(lengths) / len(lengths), "p50": q(0.5),
                        "p95": q(0.95), "max": lengths[-1]},
        "duplicate_states": sum(c - 1 for c in states.values() if c > 1),
        "images": has_images(spec),
        # Longest edge, since that is what --image-max-side scales and what
        # the image's token cost follows.
        "image_longest_side": ({"p50": sorted(pixels)[len(pixels) // 2],
                                "max": max(pixels)} if pixels else None),
        "stats_sample": len(sample),
        "family": spec.family, "language": spec.language, "tier": spec.tier,
        "ladder": list(spec.ladder) if spec.ladder else None,
        "instructions": spec.instructions,
        "rename": spec.rename,
    }


def problems(report: dict, spec: Spec) -> list[str]:
    out = []
    if not report["options_stable"]:
        out.append(f"option vocabulary still growing after {VOCAB_SCAN} rows")
    if spec.options_hint and report["option_count"] > spec.options_hint:
        out.append(f"expected at most {spec.options_hint} options, found "
                   f"{report['option_count']}")
    elif spec.options_hint and report["option_count"] < spec.options_hint:
        out.append(f"{spec.options_hint - report['option_count']} of the "
                   f"dataset's classes never occur in this split; the task is "
                   f"{report['option_count']}-way here, and is reported as such")
    if report["unknown_gold_in_sample"]:
        out.append(f"{report['unknown_gold_in_sample']} sampled golds outside "
                   "the option set")
    if report["classes_in_sample"] < report["option_count"] * 0.5:
        out.append(f"only {report['classes_in_sample']} of "
                   f"{report['option_count']} classes appear in the sample")
    if report["majority_share"] > 0.7:
        out.append(f"majority class is {report['majority_share']:.0%} of the "
                   "sample; accuracy alone will not be informative")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", default=None,
                        help="datasets to curate (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="re-curate sets that already have a frozen file")
    args = parser.parse_args(argv)

    names = args.names or [spec.name for spec in SPECS]
    out_dir = frozen_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name in names:
        path = out_dir / f"{name}.json"
        if path.exists() and not args.force:
            print(f"  {name}: already frozen", file=sys.stderr)
            continue
        try:
            report = curate(name)
        except Exception as error:
            print(f"! {name}: {type(error).__name__}: {error}", file=sys.stderr)
            failures += 1
            continue
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        flags = problems(report, BY_NAME[name])
        state = report["state_chars"]
        print(f"  {name}: {report['option_count']} options, "
              f"{report['rows']} rows, majority "
              f"{report['majority_share']:.0%}, "
              f"state p50 {state['p50']} / max {state['max']} chars"
              + ("".join(f"\n      ! {flag}" for flag in flags) if flags else ""),
              file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
