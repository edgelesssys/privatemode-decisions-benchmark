"""Turn a run's JSONL into the table you would put in front of someone.

Reads a file written by ``bench.run`` (or several, to pool runs) and prints
markdown. Kept separate from the run on purpose: every metric here can be
recomputed from rows that are already paid for, so a new question about an
old run costs nothing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from itertools import combinations

from .metrics import (accuracy, auroc, bootstrap_diff, brier,
                      expected_calibration_error, macro_f1, mcnemar, quantile,
                      selective)
from .pricing import embedding_cost, jev_cost, local_cost, privatemode_cost

def _privatemode_priced(arm: str):
    """Arms served by the Privatemode proxy bill at that model's rates."""
    return lambda row, meta, fx: privatemode_cost(
        meta["arms"][arm], row["input_tokens"], row["output_tokens"],
        row.get("cached_tokens", 0)).eur


COST = {"privatemode": _privatemode_priced("privatemode"),
        "glm-cot": _privatemode_priced("glm-cot"),
        "embed-nn": lambda row, meta, fx: embedding_cost(row["input_tokens"]).eur,
        "jev": lambda row, meta, fx: jev_cost(
            row["input_tokens"], row["output_tokens"], fx).eur,
        "laya": lambda row, meta, fx: local_cost().eur}

#: Not products. They answer "better than what, and because of what"; the
#: report marks them so no table reads as a five-way product comparison.
CONTROLS = {"glm-cot", "embed-nn"}

#: Arms that run on the machine the benchmark runs on. Their latency is
#: measured against a laptop and their cost has no list price, so both
#: columns are marked rather than compared.
LOCAL = {"laya"}


def read(path: Path) -> tuple[dict, dict[str, list[dict]], dict]:
    meta: dict = {}
    end: dict = {}
    rows: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") == "meta":
            meta = record
        elif record.get("kind") == "end":
            end = record
        elif record.get("kind") == "row":
            rows.setdefault(record["arm"], []).append(record)
    if not meta:
        raise ValueError(f"{path} has no meta line")
    return meta, rows, end


#: Errors that are the arm's own failure to answer, not the vendor's or the
#: network's: the reasoning control running out of its token budget, or
#: replying without naming an option. Dropping these would take exactly the
#: hard examples out of that arm's score, so they count as wrong answers.
ARM_FAILURES = ("RuntimeError: ran out of tokens while reasoning",
                "RuntimeError: no option in the reply")


def as_wrong_answer(row: dict, meta: dict) -> dict:
    """An arm's failure to answer, scored as a wrong answer.

    It gets no probability on any real option, no latency (there is no
    answer to time), and the reasoning budget as its output tokens, which
    is what running out of it cost.
    """
    budget = (meta.get("identity") or {}).get("cot_max_tokens", 0)
    out = {key: value for key, value in row.items() if key != "error"}
    out.update(choice=None, correct=False, probabilities={"(no answer)": 1.0},
               confidence=0.0, latency_s=None, input_tokens=0,
               output_tokens=budget if row["error"].startswith(ARM_FAILURES[0]) else 0,
               cached_tokens=0, failed=True)
    return out


def paired(rows: dict[str, list[dict]],
           meta: dict | None = None) -> tuple[list[str], dict[str, list[dict]]]:
    """Only the examples every arm answered, an arm's own failures included.

    Dropping examples per arm instead would let a failure on the hard half of
    the data look like accuracy. Infrastructure errors are dropped for every
    arm alike; an arm's failure to answer (ARM_FAILURES) is kept as a wrong
    answer, for the same reason.
    """
    meta = meta or {}
    def usable(row: dict) -> dict | None:
        if "error" not in row:
            return row
        if row["error"].startswith(ARM_FAILURES):
            return as_wrong_answer(row, meta)
        return None
    by_index = {arm: {row["index"]: kept for row in rows[arm]
                      if (kept := usable(row)) is not None}
                for arm in sorted(rows)}
    # An arm that answered nothing is dropped rather than intersected: an
    # empty set would take every other arm's examples with it, turning one
    # arm's outage into a run with no data at all.
    arms = [arm for arm, answered in by_index.items() if answered]
    shared = sorted(set.intersection(*(set(by_index[arm]) for arm in arms))) \
        if arms else []
    return arms, {arm: [by_index[arm][i] for i in shared] for arm in arms}


def measure(rows: list[dict], arm: str, meta: dict, fx: float) -> dict:
    gold = [row["gold"] for row in rows]
    predicted = [row["choice"] for row in rows]
    correct = [bool(row["correct"]) for row in rows]
    distributions = [row["probabilities"] for row in rows]
    # The common confidence: the probability the arm put on the option it
    # picked. Both arms return a distribution, so this is the one calibration
    # number that is defined the same way on both sides.
    top = [max(probs.values()) for probs in distributions]
    vendor = [row["confidence"] for row in rows]
    latency = [row["latency_s"] for row in rows if row.get("latency_s") is not None]
    costs = [COST[arm](row, meta, fx) for row in rows]
    return {
        "n": len(rows),
        "accuracy": accuracy(correct),
        "macro_f1": macro_f1(gold, predicted),
        "ece_top": expected_calibration_error(top, correct),
        "brier": brier(distributions, gold),
        "auroc_vendor_confidence": auroc(vendor, correct),
        "selective": selective(vendor, correct),
        # p10 is the arm's unqueued floor. A p50 far above it is the
        # signature of something waiting -- a vendor queue, a throttle, or
        # another process on this machine -- rather than a slower model.
        "p10_ms": quantile(latency, 0.1) * 1000,
        "p50_ms": quantile(latency, 0.5) * 1000,
        "p95_ms": quantile(latency, 0.95) * 1000,
        "mean_ms": sum(latency) / len(latency) * 1000,
        "throttled": sum(row.get("throttled", 0) for row in rows),
        "gate_wait_ms": sum(row.get("gate_wait_s", 0.0)
                            for row in rows) / len(rows) * 1000,
        "input_tokens": sum(row["input_tokens"] for row in rows) / len(rows),
        "eur_per_1k": sum(costs) / len(costs) * 1000,
        "retries": sum(row.get("attempts", 1) - 1 for row in rows),
        "correct": correct,
        # Kept per example so bench.aggregate can recompute coverage curves
        # without re-reading every run.
        "confidence": vendor,
    }


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) \
        else f"{value:.{digits}f}"


def summarize(*paths: Path | str) -> str:
    out: list[str] = []
    for path in paths:
        path = Path(path)
        meta, raw, end = read(path)
        arms, rows = paired(raw, meta)
        fx = meta.get("eur_per_usd", 0.92)
        scored = {arm: measure(rows[arm], arm, meta, fx) for arm in arms}
        dropped = {arm: sum(1 for row in raw[arm] if "error" in row) for arm in arms}

        out.append(f"## {meta['dataset']} — {meta['n']} examples, "
                   f"{meta['options']} options, seed {meta['seed']}")
        out.append("")
        settings = [f"Concurrency {meta['concurrency']}",
                    f"permutations {meta.get('permutations', 1)}",
                    f"laya shortlist {meta.get('laya_shortlist', 0) or 'off'}"]
        if meta.get("images"):
            settings.append(f"images scaled to {meta['image_max_side']}px")
        if meta.get("perturb", "none") not in ("none", None):
            settings.append(f"PERTURBED: {meta['perturb']}")
        if meta.get("skipped_arms"):
            left_out = meta["skipped_arms"]
            if isinstance(left_out, dict):
                left_out = [f"{name} ({why})" for name, why in left_out.items()]
            settings.append("arms that cannot answer this set: "
                            + ", ".join(left_out))
        settings.append(f"1 USD = {fx} EUR")
        out.append(f"`{meta['hf']}` / {meta['split']}. "
                   f"Models: {', '.join(f'{k} = {v}' for k, v in meta['arms'].items())}. "
                   + ", ".join(settings) + ".")
        out.append("")

        def cell(arm: str, value: str, local: str | None = None) -> str:
            return local if (local and arm in LOCAL) else value

        header = ["metric", *arms]
        rows_md = [
            ("examples scored", *[str(scored[a]["n"]) for a in arms]),
            ("errors dropped", *[str(dropped[a]) for a in arms]),
            ("**accuracy**", *[_fmt(scored[a]["accuracy"]) for a in arms]),
            ("macro F1", *[_fmt(scored[a]["macro_f1"]) for a in arms]),
            ("ECE (top prob)", *[_fmt(scored[a]["ece_top"]) for a in arms]),
            ("Brier", *[_fmt(scored[a]["brier"]) for a in arms]),
            ("AUROC (vendor confidence)",
             *[_fmt(scored[a]["auroc_vendor_confidence"]) for a in arms]),
            ("acc @ 90% coverage",
             *[_fmt(scored[a]["selective"]["acc@0.9"]) for a in arms]),
            ("acc @ 50% coverage",
             *[_fmt(scored[a]["selective"]["acc@0.5"]) for a in arms]),
            ("latency p10 (ms)",
             *[cell(a, _fmt(scored[a]["p10_ms"], 0), _fmt(scored[a]["p10_ms"], 0) + "*")
               for a in arms]),
            ("latency p50 (ms)",
             *[cell(a, _fmt(scored[a]["p50_ms"], 0), _fmt(scored[a]["p50_ms"], 0) + "*")
               for a in arms]),
            ("latency p95 (ms)",
             *[cell(a, _fmt(scored[a]["p95_ms"], 0), _fmt(scored[a]["p95_ms"], 0) + "*")
               for a in arms]),
            ("p50 / p10 (1.0 = unqueued)",
             *[_fmt(scored[a]["p50_ms"] / scored[a]["p10_ms"], 2)
               if scored[a]["p10_ms"] else "n/a" for a in arms]),
            ("throttled responses",
             *[str(scored[a]["throttled"]) for a in arms]),
            ("queued on our own gate (ms)",
             *[_fmt(scored[a]["gate_wait_ms"], 1) for a in arms]),
            ("mean input tokens", *[_fmt(scored[a]["input_tokens"], 0) for a in arms]),
            ("**EUR / 1000 decisions**",
             *[cell(a, _fmt(scored[a]["eur_per_1k"], 4), "local") for a in arms]),
            ("retries", *[str(scored[a]["retries"]) for a in arms]),
        ]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for row in rows_md:
            out.append("| " + " | ".join(row) + " |")
        out.append("")
        if meta.get("concurrency", 1) != 1:
            out.append(f"**Latency here is not a product measurement.** This "
                       f"run used concurrency {meta['concurrency']}, so "
                       f"requests competed with each other. Take latency from "
                       f"a concurrency-1 run; accuracy and cost are unaffected.")
            out.append("")
        if any(scored[a]["throttled"] for a in arms):
            hit = ", ".join(f"{a} ({scored[a]['throttled']})" for a in arms
                            if scored[a]["throttled"])
            out.append(f"**Throttling seen**: {hit}. A 429 or 503 is a closed "
                       f"door rather than a slow model; the retry's time is "
                       f"excluded from the latency above, but an arm that was "
                       f"throttled was not measured under the same conditions "
                       f"as one that was not.")
            out.append("")
        if set(arms) & CONTROLS:
            out.append("Control arms (not products): "
                       + ", ".join(sorted(set(arms) & CONTROLS))
                       + ". `glm-cot` is the same GLM-5.3-Flash asked normally "
                         "with reasoning, which separates the technique from "
                         "the model; `embed-nn` is zero-shot nearest-centroid "
                         "over Qwen3-Embedding-4B, which is the floor.")
            out.append("")
        if set(arms) & LOCAL:
            out.append("\\* Measured on the machine that ran the benchmark, not in a "
                       "datacentre. It is not comparable with the hosted arms and is "
                       "printed only so the run is fully recorded.")
            out.append("")

        for a, b in combinations(arms, 2):
            test = mcnemar(scored[a]["correct"], scored[b]["correct"])
            point, low, high = bootstrap_diff(scored[a]["correct"],
                                              scored[b]["correct"])
            line = (f"**{a} vs {b}** — right where the other was wrong on "
                    f"{test['only_a']} / {test['only_b']} examples; McNemar "
                    f"p = {test['p']:.4f}. Accuracy difference "
                    f"{point:+.3f} [95% CI {low:+.3f}, {high:+.3f}].")
            if not ({a, b} & LOCAL):
                ratio = (scored[a]["eur_per_1k"] / scored[b]["eur_per_1k"]
                         if scored[b]["eur_per_1k"] else float("inf"))
                line += (f" Cost {ratio:.2f}×, latency p50 "
                         f"{scored[a]['p50_ms'] / scored[b]['p50_ms']:.2f}×.")
            out.append(line)
            out.append("")

        if end:
            out.append(f"Wall clock for the run: {end['wall_s']:.1f}s.")
            out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--write", help="also write the markdown to this file")
    args = parser.parse_args(argv)
    text = summarize(*args.paths)
    print(text)
    if args.write:
        Path(args.write).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
