"""Turn a suite of runs into one result.

Reads every run JSONL under a directory, groups them by dataset and
replicate, and answers the question the suite exists to answer: across two
dozen tasks, which arm is ahead, by how much, and where does that stop being
true.

What it deliberately does *not* do is print a p-value per dataset. With 29
sets, "significant somewhere" is guaranteed by chance, and a reader who
scans for stars will find whichever story they came for. The headline is one
test per pair of arms whose unit is the dataset -- Wilcoxon signed-rank over
the per-dataset differences -- and per-dataset numbers are reported with
their replicate spread instead, which is the honest local uncertainty.

Three views, in the order they should be read:

*Overall.* Normalised accuracy against the majority-class baseline, because
averaging raw accuracy over sets whose chance levels are 50% and 1.3% is
meaningless, and win/tie/loss counts beside it because they survive a single
pathological set.

*By option count.* The pilot's finding was that the ranking reverses with
the width of the option set. This is where that is confirmed or refuted.

*The ladders.* Same examples, two label granularities. Every cross-dataset
comparison confounds option count with task, domain and text at once; a
ladder changes the option count and nothing else, so it is the only place
the effect is measured rather than inferred.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from .metrics import normalised, quantile, wilcoxon
from .report import CONTROLS, LOCAL, measure, paired, read
from .specs import BY_NAME, ladders

#: A per-dataset difference below this is treated as a tie when counting
#: wins. It is the run-to-run flip noise measured in the pilot, so calling a
#: half-point difference a "win" would be counting the servers' mood.
TIE = 0.01


def consensus(rows: dict[str, list[dict]]) -> dict[str, float]:
    """How much of a dataset's residual error is the labels, not the arms.

    Every arm is a different system built by different people, so when most
    of them return the *same* answer and the gold label disagrees, the
    likeliest explanation is that the gold label is one of two defensible
    ones. banking77 is full of these -- ``get_physical_card`` against
    ``order_physical_card``, ``declined_transfer`` against
    ``failed_transfer`` -- and on a 40-example probe three independent arms
    returned the identical wrong answer on eleven of them.

    Agreement is measured on the *modal* prediction, held by at least half
    the arms, not on unanimity. Unanimity looks stricter and is in fact
    useless: adding one weak arm that disagrees with everything drives it to
    zero and takes the signal with it. Measured with all five arms on that
    same probe, unanimity found nothing while the modal rule found the
    eleven.

    ``consensus_wrong`` is a floor estimate for label noise. ``ceiling`` is
    the share at least one arm got right -- the most any of them could have
    scored. The gap between an arm and the ceiling is contested ground; the
    gap from the ceiling to 1.0 largely is not.

    It is an estimate, and it errs in one direction: arms can agree on a
    wrong answer for reasons that are not label noise, a shared pretraining
    corpus among them. It belongs beside accuracy as context, never
    subtracted from it.
    """
    arms = sorted(rows)
    if len(arms) < 2:
        return {}
    by_index = {arm: {row["index"]: row for row in rows[arm]} for arm in arms}
    shared = sorted(set.intersection(*(set(by_index[a]) for a in arms)))
    if not shared:
        return {}
    needed = max(2, (len(arms) + 1) // 2)
    agreed = agreed_wrong = any_correct = 0
    for index in shared:
        picks = Counter(by_index[arm][index]["choice"] for arm in arms)
        modal, count = picks.most_common(1)[0]
        if count >= needed:
            agreed += 1
            if modal != by_index[arms[0]][index]["gold"]:
                agreed_wrong += 1
        if any(by_index[arm][index]["correct"] for arm in arms):
            any_correct += 1
    return {"agreed": agreed / len(shared),
            "consensus_wrong": agreed_wrong / len(shared),
            "ceiling": any_correct / len(shared),
            "arms": len(arms)}


def collect(root: Path, perturb: str = "none") -> dict[str, dict[int, tuple[dict, dict]]]:
    """dataset -> replicate -> (meta, {arm: measures}).

    ``perturb`` selects which runs to read. The like-for-like suite is the
    unperturbed one; the variants are read separately and compared, never
    pooled.

    Runs are grouped by their full identity, not by dataset alone, and a
    run written before identities existed is skipped. Both rules were
    bought the hard way: the pilot's files carry no replicate number, so
    two runs of the same dataset collapsed onto one key and the later
    silently replaced the earlier -- which is how a hosted-only rerun
    erased Laya from boolq and ag_news, and how 200-example pilot numbers
    came to be printed as 1000-example suite results. Same dataset is not
    the same experiment.

    When a dataset has several identity groups -- a 100-example latency run
    beside a 1000-example accuracy run, say -- the largest is used and the
    rest are reported as ignored rather than merged.
    """
    groups: dict[str, dict[str, dict[int, tuple[dict, dict]]]] = defaultdict(
        lambda: defaultdict(dict))
    skipped_legacy = 0
    for path in sorted(root.rglob("*.jsonl")):
        try:
            meta, raw, _ = read(path)
        except (ValueError, KeyError):
            continue
        identity = meta.get("identity")
        if not identity:
            skipped_legacy += 1
            continue
        if meta.get("permutations", 1) != 1 or meta.get("laya_shortlist", 0):
            continue  # labelled variants are not part of the like-for-like suite
        if (meta.get("perturb") or "none") != perturb:
            continue
        arms, rows = paired(raw, meta)
        if not arms or not rows[arms[0]]:
            continue
        fx = meta.get("eur_per_usd", 0.92)
        scored = {arm: measure(rows[arm], arm, meta, fx) for arm in arms}
        for arm in arms:
            golds = [row["gold"] for row in rows[arm]]
            counts = Counter(golds)
            scored[arm]["majority"] = max(counts.values()) / len(golds)
        meta = dict(meta, consensus=consensus(rows))
        # The shape is everything that decides *which questions were
        # asked* -- dataset, frozen option set, sample, seed, truncation,
        # image resolution, variant knobs. Not the replicate, and not the
        # arms: a run of the control arms over the same sample belongs in
        # the same group as the products that answered it, otherwise the
        # controls form a group of their own, lose the largest-n tiebreak,
        # and vanish from the report they exist to inform.
        shape = json.dumps({k: v for k, v in identity.items()
                            if k not in ("replicate", "arms", "concurrency")},
                           sort_keys=True)
        # Merge on arrival: products and controls over the same sample share
        # a shape and a replicate number, so assigning would drop whichever
        # file was read first.
        slot = groups[meta["dataset"]][shape]
        replicate = identity.get("replicate", 0)
        if replicate in slot:
            kept_meta, kept = slot[replicate]
            kept.update(scored)
            slot[replicate] = (kept_meta, kept)
        else:
            slot[replicate] = (meta, dict(scored))

    if skipped_legacy:
        print(f"note: skipped {skipped_legacy} run(s) written before runs had "
              f"identities", file=sys.stderr)
    out: dict[str, dict[int, tuple[dict, dict]]] = {}
    for dataset, shapes in groups.items():
        best = max(shapes.values(),
                   key=lambda reps: (next(iter(reps.values()))[0]["n"], len(reps)))
        if len(shapes) > 1:
            other = sum(len(reps) for reps in shapes.values()) - len(best)
            print(f"note: {dataset} has {len(shapes)} run shapes; using the "
                  f"largest ({next(iter(best.values()))[0]['n']} examples, "
                  f"{len(best)} replicate(s)) and ignoring {other} other run(s)",
                  file=sys.stderr)
        out[dataset] = best
    return out


def latencies(root: Path) -> dict[str, dict[str, dict]]:
    """dataset -> arm -> p10/p50/p95, from concurrency-1 runs only.

    Collected separately from everything else, and deliberately so. The
    accuracy runs are large and concurrent; the latency runs are small and
    quiet. Asking one collection to serve both means choosing a run shape
    per dataset, and whichever way that choice goes one of the two numbers
    is wrong -- pick the big run and latency measures a queue, pick the
    quiet one and accuracy is measured on a tenth of the data.

    So accuracy takes the largest run shape and latency takes every
    concurrency-1 run, whatever its size. A timing does not care how many
    examples followed it.
    """
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.rglob("*.jsonl")):
        try:
            meta, raw, _ = read(path)
        except (ValueError, KeyError):
            continue
        if not meta.get("identity") or meta.get("concurrency") != 1:
            continue
        if (meta.get("perturb") or "none") != "none":
            continue
        if meta.get("permutations", 1) != 1 or meta.get("laya_shortlist", 0):
            continue
        arms, rows = paired(raw, meta)
        fx = meta.get("eur_per_usd", 0.92)
        for arm in arms:
            out[meta["dataset"]][arm].append(
                measure(rows[arm], arm, meta, fx))
    return {dataset: {arm: {
                "p10_ms": sum(r["p10_ms"] for r in runs) / len(runs),
                "p50_ms": sum(r["p50_ms"] for r in runs) / len(runs),
                "p95_ms": sum(r["p95_ms"] for r in runs) / len(runs),
                "throttled": sum(r["throttled"] for r in runs),
                "runs": len(runs)}
            for arm, runs in per_arm.items()}
            for dataset, per_arm in out.items()}


def average(collected: dict) -> dict[str, dict[str, dict]]:
    """dataset -> arm -> metrics averaged over replicates, with the spread.

    Accuracy, calibration and cost come from every replicate. **Latency
    comes only from concurrency-1 runs**, and is ``None`` when there are
    none: at higher concurrency the requests compete with each other, so
    the number measures the harness rather than the vendor. Rolling the two
    together would let a throughput run quietly set the headline latency.
    """
    out: dict[str, dict[str, dict]] = {}
    for dataset, replicates in collected.items():
        per_arm: dict[str, list[dict]] = defaultdict(list)
        quiet: dict[str, list[dict]] = defaultdict(list)
        for meta, scored in replicates.values():
            for arm, values in scored.items():
                per_arm[arm].append(values)
                if meta.get("concurrency", 1) == 1:
                    quiet[arm].append(values)
        out[dataset] = {}
        for arm, runs in per_arm.items():
            accuracies = [run["accuracy"] for run in runs]
            out[dataset][arm] = {
                "replicates": len(runs),
                "accuracy": sum(accuracies) / len(accuracies),
                # The gap between replicates of the identical sample: the
                # local noise floor, printed rather than averaged away.
                "spread": max(accuracies) - min(accuracies),
                "majority": runs[0]["majority"],
                "eur_per_1k": sum(r["eur_per_1k"] for r in runs) / len(runs),
                "input_tokens": sum(r["input_tokens"] for r in runs) / len(runs),
                "throttled": sum(r.get("throttled", 0) for r in runs),
                "p50_ms": (sum(r["p50_ms"] for r in quiet[arm]) / len(quiet[arm])
                           if quiet[arm] else None),
                "p10_ms": (sum(r["p10_ms"] for r in quiet[arm]) / len(quiet[arm])
                           if quiet[arm] else None),
                "ece": sum(r["ece_top"] for r in runs) / len(runs),
                "coverage_95": sum(coverage_at(r, 0.95) for r in runs) / len(runs),
            }
    return out


def coverage_at(scored: dict, target: float) -> float:
    """Largest fraction of traffic answerable at ``target`` accuracy.

    The buyer's question, and the reason calibration is a headline rather
    than a footnote: route automatically above a confidence threshold, send
    the rest to a human. Zero means the arm never reaches the target, at any
    threshold.
    """
    order = sorted(range(len(scored["correct"])),
                   key=lambda i: -scored["confidence"][i])
    best, hits = 0.0, 0
    for taken, index in enumerate(order, start=1):
        hits += scored["correct"][index]
        if hits / taken >= target:
            best = taken / len(order)
    return best


def table(rows: list[tuple], header: list[str]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return out + [""]


def summarise(root: Path) -> str:
    collected = collect(root)
    if not collected:
        return "no runs found"
    averaged = average(collected)
    datasets = sorted(averaged, key=lambda d: _options(collected, d))
    arms = sorted({arm for values in averaged.values() for arm in values})

    out = ["# Suite result", ""]
    products = [arm for arm in arms if arm not in CONTROLS]
    controls = [arm for arm in arms if arm in CONTROLS]
    if controls:
        out.append(f"Products: {', '.join(products)}. Controls (not products, "
                   f"and not part of the like-for-like comparison): "
                   f"{', '.join(controls)}.")
        out.append("")
    out.append(f"{len(datasets)} datasets, arms: {', '.join(arms)}. "
               f"Per-dataset figures are averaged over replicates and carry "
               f"the spread between them; there are deliberately no "
               f"per-dataset p-values.")
    out.append("")

    # -- per dataset ------------------------------------------------------
    out.append("## Per dataset")
    out.append("")
    rows = []
    for dataset in datasets:
        options = _options(collected, dataset)
        spec = BY_NAME.get(dataset)
        cells = [f"{dataset} ({options})", f"{spec.language if spec else '?'}",
                 f"{next(iter(averaged[dataset].values()))['majority']:.2f}"]
        for arm in arms:
            values = averaged[dataset].get(arm)
            cells.append("—" if values is None else
                         f"{values['accuracy']:.3f} ±{values['spread']:.3f}")
        rows.append(tuple(cells))
    out += table(rows, ["dataset (options)", "lang", "majority", *arms])

    # -- what the labels allow -------------------------------------------
    agreed = [d for d in datasets if _consensus(collected, d)]
    if agreed:
        out.append("## What the labels allow")
        out.append("")
        out.append("Where most arms return the same answer and the gold "
                   "label disagrees, the likeliest explanation is an "
                   "ambiguous label rather than a shared failure. "
                   "`consensus wrong` is a floor estimate for that; "
                   "`ceiling` is the share at least one arm got right, which "
                   "is the most any of them could have scored. Context for "
                   "the accuracy column, never subtracted from it.")
        out.append("")
        rows = []
        for dataset in agreed:
            stats = _consensus(collected, dataset)
            best = max(averaged[dataset][arm]["accuracy"]
                       for arm in averaged[dataset])
            rows.append((f"{dataset} ({_options(collected, dataset)})",
                         f"{stats['agreed']:.2f}",
                         f"{stats['consensus_wrong']:.3f}",
                         f"{stats['ceiling']:.3f}",
                         f"{best:.3f}",
                         f"{stats['ceiling'] - best:+.3f}"))
        out += table(rows, ["dataset (options)", "arms agreed",
                            "consensus wrong", "ceiling", "best arm",
                            "headroom to ceiling"])

    # -- overall ----------------------------------------------------------
    out.append("## Overall")
    out.append("")
    rows = []
    for arm in arms:
        present = [d for d in datasets if arm in averaged[d]]
        norm = [normalised(averaged[d][arm]["accuracy"],
                           averaged[d][arm]["majority"]) for d in present]
        below = sum(1 for value in norm if value <= 0)
        spreads = [averaged[d][arm]["spread"] for d in present]
        rows.append((arm, len(present), f"{sum(norm) / len(norm):.3f}",
                     f"{below}", f"{quantile(spreads, 0.5):.3f}",
                     f"{sum(averaged[d][arm]['coverage_95'] for d in present) / len(present):.3f}"))
    out += table(rows, ["arm", "datasets", "mean normalised accuracy",
                        "sets lost to majority baseline", "median replicate spread",
                        "mean coverage at 95% accuracy"])

    # -- pairwise ---------------------------------------------------------
    out.append("## Pairwise, across datasets")
    out.append("")
    for a, b in combinations(arms, 2):
        shared = [d for d in datasets if a in averaged[d] and b in averaged[d]]
        if len(shared) < 3:
            continue
        diffs = [averaged[d][a]["accuracy"] - averaged[d][b]["accuracy"]
                 for d in shared]
        wins = sum(1 for value in diffs if value > TIE)
        losses = sum(1 for value in diffs if value < -TIE)
        test = wilcoxon(diffs)
        out.append(f"**{a} vs {b}** over {len(shared)} datasets: "
                   f"{wins} win / {len(shared) - wins - losses} tie / "
                   f"{losses} loss (ties within ±{TIE:.2f}, the measured "
                   f"run-to-run noise). Median difference "
                   f"{quantile(diffs, 0.5):+.3f}; Wilcoxon signed-rank "
                   f"p = {test['p']:.4f} over {test['n']} non-zero.")
        out.append("")

    # -- by option count --------------------------------------------------
    out.append("## By option count")
    out.append("")
    out.append("Paired: a band averages only the datasets on which every "
               "product arm answered, so each column in a row covers the same "
               "sets. Averaging each arm over whatever it could answer instead "
               "would credit an arm with the sets its rivals cannot run -- "
               "clinc150 for Jev at the wide end, rvl_cdip for Privatemode in "
               "the middle -- and compare different questions under one "
               "heading. A control arm is shown only where it ran on all of a "
               "band's sets.")
    out.append("")
    rows = []
    for label, members, means in option_bands(
            averaged, datasets, lambda d: _options(collected, d), products, arms):
        rows.append((label, len(members),
                     *("—" if means[arm] is None else f"{means[arm]:.3f}" for arm in arms)))
    out += table(rows, ["options", "sets", *arms])

    # -- ladders ----------------------------------------------------------
    out.append("## Ladders — the same examples at two option counts")
    out.append("")
    out.append("The only place the option-count effect is measured rather "
               "than inferred: identical examples, identical arms, one "
               "variable.")
    out.append("")
    rows = []
    for name, rungs in ladders().items():
        have = [rung for rung in rungs if rung.name in averaged]
        if len(have) < 2:
            continue
        narrow, wide = have[0], have[-1]
        for arm in arms:
            if arm not in averaged[narrow.name] or arm not in averaged[wide.name]:
                continue
            low = averaged[narrow.name][arm]["accuracy"]
            high = averaged[wide.name][arm]["accuracy"]
            rows.append((name, arm,
                         f"{_options(collected, narrow.name)} → "
                         f"{_options(collected, wide.name)}",
                         f"{low:.3f}", f"{high:.3f}", f"{high - low:+.3f}"))
    out += table(rows, ["ladder", "arm", "options", "narrow", "wide", "change"])

    # -- memorisation -----------------------------------------------------
    renamed = average(collect(root, "rename"))
    shared = [d for d in datasets if d in renamed]
    if shared:
        out.append("## Memorisation: the same task with renamed labels")
        out.append("")
        out.append("Every option swapped for a frozen synonym, nothing else "
                   "changed. An arm that reads the state and reasons about "
                   "the options should barely move; one that has learned the "
                   "label string should fall.")
        out.append("")
        out.append("A renaming is never perfectly neutral -- some synonyms "
                   "are simply harder words -- so every arm loses a little. "
                   "That common loss is the perturbation, not memorisation. "
                   "`excess` is the drop beyond the median arm's on the same "
                   "dataset, and it is the column to read. It needs at least "
                   "three arms: the median of two is their mean, so the "
                   "excess would be symmetric by construction and would say "
                   "nothing at all. With fewer it is left blank and only the "
                   "raw drop is reported.")
        out.append("")
        rows = []
        for dataset in shared:
            common = sorted(set(averaged[dataset]) & set(renamed[dataset]))
            if len(common) < 2:
                continue
            drops = {arm: averaged[dataset][arm]["accuracy"]
                     - renamed[dataset][arm]["accuracy"] for arm in common}
            median = quantile(sorted(drops.values()), 0.5)
            # The median of two values is their mean, so with two arms the
            # excess is +x and -x whatever the data says. Blank is honest.
            comparable = len(common) >= 3
            for arm in common:
                rows.append((f"{dataset} ({_options(collected, dataset)})", arm,
                             f"{averaged[dataset][arm]['accuracy']:.3f}",
                             f"{renamed[dataset][arm]['accuracy']:.3f}",
                             f"{-drops[arm]:+.3f}",
                             f"{-(drops[arm] - median):+.3f}" if comparable
                             else "—"))
        out += table(rows, ["dataset (options)", "arm", "original", "renamed",
                            "drop", "excess"])

    # -- cost --------------------------------------------------------------
    out.append("## Cost per decision")
    out.append("")
    out.append("Billed from each vendor's own `usage` block at the rates in "
               "`bench/pricing.py`, never from a token count of ours. The "
               "arms serialize state and options differently, so the token "
               "counts are reported next to the prices.")
    out.append("")
    rows = []
    for dataset in datasets:
        cells = [f"{dataset} ({_options(collected, dataset)})"]
        for arm in arms:
            values = averaged[dataset].get(arm)
            if values is None:
                cells.append("—")
            elif arm in LOCAL:
                cells.append("local")
            else:
                cells.append(f"{values['input_tokens']:.0f} tok · "
                             f"{values['eur_per_1k']:.4f}")
        rows.append(tuple(cells))
    out += table(rows, ["dataset (options)", *arms])

    # Cost divided by accuracy, so the two columns can be read together
    # rather than traded off by hand. Over the datasets every priced arm
    # answered, like the paired accuracy comparison: a median over different
    # sets of datasets compares the sets as much as the arms.
    priced = [arm for arm in arms if arm not in LOCAL]
    common = [d for d in datasets if all(arm in averaged[d] for arm in priced)]
    out.append("### Cost per 1000 correct decisions")
    out.append("")
    out.append(f"Medians over the {len(common)} datasets every priced arm answered.")
    out.append("")
    rows = []
    for arm in arms:
        if arm in LOCAL:
            rows.append((arm, "local", "local"))
            continue
        present = common
        raw = [averaged[d][arm]["eur_per_1k"] for d in present]
        per_right = [averaged[d][arm]["eur_per_1k"] / averaged[d][arm]["accuracy"]
                     for d in present if averaged[d][arm]["accuracy"] > 0]
        rows.append((arm, f"{quantile(raw, 0.5):.4f}",
                     f"{quantile(per_right, 0.5):.4f}" if per_right else "—"))
    out += table(rows, ["arm", "median EUR / 1000 answers",
                        "median EUR / 1000 correct"])

    # -- latency -----------------------------------------------------------
    quiet = latencies(root)
    out.append("## Latency")
    out.append("")
    rows = []
    for arm in arms:
        if arm in LOCAL:
            rows.append((arm, "not measured", "not measured", "—", "—", "—"))
            continue
        present = [d for d in quiet if arm in quiet[d]]
        if not present:
            rows.append((arm, "no quiet run", "—", "—", "—", "0"))
            continue
        lat = [quiet[d][arm]["p50_ms"] for d in present]
        floor = [quiet[d][arm]["p10_ms"] for d in present]
        tail = [quiet[d][arm]["p95_ms"] for d in present]
        throttled = sum(quiet[d][arm]["throttled"] for d in present)
        rows.append((arm, str(len(present)),
                     f"{quantile(floor, 0.5):.0f} ms",
                     f"{quantile(lat, 0.5):.0f} ms",
                     f"{quantile(tail, 0.5):.0f} ms",
                     str(throttled)))
    out += table(rows, ["arm", "datasets", "median p10 (unqueued floor)",
                        "median p50", "median p95", "throttled"])
    out.append("From concurrency-1 runs only, whatever their size; `no quiet "
               "run` means there is none yet and nothing is reported rather "
               "than a number taken under load. A p50 well above the p10 "
               "floor, or any throttled response, means an arm was queueing "
               "rather than computing. Laya is not speed-measured: it runs on "
               "a laptop while the others run in a datacentre.")
    out.append("")
    if quiet:
        out.append("### Latency per dataset")
        out.append("")
        rows = []
        for dataset in sorted(quiet, key=lambda d: _options(collected, d)
                              if d in collected else 0):
            cells = [f"{dataset} ({_options(collected, dataset)})"
                     if dataset in collected else dataset]
            for arm in arms:
                values = quiet[dataset].get(arm)
                cells.append("—" if values is None or arm in LOCAL else
                             f"{values['p50_ms']:.0f} / {values['p10_ms']:.0f}")
            rows.append(tuple(cells))
        out += table(rows, ["dataset (options)", *arms])
        out.append("p50 / p10 in ms.")
        out.append("")
    return "\n".join(out)


BANDS = [(2, 2, "2"), (3, 6, "3–6"), (7, 20, "7–20"),
         (21, 80, "21–80"), (81, 10 ** 6, "81+")]


def option_bands(averaged: dict, datasets: list[str], options_of, products: list[str],
                 arms: list[str]) -> list[tuple[str, list[str], dict]]:
    """Mean accuracy per option-count band, over the same sets for every arm.

    A band holds only the datasets on which every product arm answered; an
    arm that did not run on all of them (a control that has not finished,
    say) gets ``None`` rather than a mean over a different set. The unpaired
    version this replaced credited Jev with clinc150 and Privatemode with
    rvl_cdip -- sets the other arms cannot run -- and turned a 0.7-point gap
    at the wide end into a 2.4-point one.
    """
    out = []
    for low, high, label in BANDS:
        members = [d for d in datasets if low <= options_of(d) <= high
                   and all(arm in averaged[d] for arm in products)]
        if not members:
            continue
        means = {arm: (sum(averaged[d][arm]["accuracy"] for d in members) / len(members)
                       if all(arm in averaged[d] for d in members) else None)
                 for arm in arms}
        out.append((label, members, means))
    return out


def _options(collected: dict, dataset: str) -> int:
    meta, _ = next(iter(collected[dataset].values()))
    return meta["options"]


def _consensus(collected: dict, dataset: str) -> dict:
    """Averaged over replicates, which smooths the run-to-run flips."""
    stats = [meta.get("consensus") or {}
             for meta, _ in collected[dataset].values()]
    stats = [s for s in stats if s]
    if not stats:
        return {}
    return {key: sum(s[key] for s in stats) / len(stats)
            for key in ("agreed", "consensus_wrong", "ceiling")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default="results")
    parser.add_argument("--write", help="also write the markdown here")
    args = parser.parse_args(argv)
    text = summarise(Path(args.root))
    print(text)
    if args.write:
        Path(args.write).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
