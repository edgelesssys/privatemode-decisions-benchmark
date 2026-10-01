"""Confirming a prompt variant on the test halves, with the suite's rules.

    python -m bench.prefill_confirm --runs runs/confirm --variant R-Q \\
        --published published/results --out report/ [--renamed runs/renamed]

``--runs`` holds ``B-r1``, ``B-r2``, ``<variant>-r1`` and ``<variant>-r2`` from
``bench.prefill --split test`` (two replicates each). Reported:

* accuracy per dataset (mean of the replicates, with their spread), wins,
  ties and losses against the baseline (ties within ±0.01, the suite's
  rule), the Wilcoxon signed-rank test across datasets, and the option-count
  bands;
* the headroom closed, ``(variant − B) / (glm-cot − B)``, and Jev and Laya
  on the same examples, from the published runs: accuracy normalised
  against the majority class of the rows scored, wins, ties and losses
  against Jev with the Wilcoxon test, and the median price per decision;
* cost and latency against the baseline and ``glm-cot``;
* whether the shipped default temperature still fits the variant's
  probabilities: excess ECE with the shipped formula against a formula
  refitted on the variant (leave one dataset out), the regression check a
  prompt change needs;
* the label-renaming control (``--renamed``: ``B`` and the variant on renamed
  options), and MMLU-Pro against the published Jev and openjev-sglang
  numbers on the same 1,000 questions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import calibration as c
from .aggregate import TIE
from .calibrate_report import fmt, table
from .metrics import normalised, wilcoxon
from .prefill_report import EUR_IN, EUR_OUT, load, matrix
from decisions.calibration import FORMULAS

BANDS = ((2, 2, "2"), (3, 6, "3–6"), (7, 20, "7–20"), (21, 80, "21–80"), (81, 10_000, "81+"))
#: The library's default for GLM-5.3-Flash: log T = a + b · log(options).
SHIPPED = tuple(FORMULAS["glm-5.3-flash"])
#: Published on the same 1,000 MMLU-Pro questions by ekzhang/openjev-sglang
#: (evals/results/mmlu-pro-2026-09-18): Jev resolved to jev-1.13-20260917.
MMLU_PUBLISHED = {"Jev (jev-1.13-20260917)": 0.829, "openjev-sglang (Qwen3.6-35B-A3B)": 0.588}


def accuracy_of(rows: dict[int, dict], order: list[int]) -> float:
    P, y, _ = matrix(rows, order)
    return c.accuracy(P, y)


#: Published on JevBench v1.4.2 (results/v1.4.2/jevbench-v1.4.2-results.json):
#: public-item accuracy (231 items) and the hard tier over 220 items.
JEVBENCH_PUBLISHED = {"Jev 1.13.0": (0.866, 0.741), "GPT-6 Luna, medium effort": (0.996, 0.986),
                      "DeepSeek V4.1 Flash, thinking": (0.978, 0.950),
                      "JevK5 v0.2": (0.853, 0.700), "openjev-sglang (Qwen3.6-35B-A3B)": (0.853, 0.714)}
JEVBENCH_HARD = 120   # items from this index on are the public hard tier (111)


def jevbench_section(args, runs, arms, result) -> list[str]:
    """Accuracy on JevBench's 231 public items, and ECE on its 111 public
    hard items with JevBench's own metric (10 equal-width bins), for our
    arms as the library returns them (default temperature); and the runs of
    JevBench's own ``openai_compat`` adapter (structured output + thinking)
    if ``--jevbench-runs`` has them. Nothing here is fitted on the items."""
    import sys
    sys.path.insert(0, args.jevbench)
    from jevbench.metrics import ece_top_label
    from decisions.calibration import default_temperature, scale

    def score(rows, temperature=True):
        out = []
        for r in rows.values():
            p = r["probabilities"]
            if temperature:
                p = scale(p, default_temperature("glm-5.3-flash", len(p)))
            out.append((r["index"], max(p.values()), max(p, key=p.get) == r["gold"],
                        r.get("generated_tokens", 0), r["latency_s"]))
        hard = [(conf, ok) for i, conf, ok, _, _ in out if i >= JEVBENCH_HARD]
        return (np.mean([ok for _, _, ok, _, _ in out]), np.mean([ok for _, ok in hard]),
                ece_top_label(hard)["ece"], np.mean([g for *_, g, _ in out]),
                float(np.median([t for *_, t in out])))

    rows = []
    for label, a in (("GLM-5.3-Flash, one pass", "B"), (f"GLM-5.3-Flash, one pass, {args.variant}", args.variant)):
        vals = [score(run["jevbench"]) for run in arms[a]]
        m = np.mean(vals, axis=0)
        rows.append([label, "no", "1", f"{m[0]:.3f}", f"{m[1]:.3f}", f"{m[2]:.3f}", f"{m[4]:.2f} s"])
    for extra in args.mmlu_extra.split(",") if args.mmlu_extra else []:
        if extra in runs and "jevbench" in runs[extra]:
            v = score(runs[extra]["jevbench"], temperature=False)
            rows.append([f"GLM-5.3-Flash, {extra[2:].split('-')[0]} thinking tokens, then the read",
                         "yes", f"{v[3]:.0f}", f"{v[0]:.3f}", f"{v[1]:.3f}", f"{v[2]:.3f} (raw)", f"{v[4]:.2f} s"])
    if args.jevbench_runs:
        from jevbench.summarize import summarize
        from jevbench.tasks import load_jsonl
        for model_dir in sorted(Path(args.jevbench_runs).iterdir()):
            records, tasks, by_tier = [], [], {}
            for tier in ("original", "easy", "hard"):
                path = model_dir / tier / "results.jsonl"
                if not path.exists():
                    break
                t = load_jsonl(f"{args.jevbench}/datasets/public/{tier}.jsonl")
                r = [json.loads(line) for line in path.open()]
                by_tier[tier] = summarize(t, r)
                records += r
                tasks += t
            else:
                total = summarize(tasks, records)
                tokens = np.mean([(r.get("usage") or {}).get("output_tokens") or 0 for r in records])
                rows.append([f"{model_dir.name}, structured output + thinking (JevBench's runner)",
                             "yes", f"{tokens:.0f}", f"{total['accuracy']:.3f}",
                             f"{by_tier['hard']['accuracy']:.3f}", f"{by_tier['hard']['ece']['ece']:.3f}",
                             f"{total['latency']['p50_s']:.2f} s"])
    rows += [[f"*{name}, published*", "", "", f"{pub:.3f}", f"*{hard:.3f} (220)*", "", ""]
             for name, (pub, hard) in JEVBENCH_PUBLISHED.items()]
    result["jevbench"] = rows
    return ["\n## JevBench, public items\n",
            "The 231 public items of [JevBench](https://github.com/fstandhartinger/jevbench) "
            "(MIT), used for comparison only: nothing was chosen, fitted or tuned on them. Yes/no "
            "questions become the options no/yes described by the false/true criteria, score "
            "questions one option per level, as JevBench's own open-model adapter maps them. "
            "*Hard* is the 111 public hard items, ECE JevBench's own (10 equal-width bins); the "
            "published hard tier also has 109 held-out items. Latency at several requests in "
            "flight.\n",
            table(rows, ["system", "reasoning", "output tokens", "public items", "public hard",
                         "ECE, public hard", "latency p50"])]


def products_section(per, text, shared, v, wtl) -> tuple[list[str], dict]:
    """The suite's headline on the test halves: normalised accuracy against
    the majority class of the rows scored, over the datasets each arm
    answers (our arms on all rows, a published arm on the rows it answered);
    against Jev, mean accuracy, wins, ties and losses on the rows both
    answered; and the median price over the datasets Jev answers (prompt
    tokens at list price)."""
    columns = {"B": "B", v: "variant", "Jev": "Jev", "Laya": "Laya"}
    rows, summary = [], {}
    for label, key in columns.items():
        present = [n for n in text if key in per[n]]
        if not present:
            continue
        majority = "majority" if key in ("B", "variant") else f"majority@{key}"
        summary[label] = {"datasets": len(present), "normalised": float(np.mean(
            [normalised(per[n][key], per[n][majority]) for n in present]))}
        at_jev = key if key == "Jev" else f"{key}@Jev"
        cells = [label, len(present), f"{summary[label]['normalised']:.3f}",
                 f"{np.mean([per[n][at_jev] for n in shared]):.3f}"
                 if all(at_jev in per[n] for n in shared) else "—"]
        if key in ("B", "variant") and shared:
            tokens = "tokens_B" if key == "B" else "tokens_v"
            diffs = [per[n][f"{key}@Jev"] - per[n]["Jev"] for n in shared]
            test = wilcoxon(diffs)
            cells += ["–".join(map(str, wtl(key))), f"{test['p']:.2g}",
                      f"{(np.median([per[n][tokens] for n in shared]) * EUR_IN + EUR_OUT) * 1000:.4f}"]
            summary[label]["p_vs_jev"] = test["p"]
        else:
            cells += ["", "", ""]
        rows.append(cells)
    return ["\n**Against Jev and Laya.** Normalised accuracy is 0 for always answering the "
            "majority class of the rows scored and 1 for all right, averaged over the datasets "
            "an arm answers (29, 28 and 27: not the same sets). The mean accuracy, the "
            "wins–ties–losses and the Wilcoxon test are against Jev on the rows both answered, "
            "and the median price is over Jev's datasets (prompt tokens at list price):\n",
            table(rows, ["arm", "datasets", "normalised accuracy", "mean accuracy, Jev's datasets",
                         "against Jev", "Wilcoxon p", "EUR / 1000, median"])], summary


def report(args) -> tuple[str, dict]:
    runs = load(Path(args.runs))
    v = args.variant
    arms = {"B": [runs[f"B-r{r}"] for r in (1, 2) if f"B-r{r}" in runs],
            v: [runs[f"{v}-r{r}"] for r in (1, 2) if f"{v}-r{r}" in runs]}
    names = sorted(set.intersection(*(set(run) for reps in arms.values() for run in reps)))
    text = [n for n in names if n not in ("mmlu_pro", "jevbench")]
    published = {}
    if args.published:
        from .calibrate_extensions import load_arm
        published = {"Jev": load_arm(Path(args.published), "jev"),
                     "glm-cot": load_arm(Path(args.published), "glm-cot"),
                     "Laya": load_arm(Path(args.published), "laya")}
    per, rows = {}, []
    for n in text:
        order = sorted(set.intersection(*(set(run[n]) for reps in arms.values() for run in reps)))
        accs = {a: [accuracy_of(run[n], order) for run in reps] for a, reps in arms.items()}
        first = arms["B"][0][n]
        k = len(first[order[0]]["probabilities"])
        golds = [first[i]["gold"] for i in order]
        entry = {"options": k, "rows": len(order),
                 "majority": max(golds.count(g) for g in set(golds)) / len(golds),
                 "B": float(np.mean(accs["B"])), "variant": float(np.mean(accs[v])),
                 "spread": float(max(np.ptp(accs["B"]), np.ptp(accs[v]))),
                 "tokens_B": float(np.mean([first[i]["prompt_tokens"] for i in order])),
                 "tokens_v": float(np.mean([arms[v][0][n][i]["prompt_tokens"] for i in order])),
                 "latency_B": float(np.median([first[i]["latency_s"] for i in order])),
                 "latency_v": float(np.median([arms[v][0][n][i]["latency_s"] for i in order]))}
        for label, data in published.items():
            d = data.get(n)
            if d is not None:
                keep = np.isin(d.index, order)
                if keep.sum() >= 20:
                    # The published arm and ours on the rows both answered.
                    both = sorted(int(i) for i in d.index[keep])
                    entry[label] = c.accuracy(d.P[keep], d.y[keep])
                    entry[f"{label} rows"] = len(both)
                    entry[f"B@{label}"] = float(np.mean([accuracy_of(r[n], both) for r in arms["B"]]))
                    entry[f"variant@{label}"] = float(np.mean([accuracy_of(r[n], both)
                                                               for r in arms[v]]))
                    golds = [first[i]["gold"] for i in both]
                    entry[f"majority@{label}"] = max(golds.count(g) for g in set(golds)) / len(golds)
        per[n] = entry
        diff = entry["variant"] - entry["B"]
        rows.append([n, k, entry["rows"], f"{entry['B']:.3f}", f"{entry['variant']:.3f}",
                     f"{diff * 100:+.1f}", f"{entry['spread'] * 100:.1f}",
                     fmt(entry.get("Jev"), 3), fmt(entry.get("Laya"), 3), fmt(entry.get("glm-cot"), 3)])
    diffs = [per[n]["variant"] - per[n]["B"] for n in text]
    wins = sum(d > TIE for d in diffs)
    losses = sum(d < -TIE for d in diffs)
    test = wilcoxon(diffs)
    md = [f"# `{v}` against the baseline on the test halves\n",
          f"GLM-5.3-Flash, the test halves of {len(text)} datasets, two replicates of each arm; "
          f"accuracy is the mean of the replicates. Jev, Laya and glm-cot are their published "
          f"runs on the rows of these they answered.\n",
          f"**{wins} wins, {len(text) - wins - losses} ties, {losses} losses** (ties within "
          f"±{TIE:.2f}); median difference {np.median(diffs) * 100:+.1f} points, mean "
          f"{np.mean(diffs) * 100:+.2f}; Wilcoxon signed-rank p = {test['p']:.3g}.\n"]
    md.append(table(rows, ["dataset", "options", "rows", "B", v, "points", "replicate spread",
                           "Jev", "Laya", "glm-cot"]))

    # Bands.
    band_rows = []
    for low, high, label in BANDS:
        members = [n for n in text if low <= per[n]["options"] <= high]
        if not members:
            continue
        jev = [per[n]["Jev"] for n in members if "Jev" in per[n]]
        band_rows.append([label, len(members), f"{np.mean([per[n]['B'] for n in members]):.3f}",
                          f"{np.mean([per[n]['variant'] for n in members]):.3f}",
                          f"{np.mean(jev):.3f}" if len(jev) == len(members) else "—"])
    md.append("\nBy option count (Jev where it answered every dataset in the band):\n")
    md.append(table(band_rows, ["options", "datasets", "B", v, "Jev"]))

    # Headroom, Jev, cost.
    # glm-cot against B and the variant on the rows glm-cot answered.
    closed = [(per[n]["variant@glm-cot"] - per[n]["B@glm-cot"]) / (per[n]["glm-cot"] - per[n]["B@glm-cot"])
              for n in text if "glm-cot" in per[n] and per[n]["glm-cot"] - per[n]["B@glm-cot"] > 0.02]
    shared = [n for n in text if "Jev" in per[n]]
    eur_b = np.mean([per[n]["tokens_B"] * EUR_IN + EUR_OUT for n in text]) * 1000
    eur_v = np.mean([per[n]["tokens_v"] * EUR_IN + EUR_OUT for n in text]) * 1000
    md.append(f"\n**Headroom closed** on the {len(closed)} datasets where glm-cot is more than 2 "
              f"points ahead of the baseline: median {np.median(closed):.0%}.\n" if closed else "")
    products = {}
    if shared:
        # Against Jev on the rows both answered (key@Jev), not on all of ours.
        wtl = lambda key: (sum(per[n][f"{key}@Jev"] > per[n]["Jev"] + TIE for n in shared),
                           sum(abs(per[n][f"{key}@Jev"] - per[n]["Jev"]) <= TIE for n in shared),
                           sum(per[n][f"{key}@Jev"] < per[n]["Jev"] - TIE for n in shared))
        md.append(f"\n**Against Jev** on the {len(shared)} datasets both answer, on the rows both "
                  f"answered ({sum(per[n]['Jev rows'] for n in shared):,} of "
                  f"{sum(per[n]['rows'] for n in shared):,}): the baseline "
                  f"{'–'.join(map(str, wtl('B')))} (wins–ties–losses), `{v}` "
                  f"{'–'.join(map(str, wtl('variant')))}; mean accuracy "
                  f"{np.mean([per[n]['B@Jev'] for n in shared]):.3f} and "
                  f"{np.mean([per[n]['variant@Jev'] for n in shared]):.3f} against Jev's "
                  f"{np.mean([per[n]['Jev'] for n in shared]):.3f}.\n")
        products_md, products = products_section(per, text, shared, v, wtl)
        md += products_md
    md.append(f"\n**Cost and latency.** Prompt tokens {np.mean([per[n]['tokens_B'] for n in text]):.0f} → "
              f"{np.mean([per[n]['tokens_v'] for n in text]):.0f} on average, EUR {eur_b:.3f} → "
              f"{eur_v:.3f} per 1,000 decisions (glm-cot: about 0.35). Median latency "
              f"{np.median([per[n]['latency_B'] for n in text]) * 1000:.0f} → "
              f"{np.median([per[n]['latency_v'] for n in text]) * 1000:.0f} ms at 4 requests in "
              f"flight, with both replicates running at once: that is load, not the prompt. The "
              f"concurrency-1 measurement below is the one to use.\n")
    if args.latency:
        lat = load(Path(args.latency))
        rows = []
        for n in sorted(lat["B"], key=lambda m: len(next(iter(lat["B"][m].values()))["probabilities"])):
            cells = [n, len(next(iter(lat["B"][n].values()))["probabilities"])]
            for a in ("B", "B2", v):
                if n in lat.get(a, {}):
                    values = np.array([r["latency_s"] for r in lat[a][n].values()])
                    tokens = np.mean([r["prompt_tokens"] for r in lat[a][n].values()])
                    cells.append(f"{np.median(values) * 1000:.0f} / {np.percentile(values, 95) * 1000:.0f} "
                                 f"({tokens:.0f})")
            rows.append(cells)
        md.append("\n**Latency at concurrency 1** (one request at a time, 80 test rows per dataset; "
                  "p50 / p95 in ms, prompt tokens in brackets; B2 is the baseline again, for the "
                  "noise):\n")
        md.append(table(rows, ["dataset", "options", "B", "B2", v]))

    # Does the shipped temperature still fit?
    options = {n: per[n]["options"] for n in text}
    parts = {a: {n: matrix(arms[a][0][n], sorted(arms[a][0][n])) for n in text} for a in arms}
    rows, calib = [], {}
    for a in arms:
        halves = {}
        for n in text:
            P, y, _ = parts[a][n]
            idx = np.arange(len(y))
            halves[n] = ((P[idx % 2 == 0], y[idx % 2 == 0]), (P[idx % 2 == 1], y[idx % 2 == 1]))
        oracle = {n: c.fit_temperature([halves[n][0]]) for n in text}
        refit = c.fit_formula(oracle, options)
        shipped_ex, refit_ex, raw_ex = [], [], []
        for n in text:
            a_, b_ = c.fit_formula({m: oracle[m] for m in text if m != n}, options)
            P, y = halves[n][1]
            for store, t in ((raw_ex, 1.0), (shipped_ex, c.formula_temperature(*SHIPPED, options[n])),
                             (refit_ex, c.formula_temperature(a_, b_, options[n]))):
                Q = c.scale(P, t)
                store.append(c.ece(Q, y) - c.ece_floor(Q, draws=50))
        calib[a] = {"formula": refit, "median_t": float(np.median(list(oracle.values()))),
                    "raw": float(np.mean(raw_ex)), "shipped": float(np.mean(shipped_ex)),
                    "refit": float(np.mean(refit_ex))}
        rows.append([a, f"{calib[a]['median_t']:.2f}", f"{refit[0]:.3f}, {refit[1]:+.3f}",
                     f"{calib[a]['raw']:.3f}", f"{calib[a]['shipped']:.3f}", f"{calib[a]['refit']:.3f}"])
    md.append("\n## Does the default temperature still fit?\n")
    md.append(f"Each dataset's test rows split in two: temperatures fitted on one half, scored on "
              f"the other, excess ECE (ECE minus the sampling floor). *Shipped* is the library's "
              f"formula (a = {SHIPPED[0]}, b = {SHIPPED[1]}), fitted on the baseline prompt; "
              f"*refitted* is the formula fitted on this arm, leaving the dataset out.\n")
    md.append(table(rows, ["arm", "task T, median", "formula a, b", "excess ECE raw",
                           "with the shipped formula", "with a refitted formula"]))

    # Renaming control.
    result = {"per": per, "wins": wins, "losses": losses, "p": test["p"],
              "median": float(np.median(diffs)), "mean": float(np.mean(diffs)),
              "closed": float(np.median(closed)) if closed else None, "calibration": calib,
              "eur": (float(eur_b), float(eur_v)), "products": products}
    if args.renamed:
        renamed = load(Path(args.renamed))
        rows, drops = [], {}
        for a in ("B", v):
            if a not in renamed:
                continue
            d = []
            for n in sorted(renamed[a]):
                if n not in per:
                    continue
                order = sorted(set(renamed[a][n]) & set(arms[a][0][n]))
                d.append(accuracy_of(renamed[a][n], order) - accuracy_of(arms[a][0][n], order))
            drops[a] = float(np.mean(d))
            rows.append([a, len(d), f"{drops[a] * 100:+.2f}"])
        md.append("\n## Label-renaming control\n")
        md.append("The same test rows with every option renamed to a synonym (the suite's "
                  "contamination control): accuracy with renamed options minus with the "
                  "originals. A variant that leans more on memorised label strings loses more.\n")
        md.append(table(rows, ["arm", "datasets", "points, renamed − original"]))
        result["renamed"] = drops

    # MMLU-Pro.
    if all("mmlu_pro" in run for reps in arms.values() for run in reps):
        order = sorted(set.intersection(*(set(run["mmlu_pro"]) for reps in arms.values() for run in reps)))
        rows = [[a, f"{np.mean([accuracy_of(run['mmlu_pro'], order) for run in reps]):.1%}",
                 "this run"] for a, reps in arms.items()]
        for extra in args.mmlu_extra.split(",") if args.mmlu_extra else []:
            if extra in runs and "mmlu_pro" in runs[extra]:
                o = sorted(runs[extra]["mmlu_pro"])
                rows.append([extra, f"{accuracy_of(runs[extra]['mmlu_pro'], o):.1%}",
                             f"this run, {len(o)} questions"])
        rows += [[name, f"{acc:.1%}", "published, same questions"] for name, acc in MMLU_PUBLISHED.items()]
        md.append("\n## MMLU-Pro\n")
        md.append(f"The {len(order)} test questions openjev-sglang sampled (seed 42, revision "
                  f"b189ec76), asked the same way: the question as the state, the options as "
                  f"letters with their texts as descriptions, zero-shot. Jev's and openjev's "
                  f"numbers are theirs, not paired with ours.\n")
        md.append(table(rows, ["system", "accuracy", "source"]))
        result["mmlu_pro"] = {r[0]: r[1] for r in rows}
    # JevBench's public items.
    if args.jevbench and all("jevbench" in run for reps in arms.values() for run in reps):
        md += jevbench_section(args, runs, arms, result)
    return "\n".join(md) + "\n", result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--published")
    parser.add_argument("--renamed")
    parser.add_argument("--latency", help="bench.prefill --threads 1 runs of B, B2 and the variant")
    parser.add_argument("--mmlu-extra", default="", help="more arms under --runs to list for MMLU-Pro")
    parser.add_argument("--jevbench", help="a JevBench checkout, for its public items and metrics")
    parser.add_argument("--jevbench-runs", help="JevBench runner output, <model>/<tier>/results.jsonl")
    args = parser.parse_args()
    text, data = report(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "confirm.md").write_text(text)
    (out / "confirm.json").write_text(json.dumps(data, indent=1, default=float))
    print(f"wrote {out / 'confirm.md'}")


if __name__ == "__main__":
    main()
