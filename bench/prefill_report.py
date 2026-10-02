"""The screening report for ``bench.prefill``: every arm against the baseline.

    python -m bench.prefill_report --runs runs/prefill --out report/ [--baseline B]

Everything is paired: an arm is compared with the baseline on the rows both
answered. Per arm:

* accuracy change, pooled with a paired bootstrap interval and per dataset
  with an exact McNemar test, and per task family;
* NLL and ECE after a temperature fitted *per arm* on half of the rows and
  scored on the other half, so an arm that only sharpens or softens doesn't
  look better or worse, and the fitted temperature itself;
* option mass, how far the mean answer moves from the baseline's (total
  variation of the mean distribution, which is also the position prior since
  the option order is fixed), extra prompt tokens, latency and EUR per 1,000;
* the accuracy change by the baseline's confidence, and two offline
  ensembles: the mean of baseline and arm, and the arm only where the
  baseline is unsure.

``B2`` is the baseline asked again, the noise floor between runs at
temperature 0.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import calibration as c
from .calibrate_report import fmt, table
from .pricing import PRIVATEMODE_EUR_PER_MTOK
from .specs import BY_NAME

#: GLM-5.3-Flash's list prices per token, input and output (bench.pricing).
EUR_IN, EUR_OUT = (eur / 1e6 for eur in PRIVATEMODE_EUR_PER_MTOK["glm-5.3-flash"][:2])
CONTROLS = ("sst2", "dbpedia_14", "banking77", "clinc150")
GROUPS = {"nli": "NLI / QA", "qa": "NLI / QA", "sentiment": "sentiment", "moderation": "sentiment",
          "topic": "topic", "legal": "topic", "intent": "intent"}


def load(directory: Path) -> dict[str, dict[str, dict[int, dict]]]:
    arms = {}
    for arm_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        data = {}
        for path in sorted(arm_dir.glob("*.jsonl")):
            rows = {}
            for line in path.open():
                row = json.loads(line)
                rows[row["index"]] = row
            if rows:
                data[path.stem] = rows
        if data:
            arms[arm_dir.name] = data
    return arms


def matrix(rows: dict[int, dict], order: list[int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Probabilities per row over the dataset's options. Where the options
    differ per question (MMLU-Pro: 3 to 10 letters), over all of them, with 0
    for a letter a question doesn't have."""
    options = max((list(rows[i]["probabilities"]) for i in order), key=len)
    for i in order:
        options += [o for o in rows[i]["probabilities"] if o not in options]
    P = np.array([[rows[i]["probabilities"].get(o, 0.0) for o in options] for i in order])
    y = np.array([options.index(rows[i]["gold"]) for i in order])
    return P / P.sum(axis=1, keepdims=True), y, options


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar p-value for paired correctness vectors."""
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * c.binomial_cdf(k, n, 0.5))


def bootstrap(diffs: np.ndarray, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(draws)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(base: dict[str, dict[int, dict]], arm: dict[str, dict[int, dict]]) -> dict:
    per, pooled_a, pooled_b, pooled_conf = {}, [], [], []
    for name in base:
        if name not in arm:
            continue
        order = sorted(set(base[name]) & set(arm[name]))
        if len(order) < 20:
            continue
        Pb, y, options = matrix(base[name], order)
        Pa, _, _ = matrix(arm[name], order)
        right_b, right_a = Pb.argmax(1) == y, Pa.argmax(1) == y
        # A temperature per arm on half the rows, scored on the other half.
        half = np.arange(len(order)) % 2 == 0
        scores = {}
        for label, P in (("base", Pb), ("arm", Pa)):
            t = c.fit_temperature([(P[half], y[half])])
            Q = c.scale(P[~half], t)
            scores[label] = (t, c.nll(Q, y[~half]), c.ece(Q, y[~half]))
        mass = [arm[name][i].get("option_mass", 1.0) for i in order]
        extra = [arm[name][i]["prompt_tokens"] - base[name][i]["prompt_tokens"] for i in order]
        per[name] = {
            "rows": len(order), "base": float(right_b.mean()), "arm": float(right_a.mean()),
            "p": mcnemar(right_a, right_b),
            "t_base": scores["base"][0], "t_arm": scores["arm"][0],
            "nll_base": scores["base"][1], "nll_arm": scores["arm"][1],
            "ece_base": scores["base"][2], "ece_arm": scores["arm"][2],
            "mass": float(np.mean(mass)), "mass_min": float(np.min(mass)),
            "shift": float(0.5 * np.abs(Pa.mean(0) - Pb.mean(0)).sum()),
            "extra_tokens": float(np.median(extra)),
            "latency_base": float(np.median([base[name][i]["latency_s"] for i in order])),
            "latency_arm": float(np.median([arm[name][i]["latency_s"] for i in order])),
            "eur": float(np.mean([arm[name][i]["prompt_tokens"] * EUR_IN
                                  + (1 + arm[name][i].get("generated_tokens", 0)) * EUR_OUT
                                  for i in order])) * 1000,
            "eur_base": float(np.mean([base[name][i]["prompt_tokens"] * EUR_IN + EUR_OUT
                                       for i in order])) * 1000,
            "group": (GROUPS.get(BY_NAME[name].family, BY_NAME[name].family)
                      if name in BY_NAME else "knowledge"),
            # For the ensembles and the confidence breakdown.
            "Pb": Pb, "Pa": Pa, "y": y,
        }
        pooled_a.append(right_a)
        pooled_b.append(right_b)
        pooled_conf.append(Pb.max(1))
    a, b = np.concatenate(pooled_a), np.concatenate(pooled_b)
    diffs = a.astype(float) - b.astype(float)
    lo, hi = bootstrap(diffs)
    return {"per": per, "diff": float(diffs.mean()), "lo": lo, "hi": hi, "rows": len(diffs),
            "conf": np.concatenate(pooled_conf), "right_a": a, "right_b": b}


def report(args) -> tuple[str, dict]:
    arms = load(Path(args.runs))
    base = arms[args.baseline]
    results = {name: compare(base, data) for name, data in arms.items() if name != args.baseline}
    noise = results.get("B2")
    md = ["# Extra positions before `answer=`: screening\n",
          f"GLM-5.3-Flash, {len(base)} dev datasets, rows from the calibration halves. Every arm "
          f"is compared with `{args.baseline}` on the same rows. *B2* is the baseline asked "
          f"again: the noise floor between runs.\n"]
    rows, summary = [], {}
    for name, r in results.items():
        per = r["per"]
        mean = lambda key: float(np.mean([v[key] for v in per.values()]))
        controls = [per[n]["arm"] - per[n]["base"] for n in CONTROLS if n in per]
        summary[name] = {
            "diff": r["diff"], "lo": r["lo"], "hi": r["hi"],
            "wins": sum(v["arm"] > v["base"] for v in per.values()),
            "losses": sum(v["arm"] < v["base"] for v in per.values()),
            "significant": sum(v["p"] < 0.05 for v in per.values()),
            "nll_base": mean("nll_base"), "nll_arm": mean("nll_arm"),
            "ece_base": mean("ece_base"), "ece_arm": mean("ece_arm"),
            "t_base": float(np.median([v["t_base"] for v in per.values()])),
            "t_arm": float(np.median([v["t_arm"] for v in per.values()])),
            "mass": mean("mass"), "shift": mean("shift"), "extra": mean("extra_tokens"),
            "latency": mean("latency_arm") - mean("latency_base"),
            "eur": mean("eur"), "eur_base": mean("eur_base"),
            "worst_control": min(controls) if controls else float("nan"),
        }
        s = summary[name]
        rows.append([name, f"{s['diff'] * 100:+.2f} [{s['lo'] * 100:+.2f}, {s['hi'] * 100:+.2f}]",
                     f"{s['wins']} / {s['losses']} ({s['significant']})",
                     f"{s['worst_control'] * 100:+.1f}",
                     f"{s['nll_arm']:.3f} (B {s['nll_base']:.3f})", f"{s['t_arm']:.2f}",
                     f"{s['mass']:.3f}", f"{s['shift']:.3f}", f"{s['extra']:.0f}",
                     f"{s['latency'] * 1000:+.0f}", f"{s['eur']:.3f}"])
    md.append("## Screening\n")
    md.append("Accuracy in points against the baseline, pooled over all rows with a paired "
              "bootstrap 95% interval; datasets better / worse (in brackets: significant by "
              "McNemar, p < 0.05); the worst control dataset (sst2, dbpedia_14, banking77, "
              "clinc150); NLL after a temperature fitted per arm on half the rows, on the other "
              "half; the median of those temperatures (the baseline's is "
              f"{np.median([v['t_base'] for v in next(iter(results.values()))['per'].values()]):.2f}); "
              "option mass; how far the mean answer moves from the baseline's (total variation; "
              "anchoring or a position prior); extra prompt tokens; the change in median latency "
              "(4 requests in flight); EUR per 1,000 decisions (baseline "
              f"{np.mean([s['eur_base'] for s in summary.values()]):.3f}).\n")
    md.append(table(rows, ["arm", "accuracy points [95% CI]", "datasets + / − (sig.)",
                           "worst control", "NLL after own T", "own T", "option mass",
                           "mean-answer shift", "extra tokens", "latency, ms", "EUR / 1,000"]))

    # Gate.
    floor = max((abs(v["arm"] - v["base"]) for n, v in noise["per"].items() if n in CONTROLS),
                default=0.0) if noise else 0.0
    md.append(f"\n**Gate** (fixed before the screening): pooled ≥ +1.0 point with the interval above "
              f"0; no control dataset losing more than the noise floor (the largest control "
              f"difference between B and B2, {floor * 100:.1f} points, or 1 point if larger); NLL "
              f"after temperature not worse; latency +30 ms at most; for filler, the placement "
              f"control (F-before) must not show the same gain.\n")
    gate_rows = []
    before = summary.get("F-before", {}).get("diff", 0.0)
    for name, s in summary.items():
        checks = {
            "accuracy": s["diff"] >= 0.01 and s["lo"] > 0,
            "controls": s["worst_control"] >= -max(floor, 0.01),
            "NLL": s["nll_arm"] <= s["nll_base"] + 1e-3,
            "latency": s["latency"] <= 0.030,
        }
        if name.startswith(("F-", "G")) and name != "F-before":
            checks["placement"] = before < s["diff"] - 0.005
        if name.startswith("H-"):
            checks["latency"] = True      # the reference; its latency is reported, not gated
        mark = lambda key: "—" if key not in checks else "pass" if checks[key] else "**fail**"
        gate_rows.append([name] + [mark(k) for k in ("accuracy", "controls", "NLL", "latency", "placement")]
                         + ["**passes**" if all(checks.values()) else "fails"])
        summary[name]["passes"] = all(checks.values())
    md.append(table(gate_rows, ["arm", "accuracy", "controls", "NLL", "latency", "placement", "verdict"]))

    # Per dataset.
    names = list(base)
    md.append("\n## Per dataset\n")
    md.append("Accuracy points against the baseline (McNemar p < 0.05 marked *):\n")
    head = ["dataset", "group", "B"] + list(results)
    rows = []
    for n in names:
        first = next((r["per"][n] for r in results.values() if n in r["per"]), None)
        if first is None:
            continue
        cells = [n, first["group"], f"{first['base']:.3f}"]
        for name, r in results.items():
            v = r["per"].get(n)
            cells.append("—" if v is None else f"{(v['arm'] - v['base']) * 100:+.1f}" + ("*" if v["p"] < 0.05 else ""))
        rows.append(cells)
    md.append(table(rows, head))

    # Against Jev and the chain-of-thought arm, on the same rows.
    if args.published:
        from .calibrate_extensions import load_arm
        others = {label: load_arm(Path(args.published), arm)
                  for label, arm in (("Jev", "jev"), ("glm-cot", "glm-cot"))}
        best = max((n for n in summary if not n.startswith(("B2", "H-"))),
                   key=lambda n: summary[n]["diff"], default=None)
        rows, closed = [], []
        for n in names:
            order = sorted(base.get(n, {}))
            if not order:
                continue
            Pb, y, options = matrix(base[n], order)
            cells = [n, f"{c.accuracy(Pb, y):.3f}"]
            if best and n in results[best]["per"]:
                v = results[best]["per"][n]
                cells.append(f"{v['arm']:.3f}")
            else:
                cells.append("—")
            accs = {}
            for label, data in others.items():
                d = data.get(n)
                if d is None:
                    cells.append("—")
                    continue
                keep = np.isin(d.index, order)
                accs[label] = c.accuracy(d.P[keep], d.y[keep]) if keep.any() else float("nan")
                cells.append(f"{accs[label]:.3f} ({int(keep.sum())})")
            if best and n in results[best]["per"] and "glm-cot" in accs:
                # All three on the rows glm-cot answered, not B on ours.
                d = others["glm-cot"][n]
                both = sorted(set(map(int, d.index)) & set(order) & set(arms[best].get(n, {})))
                if len(both) >= 20:
                    on = lambda rows: c.accuracy(*matrix(rows, both)[:2])  # noqa: E731
                    gap = c.accuracy(d.P[np.isin(d.index, both)], d.y[np.isin(d.index, both)]) - on(base[n])
                    if gap > 0.02:
                        closed.append((on(arms[best][n]) - on(base[n])) / gap)
            rows.append(cells)
        md.append(f"\n## Against Jev and chain of thought\n")
        md.append(f"Accuracy on the dev rows: the baseline, the best screened arm (`{best}`), and the "
                  f"published Jev and `glm-cot` runs on the same examples (rows they answered in "
                  f"brackets). `glm-cot` lets GLM-5.3-Flash reason before answering, with its own "
                  f"prompt.\n")
        md.append(table(rows, ["dataset", "B", best or "best arm", "Jev", "glm-cot"]))
        if closed:
            md.append(f"\nHeadroom closed by `{best}`, `(arm − B) / (glm-cot − B)` on the datasets "
                      f"where glm-cot is more than 2 points ahead: median {np.median(closed):.0%}.\n")

    # By task family.
    md.append("\n## By task family\n")
    groups = sorted({v["group"] for r in results.values() for v in r["per"].values()})
    rows = []
    for name, r in results.items():
        cells = [name]
        for g in groups:
            d = [v["arm"] - v["base"] for v in r["per"].values() if v["group"] == g]
            cells.append(f"{np.mean(d) * 100:+.1f}" if d else "—")
        rows.append(cells)
    md.append(table(rows, ["arm"] + groups))

    # By the baseline's confidence.
    md.append("\n## Where the gains are: by the baseline's confidence\n")
    md.append("Rows pooled over datasets, split into five equal groups by the baseline's top "
              "probability; accuracy points against the baseline in each:\n")
    rows = []
    for name, r in results.items():
        order = np.argsort(r["conf"], kind="stable")
        cells = [name]
        for chunk in np.array_split(order, 5):
            cells.append(f"{(r['right_a'][chunk].mean() - r['right_b'][chunk].mean()) * 100:+.1f}")
        rows.append(cells)
    md.append(table(rows, ["arm", "least sure fifth", "2", "3", "4", "surest fifth"]))

    # Offline ensembles.
    md.append("\n## Offline ensembles\n")
    md.append("From the stored distributions, no extra requests beyond the arm's: the mean of the "
              "baseline and the arm (two reads for every decision), and the arm's answer only "
              "where the baseline's top probability is below 0.9 (the share of decisions that "
              "need the second read in brackets). Accuracy points against the baseline:\n")
    rows = []
    for name, r in results.items():
        mean_d, adaptive_d, share = [], [], []
        for v in r["per"].values():
            Pb, Pa, y = v["Pb"], v["Pa"], v["y"]
            unsure = Pb.max(1) < 0.9
            Q = np.where(unsure[:, None], Pa, Pb)
            mean_d.append(c.accuracy((Pb + Pa) / 2, y) - c.accuracy(Pb, y))
            adaptive_d.append(c.accuracy(Q, y) - c.accuracy(Pb, y))
            share.append(unsure.mean())
        rows.append([name, f"{np.mean(mean_d) * 100:+.2f}", f"{np.mean(adaptive_d) * 100:+.2f} "
                           f"({np.mean(share):.0%})"])
    md.append(table(rows, ["arm", "mean of B and arm", "arm where B < 0.9"]))
    for name, r in results.items():
        for v in r["per"].values():
            for key in ("Pb", "Pa", "y"):
                v.pop(key)
    return "\n".join(md) + "\n", {"summary": summary,
                                   "per_dataset": {k: r["per"] for k, r in results.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", default="B")
    parser.add_argument("--published", help="the published runs, for Jev and glm-cot")
    args = parser.parse_args()
    text, data = report(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(data, indent=1, default=float))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
