"""The calibration report: every phase of the plan, from raw runs.

    python -m bench.calibrate_report --run results-a --second results-b \\
        --priors priors.json --published published/results --out report/

``--run`` is fitted and evaluated on (calibration/test halves), ``--second``
is a second run of the same rows for the jitter check, ``--priors`` holds the
neutral-content priors from ``bench.neutral_priors``, and ``--published`` is
the unpacked release with the other arms, used for label consensus. All but
``--run`` are optional; their sections are skipped without them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import calibration as c
from .specs import SPECS

SPEC = {s.name: s for s in SPECS}
DOCUMENT = "rvl_cdip"      # scanned pages: kept apart from the text sets throughout
SMALL = 300                # test halves at or below this are weak evidence
COVERAGES = (0.90, 0.95)
LABEL_COUNTS = (50, 100, 250, 500)


def fmt(x: float, digits: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def table(rows: list[list], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def mean(values) -> float:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


# -- pooled reliability, every dataset weighted equally --------------------

def pooled_reliability(parts: list[tuple[np.ndarray, np.ndarray]], bins: int = c.BINS):
    conf = np.concatenate([P.max(1) for P, _ in parts])
    right = np.concatenate([(P.argmax(1) == y).astype(float) for P, y in parts])
    weight = np.concatenate([np.full(len(y), 1 / len(y)) for _, y in parts])
    order = np.argsort(conf, kind="stable")
    edges = np.searchsorted(np.cumsum(weight[order]) / weight.sum(), np.linspace(0, 1, bins + 1)[1:-1])
    out = []
    for chunk in np.split(order, edges):
        if len(chunk):
            w = weight[chunk]
            out.append((float(np.average(conf[chunk], weights=w)),
                        float(np.average(right[chunk], weights=w)), float(w.sum() / weight.sum())))
    return out


def pooled_ece(parts) -> float:
    return float(sum(abs(cf - ac) * w for cf, ac, w in pooled_reliability(parts)))


# -- plots ----------------------------------------------------------------

def plots(out: Path, figures: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = {"raw": "#c0504d", "formula T": "#4f81bd", "family T": "#f79646",
              "context + T": "#9bbb59", "oracle T": "#8064a2"}

    if "reliability" in figures:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
        for label, curve in figures["reliability"].items():
            xs, ys, _ = zip(*curve)
            ax.plot(xs, ys, marker="o", ms=3, lw=1.5, label=label, color=colors.get(label))
        ax.set(xlabel="confidence", ylabel="accuracy", xlim=(0, 1.02), ylim=(0, 1.02),
               title="Reliability, 28 text datasets weighted equally")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "reliability.png")
        plt.close(fig)

    if "temperatures" in figures:
        temps, options, formula = figures["temperatures"]
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        palette = ["#4f81bd", "#c0504d", "#9bbb59", "#8064a2", "#f79646", "#4bacc6", "#7f7f7f"]
        families = sorted({SPEC[n].family for n in temps})
        for i, family in enumerate(families):
            names = [n for n in temps if SPEC[n].family == family]
            ax.scatter([options[n] for n in names], [temps[n] for n in names], s=22,
                       color=palette[i % len(palette)], label=family, zorder=3)
        for name, t in temps.items():
            ax.annotate(name, (options[name], t), fontsize=5.5, xytext=(3, 2),
                        textcoords="offset points", color="#666")
        xs = np.geomspace(2, 160, 50)
        if formula:
            ax.plot(xs, [c.formula_temperature(*formula, x) for x in xs], color="#c0504d",
                    lw=1.2, label="fitted log T = a + b log(options)")
        ax.set(xscale="log", xlabel="options", ylabel="best temperature (calibration half)",
               title="Per-dataset temperature, by task family")
        ax.legend(frameon=False, fontsize=6.5, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "temperatures.png")
        plt.close(fig)

    if "ece" in figures:
        rows = figures["ece"]    # name -> {label: ece}
        names = list(rows)
        labels = list(next(iter(rows.values())))
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        width = 0.8 / len(labels)
        for i, label in enumerate(labels):
            ax.bar(np.arange(len(names)) + i * width, [rows[n][label] for n in names], width,
                   label=label, color=colors.get(label))
        ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2, names, rotation=70, fontsize=6.5)
        ax.set(ylabel="ECE (test half)", title="Calibration error per dataset (held out)")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "ece.png")
        plt.close(fig)

    if "sets" in figures:
        rows = figures["sets"]   # name -> {label: size}
        names = list(rows)
        labels = list(next(iter(rows.values())))
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        width = 0.8 / len(labels)
        for i, label in enumerate(labels):
            ax.bar(np.arange(len(names)) + i * width, [rows[n][label] for n in names], width,
                   label=label, color=colors.get(label))
        ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2, names, rotation=70, fontsize=6.5)
        ax.axhline(1, color="#999", lw=0.8, ls="--")
        ax.set(ylabel="mean options per set",
               title="Conformal sets at 90% coverage (LAC, per-dataset cutoff)")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "set_sizes.png")
        plt.close(fig)

    if "labels" in figures:
        rows = figures["labels"]   # n -> (mean, p05, p95)
        fig, ax = plt.subplots(figsize=(4.4, 3.2))
        ns = list(rows)
        ax.fill_between(ns, [rows[n][1] for n in ns], [rows[n][2] for n in ns], color="#4f81bd",
                        alpha=0.25, label="5th–95th percentile")
        ax.plot(ns, [rows[n][0] for n in ns], marker="o", color="#4f81bd", label="mean")
        ax.axhline(0.9, color="#c0504d", lw=0.8, ls="--", label="target 90%")
        ax.set(xscale="log", xlabel="labelled examples for calibration", ylabel="test coverage",
               title="How many labels a conformal cutoff needs")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "labels_needed.png")
        plt.close(fig)

    if "mass" in figures:
        right, wrong = figures["mass"]
        fig, ax = plt.subplots(figsize=(4.4, 3.2))
        bins = np.linspace(-6, 0, 31)
        ax.hist(np.log10(np.maximum(right, 1e-6)), bins, alpha=0.6, density=True, label="correct",
                color="#4f81bd")
        ax.hist(np.log10(np.maximum(wrong, 1e-6)), bins, alpha=0.6, density=True, label="wrong",
                color="#c0504d")
        ax.set(xlabel="log10 probability on the options before the mask", ylabel="density",
               title="Off-option mass")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "option_mass.png")
        plt.close(fig)


# -- the report -----------------------------------------------------------

def report(args) -> str:
    run = c.load_run(args.run)
    second = c.load_run(args.second) if args.second else {}
    priors = json.loads(Path(args.priors).read_text()) if args.priors else {}
    text_names = sorted((n for n in run if n != DOCUMENT), key=lambda n: (run[n].k, n))
    halves = {n: c.split(run[n]) for n in run}
    options = {n: run[n].k for n in run}
    figures: dict = {}
    md = [f"# Calibration report\n\nRun: `{Path(args.run).name}`, {len(text_names)} text datasets"
          + (" plus rvl_cdip (scanned documents), reported separately" if DOCUMENT in run else "")
          + f". Calibration/test halves are a fixed random split, seed {c.SEED}. "
          "ECE uses 15 equal-mass bins of top-label confidence; means over datasets weight "
          "every dataset equally.\n"]

    # Phase 1 -----------------------------------------------------------
    rows, raw_parts = [], []
    for n in text_names + ([DOCUMENT] if DOCUMENT in run else []):
        _, test = halves[n]
        lo, hi = c.ece_interval(test.P, test.y)
        weak = " (small)" if len(test.y) <= SMALL else ""
        rows.append([n + weak, run[n].k, len(test.y), fmt(c.accuracy(test.P, test.y), 2),
                     fmt(test.P.max(1).mean(), 2), fmt(c.overconfidence(test.P, test.y)),
                     f"{fmt(c.ece(test.P, test.y))} [{fmt(lo)}, {fmt(hi)}]",
                     fmt(c.nll(test.P, test.y)), fmt(c.brier(test.P, test.y))])
        if n != DOCUMENT:
            raw_parts.append((test.P, test.y))
    over = mean(c.overconfidence(halves[n][1].P, halves[n][1].y) for n in text_names)
    md.append("## 1. Raw probabilities\n")
    md.append(f"Raw probabilities are **overconfident by {over * 100:.1f} points** on average "
              f"(mean confidence minus accuracy, text datasets, test halves); pooled ECE "
              f"{fmt(pooled_ece(raw_parts))}. Overconfident on "
              f"{sum(c.overconfidence(halves[n][1].P, halves[n][1].y) > 0 for n in text_names)} "
              f"of {len(text_names)} text datasets.\n")
    md.append(table(rows, ["dataset", "options", "test n", "accuracy", "confidence",
                           "overconfidence", "ECE [95% CI]", "NLL", "Brier"]))
    figures["reliability"] = {"raw": pooled_reliability(raw_parts)}

    # Phase 2 -----------------------------------------------------------
    oracle = {n: c.fit_temperature([(halves[n][0].P, halves[n][0].y)]) for n in run}
    cal_parts = {n: (halves[n][0].P, halves[n][0].y) for n in text_names}
    global_t = c.fit_temperature(list(cal_parts.values()))
    lodo = {n: c.fit_temperature([cal_parts[m] for m in text_names if m != n]) for n in text_names}
    families = defaultdict(list)
    for n in text_names:
        families[SPEC[n].family].append(n)
    loto = {n: c.fit_temperature([cal_parts[m] for m in text_names
                                  if SPEC[m].family != SPEC[n].family]) for n in text_names}
    formula_lodo = {}
    for n in text_names:
        a, b = c.fit_formula({m: oracle[m] for m in text_names if m != n}, options)
        formula_lodo[n] = c.formula_temperature(a, b, options[n])
    formula = c.fit_formula({m: oracle[m] for m in text_names}, options)
    same_family = {}
    for n in text_names:
        peers = [m for m in text_names if m != n and SPEC[m].family == SPEC[n].family]
        same_family[n] = c.fit_temperature([cal_parts[m] for m in peers]) if peers else lodo[n]
    # What the library ships: fitted on every example of every text dataset.
    full_oracle = {n: c.fit_temperature([(run[n].P, run[n].y)]) for n in text_names}
    shipped = {"formula": c.fit_formula(full_oracle, options),
               "global": c.fit_temperature([(run[n].P, run[n].y) for n in text_names]),
               "family": {f: c.fit_temperature([(run[n].P, run[n].y) for n in ns])
                          for f, ns in families.items()}}

    def scores(n, t):
        test = halves[n][1]
        P = c.scale(test.P, t)
        return c.ece(P, test.y), c.nll(P, test.y)

    def excess(n, t):
        """ECE above what a calibrated model shows on this sample: 0 is perfect."""
        test = halves[n][1]
        P = c.scale(test.P, t)
        return c.ece(P, test.y) - c.ece_floor(P, draws=100)

    rows, share, totals = [], defaultdict(list), defaultdict(lambda: [0.0, 0.0])
    excesses = defaultdict(list)
    ece_fig = {}
    for n in text_names:
        raw_e, raw_l = scores(n, 1.0)
        orc_e, orc_l = scores(n, oracle[n])
        excesses["raw"].append(excess(n, 1.0))
        excesses["per task (oracle)"].append(excess(n, oracle[n]))
        cols = {"global T (LODO)": lodo[n], "formula (LODO)": formula_lodo[n],
                "same family (LODO)": same_family[n], "leave family out": loto[n]}
        cells = [n, options[n], fmt(oracle[n], 2), fmt(raw_e), fmt(orc_e),
                 fmt(c.ece_floor(c.scale(halves[n][1].P, oracle[n])))]
        for label, t in cols.items():
            e, l = scores(n, t)
            cells.append(f"{fmt(e)} (T={t:.2f})")
            gain = raw_l - orc_l
            share[label].append((raw_l - l) / gain if gain > 1e-9 else float("nan"))
            totals[label][0] += raw_e - e
            totals[label][1] += raw_e - orc_e
            excesses[label].append(excess(n, t))
        rows.append(cells)
        ece_fig[n] = {"raw": raw_e, "formula T": scores(n, formula_lodo[n])[0],
                      "family T": scores(n, same_family[n])[0], "oracle T": orc_e}
    temps = np.array([oracle[n] for n in text_names])
    md.append("\n## 2. Temperature\n")
    md.append(f"Best temperature per dataset (fitted on the calibration half): median "
              f"{np.median(temps):.2f}, range {temps.min():.2f}–{temps.max():.2f}, all above 1 "
              f"(= overconfident): {bool((temps > 1).all())}. One global T fitted on all "
              f"{len(text_names)} text calibration halves: **T = {global_t:.2f}**. Formula over "
              f"all datasets: log T = {formula[0]:.3f} + {formula[1]:.3f}·log(options).\n")
    md.append("How each zero-label method does on datasets left out of its fit. **Excess ECE** "
              "is a dataset's ECE minus its floor, the ECE a perfectly calibrated model shows on "
              "the same number of examples, averaged over datasets: 0 means as calibrated as the "
              "sample can show. (A pooled ECE over all datasets is not used: it sits at the "
              "floor for every method and hides the differences.) The ECE share is the total "
              "reduction over the per-task reduction; the NLL share is the median per "
              "dataset.\n")
    md.append(table([["raw", fmt(mean(excesses["raw"])), "0", "0"]]
                    + [[k, fmt(mean(excesses[k])), fmt(totals[k][0] / totals[k][1], 2),
                        fmt(float(np.nanmedian(v)), 2)] for k, v in share.items()]
                    + [["per task (oracle)", fmt(mean(excesses["per task (oracle)"])), "1", "1"]],
                    ["method", "mean excess ECE", "ECE share", "NLL share (median)"]))
    summary_excess = {k: mean(v) for k, v in excesses.items()}
    md.append("\n`floor` is the ECE a perfectly calibrated model would show on this many "
              "examples (labels drawn from its own probabilities); values near it are as good "
              "as the sample can show.\n")
    md.append(table(rows, ["dataset", "options", "oracle T", "raw ECE", "oracle ECE", "floor",
                           "global T ECE", "formula ECE", "same-family ECE",
                           "leave-family-out ECE"]))
    by_family = {f: mean(math.log(oracle[n]) for n in ns) for f, ns in families.items()}
    md.append("\nOracle T by family (geometric mean): " + ", ".join(
        f"{f} {math.exp(v):.2f}" for f, v in sorted(by_family.items(), key=lambda kv: kv[1])) + ".\n")
    figures["reliability"]["formula T"] = pooled_reliability(
        [(c.scale(halves[n][1].P, formula_lodo[n]), halves[n][1].y) for n in text_names])
    figures["reliability"]["oracle T"] = pooled_reliability(
        [(c.scale(halves[n][1].P, oracle[n]), halves[n][1].y) for n in text_names])
    figures["reliability"]["family T"] = pooled_reliability(
        [(c.scale(halves[n][1].P, same_family[n]), halves[n][1].y) for n in text_names])
    md.append(f"\n**Shipped in the library** (fitted on all examples): log T = "
              f"{shipped['formula'][0]:.3f} + {shipped['formula'][1]:.3f}·log(options); per family: "
              + ", ".join(f"{f} {t:.2f}" for f, t in sorted(shipped['family'].items())) + ".\n")
    figures["temperatures"] = ({n: oracle[n] for n in text_names}, options, formula)
    figures["global_t"] = global_t

    if second:
        rows, jit = [], []
        for n in text_names:
            if n not in second:
                continue
            _, test_b = c.split(second[n])
            t_a, t_b = oracle[n], c.fit_temperature([(c.split(second[n])[0].P, c.split(second[n])[0].y)])
            e_ab = c.ece(c.scale(test_b.P, t_a), test_b.y)
            e_bb = c.ece(c.scale(test_b.P, t_b), test_b.y)
            agree = float(np.mean(run[n].P.argmax(1)[np.argsort(run[n].index)]
                                  == second[n].P.argmax(1)[np.argsort(second[n].index)])) \
                if len(run[n].index) == len(second[n].index) else float("nan")
            jit.append(abs(math.log(t_a / t_b)))
            rows.append([n, fmt(t_a, 2), fmt(t_b, 2), fmt(e_ab), fmt(e_bb), fmt(agree, 3)])
        md.append("\n### Run jitter\n\nT fitted on run 1's calibration half, evaluated on run 2's "
                  f"test half. Median |log(T1/T2)| = {np.median(jit):.3f} "
                  f"(a factor of {math.exp(np.median(jit)):.3f}).\n")
        md.append(table(rows, ["dataset", "T run 1", "T run 2", "ECE run 2, T from run 1",
                               "ECE run 2, own T", "same answer in both runs"]))

    # Phase 3 -----------------------------------------------------------
    cc_lodo = {}
    if priors:
        corrected = {n: (c.contextual(halves[n][0].P, np.array(priors[n])),
                         c.contextual(halves[n][1].P, np.array(priors[n])))
                     for n in text_names if n in priors}
        names = list(corrected)
        cc_parts = {n: (corrected[n][0], halves[n][0].y) for n in names}
        cc_global = c.fit_temperature(list(cc_parts.values()))
        cc_lodo = {n: c.fit_temperature([cc_parts[m] for m in names if m != n]) for n in names}
        rows, verdict = [], defaultdict(int)
        for n in names:
            test = halves[n][1]
            base = c.scale(test.P, formula_lodo[n])
            fixed = c.scale(corrected[n][1], cc_lodo[n])
            d_nll = c.nll(fixed, test.y) - c.nll(base, test.y)
            d_acc = c.accuracy(fixed, test.y) - c.accuracy(base, test.y)
            v = "helps" if d_nll < -0.01 else "hurts" if d_nll > 0.01 else "neutral"
            verdict[v] += 1
            ece_fig[n]["context + T"] = c.ece(fixed, test.y)
            rows.append([n, fmt(max(priors[n]), 2), fmt(c.ece(base, test.y)), fmt(c.ece(fixed, test.y)),
                         f"{d_nll:+.3f}", f"{d_acc * 100:+.1f}", v])
        md.append("\n## 3. Contextual calibration\n")
        md.append(f"Priors from neutral content (`N/A`, empty, `[MASK]`, and `k. A.` for German "
                  f"sets), divided out, then T refitted on the corrected probabilities "
                  f"(global T = {cc_global:.2f}); compared with the formula T alone, both "
                  f"leave-one-dataset-out. Unlike temperature, this changes answers, so accuracy "
                  f"moves too. Verdict by test NLL (±0.01): "
                  + ", ".join(f"{k} {v}" for k, v in verdict.items()) + ".\n")
        md.append(table(rows, ["dataset", "largest prior", "ECE, T only", "ECE, context + T",
                               "Δ NLL", "Δ accuracy (points)", "verdict"]))
        figures["reliability"]["context + T"] = pooled_reliability(
            [(c.scale(corrected[n][1], cc_lodo[n]), halves[n][1].y) for n in names])
    figures["ece"] = ece_fig

    # Phase 4 -----------------------------------------------------------
    md.append("\n## 4. Conformal prediction sets\n")
    md.append("Split conformal on each dataset's calibration half, evaluated on its test half. "
              "LAC scores 1 − p(gold); APS the mass of options at least as likely as the gold "
              "one. Probabilities: raw, and after the formula temperature fitted without "
              "the dataset (the library's default).\n")
    set_fig, rows = {}, []
    summary = defaultdict(list)
    for n in text_names + ([DOCUMENT] if DOCUMENT in run else []):
        cal, test = halves[n]
        t = formula_lodo.get(n, c.formula_temperature(*formula, options[n]))
        cells = [n, options[n]]
        for cov in COVERAGES:
            for label, (P_cal, P_test) in (("raw", (cal.P, test.P)),
                                            ("T", (c.scale(cal.P, t), c.scale(test.P, t)))):
                for method, score, sets in (("LAC", c.lac_scores, c.lac_sets),
                                            ("APS", c.aps_scores, c.aps_sets)):
                    q = c.threshold(score(P_cal, cal.y), cov)
                    st = c.set_stats(sets(P_test, q), test.y)
                    if n != DOCUMENT:
                        summary[(cov, label, method)].append(st)
                    if method == "LAC" and cov == 0.90:
                        set_fig.setdefault(n, {})["raw" if label == "raw" else "formula T"] = st["size"]
                    if method == "LAC" and label == "T":
                        cells += [fmt(st["coverage"], 3), fmt(st["size"], 2), fmt(st["single"], 2)]
        rows.append(cells)
    md.append(table([[f"{int(cov * 100)}%", label, method,
                      fmt(mean(s["coverage"] for s in v), 3), fmt(mean(s["size"] for s in v), 2),
                      fmt(mean(s["single"] for s in v), 2)]
                     for (cov, label, method), v in summary.items()],
                    ["target", "probabilities", "score", "mean coverage", "mean set size",
                     "single-option share"]))
    md.append("\nPer dataset, LAC after the formula T:\n")
    md.append(table(rows, ["dataset", "options", "coverage 90%", "size 90%", "single 90%",
                           "coverage 95%", "size 95%", "single 95%"]))
    figures["sets"] = set_fig

    rows = []
    for cov in COVERAGES:
        misses = []
        for n in text_names:
            groups = [c.lac_scores(c.scale(halves[m][0].P, formula_lodo[n]), halves[m][0].y)
                      for m in text_names if m != n]
            q = c.weighted_threshold(groups, cov)
            test = halves[n][1]
            misses.append(c.set_stats(c.lac_sets(c.scale(test.P, formula_lodo[n]), q),
                                      test.y)["coverage"] - cov)
        rows.append([f"{int(cov * 100)}%", fmt(mean(misses) + cov, 3), fmt(min(misses) + cov, 3),
                     fmt(max(misses) + cov, 3),
                     f"{sum(m < -0.02 for m in misses)} of {len(misses)}"])
    md.append("\n**Global cutoff (zero labels, heuristic, no guarantee):** LAC cutoff pooled over "
              "the other datasets' calibration halves, applied to the held-out dataset.\n")
    md.append(table(rows, ["target", "mean coverage", "worst", "best", "datasets > 2 points short"]))

    rng = np.random.default_rng(c.SEED)
    label_rows, label_fig = [], {}
    for size in LABEL_COUNTS:
        coverages = []
        for n in text_names:
            cal, test = halves[n]
            if len(cal.y) < size:
                continue
            P_cal, P_test = c.scale(cal.P, formula_lodo[n]), c.scale(test.P, formula_lodo[n])
            for _ in range(200):
                rows_ = rng.choice(len(cal.y), size, replace=False)
                q = c.threshold(c.lac_scores(P_cal[rows_], cal.y[rows_]), 0.9)
                coverages.append(c.set_stats(c.lac_sets(P_test, q), test.y)["coverage"])
        coverages = np.array(coverages)
        label_fig[size] = (float(coverages.mean()), float(np.percentile(coverages, 5)),
                           float(np.percentile(coverages, 95)))
        label_rows.append([size, fmt(coverages.mean(), 3), fmt(np.percentile(coverages, 5), 3),
                           fmt(np.percentile(coverages, 95), 3), fmt(np.mean(coverages < 0.88), 2)])
    md.append("\n**Labels needed** (90% target, 200 random calibration draws per dataset):\n")
    md.append(table(label_rows, ["labels", "mean coverage", "5th percentile", "95th percentile",
                                 "share of draws below 88%"]))
    figures["labels"] = label_fig

    # Phase 5 -----------------------------------------------------------
    md.append("\n## 5. Label noise, contamination, language\n")
    if args.published:
        consensus = label_consensus(Path(args.published))
        rows = []
        for n in text_names:
            if n not in consensus:
                continue
            _, test = halves[n]
            bad = np.array([consensus[n].get(int(i), False) for i in test.index])
            if not bad.any():
                continue
            t = formula_lodo[n]
            full = c.scale(test.P, t)
            clean = full[~bad]
            rows.append([n, fmt(bad.mean(), 3), fmt(c.overconfidence(full, test.y)),
                         fmt(c.overconfidence(clean, test.y[~bad])), fmt(c.ece(full, test.y)),
                         fmt(c.ece(clean, test.y[~bad])), fmt(c.accuracy(clean, test.y[~bad]), 3)])
        md.append("Examples where at least three of the four other published arms (Jev, Laya, "
                  "reasoning GLM, embeddings) agree on an answer that differs from the label are "
                  "removed. This is an **upper bound** on the effect of label noise: the removed "
                  "examples include ones where the label is right and GLM is confidently wrong "
                  "together with the other systems, and dropping those flatters any model. "
                  "After the formula T:\n")
        md.append(table(rows, ["dataset", "removed share", "overconfidence, all",
                               "overconfidence, cleaned", "ECE, all", "ECE, cleaned",
                               "accuracy, cleaned"]))
        if args.label_check and "banking77" in run:
            check = json.loads(Path(args.label_check).read_text())
            verdict = {r["index"]: r for r in check["rows"]}
            counts = defaultdict(int)
            for r in check["rows"]:
                counts[r["verdict"]] += 1
            d = run["banking77"]
            P = c.scale(d.P, formula_lodo["banking77"])
            flagged = np.array([int(i) in verdict for i in d.index])
            wrong_label = np.array([verdict.get(int(i), {}).get("verdict") == "consensus right"
                                    for i in d.index])
            relabelled = d.y.copy()
            position = {o: k for k, o in enumerate(d.options)}
            for k, i in enumerate(d.index):
                if wrong_label[k]:
                    relabelled[k] = position[verdict[int(i)]["consensus"]]
            variants = [("as labelled", P, d.y), ("all flagged removed", P[~flagged], d.y[~flagged]),
                        ("confirmed errors relabelled", P, relabelled)]
            md.append(f"\n**Checking the flags on banking77.** All {len(check['rows'])} flagged "
                      f"examples were read against both labels, by {check['checked_by']}. Label "
                      f"right {counts['gold right']}, label wrong {counts['consensus right']}, "
                      f"both defensible {counts['both defensible']}. So only "
                      f"{counts['consensus right'] / len(check['rows']):.0%} of flags are clear "
                      f"label errors. Correcting just those, on all examples after the formula T:\n")
            md.append(table([[k, len(y), fmt(c.accuracy(Pv, y), 3), fmt(c.overconfidence(Pv, y)),
                              fmt(c.ece(Pv, y))] for k, Pv, y in variants],
                            ["banking77", "examples", "accuracy", "overconfidence", "ECE"]))
    rows = []
    for key, attr in (("tier", "tier"), ("language", "language")):
        groups = defaultdict(list)
        for n in text_names:
            groups[getattr(SPEC[n], attr)].append(n)
        for g, ns in sorted(groups.items()):
            rows.append([key, g, len(ns), fmt(math.exp(mean(math.log(oracle[n]) for n in ns)), 2),
                         fmt(mean(scores(n, formula_lodo[n])[0] for n in ns))])
    md.append("\nOracle T and held-out ECE by contamination tier (how likely the set was in "
              "training) and by language:\n")
    md.append(table(rows, ["grouping", "group", "datasets", "oracle T (geo. mean)",
                           "ECE after formula T"]))
    # Is the tier effect the task family, or difficulty, in disguise?
    acc = {n: c.accuracy(run[n].P, run[n].y) for n in text_names}
    rows = [[f, SPEC[n].tier, n, fmt(acc[n], 2), fmt(oracle[n], 2)]
            for f in sorted(families) for n in sorted(families[f], key=lambda m: SPEC[m].tier)
            if len({SPEC[m].tier for m in families[f]}) > 1]
    md.append("\nWithin families that span more than one tier:\n")
    md.append(table(rows, ["family", "tier", "dataset", "accuracy", "oracle T"]))
    X = np.column_stack([np.ones(len(text_names)), [acc[n] for n in text_names],
                         [SPEC[n].tier == "likely" for n in text_names],
                         [SPEC[n].tier == "unclear" for n in text_names]]).astype(float)
    z = np.log([oracle[n] for n in text_names])
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    resid = z - X @ coef
    sigma2 = float(resid @ resid) / (len(z) - X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * sigma2)
    md.append(f"\nRegression over the {len(z)} text datasets, log T = a + b·accuracy + tier: "
              f"accuracy {coef[1]:+.2f} ± {se[1]:.2f}, 'likely' {coef[2]:+.2f} ± {se[2]:.2f}, "
              f"'unclear' {coef[3]:+.2f} ± {se[3]:.2f} (log T, against 'seen'; ± one standard "
              f"error). A tier coefficient within about two standard errors of 0 means the tier "
              f"adds nothing once accuracy is known.\n")
    contamination = {"accuracy": (coef[1], se[1]), "likely": (coef[2], se[2]),
                     "unclear": (coef[3], se[3])}

    # Phase 6 -----------------------------------------------------------
    with_mass = [n for n in text_names if run[n].mass is not None]
    if with_mass:
        rows, right_all, wrong_all, inverted, pooled_scores = [], [], [], [], ([], [], [])

        def with_interval(score, positive):
            lo, hi = c.auroc_interval(score, positive)
            return f"{fmt(c.auroc(score, positive), 2)} [{fmt(lo, 2)}, {fmt(hi, 2)}]", hi

        for n in with_mass:
            d = run[n]
            wrong = d.P.argmax(1) != d.y
            conf = c.scale(d.P, formula_lodo[n]).max(1)
            conf_cell, _ = with_interval(-conf, wrong)
            mass_cell, mass_hi = with_interval(-d.mass, wrong)
            if mass_hi < 0.5:
                inverted.append(n)
            rows.append([n, int(wrong.sum()), fmt(np.median(d.mass[~wrong]), 3),
                         fmt(np.median(d.mass[wrong]) if wrong.any() else float("nan"), 3),
                         conf_cell, mass_cell])
            right_all.append(d.mass[~wrong])
            wrong_all.append(d.mass[wrong])
            # Pool ranks within each dataset, so dataset-level differences in mass don't count.
            pooled_scores[0].append((np.argsort(np.argsort(-conf)) + 0.5) / len(conf))
            pooled_scores[1].append((np.argsort(np.argsort(-d.mass)) + 0.5) / len(d.mass))
            pooled_scores[2].append(wrong)
        pooled_wrong = np.concatenate(pooled_scores[2])
        pooled_conf, _ = with_interval(np.concatenate(pooled_scores[0]), pooled_wrong)
        pooled_mass, _ = with_interval(np.concatenate(pooled_scores[1]), pooled_wrong)
        rows.append(["**pooled** (ranks within dataset)", int(pooled_wrong.sum()), "", "",
                     pooled_conf, pooled_mass])
        md.append("\n## 6. Off-option mass\n")
        md.append("Probability the model put on the option tokens before the mask. AUROC for "
                  "flagging wrong answers from low confidence and from low mass, with 95% "
                  "bootstrap intervals: 0.5 is chance, below 0.5 means the signal points the "
                  "other way (low mass on *right* answers). Intervals are wide where there are "
                  "few wrong answers.\n")
        md.append(table(rows, ["dataset", "wrong answers", "mass when right", "mass when wrong",
                               "AUROC confidence", "AUROC mass"]))
        md.append("\nMass is significantly *inverted* (interval below 0.5) on: "
                  + (", ".join(inverted) or "none")
                  + ". There, answers the model is right about carry slightly less probability on "
                  "the options, so low mass is not a usable error signal in either direction.\n")
        figures["mass"] = (np.concatenate(right_all), np.concatenate(wrong_all))

    md.append("\n## Figures\n")
    for name, title in (("reliability", "Reliability"), ("temperatures", "Per-dataset temperature"),
                        ("ece", "ECE per dataset"), ("set_sizes", "Conformal set sizes"),
                        ("labels_needed", "Labels needed"), ("option_mass", "Off-option mass")):
        md.append(f"![{title}]({name}.png)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plots(out, figures)
    summary_json = {"global_t": global_t, "formula": formula, "oracle": oracle,
                    "lodo": lodo, "context_lodo": cc_lodo, "shipped": shipped,
                    "excess_ece": summary_excess, "contamination": contamination}
    (out / "summary.json").write_text(json.dumps(summary_json, indent=1))
    # What the library ships, for its scripts/update_calibration.py.
    meta = next(iter(run.values())).meta
    constants = {"model": (meta.get("arms") or {}).get("privatemode", "unknown"),
                 "formula": list(shipped["formula"]), "family": shipped["family"],
                 "source": (f"privatemode-decisions-benchmark, results/calibration/part-1/"
                            f"constants.json (run {meta.get('started', '?')})")}
    (out / "constants.json").write_text(json.dumps(constants, indent=1) + "\n")
    return "\n".join(md) + "\n"


def label_consensus(published: Path, others=("jev", "laya", "glm-cot", "embed-nn"),
                    needed: int = 3) -> dict[str, dict[int, bool]]:
    """Per dataset and example: do most other arms agree on a non-gold answer?"""
    out: dict[str, dict[int, bool]] = {}
    for directory in sorted(p for p in published.iterdir() if p.is_dir()):
        answers: dict[int, dict[str, str]] = defaultdict(dict)
        gold: dict[int, str] = {}
        for path in directory.glob("*-r0.jsonl"):
            lines = path.open()
            meta = json.loads(next(lines))
            if meta.get("perturb") not in (None, "none"):
                continue    # renamed options: different names for the same answers
            for line in lines:
                r = json.loads(line)
                if r.get("kind") == "row" and r.get("arm") in others and "choice" in r:
                    answers[r["index"]][r["arm"]] = r["choice"]
                    gold[r["index"]] = r["gold"]
        flags = {}
        for i, by_arm in answers.items():
            votes = defaultdict(int)
            for choice in by_arm.values():
                votes[choice] += 1
            top, count = max(votes.items(), key=lambda kv: kv[1])
            flags[i] = count >= needed and top != gold[i]
        if flags:
            out[directory.name] = flags
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--second")
    parser.add_argument("--priors")
    parser.add_argument("--published")
    parser.add_argument("--label-check", help="banking77 label check, results/calibration/part-1/"
                                              "banking77-label-check.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    text = report(args)
    (Path(args.out) / "report.md").write_text(text)
    print(f"wrote {Path(args.out) / 'report.md'}")


if __name__ == "__main__":
    main()
