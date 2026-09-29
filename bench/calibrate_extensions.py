"""The second calibration report: Jev, guaranteed automation, position bias.

    python -m bench.calibrate_extensions --run runs/r1 --second runs/r2 \\
        --published published/results --summary report/summary.json \\
        [--rotations runs/rotations --rotations-boolq runs/rot-boolq \\
         --rotations-renamed runs/rot-boolq-renamed] --out report2/

Uses the same calibration/test halves as ``bench.calibrate_report`` and the
per-dataset temperatures in its ``summary.json``. Everything here is offline
except the rotation runs, which ``bench.rotations`` collects.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import calibration as c
from .calibrate_report import DOCUMENT, fmt, mean, table

EPSILONS = (0.02, 0.05, 0.10)
DELTA = 0.1
LABELS = (20, 50, 100, 250, 500)
DRAWS = 200
#: Jev's probabilities come in steps of 0.01.
JEV_UNIT = 0.01
JEV_DATE = "2026-09-22"


def load_arm(published: Path, arm: str) -> dict[str, c.Dataset]:
    """One published arm's rows per dataset, unperturbed run 0 only."""
    out = {}
    for directory in sorted(p for p in published.iterdir() if p.is_dir()):
        rows = []
        for path in directory.glob("*-r0.jsonl"):
            lines = path.open()
            meta = json.loads(next(lines))
            if meta.get("perturb") not in (None, "none"):
                continue
            rows += [r for r in map(json.loads, lines) if r.get("kind") == "row"
                     and r.get("arm") == arm and r.get("probabilities") and "error" not in r]
        if len(rows) < 50:
            continue
        seen, unique = set(), []
        for r in rows:
            if r["index"] not in seen:
                seen.add(r["index"])
                unique.append(r)
        options = list(unique[0]["probabilities"])
        position = {o: i for i, o in enumerate(options)}
        unique = [r for r in unique if r["gold"] in position]
        P = np.array([[r["probabilities"].get(o, 0.0) for o in options] for r in unique], float)
        out[directory.name] = c.Dataset(directory.name, options, np.array([r["index"] for r in unique]),
                                        P / P.sum(axis=1, keepdims=True),
                                        np.array([position[r["gold"]] for r in unique]))
    return out


def common(a: c.Dataset, b: c.Dataset) -> tuple[c.Dataset, c.Dataset]:
    """Both datasets restricted to the examples they share, in the same order,
    with ``b``'s options reordered to ``a``'s."""
    shared = np.intersect1d(a.index, b.index)
    ia = np.searchsorted(a.index[np.argsort(a.index)], shared)
    ib = np.searchsorted(b.index[np.argsort(b.index)], shared)
    ra, rb = np.argsort(a.index)[ia], np.argsort(b.index)[ib]
    order = [b.options.index(o) for o in a.options]
    B = c.Dataset(b.name, a.options, b.index[rb], b.P[rb][:, order], a.y[ra], None, b.meta)
    return a.subset(ra), B


def top(P: np.ndarray, y: np.ndarray):
    return P.max(axis=1), P.argmax(axis=1) == y


def ltt(conf_cal, right_cal, conf_test, right_test, eps):
    t = c.automation_threshold(conf_cal, right_cal, eps, DELTA)
    automated = conf_test >= t
    err = float((~right_test[automated]).mean()) if automated.any() else 0.0
    return float(automated.mean()), err


def report(args) -> tuple[str, dict]:
    run = c.load_run(args.run)
    second = c.load_run(args.second) if args.second else {}
    summary = json.loads(Path(args.summary).read_text())
    oracle = summary["oracle"]
    text = sorted((n for n in run if n != DOCUMENT), key=lambda n: (run[n].k, n))
    options = {n: run[n].k for n in run}
    formula_lodo = {}
    for n in text:
        a, b = c.fit_formula({m: oracle[m] for m in text if m != n}, options)
        formula_lodo[n] = c.formula_temperature(a, b, options[n])
    shipped = summary["shipped"]["formula"]
    halves = {n: c.split(run[n]) for n in run}
    figures: dict = {}
    md = ["# Calibration report, part 2\n",
          (f"Same runs, halves and temperatures as part 1 (`{Path(args.run).name}`). GLM "
          "probabilities are softened with the option-count formula fitted without the dataset "
          "in question (the library's default); Jev and Laya are taken from the published "
          "runs on the same examples. Means weight every dataset equally.\n")]

    # 1. Jev and Laya -------------------------------------------------------
    rows, sums, sets_fig, ece_fig = [], defaultdict(list), {}, {}
    systems = {"jev": load_arm(Path(args.published), "jev"),
               "laya": load_arm(Path(args.published), "laya")} if args.published else {}

    def unzero(P):
        """Jev rounds to 0.01: put half a rounding unit where it says 0."""
        Q = np.where(P == 0, JEV_UNIT / 2, P)
        return Q / Q.sum(axis=1, keepdims=True)

    def sets(P_cal, y_cal, P_test, y_test):
        return c.set_stats(c.lac_sets(P_test, c.threshold(c.lac_scores(P_cal, y_cal), 0.9)), y_test)

    for n in text:
        if n not in systems.get("jev", {}):
            continue
        glm, jev = common(run[n], systems["jev"][n])
        cal_g, test_g = c.split(glm)
        cal_j, test_j = c.split(jev)
        tg = formula_lodo[n]
        Pg_cal, Pg = c.scale(cal_g.P, tg), c.scale(test_g.P, tg)
        # With labels: each system's own temperature, fitted on the calibration half.
        tg_own = c.fit_temperature([(Pg_cal, cal_g.y)], prior=1.0, shrinkage=c.SHRINKAGE)
        Pz_cal, Pz = unzero(cal_j.P), unzero(test_j.P)
        tj_own = c.fit_temperature([(Pz_cal, cal_j.y)])
        conf_j, right_j = top(test_j.P, test_j.y)
        iso = c.fit_isotonic(*top(cal_j.P, cal_j.y))
        row = {
            "glm acc": c.accuracy(Pg, test_g.y), "jev acc": c.accuracy(test_j.P, test_j.y),
            "glm over": c.overconfidence(Pg, test_g.y), "jev over": c.overconfidence(test_j.P, test_j.y),
            "glm raw ece": c.ece(test_g.P, test_g.y),
            "glm ece": c.ece(Pg, test_g.y), "jev ece": c.ece(test_j.P, test_j.y),
            "jev unzero ece": c.ece(Pz, test_j.y),
            "glm floor": c.ece_floor(Pg, draws=100), "jev floor": c.ece_floor(test_j.P, draws=100),
            "jev unzero floor": c.ece_floor(Pz, draws=100),
            "glm own floor": c.ece_floor(c.scale(Pg, tg_own), draws=100),
            "jev own floor": c.ece_floor(c.scale(Pz, tj_own), draws=100),
            "glm own ece": c.ece(c.scale(Pg, tg_own), test_g.y),
            "jev own ece": c.ece(c.scale(Pz, tj_own), test_j.y),
            "jev own t": tj_own,
            "jev iso ece": c.ece_top(c.apply_isotonic(iso, conf_j), right_j),
            "jev zero": float(np.mean(jev.P == 0)),
            "jev gold zero": float(np.mean(test_j.P[np.arange(len(test_j.y)), test_j.y] == 0)),
        }
        for label, st in (("glm", sets(Pg_cal, cal_g.y, Pg, test_g.y)),
                          ("jev", sets(cal_j.P, cal_j.y, test_j.P, test_j.y)),
                          ("jev own", sets(c.scale(Pz_cal, tj_own), cal_j.y, c.scale(Pz, tj_own), test_j.y))):
            for key, v in st.items():
                row[f"{label} {key}"] = v
        for k, v in row.items():
            sums[k].append(v)
        sets_fig[n] = {"GLM, default T": row["glm size"], "Jev": row["jev size"],
                       "Jev, zeros fixed + own T": row["jev own size"]}
        ece_fig[n] = (row["glm ece"], row["jev ece"], row["jev unzero ece"], row["jev own ece"])
        rows.append([n, options[n], len(test_g.y), fmt(row["glm acc"], 2), fmt(row["jev acc"], 2),
                     fmt(row["glm ece"]), fmt(row["jev ece"]), fmt(row["jev unzero ece"]),
                     fmt(row["glm own ece"]), fmt(row["jev own ece"]), fmt(row["jev gold zero"], 3),
                     fmt(row["glm size"], 2), fmt(row["jev size"], 2), fmt(row["jev own size"], 2)])
    if rows:
        laya_rows = []
        for n in text:
            if n in systems.get("laya", {}):
                glm, laya = common(run[n], systems["laya"][n])
                _, test_l = c.split(laya)
                laya_rows.append((c.overconfidence(test_l.P, test_l.y), c.ece(test_l.P, test_l.y)))
        worst = max(range(len(rows)), key=lambda i: sums["jev size"][i])
        m = lambda key: mean(sums[key])
        wins = lambda a, b: sum(x < y for x, y in zip(sums[a], sums[b]))
        md.append("## 1. Jev on the same examples\n")
        md.append(f"On the {len(rows)} text datasets both systems answer, same test halves. Jev "
                  f"is `jev-latest` as the published runs called it on {JEV_DATE}; the runs did "
                  f"not record which version that resolved to. **ECE** here and in the rest of "
                  f"this report is the mean over datasets of each dataset's ECE (15 equal-size "
                  f"bins of top-answer confidence). Part 1's headline numbers subtract each "
                  f"dataset's **floor**, the ECE a perfectly calibrated model shows on the same "
                  f"number of examples, shown here too.\n")
        md.append(table([
            ["accuracy", fmt(m("glm acc"), 3), fmt(m("jev acc"), 3), ""],
            ["overconfidence, no labels",
             (f"{m('glm over') * 100:+.1f} points (raw "
              f"{mean(c.overconfidence(halves[n][1].P, halves[n][1].y) for n in text) * 100:+.1f})"),
             f"{m('jev over') * 100:+.1f} points", ""],
            ["ECE, no labels", f"{fmt(m('glm ece'))} (default T; raw {fmt(m('glm raw ece'))})",
             fmt(m("jev ece")), f"{fmt(m('jev unzero ece'))} (zeros set to {JEV_UNIT / 2})"],
            ["sampling floor", fmt(m("glm floor")), fmt(m("jev floor")), fmt(m("jev unzero floor"))],
            ["**excess ECE, no labels** (ECE − floor)", f"**{fmt(m('glm ece') - m('glm floor'))}**",
             f"**{fmt(m('jev ece') - m('jev floor'))}**",
             f"**{fmt(m('jev unzero ece') - m('jev unzero floor'))}**"],
            ["lower ECE with no labels, datasets", str(wins("glm ece", "jev unzero ece")),
             "", str(wins("jev unzero ece", "glm ece"))],
            ["ECE with the calibration half's labels (~500)",
             f"{fmt(m('glm own ece'))} (task T)",
             f"{fmt(m('jev iso ece'))} (isotonic, top answer only)",
             (f"{fmt(m('jev own ece'))} (zeros fixed + task T, median T "
              f"{float(np.median(sums['jev own t'])):.2f})")],
            ["excess ECE with labels", fmt(m("glm own ece") - m("glm own floor")), "",
             fmt(m("jev own ece") - m("jev own floor"))],
            ["lower ECE with labels, datasets", str(wins("glm own ece", "jev own ece")), "",
             str(wins("jev own ece", "glm own ece"))],
            ["probabilities exactly 0", "none", f"{m('jev zero'):.0%}", ""],
            ["right answer at exactly 0", "never",
             f"{m('jev gold zero'):.1%} (max {max(sums['jev gold zero']):.1%})", ""],
            ["90% set: coverage", fmt(m("glm coverage"), 3), fmt(m("jev coverage"), 3),
             fmt(m("jev own coverage"), 3)],
            ["90% set: options, mean / median",
             f"{m('glm size'):.2f} / {float(np.median(sums['glm size'])):.2f}",
             f"{m('jev size'):.2f} / {float(np.median(sums['jev size'])):.2f}",
             f"{m('jev own size'):.2f} / {float(np.median(sums['jev own size'])):.2f}"],
            [f"90% set: options on {rows[worst][0]}", fmt(sums["glm size"][worst], 2),
             fmt(sums["jev size"][worst], 2), fmt(sums["jev own size"][worst], 2)],
            ["90% set: single-option share", fmt(m("glm single"), 2), fmt(m("jev single"), 2),
             fmt(m("jev own single"), 2)],
        ], ["", "GLM-5.3-Flash", "Jev, raw", "Jev, fairest fix"]))
        md.append(f"\nJev rounds to 0.01, and {m('jev zero'):.0%} of its probabilities are exactly 0, sometimes "
                  "including the right answer. Setting those zeros to half a rounding unit makes "
                  "the likelihood finite, so a temperature can then be fitted: that is the "
                  "fairest fix, and any Jev user could apply it. Isotonic regression repairs only "
                  "the top answer's stated confidence, not the distribution that prediction sets "
                  "are built from."
                  + (f" Laya, for reference: overconfidence {mean(o for o, _ in laya_rows) * 100:+.1f} "
                     f"points, ECE {fmt(mean(e for _, e in laya_rows))} on "
                     f"{len(laya_rows)} datasets." if laya_rows else "") + "\n")
        md.append(table(rows, ["dataset", "options", "test n", "GLM acc", "Jev acc",
                               "GLM ECE, default T", "Jev ECE, raw", "Jev ECE, zeros fixed",
                               "GLM ECE, task T", "Jev ECE, zeros fixed + task T",
                               "Jev: right answer at 0", "GLM 90% set", "Jev 90% set",
                               "Jev 90% set, fixed"]))
        figures["jev_sets"] = sets_fig
        figures["jev_ece"] = ece_fig

    # 2. Guaranteed automation ----------------------------------------------
    md.append("\n## 2. A guaranteed error rate on automated answers\n")
    sizes = sorted(len(halves[n][0].y) for n in text)
    md.append(f"Learn then Test style: from a task's calibration half, the lowest threshold on "
              f"the top probability whose error among automated answers is at most ε, with "
              f"probability {1 - DELTA:.0%} over the choice of labels. Evaluated on the test half. "
              f"This table uses each dataset's whole calibration half, {sizes[0]} to {sizes[-1]} "
              f"labels, so its rates differ from the fixed-size draws below.\n")
    rows, auto, best = [], defaultdict(list), defaultdict(list)
    same_summary = []      # per label count: (violations, coverage, automated), calibrate() and split
    for n in text:
        cal, test = halves[n]
        cc, rc = top(c.scale(cal.P, formula_lodo[n]), cal.y)
        ct, rt = top(c.scale(test.P, formula_lodo[n]), test.y)
        cells = [n, options[n], fmt(c.accuracy(test.P, test.y), 2)]
        order = np.argsort(-ct)
        running = np.cumsum(~rt[order]) / np.arange(1, len(ct) + 1)
        for eps in EPSILONS:
            rate, err = ltt(cc, rc, ct, rt, eps)
            auto[eps].append((rate, err))
            ok = np.where(running <= eps)[0]
            best[eps].append((ok.max() + 1) / len(ct) if len(ok) else 0.0)
            cells.append(f"{rate:.0%} ({err:.1%})")
        rows.append(cells)
    md.append(table([[f"{eps:.0%}", f"{mean(r for r, _ in auto[eps]):.0%}",
                      f"{mean(best[eps]):.0%}",
                      f"{sum(r > 0 for r, _ in auto[eps])} of {len(text)}",
                      f"{sum(e > eps for r, e in auto[eps] if r > 0)} of {len(text)}"]
                     for eps in EPSILONS],
                    ["max error ε", "mean share automated", "best possible (knowing the test labels)",
                     "datasets automating anything", "datasets over ε on the test half"]))
    md.append("\nThe gap to the best possible is the price of a guarantee from ~500 labels: to "
              "certify ε from n answers, their observed error has to be well below ε. Label "
              "errors keep even the most confident answers from being error-free, so ε of a few "
              "percent is out of reach on most tasks.\n")
    md.append("\nPer dataset, share automated (error among automated on the test half):\n")
    md.append(table(rows, ["dataset", "options", "accuracy"] + [f"ε = {e:.0%}" for e in EPSILONS]))
    figures["automation"] = {n: [auto[e][i][0] for e in EPSILONS] for i, n in enumerate(text)}

    # Does the guarantee hold, and how many labels does it need?
    rng = np.random.default_rng(c.SEED)
    rows, need = [], {}
    for size in LABELS:
        viol, naive_viol, rates = [], [], []
        same_viol, same_rates, same_cov, split_viol, split_rates = [], [], [], [], []
        for n in text:
            cal, test = halves[n]
            if len(cal.y) < size:
                continue
            P_cal, P_test = c.scale(cal.P, formula_lodo[n]), c.scale(test.P, formula_lodo[n])
            cc, rc = top(P_cal, cal.y)
            ct, rt = top(P_test, test.y)
            for _ in range(DRAWS // 4):
                pick = rng.choice(len(cal.y), size, replace=False)
                t = c.automation_threshold(cc[pick], rc[pick], 0.10, DELTA)
                m = ct >= t
                rates.append(m.mean())
                viol.append(bool(m.any()) and (~rt[m]).mean() > 0.10)
                # What calibrate() does: fit the task temperature on these labels, then
                # the cutoff and the threshold on the same labels.
                tt = c.fit_temperature([(P_cal[pick], cal.y[pick])], prior=1.0, shrinkage=c.SHRINKAGE)
                Pd, Pt = c.scale(P_cal[pick], tt), c.scale(P_test, tt)
                c2, r2 = top(Pd, cal.y[pick])
                c3, r3 = top(Pt, test.y)
                ts = c.automation_threshold(c2, r2, 0.10, DELTA)
                ms = c3 >= ts
                same_rates.append(ms.mean())
                same_viol.append(bool(ms.any()) and (~r3[ms]).mean() > 0.10)
                q = c.threshold(c.lac_scores(Pd, cal.y[pick]), 0.9)
                same_cov.append(c.set_stats(c.lac_sets(Pt, q), test.y)["coverage"])
                # The strictly valid alternative: temperature on one half, threshold on the other.
                half = size // 2
                ta = c.fit_temperature([(P_cal[pick[:half]], cal.y[pick[:half]])], prior=1.0, shrinkage=c.SHRINKAGE)
                c4, r4 = top(c.scale(P_cal[pick[half:]], ta), cal.y[pick[half:]])
                c5, r5 = top(c.scale(P_test, ta), test.y)
                mh = c5 >= c.automation_threshold(c4, r4, 0.10, DELTA)
                split_rates.append(mh.mean())
                split_viol.append(bool(mh.any()) and (~r5[mh]).mean() > 0.10)
                # Naive: the lowest threshold whose *observed* error is at most 10%.
                order = np.argsort(-cc[pick])
                errs = np.cumsum(~rc[pick][order]) / np.arange(1, size + 1)
                ok = np.where(errs <= 0.10)[0]
                tn = cc[pick][order][ok.max()] if len(ok) else np.inf
                mn = ct >= tn
                naive_viol.append(bool(mn.any()) and (~rt[mn]).mean() > 0.10)
        need[size] = (float(np.mean(rates)), float(np.mean(viol)), float(np.mean(naive_viol)))
        same_summary.append((float(np.mean(same_viol)), float(np.mean(same_cov)),
                             float(np.mean(same_rates)), float(np.mean(split_rates))))
        rows.append([size, f"{np.mean(rates):.0%}", f"{np.mean(viol):.1%}",
                     f"{np.mean(same_rates):.0%}", f"{np.mean(same_viol):.1%}",
                     f"{np.mean(split_rates):.0%}", f"{np.mean(split_viol):.1%}",
                     f"{np.mean(same_cov):.3f}", f"{np.mean(naive_viol):.1%}"])
    md.append(f"\n**Labels needed, and does calibrate() keep the promise?** ε = 10%, "
              f"{DRAWS // 4} random draws of n labels per dataset (only datasets whose "
              f"calibration half has at least n). A violation is a test half whose error among "
              f"automated answers exceeds 10%; the guarantee allows {DELTA:.0%} of draws, plus "
              f"test-half sampling noise. Three ways to use the labels: *default T* uses them only "
              f"for the threshold; *calibrate()* first fits the task temperature on the same "
              f"labels, which strictly speaking uses them twice; *split* fits the temperature on "
              f"half and the threshold on the other half, which is strictly valid. The last "
              f"columns are calibrate()'s 90% set coverage on the same draws, and the naive rule "
              f"(the lowest threshold whose observed error is at most 10%):\n")
    md.append(table(rows, ["labels", "automated, default T", "violations", "automated, calibrate()",
                           "violations", "automated, split", "violations",
                           "90% set coverage, calibrate()", "violations, naive threshold"]))
    worst_viol = max(v for v, _, _, _ in same_summary)
    covs = [cv for _, cv, _, _ in same_summary]
    split_cost = mean(a - b for _, _, a, b in same_summary)
    md.append(f"\nFitting one temperature on the same labels: at most {worst_viol:.1%} of draws "
              f"over the bound ({DELTA:.0%} allowed) and 90% sets covering "
              f"{min(covs):.3f}–{max(covs):.3f}; splitting the labels between the two steps "
              f"automates {split_cost * 100:+.0f} points less on average.\n")
    figures["automation_labels"] = need

    # 3. Isotonic regression and task temperature from few labels -------------
    md.append("\n## 3. Fitting a task from a few labels: temperature or isotonic regression\n")
    md.append("Top-answer ECE on the test half, from n random labels of the calibration half "
              f"({DRAWS // 4} draws per dataset; formula T is the zero-label default, isotonic "
              "regression is fitted on the formula-T confidence):\n")
    rows, fit_fig = [], {}
    base = mean(c.ece_top(*top(c.scale(halves[n][1].P, formula_lodo[n]), halves[n][1].y)) for n in text)
    for size in LABELS:
        t_e, s_e, i_e = [], [], []
        for n in text:
            cal, test = halves[n]
            if len(cal.y) < size:
                continue
            ct, rt = top(c.scale(test.P, formula_lodo[n]), test.y)
            cc_all = c.scale(cal.P, formula_lodo[n])
            for _ in range(DRAWS // 8):
                pick = rng.choice(len(cal.y), size, replace=False)
                t = c.fit_temperature([(cal.P[pick], cal.y[pick])])
                t_e.append(c.ece_top(*top(c.scale(test.P, t), test.y)))
                t = c.fit_temperature([(cal.P[pick], cal.y[pick])], prior=formula_lodo[n],
                                      shrinkage=c.SHRINKAGE)
                s_e.append(c.ece_top(*top(c.scale(test.P, t), test.y)))
                cc, rc = top(cc_all[pick], cal.y[pick])
                i_e.append(c.ece_top(c.apply_isotonic(c.fit_isotonic(cc, rc), ct), rt))
        fit_fig[size] = (float(np.mean(t_e)), float(np.mean(i_e)), float(np.mean(s_e)))
        rows.append([size, fmt(np.mean(t_e)), fmt(np.mean(s_e)), fmt(np.mean(i_e))])
    md.append(table([["0 (formula T)", fmt(base), fmt(base), "—"]] + rows,
                    ["labels", "task temperature", "task temperature, pulled to the formula",
                     "isotonic regression"]))
    floor = mean(c.ece_floor(c.scale(halves[n][1].P, formula_lodo[n]), draws=100) for n in text)
    most = max(fit_fig)
    md.append(f"\nThe sampling floor of these test halves is about {floor:.3f}: a perfectly "
              f"calibrated model would show that much ECE on them. The pulled task temperature "
              f"from {most} labels reaches {fit_fig[most][2]:.3f}.\n")
    caught_up = [n for n in sorted(fit_fig) if fit_fig[n][1] <= fit_fig[n][2] + 0.002]
    md.append(f"\nThe pull is worth {c.SHRINKAGE:g} examples (`calibrate()` does the same, "
              f"towards the temperature the answers already have). Isotonic regression "
              + (f"catches up with the temperature from {caught_up[0]} labels on."
                 if caught_up else "stays behind the temperature at every label count here.")
              + "\n")
    figures["fit_labels"] = (base, fit_fig)

    # 4. Coverage per class ---------------------------------------------------
    md.append("\n## 4. Coverage per class on imbalanced tasks\n")
    rows = []
    for n in [m for m in text if run[m].k == 2]:
        cal, test = halves[n]
        Pc, Pt = c.scale(cal.P, formula_lodo[n]), c.scale(test.P, formula_lodo[n])
        minority = int(np.argmin(np.bincount(run[n].y, minlength=2)))
        share = float(np.mean(run[n].y == minority))
        if share > 0.35:
            continue
        q = c.threshold(c.lac_scores(Pc, cal.y), 0.9)
        overall = c.lac_sets(Pt, q)
        per_class = np.zeros_like(overall)
        for k in range(2):
            qk = c.threshold(c.lac_scores(Pc[cal.y == k], cal.y[cal.y == k]), 0.9)
            per_class[:, k] = (1 - Pt[:, k]) <= qk
        m = test.y == minority
        rows.append([n, f"{run[n].options[minority]} ({share:.0%})",
                     fmt(c.set_stats(overall[m], test.y[m])["coverage"], 3),
                     fmt(c.set_stats(per_class[m], test.y[m])["coverage"], 3),
                     fmt(c.set_stats(overall, test.y)["size"], 2),
                     fmt(c.set_stats(per_class, test.y)["size"], 2)])
    md.append("Binary tasks whose rarer class is under 35%. With one cutoff, 90% coverage holds "
              "on average but can fail for the rare class, often the one that matters. A cutoff "
              "per class (Mondrian conformal) restores it at the cost of larger sets:\n")
    md.append(table(rows, ["dataset", "rare class", "rare-class coverage, one cutoff",
                           "rare-class coverage, per class", "mean set, one cutoff",
                           "mean set, per class"]))

    # 5. Documents --------------------------------------------------------------
    if DOCUMENT in run:
        cal, test = halves[DOCUMENT]
        t_formula = c.formula_temperature(*shipped, options[DOCUMENT])
        t_doc = c.fit_temperature([(cal.P, cal.y)])
        cells = [["raw", fmt(c.ece(test.P, test.y)), fmt(c.overconfidence(test.P, test.y))],
                 [f"formula T = {t_formula:.2f}", fmt(c.ece(c.scale(test.P, t_formula), test.y)),
                  fmt(c.overconfidence(c.scale(test.P, t_formula), test.y))],
                 [f"own T = {t_doc:.2f}", fmt(c.ece(c.scale(test.P, t_doc), test.y)),
                  fmt(c.overconfidence(c.scale(test.P, t_doc), test.y))]]
        if DOCUMENT in second:
            _, t2 = c.split(second[DOCUMENT])
            cells.append(["formula T, run 2", fmt(c.ece(c.scale(t2.P, t_formula), t2.y)),
                          fmt(c.overconfidence(c.scale(t2.P, t_formula), t2.y))])
        md.append("\n## 5. Scanned documents\n")
        md.append(f"rvl_cdip (16 options) was not used to fit the formula. Its own best T is "
                  f"{t_doc:.2f}, the formula gives {t_formula:.2f}:\n")
        md.append(table(cells, ["rvl_cdip test half", "ECE", "overconfidence"]))

    # 6. Position bias ------------------------------------------------------------
    if args.rotations:
        md += position_bias(args, run, formula_lodo, shipped, figures)

    return "\n".join(md) + "\n", figures


# -- position bias (PriDe) ------------------------------------------------------

def load_rotations(path: Path) -> dict[str, list[dict]]:
    out = {}
    for f in sorted(Path(path).glob("*.jsonl")):
        rows = [json.loads(line) for line in f.open()]
        if rows:
            out[f.stem] = rows
    return out


def by_position(rows: list[dict], rotation: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probabilities in the caller's option order, each option's position, gold."""
    options = rows[0]["options"]
    P = np.array([[r["rotations"][rotation]["probabilities"][o] for o in options] for r in rows])
    pos = np.array([[r["rotations"][rotation]["position"][o] for o in options] for r in rows])
    y = np.array([options.index(r["gold"]) for r in rows])
    return P / P.sum(axis=1, keepdims=True), pos, y


def position_prior(rows: list[dict]) -> np.ndarray:
    """PriDe: log-prior per position, from every rotation of the given rows.

    Across rotations each option visits several positions, so averaging the
    log probability *by position* averages the content out and leaves what
    the model gives a position regardless of what is there.
    """
    k = len(rows[0]["options"])
    total, count = np.zeros(k), np.zeros(k)
    for rotation in range(len(rows[0]["rotations"])):
        P, pos, _ = by_position(rows, rotation)
        logs = np.log(np.maximum(P, c.FLOOR))
        logs -= logs.mean(axis=1, keepdims=True)
        np.add.at(total, pos.ravel(), logs.ravel())
        np.add.at(count, pos.ravel(), 1)
    prior = total / np.maximum(count, 1)
    return prior - prior.mean()


def debias(P: np.ndarray, pos: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    return c.softmax(np.log(np.maximum(P, c.FLOOR)) - log_prior[pos])


def position_bias(args, run, formula_lodo, shipped, figures) -> list[str]:
    rot = load_rotations(Path(args.rotations))
    md = ["\n## 6. Position bias: rotations and PriDe\n"]
    rows, deltas = [], defaultdict(list)
    paired = {"one order": [], "all rotations": []}   # per-row correctness, pooled over datasets
    refit = {"one order": [], "all rotations": []}    # NLL after each method's own temperature
    rng = np.random.default_rng(c.SEED)
    for n in sorted(rot, key=lambda m: (len(rot[m][0]["options"]), m)):
        data = rot[n]
        k = len(data[0]["options"])
        t = formula_lodo.get(n, c.formula_temperature(*shipped, k))
        P0, pos0, y = by_position(data, 0)
        rotations = len(data[0]["rotations"])
        averaged = np.mean([by_position(data, r)[0] for r in range(rotations)], axis=0)
        # PriDe: estimate the prior on 10% of the rows (all rotations), apply to the rest
        # at the cost of one order. Repeated over disjoint estimation sets.
        pride_acc, pride_nll, pride_ece = [], [], []
        order = rng.permutation(len(data))
        folds = np.array_split(order, 10)
        for fold in folds:
            prior = position_prior([data[i] for i in fold])
            rest = np.setdiff1d(order, fold)
            Pd = c.scale(debias(P0[rest], pos0[rest], prior), t)
            pride_acc.append(c.accuracy(Pd, y[rest]))
            pride_nll.append(c.nll(Pd, y[rest]))
            pride_ece.append(c.ece(Pd, y[rest]))
        single, avg = c.scale(P0, t), c.scale(averaged, t)
        paired["one order"].append(single.argmax(1) == y)
        paired["all rotations"].append(avg.argmax(1) == y)
        # Averaging distributions softens them, which lowers NLL by itself. Give each method
        # its own best temperature on these rows before comparing NLL.
        for label, P in (("one order", P0), ("all rotations", averaged)):
            refit[label].append(c.nll(c.scale(P, c.fit_temperature([(P, y)])), y))
        cells = {"one order": (c.accuracy(single, y), c.nll(single, y), c.ece(single, y)),
                 "all rotations": (c.accuracy(avg, y), c.nll(avg, y), c.ece(avg, y)),
                 "PriDe": (float(np.mean(pride_acc)), float(np.mean(pride_nll)), float(np.mean(pride_ece)))}
        for label, v in cells.items():
            deltas[label].append(v)
        prior = position_prior(data)
        # Only where the rotations put every option in every position does the
        # average by position cancel the content; with more options it doesn't.
        covered = rotations >= k
        rows.append([n, k, len(data)] + [f"{v[0]:.2f} / {v[1]:.2f}" for v in cells.values()]
                    + [fmt(float(np.exp(prior.max() - prior.min())), 2) if covered else "—"])
        if covered:
            deltas["_small"].append((cells["one order"][0], cells["all rotations"][0], cells["PriDe"][0]))
    counts = sorted({len(rot[n][0]["rotations"]) for n in rot})
    per_set = int(np.median([len(rot[n]) for n in rot]))
    md.append(f"Every row asked in {'/'.join(map(str, counts))} rotated option orders (about "
              f"{per_set} rows per text dataset). Compared "
              "on the same rows, after the formula T: one order (the default), the average of "
              "all rotations (4× the cost; the strongest standard position fix), and PriDe "
              "(position prior estimated from 10% of the rows in all rotations, applied to the "
              "other 90% at the cost of one order).\n")
    small = deltas.pop("_small", [])
    labels = list(deltas)
    md.append(table([[label, fmt(mean(v[0] for v in deltas[label]), 4),
                      f"{(mean(v[0] for v in deltas[label]) - mean(v[0] for v in deltas['one order'])) * 100:+.2f}",
                      fmt(mean(v[1] for v in deltas[label]), 3), fmt(mean(v[2] for v in deltas[label]), 3)]
                     for label in labels],
                    ["method", "mean accuracy", "points vs one order", "mean NLL", "mean ECE"]))
    one, rot_all = np.concatenate(paired["one order"]), np.concatenate(paired["all rotations"])
    diffs = []
    for _ in range(2000):
        rows_ = rng.integers(0, len(one), len(one))
        diffs.append(rot_all[rows_].mean() - one[rows_].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = ("**No significant difference.**" if lo <= 0 <= hi else
               "**All rotations are significantly better.**" if lo > 0 else
               "**All rotations are significantly worse.**")
    md.append(f"\n{verdict} Over all {len(one)} rows, all "
              f"rotations minus one order is {(rot_all.mean() - one.mean()) * 100:+.2f} points of "
              f"accuracy, 95% interval [{lo * 100:+.2f}, {hi * 100:+.2f}] (paired bootstrap)"
              + (f": a gain of more than {hi * 100:.1f} points is unlikely" if lo <= 0 <= hi else "")
              + f". PriDe: {mean(v[0] for v in deltas['PriDe']):.4f} mean accuracy. On the {len(small)} datasets with at most {max(counts)} options, where "
              f"{max(counts)} rotations cover every position, accuracy is {mean(v[0] for v in small):.3f} for "
              f"one order, {mean(v[1] for v in small):.3f} for all rotations and "
              f"{mean(v[2] for v in small):.3f} for PriDe. Averaging also softens the "
              f"distribution; with each method's own temperature, which takes that out, NLL "
              f"is {mean(refit['one order']):.3f} for one order and "
              f"{mean(refit['all rotations']):.3f} for all rotations. With more options than "
              f"rotations the position prior can't be separated from content, which limits "
              f"PriDe on the many-option sets.\n")
    md.append("\nPer dataset, accuracy / NLL; the last column is how much more the model likes "
              "its favourite position than its least favourite, shown only where the rotations "
              "cover every position:\n")
    md.append(table(rows, ["dataset", "options", "rows"] + [f"{l} (acc / NLL)" for l in labels]
                    + ["position preference, max/min"]))
    figures["rotations"] = {label: [v[0] for v in deltas[label]] for label in labels}

    for label, path in (("original", args.rotations_boolq), ("renamed", args.rotations_renamed)):
        if not path:
            continue
        rr = load_rotations(Path(path)).get("boolq")
        if not rr:
            continue
        P0, pos0, y = by_position(rr, 0)
        rotations = boolq_rotations = len(rr[0]["rotations"])
        averaged = np.mean([by_position(rr, r)[0] for r in range(rotations)], axis=0)
        accs = []
        for fold in np.array_split(rng.permutation(len(rr)), 20):
            prior = position_prior([rr[i] for i in fold])
            rest = np.setdiff1d(np.arange(len(rr)), fold)
            accs.append(c.accuracy(debias(P0[rest], pos0[rest], prior), y[rest]))
        figures.setdefault("boolq", {})[label] = (c.accuracy(P0, y), c.accuracy(averaged, y), float(np.mean(accs)))
    if "boolq" in figures:
        b = figures["boolq"]
        md.append(f"\n**boolq, renamed options** (`true`/`false` → `correct`/`wrong`, all rows, "
                  f"{boolq_rotations} rotations each):\n")
        md.append(table([[k, fmt(v[0], 3), fmt(v[1], 3), fmt(v[2], 3)] for k, v in b.items()],
                        ["options", "one order", "all rotations", "PriDe (5% to estimate)"]))
        if "renamed" in b and "original" in b:
            loss = b["original"][0] - b["renamed"][0]
            helps = b["renamed"][1] - b["renamed"][0]
            md.append(f"\nRenaming costs {loss * 100:.1f} points in one order; rotating the "
                      f"renamed options changes that by {helps * 100:+.1f}. A drop that comes "
                      f"from the option *names* rather than their positions is one a position "
                      f"fix can't recover.\n")
    return md


# -- plots ----------------------------------------------------------------------

def plots(out: Path, figures: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})

    if "jev_sets" in figures:
        rows = figures["jev_sets"]
        names = list(rows)
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        series = (("GLM, default T", "#4f81bd"), ("Jev", "#c0504d"),
                  ("Jev, zeros fixed + own T", "#9bbb59"))
        for i, (label, color) in enumerate(series):
            ax.bar(np.arange(len(names)) + i * 0.27, [rows[n][label] for n in names], 0.27,
                   label=label, color=color)
        ax.set_xticks(np.arange(len(names)) + 0.27, names, rotation=70, fontsize=6.5)
        ax.axhline(1, color="#999", lw=0.8, ls="--")
        ax.set(ylabel="mean options per set (log)", yscale="log",
               title="90% conformal sets, same examples")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "jev_sets.png")
        plt.close(fig)

    if "jev_ece" in figures:
        rows = figures["jev_ece"]
        names = list(rows)
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        series = (("GLM, default T (no labels)", "#4f81bd"), ("Jev, raw", "#c0504d"),
                  ("Jev, zeros fixed (no labels)", "#f79646"),
                  ("Jev, zeros fixed + task T (labels)", "#9bbb59"))
        for i, (label, color) in enumerate(series):
            ax.bar(np.arange(len(names)) + i * 0.2, [float(rows[n][i]) for n in names], 0.2,
                   label=label, color=color)
        ax.set_xticks(np.arange(len(names)) + 0.3, names, rotation=70, fontsize=6.5)
        ax.set(ylabel="ECE (test half)", title="Calibration error, same examples")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "jev_ece.png")
        plt.close(fig)

    if "automation" in figures:
        rows = figures["automation"]
        names = list(rows)
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        for i, (eps, color) in enumerate(zip(EPSILONS, ("#c0504d", "#f79646", "#4f81bd"))):
            ax.bar(np.arange(len(names)) + i * 0.27, [rows[n][i] for n in names], 0.27,
                   label=f"error ≤ {eps:.0%}", color=color)
        ax.set_xticks(np.arange(len(names)) + 0.27, names, rotation=70, fontsize=6.5)
        ax.set(ylabel="share of answers automated", ylim=(0, 1),
               title=f"Automation with a guaranteed error rate ({1 - DELTA:.0%} confidence)")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "automation.png")
        plt.close(fig)

    if "automation_labels" in figures:
        rows = figures["automation_labels"]
        ns = list(rows)
        fig, ax = plt.subplots(figsize=(4.8, 3.3))
        x = np.arange(len(ns))
        ax.bar(x - 0.2, [rows[n][2] for n in ns], 0.4, color="#c0504d",
               label="naive: observed error ≤ 10%")
        ax.bar(x + 0.2, [rows[n][1] for n in ns], 0.4, color="#4f81bd",
               label="Learn then Test")
        ax.axhline(DELTA, color="#999", lw=0.8, ls="--", label=f"allowed ({DELTA:.0%})")
        ax.set_xticks(x, [str(n) for n in ns])
        ax.set(xlabel="labels", ylabel="share of draws whose automated error > 10%",
               title="Does the error promise hold?", ylim=(0, 0.6))
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "guarantee.png")
        plt.close(fig)

    if "fit_labels" in figures:
        base, rows = figures["fit_labels"]
        ns = list(rows)
        fig, ax = plt.subplots(figsize=(4.4, 3.2))
        ax.plot(ns, [rows[n][0] for n in ns], marker="o", label="task temperature", color="#4f81bd")
        ax.plot(ns, [rows[n][2] for n in ns], marker="o", label="task temperature, pulled to formula",
                color="#9bbb59")
        ax.plot(ns, [rows[n][1] for n in ns], marker="o", label="isotonic regression", color="#f79646")
        ax.axhline(base, color="#999", lw=0.8, ls="--", label="formula T, no labels")
        ax.set(xscale="log", xlabel="labels", ylabel="top-answer ECE (test half)",
               title="Fitting a task from few labels")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fit_labels.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for flag in ("--run", "--summary", "--out"):
        parser.add_argument(flag, required=True)
    for flag in ("--second", "--published", "--rotations", "--rotations-boolq", "--rotations-renamed"):
        parser.add_argument(flag)
    args = parser.parse_args()
    text, figures = report(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plots(out, figures)
    names = [p.stem for p in sorted(out.glob("*.png"))]
    text += "\n## Figures\n\n" + "\n".join(f"![{n}]({n}.png)" for n in names) + "\n"
    (out / "full-report.md").write_text(text)
    print(f"wrote {out / 'full-report.md'}")


if __name__ == "__main__":
    main()
