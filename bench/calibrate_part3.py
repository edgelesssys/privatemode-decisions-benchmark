"""The third calibration report: a bias per option, several option orders,
many options, other models.

    python -m bench.calibrate_part3 --run runs/r1 --summary part-1/summary.json \\
        --published published/results --rotations runs/rotations \\
        [--model kimi=runs/kimi --model glm=runs/glm] --out report3/

Uses the same calibration/test halves as parts 1 and 2 and the per-dataset
temperatures in part 1's ``summary.json``. Jev's rows come from the published
runs, and every method applied to GLM is applied to Jev too, on the same
examples. ``--model`` adds a run of another model (``name=directory``) for
the section on other models; each also gets its own ``bench.calibrate_report``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import calibration as c
from .calibrate_extensions import DELTA, JEV_UNIT, common, load_arm, load_rotations, top
from .calibrate_report import DOCUMENT, fmt, mean, table

LABELS = (20, 50, 100, 250, 500)
DRAWS = 20
#: The bias's pull towards 0, in examples (``strength / n · |b|²``).
STRENGTH = 2.0
STRENGTHS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
SHRINKAGE = 5.0
EPSILON = 0.10
COVERAGE = 0.9
MANY = ("banking77", "clinc150", "ledgar")


def unzero(P: np.ndarray) -> np.ndarray:
    """Jev rounds to 0.01: put half a rounding unit where it says 0."""
    Q = np.where(P == 0, JEV_UNIT / 2, P)
    return Q / Q.sum(axis=1, keepdims=True)


def band(k: int) -> str:
    return "2" if k == 2 else "3–6" if k <= 6 else "7–20" if k <= 20 else "21+"


def excess(P: np.ndarray, y: np.ndarray) -> float:
    return c.ece(P, y) - c.ece_floor(P, draws=50)


def fit(P: np.ndarray, y: np.ndarray, bias: bool, strength: float = STRENGTH):
    """What calibrate() fits: a temperature pulled towards no change, and
    optionally a bias per option. Returns a function that applies it."""
    if bias:
        t, b = c.fit_temperature_bias(P, y, prior=1.0, shrinkage=SHRINKAGE, strength=strength)
        return lambda Q: c.scale_bias(Q, t, b)
    t = c.fit_temperature([(P, y)], prior=1.0, shrinkage=SHRINKAGE)
    return lambda Q: c.scale(Q, t)


FOLDS = 5


def out_of_fold(P: np.ndarray, y: np.ndarray, bias: bool, folds: int = FOLDS) -> np.ndarray:
    """Each labelled answer corrected by a fit on the other folds: what the
    cutoffs and the threshold are set on when the bias is fitted, so they
    don't see answers the fit has already bent towards their labels."""
    out = np.zeros_like(P)
    order = np.array(fold_order(len(y)))
    for part in np.array_split(order, folds):
        rest = np.setdiff1d(order, part)
        out[part] = fit(P[rest], y[rest], bias)(P[part])
    return out


def fold_order(n: int, seed: int = 0) -> list[int]:
    """The library's fold order (decisions.calibration.fold_order): a seeded
    permutation, so folds don't depend on how the labels are sorted."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def evaluate_fit(apply, P_fit, y_fit, P_test, y_test, P_cut=None) -> dict:
    """calibrate()'s whole path on one draw: the fit, then the conformal
    cutoff and the automation threshold, on the corrected probabilities of
    the same labels (or on ``P_cut``, their out-of-fold version)."""
    Pf, Pt = apply(P_fit) if P_cut is None else P_cut, apply(P_test)
    cutoff = c.threshold(c.lac_scores(Pf, y_fit), COVERAGE)
    sets = c.set_stats(c.lac_sets(Pt, cutoff), y_test)
    cf, rf = top(Pf, y_fit)
    ct, rt = top(Pt, y_test)
    automated = ct >= c.automation_threshold(cf, rf, EPSILON, DELTA)
    error = float((~rt[automated]).mean()) if automated.any() else 0.0
    return {"accuracy": c.accuracy(Pt, y_test), "nll": c.nll(Pt, y_test), "ece": c.ece(Pt, y_test),
            "coverage": sets["coverage"], "size": sets["size"],
            "automated": float(automated.mean()), "violation": float(automated.any() and error > EPSILON)}


# -- 1. a bias per option ---------------------------------------------------------

def bias_section(run, halves, text, lodo, jev, figures, rng) -> tuple[list[str], dict]:
    md = ["## 1. A bias per option\n"]
    md.append(f"`softmax(log p / T + b)`: a temperature and one bias per option, fitted together "
              f"on n random labels of the calibration half ({DRAWS} draws per dataset), with the "
              f"temperature pulled towards the default (worth {SHRINKAGE:g} examples, as before) "
              f"and the bias towards 0 (worth {STRENGTH:g} examples: `{STRENGTH:g} / n · |b|²`). "
              f"Then, as `calibrate()` does, the 90% cutoff and the automation threshold "
              f"(ε = {EPSILON:.0%}) on the same labels and the corrected probabilities. Evaluated "
              f"on the test half. *T* is the temperature alone (what part 2 shipped), *T + b* adds "
              f"the bias and sets the cutoff and the threshold on out-of-fold probabilities "
              f"({FOLDS} folds: each label's answer corrected by a fit on the other folds), then "
              f"fits the final correction on all labels. Excess ECE is ECE minus the sampling "
              f"floor of the same probabilities.\n")
    per = {}   # (system, method, size) -> dataset -> mean metrics
    systems = {"glm": {n: halves[n] for n in text}}
    if jev:
        systems["glm-jev"], systems["jev"] = {}, {}
        for n in text:
            if n in jev:
                g, j = common(run[n], jev[n])
                systems["glm-jev"][n] = c.split(g)
                systems["jev"][n] = tuple(c.Dataset(d.name, d.options, d.index, unzero(d.P), d.y)
                                          for d in c.split(j))
        # Jev's own zero-label default, as part 1's overview gives it: the
        # option formula fitted on Jev's per-task temperatures, leaving the
        # dataset out.
        own = {n: c.fit_temperature([(cal.P, cal.y)]) for n, (cal, _) in systems["jev"].items()}
        k = {n: run[n].k for n in own}
        jev_default = {n: c.formula_temperature(*c.fit_formula({m: own[m] for m in own if m != n}, k), k[n])
                       for n in own}
    for system, data in systems.items():
        for size in LABELS:
            for n, (cal, test) in data.items():
                if len(cal.y) < size:
                    continue
                t0 = jev_default[n] if system == "jev" else lodo[n]
                Pc, Pt = c.scale(cal.P, t0), c.scale(test.P, t0)
                rows = defaultdict(list)
                for _ in range(DRAWS):
                    pick = rng.choice(len(cal.y), size, replace=False)
                    Pp, yp = Pc[pick], cal.y[pick]
                    rows["T"].append(evaluate_fit(fit(Pp, yp, False), Pp, yp, Pt, test.y))
                    with_bias = fit(Pp, yp, True)
                    rows["T + b"].append(evaluate_fit(with_bias, Pp, yp, Pt, test.y,
                                                      out_of_fold(Pp, yp, True)))
                    rows["T + b, same labels"].append(evaluate_fit(with_bias, Pp, yp, Pt, test.y))
                for method, values in rows.items():
                    m = {key: float(np.mean([v[key] for v in values])) for key in values[0]}
                    # Excess ECE from the mean fit's probabilities would need them; the floor
                    # barely moves with the fit, so use the floor of the first draw's.
                    per.setdefault((system, method, size), {})[n] = m
    floors = {}
    for system, data in systems.items():
        for n, (cal, test) in data.items():
            t0 = jev_default[n] if system == "jev" else lodo[n]
            floors[(system, n)] = c.ece_floor(c.scale(test.P, t0), draws=50)

    def agg(system, method, size, key):
        values = per.get((system, method, size), {})
        return mean(v[key] for v in values.values())

    def ex(system, method, size):
        values = per.get((system, method, size), {})
        return mean(v["ece"] - floors[(system, n)] for n, v in values.items())

    rows = []
    for size in LABELS:
        base, new = per[("glm", "T", size)], per[("glm", "T + b", size)]
        diffs = np.array([new[n]["accuracy"] - base[n]["accuracy"] for n in base])
        rows.append([size, len(base), f"{agg('glm', 'T', size, 'accuracy'):.1%}",
                     f"{agg('glm', 'T + b', size, 'accuracy'):.1%}",
                     f"{diffs.mean() * 100:+.1f}", f"{(diffs > 0.001).sum()} / {(diffs < -0.001).sum()}",
                     f"{diffs.min() * 100:+.1f}",
                     f"{agg('glm', 'T', size, 'nll'):.3f} → {agg('glm', 'T + b', size, 'nll'):.3f}",
                     f"{ex('glm', 'T', size):.3f} → {ex('glm', 'T + b', size):.3f}",
                     f"{agg('glm', 'T', size, 'coverage'):.3f} → {agg('glm', 'T + b', size, 'coverage'):.3f}",
                     f"{agg('glm', 'T', size, 'size'):.2f} → {agg('glm', 'T + b', size, 'size'):.2f}",
                     f"{agg('glm', 'T', size, 'automated'):.0%} → {agg('glm', 'T + b', size, 'automated'):.0%}",
                     f"{agg('glm', 'T', size, 'violation'):.1%} → {agg('glm', 'T + b', size, 'violation'):.1%}"])
    md.append("**GLM-5.3-Flash**, mean over datasets (datasets whose calibration half has at "
              "least n examples):\n")
    md.append(table(rows, ["labels", "datasets", "accuracy, T", "accuracy, T + b", "points",
                           "datasets better / worse", "worst dataset", "NLL", "excess ECE",
                           "90% set coverage", "90% set size", f"automated at ε = {EPSILON:.0%}",
                           "draws over ε"]))
    rows = []
    for size in LABELS:
        rows.append([size] + [f"{agg('glm', m, size, 'automated'):.0%} / {agg('glm', m, size, 'violation'):.1%}"
                              f" / {agg('glm', m, size, 'coverage'):.3f}"
                              for m in ("T", "T + b, same labels", "T + b")])
    md.append("\n**Why out-of-fold.** With a bias per option, setting the cutoff and the threshold "
              "on the same labels the bias was fitted on makes the answers look better than they "
              "are, and the error bound starts to slip as the fit gets more room: at 500 labels "
              "it was broken on 2 of 23 datasets. Automated share / share of draws over ε / 90% "
              "coverage:\n")
    md.append(table(rows, ["labels", "T, same labels (part 2)", "T + b, same labels",
                           "T + b, out-of-fold (shipped)"]))
    figures["bias_by_dataset"] = {size: {n: per[("glm", "T + b", size)][n]["accuracy"]
                                            - per[("glm", "T", size)][n]["accuracy"]
                                            for n in per[("glm", "T", size)]} for size in LABELS}
    figures["bias"] = {size: [per[("glm", "T + b", size)][n]["accuracy"] - per[("glm", "T", size)][n]["accuracy"]
                              for n in per[("glm", "T", size)]] for size in LABELS}

    # By option count, at 100 labels.
    rows = []
    for name in ("2", "3–6", "7–20", "21+"):
        chosen = [n for n in per[("glm", "T", 100)] if band(run[n].k) == name]
        d = [per[("glm", "T + b", 100)][n]["accuracy"] - per[("glm", "T", 100)][n]["accuracy"] for n in chosen]
        rows.append([name, len(chosen), f"{np.mean(d) * 100:+.1f}", f"{min(d) * 100:+.1f}",
                     ", ".join(f"{n} {v * 100:+.1f}" for n, v in
                               sorted(zip(chosen, d), key=lambda x: -x[1])[:2])])
    md.append("\nBy option count, 100 labels (accuracy points, T + b minus T):\n")
    md.append(table(rows, ["options", "datasets", "mean", "worst", "largest gains"]))

    # Binary tasks: T + b is Platt scaling.
    binary = [n for n in text if run[n].k == 2]
    rows = []
    for n in binary:
        cells = [n]
        for size in (100, 500):
            if n in per[("glm", "T", size)]:
                a, b = per[("glm", "T", size)][n], per[("glm", "T + b", size)][n]
                cells.append(f"{a['accuracy']:.1%} → {b['accuracy']:.1%} (NLL {a['nll']:.3f} → {b['nll']:.3f})")
            else:
                cells.append("—")
        rows.append(cells)
    md.append("\n**Binary tasks.** With two options, temperature plus a bias is Platt scaling, "
              "which openjev's server applies to yes/no questions. Accuracy (NLL), temperature "
              "alone → Platt:\n")
    md.append(table(rows, ["dataset", "100 labels", "500 labels"]))

    # Strength sweep at 100 labels.
    md.append(f"\n**How strong a pull.** 100 labels, {DRAWS // 2} draws per dataset, accuracy "
              f"points against the temperature alone and the worst dataset:\n")
    rows = []
    for strength in STRENGTHS:
        d, worst, nlls = [], [], []
        for n in text:
            cal, test = halves[n]
            if len(cal.y) < 100:
                continue
            Pc, Pt = c.scale(cal.P, lodo[n]), c.scale(test.P, lodo[n])
            acc = []
            for _ in range(DRAWS // 2):
                pick = rng.choice(len(cal.y), 100, replace=False)
                Q = fit(Pc[pick], cal.y[pick], True, strength)(Pt)
                acc.append((c.accuracy(Q, test.y) - c.accuracy(Pt, test.y), c.nll(Q, test.y)))
            d.append(np.mean([a for a, _ in acc]))
            nlls.append(np.mean([v for _, v in acc]))
        rows.append([f"{strength:g}" + (" (chosen)" if strength == STRENGTH else ""),
                     f"{np.mean(d) * 100:+.2f}", f"{min(d) * 100:+.1f}", f"{np.mean(nlls):.3f}"])
    md.append(table(rows, ["strength (examples)", "accuracy points", "worst dataset", "NLL"]))

    # Jev, same method.
    if jev:
        rows = []
        for size in (100, 500):
            if ("jev", "T", size) not in per:
                continue
            for system, label in (("glm-jev", "GLM-5.3-Flash"), ("jev", "Jev, zeros fixed")):
                rows.append([size, label, f"{agg(system, 'T', size, 'accuracy'):.1%}",
                             f"{agg(system, 'T + b', size, 'accuracy'):.1%}",
                             f"{ex(system, 'T', size):.3f}", f"{ex(system, 'T + b', size):.3f}",
                             f"{agg(system, 'T + b', size, 'coverage'):.3f}",
                             f"{agg(system, 'T + b', size, 'size'):.2f}",
                             f"{agg(system, 'T', size, 'automated'):.0%}",
                             f"{agg(system, 'T + b', size, 'automated'):.0%}",
                             f"{agg(system, 'T + b', size, 'violation'):.1%}"])
        md.append(f"\n**Jev with the same method**, on the {len(systems['jev'])} text datasets both "
                  f"answer, same examples. Jev's zeros are set to half a rounding unit first, the "
                  f"fairest fix from part 2, so that a temperature and a bias can be fitted, and "
                  f"Jev starts from its own zero-label default temperature (the option formula "
                  f"fitted on Jev's per-task temperatures, leaving the dataset out), as GLM starts "
                  f"from its own:\n")
        md.append(table(rows, ["labels", "system", "accuracy, T", "accuracy, T + b",
                               "excess ECE, T", "excess ECE, T + b", "90% coverage, T + b",
                               "90% set, T + b", f"automated, T", "automated, T + b",
                               "draws over ε, T + b"]))
    summary = {size: {system: {method: {key: agg(system, method, size, key)
                                        for key in ("accuracy", "nll", "coverage", "size",
                                                    "automated", "violation")}
                                        | {"excess_ece": ex(system, method, size)}
                                for method in ("T", "T + b")}
                      for system in systems if (system, "T", size) in per}
               for size in LABELS}
    return md, summary


# -- 2. several option orders -----------------------------------------------------

def orders_of(rows: list[dict], count: int) -> list[int]:
    """Which stored rotations the library's ``permutations=count`` would ask:
    the ones whose offset matches ``decisions.inference.rotations``."""
    from decisions.inference import rotations

    names = rows[0]["options"]
    wanted = [order[0] for order in rotations(len(names), count)]   # the option at position 0
    stored = []
    for r, rotation in enumerate(rows[0]["rotations"]):
        # position p holds option (p + offset) % k, so position 0 holds the offset
        first = min(rotation["position"], key=rotation["position"].get)
        stored.append((names.index(first), r))
    by_offset = dict(stored)
    return [by_offset[o] for o in wanted if o in by_offset]


def averaged(rows: list[dict], which: list[int]) -> tuple[np.ndarray, np.ndarray]:
    from .calibrate_extensions import by_position

    P = np.mean([by_position(rows, r)[0] for r in which], axis=0)
    return P, by_position(rows, 0)[2]


def entropy(P: np.ndarray) -> np.ndarray:
    return -np.sum(np.where(P > 0, P * np.log(np.maximum(P, c.FLOOR)), 0.0), axis=1)


def best_automation(P: np.ndarray, y: np.ndarray, eps: float = EPSILON) -> float:
    """Share of answers that can be automated at an observed error of at most
    ``eps``, most confident first: how well the confidence ranks, knowing the
    labels (no guarantee)."""
    conf, right = top(P, y)
    order = np.argsort(-conf, kind="stable")
    errors = np.cumsum(~right[order]) / np.arange(1, len(y) + 1)
    ok = np.where(errors <= eps)[0]
    return (ok.max() + 1) / len(y) if len(ok) else 0.0


def permutation_section(rot, text, shipped, figures) -> tuple[list[str], dict]:
    md = ["\n## 2. Several option orders: their own temperature\n"]
    names = sorted((n for n in rot if n in text), key=lambda n: (len(rot[n][0]["options"]), n))
    options = {n: len(rot[n][0]["options"]) for n in names}
    counts = (1, 2, 4)
    P_by = {m: {n: averaged(rot[n], orders_of(rot[n], m)) for n in names} for m in counts}
    oracle = {m: {n: c.fit_temperature([P_by[m][n]]) for n in names} for m in counts}
    formula = {m: c.fit_formula(oracle[m], options) for m in counts}
    lodo = {m: {} for m in counts}
    for m in counts:
        for n in names:
            a, b = c.fit_formula({d: oracle[m][d] for d in names if d != n}, options)
            lodo[m][n] = c.formula_temperature(a, b, options[n])
    # The ratio of each count's best T to one order's, the simplest rule.
    ratio = {m: float(np.exp(np.median([math.log(oracle[m][n] / oracle[1][n]) for n in names])))
             for m in counts}
    rows, results = [], {}
    for m in counts:
        cells = {}
        for label, temps in (("one-order formula", lodo[1]),
                             ("own formula (LODO)", lodo[m]),
                             ("one-order formula × ratio", {n: lodo[1][n] * ratio[m] for n in names}),
                             ("own T per dataset", oracle[m])):
            if m == 1 and "ratio" in label:
                continue
            Ps = {n: c.scale(P_by[m][n][0], temps[n]) for n in names}
            ys = {n: P_by[m][n][1] for n in names}
            cells[label] = (mean(c.nll(Ps[n], ys[n]) for n in names),
                            mean(c.ece(Ps[n], ys[n]) - c.ece_floor(Ps[n], draws=50) for n in names),
                            mean(best_automation(Ps[n], ys[n]) for n in names),
                            mean(c.accuracy(Ps[n], ys[n]) for n in names))
        results[m] = cells
        for label, (nll_, ex_, auto_, acc_) in cells.items():
            rows.append([m, label, f"{acc_:.2%}", f"{nll_:.3f}", f"{ex_:.3f}", f"{auto_:.0%}"])
    a1, b1 = formula[1]
    md.append(f"`SystemOne(permutations=k)` averages k rotated option orders and then applied the "
              f"one-order default temperature. From the rotation runs ({len(names)} text datasets, "
              f"100 rows each, 4 orders), the orders the library would ask for k = 2 and 4. "
              f"Temperatures fitted per dataset (*own T*) or by the option formula leaving the "
              f"dataset out (*LODO*). Averaging softens, so the best temperature falls: the median "
              f"ratio to one order's is {ratio[2]:.2f} for 2 orders and {ratio[4]:.2f} for 4. "
              f"*Automatable* is the share of answers that can be automated at an observed error "
              f"of at most {EPSILON:.0%}, most confident first, knowing the labels: how well the "
              f"confidence ranks answers.\n")
    md.append(table(rows, ["orders", "temperature", "accuracy", "NLL", "excess ECE", "automatable"]))
    md.append("\nFormulas `log T = a + b · log(options)` fitted on all datasets: " + "; ".join(
        f"{m} order{'s' if m > 1 else ''}: a = {formula[m][0]:.3f}, b = {formula[m][1]:.3f}"
        for m in counts) + ".\n")

    # Re-read only uncertain answers.
    md.append("\n**Re-reading only uncertain answers.** razorback16/openjev asks a question three "
              "more times when the entropy of its answer is above 0.1 nats and averages the four. "
              "The same with rotations: the first order's raw distribution decides, the answers "
              "above the entropy threshold are asked in all 4 orders and averaged, and each kind "
              "gets its own temperature (the LODO formulas above).\n")
    rows, reread = [], {}
    for tau in (None, 0.5, 0.3, 0.1, 0.05, 0.0):
        nlls, exs, autos, accs, shares = [], [], [], [], []
        for n in names:
            P1, y = P_by[1][n]
            P4, _ = P_by[4][n]
            if tau is None:
                pick = np.zeros(len(y), bool)
            else:
                pick = entropy(P1) > tau
            Q = np.where(pick[:, None], c.scale(P4, lodo[4][n]), c.scale(P1, lodo[1][n]))
            nlls.append(c.nll(Q, y))
            exs.append(c.ece(Q, y) - c.ece_floor(Q, draws=50))
            autos.append(best_automation(Q, y))
            accs.append(c.accuracy(Q, y))
            shares.append(pick.mean())
        extra = 3 * np.mean(shares)
        reread[tau] = (np.mean(shares), float(np.mean(accs)), float(np.mean(nlls)), float(np.mean(exs)))
        rows.append(["never (one order)" if tau is None else "always (4 orders)" if tau == 0 else f"> {tau:g} nats",
                     f"{np.mean(shares):.0%}", f"+{extra:.0%}", f"{np.mean(accs):.2%}",
                     f"{np.mean(nlls):.3f}", f"{np.mean(exs):.3f}", f"{np.mean(autos):.0%}"])
    md.append(table(rows, ["re-read when entropy", "answers re-read", "extra requests", "accuracy",
                           "NLL", "excess ECE", "automatable"]))

    # Order stability.
    rows, flips = [], {}
    for n in names:
        data = rot[n]
        which = list(range(len(data[0]["rotations"])))
        choices = np.array([averaged(data, [r])[0].argmax(1) for r in which])
        flips[n] = float(np.mean(choices[1:] != choices[0]))
        rows.append([n, options[n], f"{flips[n]:.1%}",
                     f"{c.accuracy(P_by[1][n][0], P_by[1][n][1]):.2f}"])
    values = list(flips.values())
    md.append(f"\n**Order stability.** How often the answer changes when the options are rotated: "
              f"the share of the other orders whose answer differs from the first order's. GLM "
              f"Flash changes its answer in {np.mean(values):.1%} of rotations (median "
              f"{np.median(values):.1%}). openjev reports 18.5% for its untuned base and 2.3% after "
              f"fine-tuning, measured by shuffling on its own questions, so the numbers are "
              f"indicative only.\n")
    md.append(table(sorted(rows, key=lambda r: -float(r[2].rstrip('%'))),
                    ["dataset", "options", "answer changes", "accuracy, one order"]))
    figures["flips"] = flips
    return md, {"formula": {m: formula[m] for m in counts}, "ratio": ratio, "results": results,
                "reread": reread, "flips": flips}


# -- 3. many options, few labels ----------------------------------------------------

SPLITS = 20


def sets_for(Q: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    member = (1 - Q) <= cutoffs[None, :]
    member[np.arange(len(Q)), Q.argmax(axis=1)] = True     # never empty, as the library
    return member


def class_conditional(datasets: dict[str, c.Dataset], t0: dict[str, float], rng) -> dict:
    """Coverage per class over repeated random halves, for each way to set
    the cutoffs."""
    out = {}
    for n, data in datasets.items():
        k = data.k
        covered = defaultdict(lambda: np.zeros(k))
        seen = np.zeros(k)
        sizes, marginal = defaultdict(list), defaultdict(list)
        for _ in range(SPLITS):
            order = rng.permutation(len(data.y))
            cal, test = order[: len(order) // 2], order[len(order) // 2:]
            P = c.scale(data.P, t0[n])
            t = c.fit_temperature([(P[cal], data.y[cal])], prior=1.0, shrinkage=SHRINKAGE)
            Q = c.scale(P, t)
            Qc, yc, Qt, yt = Q[cal], data.y[cal], Q[test], data.y[test]
            scores = c.lac_scores(Qc, yc)
            per_class = np.ones(k)
            for cls in range(k):
                q = c.threshold(scores[yc == cls], COVERAGE)
                per_class[cls] = 1.0 if q == float("inf") else q
            groups, rest = c.ding_groups(Qc, yc, COVERAGE)
            methods = {
                "one cutoff": np.full(k, c.threshold(scores, COVERAGE)),
                "per class": per_class,
                "clustered (Ding)": c.group_cutoffs(Qc[rest], yc[rest], groups, COVERAGE),
                "grouped by answers": c.group_cutoffs(Qc, yc, c.unlabelled_groups(Qc, len(yc)), COVERAGE),
            }
            seen += np.bincount(yt, minlength=k)
            for label, cutoffs in methods.items():
                S = sets_for(Qt, cutoffs)
                hit = S[np.arange(len(yt)), yt]
                covered[label] += np.bincount(yt, weights=hit, minlength=k)
                sizes[label].append(S.sum(axis=1).mean())
                marginal[label].append(hit.mean())
        enough = seen >= 10
        out[n] = {label: {"coverage": float(np.mean(marginal[label])),
                          "size": float(np.mean(sizes[label])),
                          "gap": float(np.mean(np.abs(covered[label][enough] / seen[enough] - COVERAGE))),
                          "under": float(np.mean(covered[label][enough] / seen[enough] < 0.8)),
                          "classes": int(enough.sum())}
                  for label in covered}
    return out


def many_section(run, text, lodo, jev, rng) -> tuple[list[str], dict]:
    md = ["\n## 3. Many options, few labels\n"]
    wide = [n for n in text if run[n].k > 20]
    glm = class_conditional({n: run[n] for n in wide}, lodo, rng)
    methods = list(next(iter(glm.values())))
    md.append(f"Prediction sets at 90% on the text datasets with more than 20 options, "
              f"{SPLITS} random 50/50 splits each (the calibration half, about 500 labels, is "
              f"also what fits the task temperature). *Per class* is `per_class=True`: an option "
              f"with fewer than 9 labels is in every set. *Clustered (Ding)* is Ding et al. "
              f"(2023): half of the labels describe each class by quantiles of its scores, "
              f"k-means groups classes that behave alike (one group per 50 remaining labels), "
              f"and the other half sets a cutoff per group; classes with too few labels share "
              f"the marginal cutoff. *Grouped by answers* groups classes the same way but from "
              f"the unlabelled answers (quantiles of the probability a class gets where it is "
              f"the answer), so every label goes into the cutoffs. **Class gap** is the mean "
              f"distance of a class's coverage from 90%, over classes seen at least 10 times "
              f"across the splits; **under 80%** is the share of those classes covered less "
              f"than 80% of the time.\n")
    rows = []
    for n in wide:
        for label in methods:
            v = glm[n][label]
            rows.append([n if label == methods[0] else "", run[n].k if label == methods[0] else "",
                         label, f"{v['coverage']:.3f}", f"{v['size']:.2f}", f"{v['gap']:.3f}",
                         f"{v['under']:.0%}"])
    md.append(table(rows, ["dataset", "options", "cutoffs", "coverage", "set size", "class gap",
                           "classes under 80%"]))
    agg = {label: {key: mean(glm[n][label][key] for n in wide) for key in ("coverage", "size", "gap", "under")}
           for label in methods}
    md.append("\nMean over these datasets:\n")
    md.append(table([[label, f"{v['coverage']:.3f}", f"{v['size']:.2f}", f"{v['gap']:.3f}", f"{v['under']:.0%}"]
                     for label, v in agg.items()],
                    ["cutoffs", "coverage", "set size", "class gap", "classes under 80%"]))
    summary = {"glm": agg, "datasets": wide}
    if jev:
        shared = {}
        for n in wide:
            if n in jev:
                _, j = common(run[n], jev[n])
                shared[n] = c.Dataset(n, j.options, j.index, unzero(j.P), j.y)
        if shared:
            jv = class_conditional(shared, {n: 1.0 for n in shared}, rng)
            gl = {n: glm[n] for n in shared}
            rows = []
            for label in methods:
                rows.append([label] + [f"{mean(d[n][label][key] for n in shared):{f}}"
                                       for d in (gl, jv) for key, f in (("size", ".2f"), ("gap", ".3f"))])
            md.append(f"\n**Jev with the same cutoffs** ({', '.join(shared)}; zeros set to half a "
                      f"rounding unit, then its own task temperature). GLM on all its rows, Jev on "
                      f"the rows it answered:\n")
            md.append(table(rows, ["cutoffs", "GLM set size", "GLM class gap", "Jev set size",
                                   "Jev class gap"]))
            summary["jev"] = {label: {key: mean(jv[n][label][key] for n in shared)
                                      for key in ("coverage", "size", "gap", "under")} for label in methods}
    return md, summary


def prior_section(run, halves, text, lodo, bias_by_dataset, rng) -> tuple[list[str], dict]:
    """Known class rates, no per-example labels: p' ∝ p · π / π̂, where π̂ is
    the mean answer over unlabelled traffic."""
    md = ["\n**Known class rates, from logs only.** `p′ ∝ p · π / π̂`: π are the class rates "
          "someone knows from logs, π̂ the mean answer over unlabelled traffic (the test half). "
          "The rates are the dataset's true ones, each multiplied by a random factor of up to "
          "±25% or up to 2× either way, then renormalized; 20 draws. Text datasets with up to 20 "
          "options, accuracy points against the default temperature, and the bias from 50 and "
          "100 labels (section 1) on the same datasets:\n"]
    chosen = [n for n in text if run[n].k <= 20]
    results = {}
    for label, spread in (("exact rates", 1.0), ("within 25%", 1.25), ("up to 2× off", 2.0)):
        d = []
        for n in chosen:
            _, test = halves[n]
            P = c.scale(test.P, lodo[n])
            true = np.bincount(run[n].y, minlength=run[n].k) / len(run[n].y)
            est = P.mean(axis=0)
            values = []
            for _ in range(20 if spread > 1 else 1):
                pi = true * np.exp(rng.uniform(-math.log(spread), math.log(spread), len(true)))
                pi /= pi.sum()
                Q = P * (pi / est)[None, :]
                values.append(c.accuracy(Q, test.y) - c.accuracy(P, test.y))
            d.append(float(np.mean(values)))
        results[label] = d
    rows = [[label, f"{np.mean(d) * 100:+.1f}", f"{min(d) * 100:+.1f}",
             f"{sum(v > 0.001 for v in d)} / {sum(v < -0.001 for v in d)}"] for label, d in results.items()]
    for size in (50, 100):
        d = [bias_by_dataset[size][n] for n in chosen if n in bias_by_dataset[size]]
        rows.append([f"bias from {size} labels", f"{np.mean(d) * 100:+.1f}", f"{min(d) * 100:+.1f}",
                     f"{sum(v > 0.001 for v in d)} / {sum(v < -0.001 for v in d)}"])
    md.append(table(rows, ["correction", "accuracy points", "worst dataset", "datasets better / worse"]))
    return md, {k: float(np.mean(v)) for k, v in results.items()}


# -- 4. other models ------------------------------------------------------------------

def model_profile(run: dict[str, c.Dataset]) -> dict:
    """Accuracy, raw calibration and the option formula's fit for one model."""
    text = sorted(n for n in run if n != DOCUMENT)
    options = {n: run[n].k for n in run}
    halves = {n: c.split(run[n]) for n in run}
    oracle = {n: c.fit_temperature([(halves[n][0].P, halves[n][0].y)]) for n in text}
    a, b = c.fit_formula(oracle, options)
    x, z = np.log([options[n] for n in text]), np.log([oracle[n] for n in text])
    r = float(np.corrcoef(x, z)[0, 1])
    raw_ex, lodo_ex, own_ex = [], [], []
    for n in text:
        a_, b_ = c.fit_formula({m: oracle[m] for m in text if m != n}, options)
        test = halves[n][1]
        for store, t in ((raw_ex, 1.0), (lodo_ex, c.formula_temperature(a_, b_, options[n])),
                         (own_ex, oracle[n])):
            P = c.scale(test.P, t)
            store.append(c.ece(P, test.y) - c.ece_floor(P, draws=50))
    served = sorted({m for d in run.values() for m in d.meta.get("served_model", [])})
    out = {"served": served, "datasets": len(text),
           "accuracy": mean(c.accuracy(run[n].P, run[n].y) for n in text),
           "formula": (a, b), "r": r, "median_t": float(np.median(list(oracle.values()))),
           "t_range": (min(oracle.values()), max(oracle.values())),
           "excess_raw": mean(raw_ex), "excess_formula": mean(lodo_ex), "excess_own": mean(own_ex),
           "oracle": oracle,
           "mass": mean(float(run[n].mass.mean()) for n in text if run[n].mass is not None)}
    if DOCUMENT in run:
        cal, test = halves[DOCUMENT]
        t = c.formula_temperature(a, b, options[DOCUMENT])
        out["document"] = {"accuracy": c.accuracy(run[DOCUMENT].P, run[DOCUMENT].y),
                           "own_t": c.fit_temperature([(cal.P, cal.y)]), "formula_t": t,
                           "ece_raw": c.ece(test.P, test.y), "ece_formula": c.ece(c.scale(test.P, t), test.y)}
    return out


def models_section(profiles: dict[str, dict]) -> list[str]:
    md = ["\n## 4. Other models\n"]
    md.append("One run of each model on all 29 datasets (up to 1,000 examples, 4 in flight), the "
              "same halves. *Served* is the model the endpoint reported answering, recorded in "
              "every row. The formula is fitted on all text datasets; excess ECE after it is "
              "leave-one-dataset-out, as for GLM Flash in part 1.\n")
    rows = []
    for name, p in profiles.items():
        a, b = p["formula"]
        doc = p.get("document")
        rows.append([name, ", ".join(p["served"]) or "—", p["datasets"], f"{p['accuracy']:.1%}",
                     f"{p['mass']:.3f}", f"{p['median_t']:.2f} ({p['t_range'][0]:.2f}–{p['t_range'][1]:.2f})",
                     f"{a:.3f}, {b:+.3f} (r = {p['r']:+.2f})",
                     f"{p['excess_raw']:.3f}", f"{p['excess_formula']:.3f}", f"{p['excess_own']:.3f}",
                     "—" if not doc else f"{doc['accuracy']:.1%}, T {doc['own_t']:.2f} vs {doc['formula_t']:.2f}"])
    md.append(table(rows, ["run", "served", "text datasets", "accuracy", "option mass",
                           "task T, median (range)", "formula a, b", "excess ECE raw",
                           "excess ECE, formula (LODO)", "excess ECE, task T",
                           "rvl_cdip: accuracy, own vs formula T"]))
    return md


# -- plots ------------------------------------------------------------------------------

def plots(out: Path, figures: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    if "bias" in figures:
        data = figures["bias"]
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        sizes = list(data)
        ax.boxplot([np.array(data[s]) * 100 for s in sizes], tick_labels=[str(s) for s in sizes],
                   widths=0.5, showfliers=True)
        ax.axhline(0, color="#999", lw=0.8, ls="--")
        ax.plot(range(1, len(sizes) + 1), [np.mean(data[s]) * 100 for s in sizes], marker="o",
                color="#4f81bd", label="mean over datasets")
        ax.set(xlabel="labels", ylabel="accuracy points, T + b minus T",
               title="A bias per option, per dataset")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "bias.png")
        plt.close(fig)
    if "flips" in figures:
        flips = figures["flips"]
        names = sorted(flips, key=flips.get, reverse=True)
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ax.bar(range(len(names)), [flips[n] * 100 for n in names], color="#4f81bd")
        ax.axhline(np.mean(list(flips.values())) * 100, color="#c0504d", lw=0.8, ls="--",
                   label="mean")
        ax.set_xticks(range(len(names)), names, rotation=70, fontsize=6.5)
        ax.set(ylabel="% of rotations changing the answer", title="Order stability, GLM-5.3-Flash")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "order_stability.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for flag in ("--run", "--summary", "--out"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--published")
    parser.add_argument("--rotations")
    parser.add_argument("--model", action="append", default=[], help="name=run directory")
    parser.add_argument("--sections", default="1,2,3,4")
    args = parser.parse_args()
    sections = set(args.sections.split(","))
    run = c.load_run(args.run)
    summary = json.loads(Path(args.summary).read_text())
    oracle = summary["oracle"]
    text = sorted((n for n in run if n != DOCUMENT), key=lambda n: (run[n].k, n))
    options = {n: run[n].k for n in run}
    lodo = {}
    for n in text:
        a, b = c.fit_formula({m: oracle[m] for m in text if m != n}, options)
        lodo[n] = c.formula_temperature(a, b, options[n])
    halves = {n: c.split(run[n]) for n in run}
    jev = load_arm(Path(args.published), "jev") if args.published else {}
    rng = np.random.default_rng(c.SEED)
    figures: dict = {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Calibration report, part 3\n",
          f"Same runs, halves and temperatures as parts 1 and 2 (`{Path(args.run).name}`). GLM "
          "probabilities start from the library's default temperature (the option formula "
          "fitted without the dataset in question); Jev's come from the published runs on the "
          "same examples. Means weight every dataset equally.\n"]
    result: dict = {}
    if "1" in sections:
        part, result["bias"] = bias_section(run, halves, text, lodo, jev, figures, rng)
        md += part
    if "2" in sections and args.rotations:
        part, result["permutations"] = permutation_section(
            load_rotations(Path(args.rotations)), text, summary["shipped"]["formula"], figures)
        md += part
    if "3" in sections:
        part, result["many"] = many_section(run, text, lodo, jev, rng)
        md += part
        if "bias_by_dataset" in figures:
            part, result["prior"] = prior_section(run, halves, text, lodo, figures["bias_by_dataset"], rng)
            md += part
    if "4" in sections and args.model:
        profiles = {"glm-5.3-flash (part 1 run)": model_profile(run)}
        for spec in args.model:
            name, directory = spec.split("=", 1)
            profiles[name] = model_profile(c.load_run(directory))
        md += models_section(profiles)
        result["models"] = {k: {kk: vv for kk, vv in v.items() if kk != "oracle"} for k, v in profiles.items()}
    plots(out, figures)
    names = [p.stem for p in sorted(out.glob("*.png"))]
    md.append("\n## Figures\n\n" + "\n".join(f"![{n}]({n}.png)" for n in names))
    (out / "report.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
