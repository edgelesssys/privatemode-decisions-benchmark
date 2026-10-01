"""The pre-registered final test on untouched tasks (part-3/holdout-plan.md).

    python -m bench.holdout run --task fin_topic --out runs/holdout [--concurrency 4]
    python -m bench.holdout report --run runs/holdout --out results/calibration/holdout

``run`` asks the Privatemode arm for raw probabilities (T = 1) on one frozen
task from ``bench.holdout_data`` and writes a run file ``load_run`` reads.
``report`` scores the test halves (``bench.calibration.split``) with every
zero-label temperature the library has or had, the per-task ceiling, and
the library's own ``calibrate()`` from labels, then checks the criteria the
plan fixed before the tasks were chosen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np
from decisions import calibration as lib
from decisions.inference import PREFIX
from decisions.types import ChoiceAnswer

from . import calibration as c
from .adapters import PrivatemodeArm, library_version
from .holdout_data import FROZEN, TASKS, load

MODEL = "glm-5.3-flash"
#: The family each task is scored under for the family temperature, fixed
#: in the plan before the run. PubMed and GitHub are families the benchmark
#: doesn't have; they get the nearest one.
FAMILY = {"fin_topic": "topic", "fin_sentiment": "sentiment", "arxiv_field": "topic",
          "pubmed_study": "topic", "github_issue": "intent"}
#: One temperature for every task, fitted on all 28 text datasets of part 1
#: (results/calibration/part-1/summary.json, "shipped.global").
GLOBAL_T = 2.1477
COVERAGE = 0.9
EPSILON = 0.10
DRAWS = 50
LABELS = 100


# -- run ------------------------------------------------------------------------------

def run(task: str, out: Path, concurrency: int) -> Path:
    from .run import ask_with_retry, load_env
    load_env(Path(__file__).resolve().parent.parent / ".env")
    tasks = load(task, strict=False)
    arm = PrivatemodeArm(model=MODEL)
    frozen = (FROZEN / f"{task}.json").read_bytes()
    # Concurrency changes no answer, and the plan allows one run: it goes in
    # the meta block, not the key, so a resume at another setting continues
    # the same file. The library does go in: the default temperatures and
    # calibrate() the report scores come from it.
    identity = {"dataset": task, "frozen_sha1": hashlib.sha1(frozen).hexdigest()[:12],
                "arms": {arm.name: arm.model}, "prefill": PREFIX,
                "privatemode_temperature": arm.temperature, "library": library_version()}
    key = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    folder = out / task
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.jsonl"
    done = set()
    if path.exists():
        done = {r["index"] for r in map(json.loads, path.read_text().splitlines())
                if r.get("kind") == "row" and "error" not in r}
    handle = path.open("a")
    if not path.stat().st_size:
        handle.write(json.dumps({"kind": "meta", "identity": identity, "dataset": task,
                                 "n": len(tasks), "options": len(tasks[0].criteria),
                                 "concurrency": concurrency,
                                 "started": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")})
                     + "\n")
    lock = Lock()
    todo = [t for t in tasks if t.index not in done]

    def ask(t) -> None:
        row = {"kind": "row", "arm": arm.name, "index": t.index, "gold": t.gold}
        try:
            answer, attempts = ask_with_retry(arm, t)
            row.update(choice=answer.choice, correct=answer.choice == t.gold,
                       probabilities=answer.probabilities, option_mass=answer.option_mass,
                       latency_s=answer.latency_s, input_tokens=answer.input_tokens,
                       attempts=attempts, served_model=answer.served_model)
        except Exception as error:   # noqa: BLE001 - recorded, retried on resume
            row["error"] = f"{type(error).__name__}: {error}"[:300]
        with lock:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()

    started = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(ask, todo))
    handle.write(json.dumps({"kind": "end", "wall_s": time.perf_counter() - started,
                             "answered": len(todo)}) + "\n")
    handle.close()
    arm.close()
    print(f"{task}: {len(todo)} asked, wrote {path}", file=sys.stderr)
    return path


# -- report ---------------------------------------------------------------------------

def answers(P: np.ndarray, options: list[str], temperature: float) -> list[ChoiceAnswer]:
    """Rows as the library returns them at ``temperature``."""
    out = []
    for row in c.scale(P, temperature):
        probabilities = dict(zip(options, map(float, row)))
        out.append(ChoiceAnswer(choice=max(probabilities, key=probabilities.get),
                                probabilities=probabilities,
                                confidence=lib.peakedness(probabilities.values()),
                                temperature=temperature))
    return out


def excess(P: np.ndarray, y: np.ndarray) -> float:
    return c.ece(P, y) - c.ece_floor(P)


def score_calibration(fit: lib.Calibration, test: list[ChoiceAnswer], y: np.ndarray,
                      options: list[str]) -> dict:
    """The fitted calibration on the test half, as a user of the library sees it."""
    P = np.array([[fit.probabilities(a)[o] for o in options] for a in test])
    sets = [fit.predict_set(a) for a in test]
    covered = np.array([options[g] in s for s, g in zip(sets, y)])
    right = P.argmax(1) == y
    automated = np.array([fit.automate(a) for a in test]) if fit.max_error else np.zeros(len(y), bool)
    error = float((~right[automated]).mean()) if automated.any() else 0.0
    return {"accuracy": float(right.mean()), "excess_ece": excess(P, y),
            "coverage": float(covered.mean()), "size": float(np.mean([len(s) for s in sets])),
            "singletons": float(np.mean([len(s) == 1 for s in sets])),
            "automated": float(automated.mean()), "error": error,
            "violation": bool(automated.any() and error > EPSILON)}


def labelled(cal: c.Dataset, test: c.Dataset, base: float, labels: int | None, bias: bool,
             draws: int, seed: int) -> dict:
    """``calibrate()`` on ``draws`` random samples of ``labels`` calibration
    labels (all of them, in a new random order, for ``None``)."""
    options = cal.options
    cal_answers, test_answers = answers(cal.P, options, base), answers(test.P, options, base)
    rng = random.Random(seed)
    results = []
    for _ in range(draws):
        rows = list(range(len(cal.y)))
        rng.shuffle(rows)
        rows = rows if labels is None else rows[:labels]
        fit = lib.calibrate([cal_answers[i] for i in rows], [options[cal.y[i]] for i in rows],
                            coverage=COVERAGE, max_error=EPSILON, bias=bias)
        results.append(score_calibration(fit, test_answers, test.y, options))
    keys = results[0]
    return {k: float(np.mean([r[k] for r in results])) for k in keys} | {
        "draws": draws, "violations": int(sum(r["violation"] for r in results))}


def evaluate(run: dict[str, c.Dataset]) -> dict:
    out = {}
    for name, data in run.items():
        cal, test = c.split(data)
        k = len(data.options)
        temperatures = {
            "raw": 1.0,
            "global T": GLOBAL_T,
            "option formula (default)": lib.default_temperature(MODEL, k),
            "family T": lib.default_temperature(MODEL, k, FAMILY[name]),
            "task T (calibration half)": c.fit_temperature([(cal.P, cal.y)]),
        }
        zero = {m: {"T": t, "accuracy": c.accuracy(test.P, test.y),
                    "confidence": float(c.scale(test.P, t).max(1).mean()),
                    "ece": c.ece(c.scale(test.P, t), test.y),
                    "excess_ece": excess(c.scale(test.P, t), test.y),
                    "nll": c.nll(c.scale(test.P, t), test.y)}
                for m, t in temperatures.items()}
        base = temperatures["option formula (default)"]
        seed = sum(map(ord, name))
        fits = {f"{'T + bias' if bias else 'T'}, {labels or 'all'} labels":
                labelled(cal, test, base, labels, bias, DRAWS, seed)
                for bias in (False, True) for labels in (LABELS, None)}
        # The plan's coverage check: the full calibration half, once.
        cal_answers = answers(cal.P, data.options, base)
        full = lib.calibrate(cal_answers, [data.options[g] for g in cal.y], coverage=COVERAGE)
        coverage = score_calibration(full, answers(test.P, data.options, base), test.y, data.options)
        out[name] = {"options": k, "examples": len(data.y), "calibration": len(cal.y),
                     "test": len(test.y), "family": FAMILY[name],
                     "option_mass": float(np.mean(data.mass)) if data.mass is not None else None,
                     "served_model": data.meta.get("served_model"), "dropped": data.meta.get("dropped"),
                     "zero_label": zero, "labelled": fits, "coverage_check": coverage}
    return out


def verdict(result: dict) -> list[tuple[str, str, str, bool]]:
    names = list(result)
    mean = lambda f: float(np.mean([f(result[n]) for n in names]))  # noqa: E731
    default = "option formula (default)"
    ece = {n: result[n]["zero_label"][default]["excess_ece"] for n in names}
    worst = max(ece, key=ece.get)
    coverage = mean(lambda r: r["coverage_check"]["coverage"])
    draws = {labels: [result[n]["labelled"][f"T + bias, {labels} labels"] for n in names]
             for labels in (LABELS, "all")}
    over = {labels: sum(d["violations"] for d in ds) / sum(d["draws"] for d in ds)
            for labels, ds in draws.items()}
    given = mean(lambda r: r["zero_label"]["raw"]["accuracy"])
    biased = mean(lambda r: r["labelled"][f"T + bias, {LABELS} labels"]["accuracy"])
    return [
        ("excess ECE, no labels (default temperature)", "mean ≤ 0.06",
         f"{mean(lambda r: r['zero_label'][default]['excess_ece']):.3f}",
         mean(lambda r: r["zero_label"][default]["excess_ece"]) <= 0.06),
        ("", "no dataset above 0.12", f"highest {ece[worst]:.3f} ({worst})", ece[worst] <= 0.12),
        ("coverage of 90% sets, calibrate() on the calibration half", "mean 0.87–0.93",
         f"{coverage:.3f}", 0.87 <= coverage <= 0.93),
        (f"automation at 10% error, {DRAWS} draws of {LABELS} labels", "≤ 10% of draws over",
         f"{over[LABELS]:.1%}", over[LABELS] <= 0.10),
        (f"automation at 10% error, {DRAWS} draws of all labels", "≤ 10% of draws over",
         f"{over['all']:.1%}", over["all"] <= 0.10),
        ("accuracy, as given → after the bias from 100 labels", "no loss on average",
         f"{given:.1%} → {biased:.1%}", biased >= given),
    ]


def write_report(result: dict, meta: dict, out: Path) -> None:
    names = list(result)
    default = "option formula (default)"
    lines = ["# Held-out test: results", "",
             f"Generated by `bench.holdout report` from run {meta['run']}; the model "
             f"reported serving {', '.join(meta['served'])}. Test halves, each dataset "
             "weighted equally. The plan and the criteria: "
             "[../part-3/holdout-plan.md](../part-3/holdout-plan.md).", "",
             "## Against the pre-registered criteria", "",
             "| measure | pass | result | |", "|---|---|---|---|"]
    for measure, rule, value, ok in verdict(result):
        lines.append(f"| {measure} | {rule} | {value} | {'pass' if ok else '**fail**'} |")

    lines += ["", "## Without labels", "",
              "Every row divides the same raw log probabilities by a temperature, so "
              "accuracy doesn't change. The task T is fitted on the dataset's own "
              "calibration half: the ceiling a temperature can reach, not a zero-label "
              "method.", "",
              "| method | T (median) | confidence | accuracy | ECE | excess ECE | NLL | share of gain |",
              "|---|---|---|---|---|---|---|---|"]
    methods = list(result[names[0]]["zero_label"])
    mean = lambda m, k: float(np.mean([result[n]["zero_label"][m][k] for n in names]))  # noqa: E731
    raw, best = mean("raw", "excess_ece"), mean("task T (calibration half)", "excess_ece")
    for m in methods:
        share = (raw - mean(m, "excess_ece")) / (raw - best) if raw != best else float("nan")
        lines.append(
            f"| {m} | {np.median([result[n]['zero_label'][m]['T'] for n in names]):.2f} | "
            f"{mean(m, 'confidence'):.1%} | {mean(m, 'accuracy'):.1%} | {mean(m, 'ece'):.3f} | "
            f"{mean(m, 'excess_ece'):.3f} | {mean(m, 'nll'):.3f} | {share:.0%} |")

    lines += ["", "Per dataset, excess ECE (T in brackets):", "",
              "| dataset | options | family T from | accuracy | option mass | "
              + " | ".join(methods) + " |",
              "|---|---|---|---|---|" + "---|" * len(methods)]
    for n in names:
        r = result[n]
        cells = [f"{r['zero_label'][m]['excess_ece']:.3f} ({r['zero_label'][m]['T']:.2f})"
                 for m in methods]
        mass = f"{r['option_mass']:.1%}" if r["option_mass"] is not None else "–"
        lines.append(f"| {n} | {r['options']} | {r['family']} | "
                     f"{r['zero_label']['raw']['accuracy']:.1%} | {mass} | " + " | ".join(cells) + " |")

    fits = list(result[names[0]]["labelled"])
    lines += ["", "## With labels: `calibrate()`", "",
              f"The library's `calibrate(coverage=0.9, max_error=0.10)` on answers at the "
              f"default temperature, {DRAWS} draws each of {LABELS} random calibration "
              "labels or of all of them (in a new order, which moves the folds), scored "
              "on the test half. Violations count draws whose automated answers erred "
              "more than 10% of the time.", "",
              "| fit | accuracy | excess ECE | coverage | set size | single option | automated | "
              "draws over 10% |", "|---|---|---|---|---|---|---|---|"]
    for f in fits:
        m = lambda k: float(np.mean([result[n]["labelled"][f][k] for n in names]))  # noqa: E731
        v = sum(result[n]["labelled"][f]["violations"] for n in names)
        d = sum(result[n]["labelled"][f]["draws"] for n in names)
        lines.append(f"| {f} | {m('accuracy'):.1%} | {m('excess_ece'):.3f} | {m('coverage'):.3f} | "
                     f"{m('size'):.2f} | {m('singletons'):.0%} | {m('automated'):.0%} | "
                     f"{v}/{d} ({v / d:.1%}) |")
    lines += ["", "Per dataset, T + bias from 100 labels:", "",
              "| dataset | accuracy as given | accuracy | excess ECE | coverage | set size | "
              "automated | draws over |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        r, f = result[n], result[n]["labelled"][f"T + bias, {LABELS} labels"]
        lines.append(f"| {n} | {r['zero_label']['raw']['accuracy']:.1%} | {f['accuracy']:.1%} | "
                     f"{f['excess_ece']:.3f} | {f['coverage']:.3f} | {f['size']:.2f} | "
                     f"{f['automated']:.0%} | {f['violations']}/{f['draws']} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps({"meta": meta, "datasets": result}, indent=1,
                                                 default=float) + "\n")


def report(run_dir: Path, out: Path) -> None:
    data = c.load_run(run_dir)
    missing = set(TASKS) - set(data)
    if missing:
        raise SystemExit(f"no run for {sorted(missing)}")
    data = {n: data[n] for n in TASKS}
    unrecorded = [n for n, d in data.items() if not d.meta.get("served_model")]
    if unrecorded:
        raise SystemExit(f"no served model recorded for {unrecorded}")
    served = sorted({s for d in data.values() for s in d.meta["served_model"]})
    if any(MODEL not in s for s in served):
        raise SystemExit(f"served {served}, not {MODEL}")
    starts = sorted(d.meta.get("started", "") for d in data.values())
    meta = {"run": f"{starts[0]}–{starts[-1]}", "served": served, "model": MODEL,
            "scored_with_library": library_version(),
            "global_t": GLOBAL_T, "families": FAMILY, "draws": DRAWS, "labels": LABELS}
    write_report(evaluate(data), meta, out)
    print((out / "report.md").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True, choices=sorted(TASKS))
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--concurrency", type=int, default=4)
    p = sub.add_parser("report")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.task, args.out, args.concurrency)
    else:
        report(args.run, args.out)


if __name__ == "__main__":
    main()
