"""Calibration of the Privatemode arm's probabilities, from raw runs.

Everything here works on the stored per-example distributions, so it needs
no model: a run is loaded once and every metric, fit and prediction set is
computed offline. Only :func:`neutral_priors` in ``calibrate_report`` asks
the model anything.

Conventions:

* ``P`` is an ``n x k`` array of probabilities over one dataset's options
  (rows sum to 1), ``y`` the index of the gold option per row.
* Temperature scales log probabilities: ``p_T ∝ p ** (1 / T)``. ``T > 1``
  softens, ``T < 1`` sharpens; accuracy never changes.
* Calibration and test halves are a fixed random split per dataset, seeded
  from ``SEED`` and the dataset name, so every number is reproducible.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEED = 0
BINS = 15
FLOOR = 1e-300   # probabilities are stored as floats; log(0) is not an option


@dataclass
class Dataset:
    name: str
    options: list[str]
    index: np.ndarray            # example ids, for joining runs
    P: np.ndarray                # n x k probabilities
    y: np.ndarray                # gold option index
    mass: np.ndarray | None = None   # probability on the options before the mask
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.options)

    def subset(self, rows: np.ndarray) -> Dataset:
        return Dataset(self.name, self.options, self.index[rows], self.P[rows], self.y[rows],
                       None if self.mass is None else self.mass[rows], self.meta)


# -- loading --------------------------------------------------------------

def load_run(directory: Path, arm: str = "privatemode") -> dict[str, Dataset]:
    """Every dataset in a suite output directory, rows of one arm only."""
    out = {}
    for path in sorted(Path(directory).glob("*/*.jsonl")):
        meta, rows = None, []
        for line in path.open():
            record = json.loads(line)
            if record.get("kind") == "meta":
                meta = record
            elif (record.get("kind") == "row" and record.get("arm") == arm
                  and "error" not in record and record.get("probabilities")):
                rows.append(record)
        if not rows:
            continue
        options = list(rows[0]["probabilities"])
        position = {name: i for i, name in enumerate(options)}
        rows = [r for r in rows if r["gold"] in position]
        P = np.array([[r["probabilities"][o] for o in options] for r in rows], dtype=float)
        mass = (np.array([r["option_mass"] for r in rows], dtype=float)
                if all("option_mass" in r for r in rows) else None)
        out[path.parent.name] = Dataset(
            path.parent.name, options, np.array([r["index"] for r in rows]),
            P / P.sum(axis=1, keepdims=True), np.array([position[r["gold"]] for r in rows]),
            mass, meta or {})
    return out


def split(data: Dataset, seed: int = SEED) -> tuple[Dataset, Dataset]:
    """Fixed random 50/50 calibration/test split, by example id."""
    order = np.argsort(data.index)
    rng = np.random.default_rng(seed + zlib.crc32(data.name.encode()))
    rows = order[rng.permutation(len(order))]
    half = len(rows) // 2
    return data.subset(np.sort(rows[:half])), data.subset(np.sort(rows[half:]))


# -- transforms -----------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def scale(P: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.log(np.maximum(P, FLOOR)) / temperature)


def contextual(P: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Divide out what the model prefers with no content, then renormalize."""
    Q = P / np.maximum(prior, FLOOR)
    return Q / Q.sum(axis=1, keepdims=True)


# -- metrics --------------------------------------------------------------

def nll(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.maximum(P[np.arange(len(y)), y], FLOOR))))


def brier(P: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def accuracy(P: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(P.argmax(axis=1) == y))


def reliability(P: np.ndarray, y: np.ndarray, bins: int = BINS):
    """Equal-mass bins of top-label confidence: (confidence, accuracy, count)."""
    conf = P.max(axis=1)
    right = (P.argmax(axis=1) == y).astype(float)
    order = np.argsort(conf, kind="stable")
    chunks = [c for c in np.array_split(order, bins) if len(c)]
    return [(float(conf[c].mean()), float(right[c].mean()), len(c)) for c in chunks]


def ece(P: np.ndarray, y: np.ndarray, bins: int = BINS) -> float:
    n = len(y)
    return float(sum(abs(c - a) * m / n for c, a, m in reliability(P, y, bins)))


def overconfidence(P: np.ndarray, y: np.ndarray) -> float:
    """Mean confidence minus accuracy: positive means overconfident."""
    return float(P.max(axis=1).mean() - accuracy(P, y))


def ece_floor(P: np.ndarray, draws: int = 200, seed: int = SEED) -> float:
    """Expected ECE if ``P`` were perfectly calibrated: labels drawn from P.

    With a few hundred examples, sampling noise alone gives a few points of
    ECE, so a measured ECE near this floor is as good as it can get.
    """
    rng = np.random.default_rng(seed)
    cumulative = np.cumsum(P, axis=1)
    values = []
    for _ in range(draws):
        u = rng.random((len(P), 1))
        y = np.minimum((u > cumulative).sum(axis=1), P.shape[1] - 1)
        values.append(ece(P, y))
    return float(np.mean(values))


def ece_interval(P: np.ndarray, y: np.ndarray, draws: int = 500, seed: int = SEED):
    rng = np.random.default_rng(seed)
    values = [ece(P[rows], y[rows]) for rows in
              (rng.integers(0, len(y), len(y)) for _ in range(draws))]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Probability a random positive scores higher than a random negative."""
    pos, neg = score[positive], score[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable")) + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_interval(score: np.ndarray, positive: np.ndarray, draws: int = 500,
                   seed: int = SEED) -> tuple[float, float]:
    """95% bootstrap interval; wide when there are few positives."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        rows = rng.integers(0, len(score), len(score))
        v = auroc(score[rows], positive[rows])
        if not math.isnan(v):
            values.append(v)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# -- temperature ----------------------------------------------------------

def fit_temperature(parts: list[tuple[np.ndarray, np.ndarray]],
                    low: float = 0.05, high: float = 50.0,
                    prior: float = 1.0, shrinkage: float = 0.0) -> float:
    """T minimizing the mean NLL, each (P, y) part weighted equally.

    NLL is convex in 1/T, so a golden-section search on log T finds the
    minimum; 60 steps narrow [0.05, 50] far below any meaningful digit.
    """
    logs = [(np.log(np.maximum(P, FLOOR)), y) for P, y in parts]

    n = sum(len(y) for _, y in logs) / len(logs)

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        fit = sum(nll(softmax(L / t), y) for L, y in logs) / len(logs)
        # Optional pull towards ``prior``, worth ``shrinkage`` examples.
        return fit + shrinkage / n * (log_t - math.log(prior)) ** 2

    a, b = math.log(low), math.log(high)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = loss(d)
    return math.exp((a + b) / 2)


def fit_formula(temps: dict[str, float], options: dict[str, int]) -> tuple[float, float]:
    """``log T = a + b * log(options)`` by least squares over datasets."""
    x = np.log([options[d] for d in temps])
    z = np.log(list(temps.values()))
    b, a = np.polyfit(x, z, 1)
    return float(a), float(b)


def formula_temperature(a: float, b: float, options: int) -> float:
    return math.exp(a + b * math.log(options))


# -- conformal prediction sets --------------------------------------------

def lac_scores(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1 - P[np.arange(len(y)), y]


def aps_scores(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mass of every option at least as likely as the gold one."""
    p_true = P[np.arange(len(y)), y][:, None]
    return np.sum(np.where(P >= p_true, P, 0.0), axis=1)


def threshold(scores: np.ndarray, coverage: float) -> float:
    """Split-conformal quantile: the ceil((n+1)·coverage)-th smallest score."""
    n = len(scores)
    rank = math.ceil((n + 1) * coverage)
    return float("inf") if rank > n else float(np.sort(scores)[rank - 1])


def weighted_threshold(groups: list[np.ndarray], coverage: float) -> float:
    """A cutoff pooled over datasets, each weighted equally (a heuristic)."""
    scores = np.concatenate(groups)
    weights = np.concatenate([np.full(len(g), 1 / len(g)) for g in groups])
    order = np.argsort(scores)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(scores[order][min(np.searchsorted(cumulative, coverage), len(scores) - 1)])


def lac_sets(P: np.ndarray, cutoff: float) -> np.ndarray:
    return (1 - P) <= cutoff


def aps_sets(P: np.ndarray, cutoff: float) -> np.ndarray:
    """Each option's APS score if it were the answer, compared with the cutoff."""
    order = np.argsort(-P, axis=1, kind="stable")
    sorted_p = np.take_along_axis(P, order, axis=1)
    inclusive = np.cumsum(sorted_p, axis=1)
    member = np.zeros_like(P, dtype=bool)
    np.put_along_axis(member, order, inclusive <= cutoff, axis=1)
    member[np.arange(len(P)), order[:, 0]] |= True   # never return an empty set
    return member


def set_stats(sets: np.ndarray, y: np.ndarray) -> dict[str, float]:
    sizes = sets.sum(axis=1)
    return {"coverage": float(np.mean(sets[np.arange(len(y)), y])),
            "size": float(sizes.mean()),
            "single": float(np.mean(sizes == 1)),
            "empty": float(np.mean(sizes == 0))}


# -- isotonic regression on top-label confidence ---------------------------

def fit_isotonic(confidence: np.ndarray, correct: np.ndarray):
    """Pool-adjacent-violators: a non-decreasing map from confidence to accuracy.

    Returns ``(knots, values)`` for :func:`apply_isotonic`. It calibrates the
    top answer's confidence only, not the whole distribution, which is why it
    can repair rounded outputs that temperature cannot.
    """
    order = np.argsort(confidence, kind="stable")
    x, y = confidence[order], correct[order].astype(float)
    values, weights, starts = [], [], []
    for i, v in enumerate(y):
        values.append(v)
        weights.append(1.0)
        starts.append(i)
        while len(values) > 1 and values[-2] > values[-1]:
            w = weights[-2] + weights[-1]
            values[-2] = (values[-2] * weights[-2] + values[-1] * weights[-1]) / w
            weights[-2] = w
            del values[-1], weights[-1], starts[-1]
    ends = starts[1:] + [len(x)]
    knots = np.array([x[s:e].mean() for s, e in zip(starts, ends)])
    return knots, np.array(values)


def apply_isotonic(model, confidence: np.ndarray) -> np.ndarray:
    knots, values = model
    return np.interp(confidence, knots, values)


def ece_top(confidence: np.ndarray, correct: np.ndarray, bins: int = BINS) -> float:
    """ECE of a top-label confidence that isn't part of a distribution."""
    order = np.argsort(confidence, kind="stable")
    n = len(confidence)
    return float(sum(abs(confidence[c].mean() - correct[c].mean()) * len(c) / n
                     for c in np.array_split(order, bins) if len(c)))


# -- a guaranteed error rate on automated answers (Learn then Test) --------

def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space."""
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0
    log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                 + (i * math.log(p) if i else 0.0) + ((n - i) * math.log1p(-p) if n - i else 0.0)
                 for i in range(k + 1)]
    top = max(log_terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in log_terms))


def automation_threshold(confidence: np.ndarray, correct: np.ndarray, max_error: float,
                         delta: float = 0.1, step: float = 0.05) -> float:
    """Lowest confidence threshold whose error among automated answers is
    at most ``max_error``, with probability ``1 - delta``.

    Learn then Test with fixed-sequence testing over a grid: the candidates
    automate 5%, 10%, ... of the calibration answers, most confident first.
    Each null hypothesis "the error above this threshold exceeds max_error"
    is tested with an exact binomial test at level ``delta``, in that order,
    and testing stops at the first failure, so no correction for multiple
    tests is needed. Levels too small to reject even with zero errors are
    skipped; which ones depends on ``max_error`` and ``delta`` only, not on
    the labels. Returns ``inf`` (automate nothing) if no level is certified.
    """
    order = np.argsort(-confidence, kind="stable")
    conf, wrong = confidence[order], (~correct[order].astype(bool)).astype(int)
    errors = np.cumsum(wrong)
    smallest = math.ceil(math.log(delta) / math.log1p(-max_error))
    chosen = float("inf")
    for level in np.arange(step, 1 + 1e-9, step):
        n = math.ceil(len(conf) * level)
        while n < len(conf) and conf[n] == conf[n - 1]:
            n += 1      # a threshold can't split a run of equal confidences
        if n < smallest:
            continue
        if binomial_cdf(int(errors[n - 1]), n, max_error) > delta:
            break
        chosen = float(conf[n - 1])
    return chosen
