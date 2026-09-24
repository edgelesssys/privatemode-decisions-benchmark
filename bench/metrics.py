"""Metrics, in plain Python so a run needs no scientific stack.

Two families, and the distinction matters for reading the report:

*Accuracy* metrics score the argmax. *Calibration* metrics score the
probabilities, and they are the reason to run this benchmark at all -- both
products return a distribution, and a distribution you can threshold is
worth more than a label you cannot. Two are reported:

``ece`` / ``brier``   computed from ``probabilities``, identically for both
                      arms, so they compare like for like.
``auroc``             computed from the vendor's own ``confidence`` field,
                      because that is the number the documentation tells you
                      to gate on -- and the two vendors derive it
                      differently, so it is a product comparison rather than
                      a mathematical one.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict


def accuracy(correct: list[bool]) -> float:
    return sum(correct) / len(correct) if correct else 0.0


def macro_f1(gold: list[str], predicted: list[str]) -> float:
    """Unweighted mean F1 over the classes that actually occur in ``gold``.

    Averaged over gold classes rather than over the union with the
    predictions: a 77-way task where an arm invents a class it was never
    asked about should lose precision on the classes it confused, not gain a
    free 0 that drags every arm down by the same arbitrary amount.
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for want, got in zip(gold, predicted):
        if want == got:
            tp[want] += 1
        else:
            fn[want] += 1
            fp[got] += 1
    scores = []
    for label in sorted(set(gold)):
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        scores.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def expected_calibration_error(confidences: list[float], correct: list[bool],
                               bins: int = 10) -> float:
    """Mean |accuracy - confidence| over equal-width confidence bins."""
    if not confidences:
        return 0.0
    buckets: list[list[int]] = [[] for _ in range(bins)]
    for i, value in enumerate(confidences):
        index = min(bins - 1, max(0, int(value * bins)))
        buckets[index].append(i)
    total = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_conf = sum(confidences[i] for i in bucket) / len(bucket)
        mean_acc = sum(correct[i] for i in bucket) / len(bucket)
        total += len(bucket) / len(confidences) * abs(mean_acc - mean_conf)
    return total


def brier(distributions: list[dict[str, float]], gold: list[str]) -> float:
    """Multiclass Brier score: mean squared error of the whole distribution.

    A distribution that does not mention the gold label at all scores as if
    it had put zero there -- which is what a shortlist that dropped the right
    answer actually did, and leaving the term out would hide exactly that
    failure.
    """
    if not gold:
        return 0.0
    total = 0.0
    for probs, want in zip(distributions, gold):
        total += sum((p - (1.0 if name == want else 0.0)) ** 2
                     for name, p in probs.items())
        if want not in probs:
            total += 1.0
    return total / len(gold)


def auroc(scores: list[float], correct: list[bool]) -> float:
    """Area under the ROC of ``scores`` separating correct from wrong.

    Mid-ranks for ties, which is not a detail here: a model that answers 1.0
    on most examples produces enormous tie groups, and breaking them
    arbitrarily would invent discrimination it does not have.
    """
    positives = sum(correct)
    negatives = len(correct) - positives
    if not positives or not negatives:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    rank_sum = sum(ranks[i] for i, ok in enumerate(correct) if ok)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def selective(confidences: list[float], correct: list[bool],
              coverages=(1.0, 0.9, 0.75, 0.5)) -> dict[str, float]:
    """Accuracy on the most confident ``c`` fraction: the abstention story.

    "Route automatically above the threshold, send the rest to a human" is
    the product, so this is the number a buyer actually acts on.
    """
    order = sorted(range(len(correct)), key=lambda i: -confidences[i])
    out = {}
    for coverage in coverages:
        keep = max(1, int(round(coverage * len(order))))
        out[f"acc@{coverage:g}"] = sum(correct[i] for i in order[:keep]) / keep
    return out


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict[str, float]:
    """Exact two-sided McNemar on the paired outcomes.

    The arms see the same examples, so the unpaired comparison throws away
    exactly the information that makes a 2-point difference readable.
    """
    only_a = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    only_b = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p": 1.0}
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"only_a": only_a, "only_b": only_b, "p": min(1.0, 2 * tail)}


def bootstrap_diff(a_correct: list[bool], b_correct: list[bool],
                   samples: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap CI for accuracy(a) - accuracy(b): (point, lo, hi)."""
    n = len(a_correct)
    point = accuracy(a_correct) - accuracy(b_correct)
    if n == 0:
        return point, 0.0, 0.0
    rng = random.Random(seed)
    diffs = []
    for _ in range(samples):
        picks = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a_correct[i] for i in picks) / n
                     - sum(b_correct[i] for i in picks) / n)
    diffs.sort()
    return point, diffs[int(0.025 * samples)], diffs[int(0.975 * samples) - 1]


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def wilcoxon(differences: list[float]) -> dict[str, float]:
    """Two-sided signed-rank test over one difference per dataset.

    The headline test for a suite. With two dozen datasets, per-dataset
    significance stars are guaranteed somewhere by chance; what is actually
    being asked is whether one arm is ahead *across* sets, and that is a
    paired test whose unit is the dataset rather than the example. Signed
    rank rather than a t-test because accuracy differences across tasks are
    nothing like normal -- one set where an arm collapses would otherwise
    carry the whole result.

    Zero differences are dropped, which is the standard treatment and is
    conservative here. Exact by enumeration while that is cheap, normal
    approximation with a tie correction beyond it.
    """
    values = [d for d in differences if d != 0.0]
    n = len(values)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "p": 1.0, "positive": 0, "negative": 0}

    order = sorted(range(n), key=lambda i: abs(values[i]))
    ranks = [0.0] * n
    i = 0
    tie_groups: list[int] = []
    while i < n:
        j = i
        while j + 1 < n and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        tie_groups.append(j - i + 1)
        i = j + 1

    w_plus = sum(rank for rank, value in zip(ranks, values) if value > 0)
    positive = sum(1 for value in values if value > 0)

    if n <= 15:
        # Enumerate every sign assignment; the null says each is equally likely.
        total = 1 << n
        target = min(w_plus, sum(ranks) - w_plus)
        hits = 0
        for mask in range(total):
            below = sum(ranks[bit] for bit in range(n) if mask >> bit & 1)
            if below <= target + 1e-9:
                hits += 1
        p = min(1.0, 2 * hits / total)
    else:
        mean = n * (n + 1) / 4
        variance = n * (n + 1) * (2 * n + 1) / 24
        variance -= sum(t ** 3 - t for t in tie_groups) / 48
        if variance <= 0:
            return {"n": n, "w_plus": w_plus, "p": 1.0,
                    "positive": positive, "negative": n - positive}
        z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
        p = min(1.0, math.erfc(z / math.sqrt(2)))
    return {"n": n, "w_plus": w_plus, "p": p,
            "positive": positive, "negative": n - positive}


def normalised(accuracy: float, baseline: float) -> float:
    """Accuracy rescaled so the trivial classifier is 0 and perfect is 1.

    Averaging raw accuracy across sets whose chance levels are 50% and 1.3%
    is meaningless. The baseline here is the majority class, not
    ``1/options``: on a corpus that is 92% one label, always guessing it is
    the classifier a buyer would compare against, and it is far stronger
    than uniform chance. A negative result means the arm lost to it.
    """
    if baseline >= 1.0:
        return 0.0
    return (accuracy - baseline) / (1.0 - baseline)
