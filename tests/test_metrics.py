"""The metrics are the claim, so they are the thing worth testing."""

import math

from bench.metrics import (accuracy, auroc, bootstrap_diff, brier,
                           expected_calibration_error, macro_f1, mcnemar,
                           normalised, quantile, selective, wilcoxon)


def test_accuracy_and_macro_f1():
    assert accuracy([True, True, False, False]) == 0.5
    # Two classes, one perfect and one missed entirely.
    assert macro_f1(["a", "a", "b"], ["a", "a", "a"]) == (1.0 * 2 / 3 * 2 / (2 / 3 + 1)
                                                          + 0.0) / 2


def test_macro_f1_ignores_classes_absent_from_gold():
    # "c" was predicted but never correct; it must not add a free zero.
    assert macro_f1(["a", "a"], ["a", "c"]) == 2 * 1.0 * 0.5 / 1.5


def test_ece_is_zero_when_confidence_matches_accuracy():
    confidences = [0.95] * 20
    correct = [True] * 19 + [False]
    assert expected_calibration_error(confidences, correct) == 0.0


def test_ece_catches_overconfidence():
    assert expected_calibration_error([1.0] * 4, [True, False, False, False]) == 0.75


def test_brier_rewards_the_right_distribution():
    confident = brier([{"a": 1.0, "b": 0.0}], ["a"])
    hedged = brier([{"a": 0.5, "b": 0.5}], ["a"])
    assert confident == 0.0 and math.isclose(hedged, 0.5)


def test_brier_charges_for_a_missing_gold_label():
    # A shortlist that dropped the right answer put zero mass on it.
    assert brier([{"a": 0.6, "b": 0.4}], ["c"]) == 0.6 ** 2 + 0.4 ** 2 + 1.0


def test_auroc_uses_midranks_for_ties():
    # All scores identical: no discrimination, not a coin flip in disguise.
    assert auroc([0.9, 0.9, 0.9, 0.9], [True, True, False, False]) == 0.5
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0


def test_auroc_is_nan_without_both_outcomes():
    assert math.isnan(auroc([0.5, 0.6], [True, True]))


def test_selective_accuracy_improves_with_a_useful_confidence():
    confidences = [0.9, 0.8, 0.3, 0.2]
    correct = [True, True, False, False]
    result = selective(confidences, correct)
    assert result["acc@1"] == 0.5 and result["acc@0.5"] == 1.0


def test_mcnemar_is_paired_and_two_sided():
    a = [True] * 10 + [False] * 2
    b = [True] * 2 + [False] * 10
    result = mcnemar(a, b)
    assert result["only_a"] == 8 and result["only_b"] == 0
    assert result["p"] < 0.01
    assert mcnemar([True, False], [True, False])["p"] == 1.0


def test_bootstrap_brackets_the_point_estimate():
    a = [True] * 80 + [False] * 20
    b = [True] * 60 + [False] * 40
    point, low, high = bootstrap_diff(a, b, samples=2000, seed=0)
    assert math.isclose(point, 0.2)
    assert low < point < high and low > 0  # a is better, significantly


def test_quantile_interpolates():
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert quantile([1.0], 0.95) == 1.0


def test_wilcoxon_is_two_sided_and_exact_when_small():
    # Eight datasets, one arm ahead on every one of them.
    result = wilcoxon([0.05, 0.02, 0.11, 0.03, 0.07, 0.01, 0.09, 0.04])
    assert result["n"] == 8 and result["positive"] == 8
    assert result["p"] < 0.01
    # Symmetric input: no evidence either way.
    assert wilcoxon([0.05, -0.05, 0.02, -0.02])["p"] == 1.0


def test_wilcoxon_drops_zero_differences():
    assert wilcoxon([0.0, 0.0, 0.0]) == {"n": 0, "w_plus": 0.0, "p": 1.0,
                                         "positive": 0, "negative": 0}
    assert wilcoxon([0.0, 0.1, 0.2])["n"] == 2


def test_wilcoxon_switches_to_the_approximation_past_fifteen():
    # Twenty consistent wins must still register, via the normal branch.
    result = wilcoxon([0.01 * (i + 1) for i in range(20)])
    assert result["n"] == 20 and result["p"] < 0.001


def test_wilcoxon_is_robust_to_one_collapsed_dataset():
    # Fourteen small wins and one catastrophic loss: the mean flips sign,
    # the signed-rank test does not. That is the reason it is the headline
    # test -- a single set where an arm collapses must not decide a suite.
    diffs = [0.02] * 14 + [-0.40]
    assert sum(diffs) / len(diffs) < 0
    result = wilcoxon(diffs)
    assert result["positive"] == 14 and result["p"] < 0.01


def test_wilcoxon_needs_more_than_a_bare_majority_of_wins():
    # Nine tied wins against one larger loss is p = 0.09, not significance.
    # Worth pinning: it is exactly the shape a suite result can take, and
    # reading it as a win would be the easiest mistake to publish.
    assert 0.05 < wilcoxon([0.02] * 9 + [-0.40])["p"] < 0.10


def test_normalised_puts_the_majority_baseline_at_zero():
    assert normalised(0.92, 0.92) == 0.0
    assert normalised(1.0, 0.92) == 1.0
    assert normalised(0.90, 0.92) < 0  # lost to always-guess-the-majority
