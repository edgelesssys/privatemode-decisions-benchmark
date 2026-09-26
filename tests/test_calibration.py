"""The calibration math, on synthetic data where the answer is known."""

import numpy as np
import pytest

from bench import calibration as c


def overconfident(n=20000, k=5, temperature=0.5, seed=1):
    rng = np.random.default_rng(seed)
    true = c.softmax(rng.normal(0, 1.5, (n, k)))
    y = np.array([rng.choice(k, p=p) for p in true])
    return true, c.scale(true, temperature), y


def test_temperature_recovers_the_sharpening():
    _, P, y = overconfident()
    assert c.fit_temperature([(P, y)]) == pytest.approx(2.0, rel=0.05)


def test_calibrated_probabilities_sit_near_the_ece_floor():
    true, P, y = overconfident()
    assert c.ece(true, y) < 0.02 < c.ece(P, y)
    assert c.ece(true, y) == pytest.approx(c.ece_floor(true, draws=20), abs=0.01)
    assert c.overconfidence(P, y) > 0.1


@pytest.mark.parametrize("scores, sets", [(c.lac_scores, c.lac_sets), (c.aps_scores, c.aps_sets)])
def test_conformal_sets_hit_their_coverage(scores, sets):
    true, _, y = overconfident()
    half = len(y) // 2
    q = c.threshold(scores(true[:half], y[:half]), 0.9)
    assert c.set_stats(sets(true[half:], q), y[half:])["coverage"] == pytest.approx(0.9, abs=0.015)


def test_contextual_calibration_divides_out_the_prior():
    P = np.array([[0.6, 0.4]])
    assert c.contextual(P, np.array([0.75, 0.25]))[0] == pytest.approx([1 / 3, 2 / 3])


def test_auroc():
    assert c.auroc(np.array([3.0, 4, 1, 2]), np.array([True, True, False, False])) == 1.0


def test_the_split_is_fixed_and_disjoint():
    d = c.Dataset("x", ["a", "b"], np.arange(100), np.full((100, 2), 0.5), np.zeros(100, int))
    cal, test = c.split(d)
    again, _ = c.split(d)
    assert set(cal.index).isdisjoint(test.index) and len(cal.index) + len(test.index) == 100
    assert (cal.index == again.index).all()


def test_learn_then_test_keeps_the_error_promise():
    rng = np.random.default_rng(3)
    violations, automated = [], []
    for _ in range(200):
        conf = rng.uniform(0.5, 1, 1500)
        right = rng.random(1500) < conf
        t = c.automation_threshold(conf[:500], right[:500], 0.10, 0.1)
        m = conf[500:] >= t
        automated.append(m.mean())
        violations.append(m.any() and (~right[500:][m]).mean() > 0.10)
    assert np.mean(violations) <= 0.1
    assert np.mean(automated) > 0.1


def test_learn_then_test_automates_nothing_it_cannot_certify():
    conf = np.linspace(0.5, 1, 20)
    assert c.automation_threshold(conf, np.ones(20, bool), 0.10, 0.1) == np.inf


def test_isotonic_regression_calibrates_a_monotone_distortion():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, 6000)
    y = rng.random(6000) < x ** 2
    model = c.fit_isotonic(x[:3000], y[:3000])
    assert c.ece_top(c.apply_isotonic(model, x[3000:]), y[3000:]) < 0.03 < c.ece_top(x[3000:], y[3000:])


def test_binomial_cdf():
    import math
    n, p = 40, 0.1
    exact = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(4))
    assert c.binomial_cdf(3, n, p) == pytest.approx(exact)
