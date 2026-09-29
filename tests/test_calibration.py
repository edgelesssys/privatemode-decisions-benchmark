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


def write_run(folder, rows, name="run-r0.jsonl"):
    import json as _json
    folder.mkdir(parents=True, exist_ok=True)
    lines = [{"kind": "meta", "started": "t0"}] + [dict(kind="row", arm="privatemode", **r) for r in rows]
    (folder / name).write_text("".join(_json.dumps(x) + "\n" for x in lines))


def test_load_run_reads_one_run_and_counts_what_it_drops(tmp_path):
    from bench.calibration import load_run

    ok = lambda i, gold="a", **extra: dict(index=i, gold=gold, probabilities={"a": 0.7, "b": 0.3},
                                           option_mass=0.99, **extra)
    write_run(tmp_path / "set", [
        ok(0, served_model="glm-5.3-flash", fingerprint="f1"), ok(1, "b"),
        dict(index=2, gold="a", error="timeout"),            # never answered
        dict(index=3, gold="a", error="timeout"), ok(3),       # answered on the retry
        ok(1, "b"),                                            # a repeated row
        ok(4, "zzz"),                                          # gold not an option
    ])
    data = load_run(tmp_path)["set"]
    assert list(data.index) == [0, 1, 3]
    assert list(data.y) == [0, 1, 0]
    assert data.mass is not None and data.P.shape == (3, 2)
    assert data.meta["dropped"] == {"failed": 1, "repeated": 1, "gold not an option or no probability": 1}
    assert data.meta["served_model"] == ["glm-5.3-flash"]


def test_load_run_refuses_several_runs_of_one_dataset(tmp_path):
    from bench.calibration import load_run

    row = dict(index=0, gold="a", probabilities={"a": 0.7, "b": 0.3})
    write_run(tmp_path / "set", [row], "one-r0.jsonl")
    write_run(tmp_path / "set", [row], "two-r1.jsonl")
    with pytest.raises(ValueError, match="2 run files"):
        load_run(tmp_path)


def test_common_aligns_two_systems_by_example_and_option():
    from bench.calibrate_extensions import common
    from bench.calibration import Dataset

    a = Dataset("set", ["x", "y"], np.array([3, 1, 2]),
                np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]), np.array([0, 1, 0]))
    # b answered 1, 2 and 4, lists the options the other way round, in another order
    b = Dataset("set", ["y", "x"], np.array([4, 2, 1]),
                np.array([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]), np.array([0, 1, 0]))
    ga, gb = common(a, b)
    assert list(ga.index) == list(gb.index) == [1, 2]
    assert gb.options == ["x", "y"]
    assert gb.P.tolist() == [[0.1, 0.9], [0.7, 0.3]]    # b's rows, reordered to a's options
    assert list(gb.y) == list(ga.y) == [1, 0]            # gold from a


def test_automation_threshold_never_splits_tied_confidences():
    from bench.calibration import automation_threshold

    # 40 answers at 0.99 (one wrong), then 60 at 0.90: a level cutting into a
    # run of equal confidences extends to the whole run.
    conf = np.array([0.99] * 40 + [0.90] * 60)
    right = np.array([False] + [True] * 39 + [True] * 60)
    t = automation_threshold(conf, right, max_error=0.10, delta=0.1)
    assert t in (0.99, 0.90)
    assert (conf >= t).sum() in (40, 100)
