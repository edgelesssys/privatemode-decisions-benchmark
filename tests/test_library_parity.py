"""The benchmark's numpy calibration code against the library's, on real data.

The reports use fast numpy versions of what the library ships; these tests
make sure they compute the same thing, on GLM-5.3-Flash probabilities from a
real run (tests/fixtures). Skipped where the installed library predates
calibrate().
"""

import json
from pathlib import Path

import numpy as np
import pytest

from bench import calibration as c

library = pytest.importorskip("decisions.calibration")
if not hasattr(library, "evaluate"):
    pytest.skip("installed privatemode-decisions has no calibration API yet", allow_module_level=True)

from decisions.types import ChoiceAnswer

#: The temperature-only path, for the tests that predate the bias.
ALONE = {"bias": False} if "bias" in library.calibrate.__code__.co_varnames else {}

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "glm-flash-probabilities.json").read_text())


def dataset(name):
    d = FIXTURE[name]
    P = np.array(d["probabilities"])
    y = np.array(d["gold"])
    answers = [ChoiceAnswer(choice=d["options"][int(np.argmax(row))],
                            probabilities=dict(zip(d["options"], row)), confidence=0.0)
               for row in P]
    return P, y, answers, [d["options"][i] for i in y]


@pytest.fixture(params=sorted(FIXTURE))
def data(request):
    return dataset(request.param)


def test_temperature_fit(data):
    P, y, answers, labels = data
    assert c.fit_temperature([(P, y)]) == pytest.approx(
        library.fit_temperature(answers, labels, shrinkage=0), rel=1e-3)
    # With the pull towards no change, as calibrate() fits it.
    assert c.fit_temperature([(P, y)], prior=1.0, shrinkage=library.SHRINKAGE) == pytest.approx(
        library.fit_temperature(answers, labels), rel=1e-3)


def test_conformal_cutoffs(data):
    P, y, answers, labels = data
    fitted = library.calibrate(answers, labels, coverage=0.9, **ALONE)
    Pt = c.scale(P, fitted.temperature)
    assert c.threshold(c.lac_scores(Pt, y), 0.9) == pytest.approx(fitted.cutoffs["*"], abs=1e-9)
    per_class = library.calibrate(answers, labels, coverage=0.9, per_class=True, **ALONE)
    for k, name in enumerate(answers[0].probabilities):
        rows = y == k
        expected = c.threshold(c.lac_scores(Pt[rows], y[rows]), 0.9) if rows.any() else 1.0
        expected = 1.0 if expected == float("inf") else expected
        assert per_class.cutoffs[name] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("max_error", [0.05, 0.10, 0.20])
def test_automation_threshold(data, max_error):
    P, y, answers, labels = data
    fitted = library.calibrate(answers, labels, max_error=max_error, **ALONE)
    Pt = c.scale(P, fitted.temperature)
    assert c.automation_threshold(Pt.max(1), Pt.argmax(1) == y, max_error, 0.1) == pytest.approx(
        fitted.threshold, abs=1e-12)


def test_ece(data):
    P, y, answers, labels = data
    assert c.ece(P, y) == pytest.approx(library.evaluate(answers, labels)["ece"], abs=1e-12)


def test_temperature_bias_fit(data):
    P, y, answers, labels = data
    if not hasattr(library, "fit_temperature_bias"):
        pytest.skip("installed privatemode-decisions has no bias yet")
    t, b = c.fit_temperature_bias(P, y, prior=1.0, shrinkage=library.SHRINKAGE,
                                  strength=library.BIAS_STRENGTH)
    t_lib, b_lib = library.fit_temperature_bias(answers, labels)
    assert t == pytest.approx(t_lib, rel=1e-4)
    assert list(b_lib.values()) == pytest.approx(list(b), abs=1e-4)


@pytest.mark.parametrize("max_error", [0.05, 0.10])
def test_bias_cutoffs_and_threshold_out_of_fold(data, max_error):
    from bench.calibrate_part3 import FOLDS, out_of_fold

    P, y, answers, labels = data
    if not hasattr(library, "fit_temperature_bias"):
        pytest.skip("installed privatemode-decisions has no bias yet")
    assert FOLDS == library.FOLDS
    fitted = library.calibrate(answers, labels, coverage=0.9, max_error=max_error)
    Pf = out_of_fold(P, y, True)
    assert c.threshold(c.lac_scores(Pf, y), 0.9) == pytest.approx(fitted.cutoffs["*"], abs=1e-5)
    expected = c.automation_threshold(Pf.max(1), Pf.argmax(1) == y, max_error, 0.1)
    assert expected == pytest.approx(fitted.threshold, abs=1e-5)
