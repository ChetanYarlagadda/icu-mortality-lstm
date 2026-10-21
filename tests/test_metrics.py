import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from icumort import metrics as M


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.15
    score = rng.normal(size=2000) + 1.2 * y
    return score, y.astype(int)


def test_auroc_and_auprc_match_sklearn(data):
    score, y = data
    rounded = np.round(score, 1)  # with ties
    for s in (score, rounded):
        assert M.auroc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
        assert M.auprc(s, y) == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_event2_follows_score_c_on_a_hand_example():
    # 20 stays, 10 groups of 2: risks 0.05, 0.10, ..., 1.00, deaths in the top 4.
    risk = np.arange(1, 21) / 20
    y = (np.arange(20) >= 16).astype(int)
    rid = np.arange(100, 120)
    pi = risk.reshape(10, 2).mean(axis=1)
    o = y.reshape(10, 2).sum(axis=1)
    h = np.sum((o - 2 * pi) ** 2 / (2 * (pi * (1 - pi) + 0.001)))
    score, h_got, d = M.event2(risk, y, rid)
    assert h_got == pytest.approx(h)
    assert d == pytest.approx(pi[-1] - pi[0])
    assert score == pytest.approx(h / d)
    # Ties in risk are broken by record id, so shuffling the input does not change the score.
    perm = np.random.default_rng(1).permutation(20)
    assert M.event2(risk[perm], y[perm], rid[perm])[0] == pytest.approx(score)


def test_event1_and_threshold():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    s, se, ppv = M.event1(p >= 0.7, y)
    assert (se, ppv, s) == (0.75, 1.0, 0.75)
    thr = M.event1_threshold(p, y)
    best = max(M.event1(p >= t, y)[0] for t in p)
    assert M.event1(p >= thr, y)[0] == best


def test_platt_and_calibration_slope_recover_known_miscalibration():
    rng = np.random.default_rng(3)
    z = rng.normal(size=20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-z))).astype(int)  # calibrated on z
    p_over = 1 / (1 + np.exp(-2 * z))  # risks twice too extreme
    a, b = M.platt_fit(p_over, y)
    assert b == pytest.approx(0.5, abs=0.05) and a == pytest.approx(0.0, abs=0.05)
    intercept, slope = M.calibration_slope_intercept(1 / (1 + np.exp(-z)), y)
    assert slope == pytest.approx(1.0, abs=0.05) and intercept == pytest.approx(0.0, abs=0.05)
    assert M.ece(M.platt_apply(p_over, (a, b)), y) < M.ece(p_over, y)


def test_holm_and_paired_p():
    assert M.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert M.paired_p(np.full(100, 0.0)) == 1.0
    assert M.paired_p(np.random.default_rng(0).normal(5, 1, 500)) < 1e-6
