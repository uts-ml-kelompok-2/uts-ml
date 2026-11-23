import numpy as np
from kode.common.metrics import pinball_loss, coverage, total_cost, cost_flag_all, cost_flag_none


def test_pinball():
    y = np.array([10., 20.]); q = np.array([12., 15.])
    # alpha 0.9: (10-12)→ (0.9-1)*(-2)=0.2 ; (20-15)→0.9*5=4.5 ; rata-rata 2.35
    assert abs(pinball_loss(y, q, 0.9) - 2.35) < 1e-9
    assert abs(pinball_loss(y, q, 0.5) - 1.75) < 1e-9


def test_coverage():
    assert coverage([1, 2, 3, 4], [2, 2, 2, 5]) == 0.75


def test_cost():
    y = np.array([1, 0, 1, 0]); nom = np.array([100_000, 50_000, 30_000, 10_000]); t = np.array([1, 1, 0, 0])
    assert total_cost(y, nom, t) == 30_000 + 2 * 25_000
    assert cost_flag_all(y, nom) == 4 * 25_000
    assert cost_flag_none(y, nom) == 130_000
