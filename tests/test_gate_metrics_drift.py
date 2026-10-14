from __future__ import annotations

import numpy as np
import pandas as pd

from rtml.dataset import labels_as_of
from rtml.gate import Scored, decide
from rtml.metrics import best_threshold, case_cost, recall_at_precision
from rtml.monitor.drift import flag_rate_z, psi


def _world(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(np.float64)
    amount = rng.lognormal(3.5, 0.6, n)
    return y, amount, [f"t{i}" for i in range(n)]


def _scores(y: np.ndarray, skill: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(0.5 * rng.random(len(y)) + skill * y, 0, 1)


def test_cost_counts_reviews_and_missed_fraud() -> None:
    y = np.array([1.0, 1.0, 0.0, 0.0])
    flagged = np.array([True, False, True, False])
    cost = case_cost(y, flagged, np.array([100.0, 50.0, 20.0, 20.0]), review_cost=5.0)
    assert cost.tolist() == [5.0, 50.0, 5.0, 0.0]


def test_threshold_and_recall_at_precision_on_a_separable_case() -> None:
    y = np.array([0.0] * 90 + [1.0] * 10)
    p = np.concatenate([np.linspace(0, 0.4, 90), np.linspace(0.6, 1, 10)])
    t = best_threshold(y, p, np.full(100, 100.0))
    assert 0.4 < t <= 0.6
    assert recall_at_precision(y, p, 0.99) == 1.0


def test_gate_promotes_a_clearly_better_challenger() -> None:
    y, amount, ids = _world()
    fast = np.full(200, 1.0)
    champ = Scored("a", ids, _scores(y, 0.2, 1), 0.5, fast)
    chall = Scored("b", ids, _scores(y, 0.6, 2), 0.5, fast)
    d = decide(champ, chall, y, amount, n_boot=500)
    assert d.promote, d.reason
    assert d.cost_delta_per_txn > 0 and d.cost_p < 0.05


def test_gate_refuses_an_equal_challenger() -> None:
    y, amount, ids = _world()
    s = _scores(y, 0.3, 1)
    fast = np.full(200, 1.0)
    d = decide(Scored("a", ids, s, 0.5, fast), Scored("b", ids, s.copy(), 0.5, fast), y, amount, n_boot=500)
    assert not d.promote
    assert "not significantly lower" in d.reason


def test_gate_refuses_a_better_but_slow_challenger() -> None:
    y, amount, ids = _world()
    champ = Scored("a", ids, _scores(y, 0.2, 1), 0.5, np.full(200, 1.0))
    chall = Scored("b", ids, _scores(y, 0.6, 2), 0.5, np.full(200, 25.0))
    d = decide(champ, chall, y, amount, latency_budget_ms=10.0, n_boot=500)
    assert not d.promote
    assert "budget" in d.reason


def test_psi_is_small_for_same_distribution_and_large_for_a_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.02
    assert psi(ref, rng.normal(0.8, 1, 5000)) > 0.2


def test_flag_rate_z() -> None:
    assert flag_rate_z(0.01, 10, 1000) == 0.0
    assert flag_rate_z(0.01, 40, 1000) > 8


def test_labels_as_of_hides_chargebacks_that_have_not_arrived() -> None:
    t0 = pd.Timestamp("2026-07-10", tz="UTC")
    rows = pd.DataFrame({"chargeback_at": [t0 - pd.Timedelta(days=1), t0 + pd.Timedelta(days=1), pd.NaT]})
    assert labels_as_of(rows, t0).tolist() == [1, 0, 0]
