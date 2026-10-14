from __future__ import annotations

import pandas as pd

from rtml.generate import GenConfig, generate


def test_same_seed_same_data() -> None:
    a, _ = generate(GenConfig(seed=11, n_customers=300, days=10))
    b, _ = generate(GenConfig(seed=11, n_customers=300, days=10))
    pd.testing.assert_frame_equal(a, b)


def test_every_fraud_pattern_appears_and_is_rare() -> None:
    tx, _ = generate(GenConfig(seed=1, n_customers=2000, days=20))
    assert set(tx.fraud_type) == {"none", "card_testing", "account_takeover", "lone_anomaly"}
    assert 0.002 < tx.is_fraud.mean() < 0.03


def test_timestamps_are_unique_and_ordered() -> None:
    tx, _ = generate(GenConfig(seed=2, n_customers=500, days=10))
    assert tx.ts.is_monotonic_increasing
    assert not tx.ts.duplicated().any()


def test_chargebacks_arrive_late_and_some_never() -> None:
    tx, _ = generate(GenConfig(seed=4, n_customers=3000, days=30))
    fraud = tx[tx.is_fraud]
    reported = fraud.chargeback_at.notna()
    assert 0.75 < reported.mean() < 0.95
    delay = (fraud.chargeback_at - fraud.ts)[reported]
    assert (delay > pd.Timedelta(0)).all()
    assert pd.Timedelta(days=3) < delay.median() < pd.Timedelta(days=14)


def test_drift_switch_raises_prices_only_after_the_drift_day() -> None:
    cfg = GenConfig(seed=5, n_customers=800, days=20)
    base, _ = generate(cfg)
    drifted, _ = generate(GenConfig(seed=5, n_customers=800, days=20, drift_day=10, drift_inflation=1.5))
    cut = pd.Timestamp(cfg.start, tz="UTC") + pd.Timedelta(days=10)
    before_a = base[(base.ts < cut) & ~base.is_fraud].amount.median()
    before_b = drifted[(drifted.ts < cut) & ~drifted.is_fraud].amount.median()
    after_a = base[(base.ts >= cut) & ~base.is_fraud].amount.median()
    after_b = drifted[(drifted.ts >= cut) & ~drifted.is_fraud].amount.median()
    assert abs(before_b / before_a - 1) < 0.05
    assert after_b / after_a > 1.3
