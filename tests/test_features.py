"""The two feature implementations must agree, and the PIT join must never look ahead."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rtml.dataset import SCORE_OFFSET, naive, pit_join
from rtml.features.defs import CUSTOMER_DEVICE, CUSTOMER_VELOCITY, DEVICE_VELOCITY, request_features
from rtml.features.online import OnlineFeatures, txns_from_frame
from rtml.store import get_store

Small = tuple[Path, pd.DataFrame, dict[str, pd.DataFrame]]


def _online(tx: pd.DataFrame, bug: bool) -> dict[str, pd.DataFrame]:
    rows = [f for _, f in OnlineFeatures(skew_bug=bug).replay(txns_from_frame(tx))]
    return {
        "customer_velocity": pd.DataFrame([r.customer_velocity for r in rows]),
        "device_velocity": pd.DataFrame([r.device_velocity for r in rows]),
        "customer_device": pd.DataFrame([r.customer_device for r in rows]),
    }


def _mismatch(online: dict[str, pd.DataFrame], offline: dict[str, pd.DataFrame]) -> dict[str, float]:
    out = {}
    for name, cols, keys in (
        ("customer_velocity", CUSTOMER_VELOCITY, ["customer_id"]),
        ("device_velocity", DEVICE_VELOCITY, ["device_id"]),
        ("customer_device", CUSTOMER_DEVICE, ["customer_id", "device_id"]),
    ):
        j = online[name].merge(offline[name], on=[*keys, "event_timestamp"], suffixes=("_on", "_off"), validate="1:1")
        assert len(j) == len(online[name])
        for c in cols:
            out[c] = float((~np.isclose(j[f"{c}_on"], j[f"{c}_off"], rtol=1e-9, atol=1e-9)).mean())
    return out


def test_stream_matches_sql(small: Small, velocity: dict[str, pd.DataFrame]) -> None:
    assert all(v == 0.0 for v in _mismatch(_online(small[1], bug=False), velocity).values())


def test_parity_check_catches_the_planted_bug(small: Small, velocity: dict[str, pd.DataFrame]) -> None:
    bad = {k: v for k, v in _mismatch(_online(small[1], bug=True), velocity).items() if v > 0}
    assert set(bad) == {"dev_distinct_cards_1h"}
    assert bad["dev_distinct_cards_1h"] > 0.01


def test_feast_pit_join_equals_merge_asof(small: Small) -> None:
    d, tx, tables = small
    rows = tx.iloc[len(tx) // 2 :]
    pit = pit_join(get_store(d, redis="", apply=False), rows).set_index("txn_id")
    ref = rows.assign(event_timestamp=rows.ts - SCORE_OFFSET).sort_values("event_timestamp")
    t = tables["device_velocity"].sort_values("event_timestamp")
    ref = pd.merge_asof(
        ref, t.astype({"event_timestamp": ref.event_timestamp.dtype}), on="event_timestamp", by="device_id"
    )
    ref = ref.set_index("txn_id").loc[pit.index]
    for c in DEVICE_VELOCITY:
        np.testing.assert_allclose(pit[c].astype(float).fillna(-1), ref[c].astype(float).fillna(-1))


def test_pit_features_come_from_before_the_transaction(small: Small) -> None:
    _, tx, tables = small
    t = tables["customer_velocity"]
    rows = tx.iloc[len(tx) // 2 :]
    first = rows.iloc[0]
    # The state stored for the transaction itself is at its own timestamp; the join asks 1 ms earlier.
    own = t[(t.customer_id == first.customer_id) & (t.event_timestamp == first.ts)]
    assert len(own) == 1
    assert (first.ts - SCORE_OFFSET) < own.event_timestamp.iloc[0]


def test_naive_join_reads_the_future(small: Small) -> None:
    _, tx, tables = small
    as_of = tx.ts.max()
    rows = tx[tx.ts < tx.ts.min() + pd.Timedelta(days=10)]
    j = naive(rows, tables, as_of)
    assert (j.cust_last_txn_epoch > (j.ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()).mean() > 0.9


def test_request_features_treat_unseen_entities_as_zero_history() -> None:
    df = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2026-07-01 13:30", tz="UTC"),
                "amount": 100.0,
                "merchant_category": "electronics",
                "country": "DE",
                "home_country": "US",
                "cust_amount_mean_7d": 25.0,
            },
            {
                "ts": pd.Timestamp("2026-07-01 02:00", tz="UTC"),
                "amount": 5.0,
                "merchant_category": "nope",
                "country": "US",
            },
        ]
    )
    out = request_features(df)
    assert out.loc[0, "is_foreign"] == 1.0
    assert out.loc[0, "amount_to_cust_mean_7d"] == 4.0
    assert out.loc[0, "hour_of_day"] == 13.0
    assert out.loc[1, "category_code"] == -1.0
    assert out.loc[1, "is_new_device"] == 1.0
    assert out.loc[1, "cust_txn_count_24h"] == 0.0
    assert np.isnan(out.loc[1, "is_foreign"])


def test_serving_row_equals_training_matrix(small: Small) -> None:
    """The pandas-free serving transform must match the training transform on every row."""
    from rtml.features.defs import MODEL_FEATURES_V2, model_matrix, request_row

    d, tx, _ = small
    rows = pit_join(get_store(d, redis="", apply=False), tx.iloc[len(tx) // 2 :])
    expected = model_matrix(rows, MODEL_FEATURES_V2).to_numpy()
    recs = rows.to_dict(orient="records")
    # what the API sees: missing features arrive as None, timestamps as ISO strings
    for r in recs:
        r["ts"] = r["ts"].isoformat()
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                r[k] = None
    got = np.vstack([request_row(r, MODEL_FEATURES_V2) for r in recs])
    # epoch seconds near 1.8e9 round differently by a few hundred nanoseconds in the two paths
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-6, equal_nan=True)
