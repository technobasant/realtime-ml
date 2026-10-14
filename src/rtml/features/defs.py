"""The feature contract: names, windows and the request-time transform.

Two implementations compute the windowed aggregates — SQL over the event log for
training (``batch.py``) and an incremental state machine for serving (``online.py``).
They are written separately on purpose, as they would be in a real platform, and the
parity test holds them to the same answer. Everything computed from the transaction
itself at request time lives here, in one function, used by training and serving alike.

Window semantics, identical in both paths: a stored row is the entity's state *after* an
event at time ``t``, aggregating that entity's events in ``[t - W, t]``. A transaction is
scored with the state after the entity's previous event — online because the stream has
not seen the current one yet, offline because the training timestamp is ``t - 1 ms``.
Windows are not re-expired at read time; ``secs_since_last_cust_txn`` tells the model how
stale they are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = 3600
DAY = 86400
CATEGORIES = ["grocery", "fashion", "home", "electronics", "gift_cards", "digital", "travel", "books"]

CUSTOMER_VELOCITY = [
    "cust_txn_count_1h",
    "cust_txn_count_24h",
    "cust_amount_sum_24h",
    "cust_txn_count_7d",
    "cust_amount_mean_7d",
    "cust_distinct_countries_7d",
    "cust_last_txn_epoch",
]
DEVICE_VELOCITY = [
    "dev_txn_count_1h",
    "dev_distinct_cards_1h",
    "dev_distinct_customers_24h",
    "dev_amount_sum_1h",
]
CUSTOMER_DEVICE = ["cd_txn_count_30d"]
CUSTOMER_PROFILE = ["home_country", "account_age_days", "cust_chargebacks_90d"]

STREAM_FEATURES = CUSTOMER_VELOCITY + DEVICE_VELOCITY + CUSTOMER_DEVICE
COUNT_FEATURES = [
    "cust_txn_count_1h",
    "cust_txn_count_24h",
    "cust_amount_sum_24h",
    "cust_txn_count_7d",
    "cust_distinct_countries_7d",
    "dev_txn_count_1h",
    "dev_distinct_cards_1h",
    "dev_distinct_customers_24h",
    "dev_amount_sum_1h",
    "cd_txn_count_30d",
]

REQUEST_FEATURES = [
    "amount_log",
    "category_code",
    "hour_of_day",
    "is_foreign",
    "amount_to_cust_mean_7d",
    "secs_since_last_cust_txn",
    "is_new_device",
]

# The v1 model sees the customer only; v2 adds the device, which is where card testing lives.
MODEL_FEATURES_V1 = [
    "cust_txn_count_1h",
    "cust_txn_count_24h",
    "cust_amount_sum_24h",
    "cust_txn_count_7d",
    "cust_amount_mean_7d",
    "cust_distinct_countries_7d",
    "account_age_days",
    "cust_chargebacks_90d",
    "amount_log",
    "category_code",
    "hour_of_day",
    "is_foreign",
    "amount_to_cust_mean_7d",
    "secs_since_last_cust_txn",
]
MODEL_FEATURES_V2 = [
    *MODEL_FEATURES_V1,
    "dev_txn_count_1h",
    "dev_distinct_cards_1h",
    "dev_distinct_customers_24h",
    "dev_amount_sum_1h",
    "cd_txn_count_30d",
    "is_new_device",
]
FEATURE_SETS = {"v1": MODEL_FEATURES_V1, "v2": MODEL_FEATURES_V2}


def request_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add request-time features to a frame of transactions joined with fetched features.

    ``df`` needs the raw columns ``ts``, ``amount``, ``merchant_category`` and ``country``
    plus whatever stored features were fetched; missing stored features mean the entity
    has no history yet, and count as zero.
    """
    out = df.copy()
    for c in COUNT_FEATURES:
        if c not in out:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype("float64")
    for c in ("cust_amount_mean_7d", "cust_last_txn_epoch", "account_age_days", "cust_chargebacks_90d"):
        if c not in out:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    if "home_country" not in out:
        out["home_country"] = None

    ts = pd.to_datetime(out["ts"], utc=True)
    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    amount = out["amount"].astype("float64")
    out["amount_log"] = np.log1p(amount)
    out["category_code"] = pd.Categorical(out["merchant_category"], categories=CATEGORIES).codes.astype("float64")
    out["hour_of_day"] = ts.dt.hour.astype("float64")
    home = out["home_country"]
    out["is_foreign"] = np.where(home.isna(), np.nan, (out["country"] != home).astype("float64"))
    mean7 = out["cust_amount_mean_7d"]
    out["amount_to_cust_mean_7d"] = np.where(mean7 > 0, amount / mean7, np.nan)
    out["secs_since_last_cust_txn"] = epoch - out["cust_last_txn_epoch"]
    out["is_new_device"] = (out["cd_txn_count_30d"] == 0).astype("float64")
    return out


def model_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Request features plus stored features, in the column order the model was trained on."""
    return request_features(df)[features].astype("float64")
