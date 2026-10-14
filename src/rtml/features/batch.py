"""Offline feature computation: SQL window functions over the event log, in DuckDB.

This is the backfill path — how the training tables would be built in a warehouse. It
shares no code with the streaming path in ``online.py``; the parity test compares them.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from rtml.features.defs import DAY, HOUR

_VELOCITY_SQL = f"""
WITH t AS (
  SELECT *, epoch_us(ts) AS us FROM tx
)
SELECT
  customer_id,
  ts AS event_timestamp,
  count(*)            OVER w1h  AS cust_txn_count_1h,
  count(*)            OVER w24h AS cust_txn_count_24h,
  sum(amount)         OVER w24h AS cust_amount_sum_24h,
  count(*)            OVER w7d  AS cust_txn_count_7d,
  avg(amount)         OVER w7d  AS cust_amount_mean_7d,
  count(DISTINCT country) OVER w7d AS cust_distinct_countries_7d,
  epoch_us(ts) / 1e6  AS cust_last_txn_epoch
FROM t
WINDOW
  w1h  AS (PARTITION BY customer_id ORDER BY us RANGE BETWEEN {HOUR * 1_000_000} PRECEDING AND CURRENT ROW),
  w24h AS (PARTITION BY customer_id ORDER BY us RANGE BETWEEN {DAY * 1_000_000} PRECEDING AND CURRENT ROW),
  w7d  AS (PARTITION BY customer_id ORDER BY us RANGE BETWEEN {7 * DAY * 1_000_000} PRECEDING AND CURRENT ROW)
"""

_DEVICE_SQL = f"""
WITH t AS (SELECT *, epoch_us(ts) AS us FROM tx)
SELECT
  device_id,
  ts AS event_timestamp,
  count(*)                    OVER w1h  AS dev_txn_count_1h,
  count(DISTINCT card_id)     OVER w1h  AS dev_distinct_cards_1h,
  count(DISTINCT customer_id) OVER w24h AS dev_distinct_customers_24h,
  sum(amount)                 OVER w1h  AS dev_amount_sum_1h
FROM t
WINDOW
  w1h  AS (PARTITION BY device_id ORDER BY us RANGE BETWEEN {HOUR * 1_000_000} PRECEDING AND CURRENT ROW),
  w24h AS (PARTITION BY device_id ORDER BY us RANGE BETWEEN {DAY * 1_000_000} PRECEDING AND CURRENT ROW)
"""

_CUSTOMER_DEVICE_SQL = f"""
WITH t AS (SELECT *, epoch_us(ts) AS us FROM tx)
SELECT
  customer_id, device_id,
  ts AS event_timestamp,
  count(*) OVER (PARTITION BY customer_id, device_id ORDER BY us
                 RANGE BETWEEN {30 * DAY * 1_000_000} PRECEDING AND CURRENT ROW) AS cd_txn_count_30d
FROM t
"""


def _con(tx: pd.DataFrame, customers: pd.DataFrame | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.register("tx", tx)
    if customers is not None:
        con.register("customers", customers)
    return con


def velocity_tables(tx: pd.DataFrame) -> dict[str, pd.DataFrame]:
    con = _con(tx)
    out = {
        "customer_velocity": con.execute(_VELOCITY_SQL).df(),
        "device_velocity": con.execute(_DEVICE_SQL).df(),
        "customer_device": con.execute(_CUSTOMER_DEVICE_SQL).df(),
    }
    for df in out.values():
        for c in df.columns:
            if "_count_" in c or "_distinct_" in c:
                df[c] = df[c].astype("int64")
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    return out


def profile_table(tx: pd.DataFrame, customers: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.DataFrame:
    """One snapshot per customer per day, taken at midnight from what was known before it.

    ``cust_chargebacks_90d`` counts chargebacks *received* in the 90 days before the
    snapshot (a chargeback received during day d is visible from the d+1 snapshot). It is
    a label-derived feature: safe when joined point-in-time, a leak when it is not.
    """
    end = start + pd.Timedelta(days=days)
    cb = tx.loc[tx.chargeback_at.notna(), ["customer_id", "chargeback_at"]]
    con = _con(cb, customers)
    con.execute(
        "CREATE TEMP TABLE days AS "
        "SELECT unnest(generate_series($s::TIMESTAMPTZ, $e::TIMESTAMPTZ, INTERVAL 1 DAY)) AS day",
        {"s": start, "e": end},
    )
    df = con.execute(
        """
        WITH daily AS (
          SELECT customer_id, date_trunc('day', chargeback_at) + INTERVAL 1 DAY AS visible_from, count(*) AS n
          FROM tx GROUP BY ALL
        )
        SELECT c.customer_id, d.day AS event_timestamp, c.home_country,
               CAST(date_diff('second', c.account_created_at, d.day) AS DOUBLE) / 86400.0 AS account_age_days,
               coalesce(sum(daily.n) FILTER (
                 WHERE daily.visible_from <= d.day AND daily.visible_from > d.day - INTERVAL 90 DAY
               ), 0) AS cust_chargebacks_90d
        FROM customers c CROSS JOIN days d
        LEFT JOIN daily ON daily.customer_id = c.customer_id
        GROUP BY ALL
        """
    ).df()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df["cust_chargebacks_90d"] = df["cust_chargebacks_90d"].astype("int64")
    return df


def write_tables(
    tx: pd.DataFrame, customers: pd.DataFrame, out: Path, start: pd.Timestamp, days: int
) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    tables = velocity_tables(tx)
    tables["customer_profile"] = profile_table(tx, customers, start, days)
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    return tables
