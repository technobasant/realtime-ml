"""Build a dataset: generate events, compute feature tables, register them with Feast.

Also the two training joins — point-in-time through Feast, and naive — and the label
view a model is allowed to see at training time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from feast import FeatureStore

from rtml.config import DATA
from rtml.features import batch
from rtml.features.naive import naive_join
from rtml.generate import GenConfig, write
from rtml.store import FEATURE_REFS, get_store

RAW_COLS = ["txn_id", "ts", "customer_id", "card_id", "device_id", "merchant_category", "amount", "country"]
LABEL_COLS = ["is_fraud", "fraud_type", "chargeback_at"]

# A transaction is scored before its own event reaches the feature pipeline. Offline, the
# same thing is expressed by asking for features one millisecond before the event.
SCORE_OFFSET = pd.Timedelta(milliseconds=1)


def build(cfg: GenConfig, data: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    tx, customers = write(cfg, data)
    tables = batch.write_tables(tx, customers, data / "features", pd.Timestamp(cfg.start, tz="UTC"), cfg.days)
    (data / "dataset.json").write_text(json.dumps(asdict(cfg), indent=2))
    return tx, tables


def load(data: Path) -> tuple[pd.DataFrame, GenConfig]:
    cfg = GenConfig(**json.loads((data / "dataset.json").read_text()))
    return pd.read_parquet(data / "transactions.parquet"), cfg


def load_tables(data: Path) -> dict[str, pd.DataFrame]:
    names = ["customer_velocity", "device_velocity", "customer_device", "customer_profile"]
    return {n: pd.read_parquet(data / "features" / f"{n}.parquet") for n in names}


def day_slice(tx: pd.DataFrame, cfg: GenConfig, first: int, last: int) -> pd.DataFrame:
    """Transactions on days ``first`` to ``last`` inclusive."""
    start = pd.Timestamp(cfg.start, tz="UTC")
    lo, hi = start + pd.Timedelta(days=first), start + pd.Timedelta(days=last + 1)
    return tx[(tx.ts >= lo) & (tx.ts < hi)]


def day_ts(cfg: GenConfig, day: int) -> pd.Timestamp:
    return pd.Timestamp(cfg.start, tz="UTC") + pd.Timedelta(days=day)


def pit_join(store: FeatureStore, rows: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features for ``rows``, through Feast's historical retrieval."""
    ent = rows[["txn_id", "customer_id", "device_id"]].copy()
    ent["event_timestamp"] = rows["ts"] - SCORE_OFFSET
    feats = store.get_historical_features(entity_df=ent, features=FEATURE_REFS).to_df()
    feats = feats.drop(columns=["customer_id", "device_id", "event_timestamp"])
    out = rows.merge(feats, on="txn_id", how="left", validate="one_to_one")
    return out.sort_values("ts").reset_index(drop=True)


def naive(rows: pd.DataFrame, tables: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    return naive_join(rows, tables, as_of).sort_values("ts").reset_index(drop=True)


def labels_as_of(rows: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """What the label table says at ``as_of``: fraud only if the chargeback has arrived."""
    return (rows["chargeback_at"].notna() & (rows["chargeback_at"] <= as_of)).astype("int8")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate events and build the feature tables.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--customers", type=int, default=8000)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--drift-day", type=int, default=None)
    p.add_argument("--out", type=Path, default=DATA)
    a = p.parse_args()
    cfg = GenConfig(seed=a.seed, n_customers=a.customers, days=a.days, drift_day=a.drift_day)
    tx, tables = build(cfg, a.out)
    get_store(a.out)
    print(
        f"{len(tx):,} transactions over {cfg.days} days, fraud rate {tx.is_fraud.mean():.3%}; "
        + ", ".join(f"{k} {len(v):,} rows" for k, v in tables.items())
    )


if __name__ == "__main__":
    main()
