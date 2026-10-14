"""Seeded synthetic payments for a fictional e-commerce company, with fraud and late labels.

The generator is the ground truth for every measurement in this repository, so it is
deterministic for a given seed and every fraud pattern is tagged with its type.

Legitimate traffic: each customer has a home country, one or two cards, one to three
devices, a typical basket size and a daily rate. Fraud arrives in three shapes:

* card testing — a fraudster tries many victims' cards with small amounts inside a
  burst, from one device or a fresh device per attempt;
* account takeover — a victim's account is used from a new device, often abroad, for a
  few larger purchases within hours;
* lone anomaly — a single larger purchase, sometimes from the victim's own stolen device.

Legitimate customers do fraud-like things too — big-ticket purchases, a new device,
travel — so the classes overlap and the model has something to learn.

Labels are late: a fraud is reported as a chargeback after a lognormal delay (median about
a week, tail past a month), 15% are never reported, and a small rate of legitimate
purchases are disputed anyway. ``drift_day`` switches fraudsters to quieter tactics and
inflates every basket from that day on, so a monitor has something real to find.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = np.array(["US", "GB", "DE", "IN", "NP", "AU", "CA", "FR", "BR", "SG"])
COUNTRY_P = np.array([0.30, 0.12, 0.10, 0.12, 0.08, 0.06, 0.07, 0.06, 0.05, 0.04])
CATEGORIES = np.array(["grocery", "fashion", "home", "electronics", "gift_cards", "digital", "travel", "books"])
LEGIT_CAT_P = np.array([0.30, 0.18, 0.15, 0.10, 0.04, 0.10, 0.05, 0.08])
N_MERCHANTS_PER_CAT = 40


@dataclass(frozen=True)
class GenConfig:
    seed: int = 7
    n_customers: int = 8000
    days: int = 60
    start: str = "2026-07-01"
    drift_day: int | None = None
    drift_inflation: float = 1.15
    card_testing_bursts_per_day: float = 3.0
    ato_per_day: float = 8.0
    lone_anomaly_per_day: float = 12.0


def _merchant(rng: np.random.Generator, cats: np.ndarray) -> np.ndarray:
    idx = pd.Index(CATEGORIES).get_indexer(cats)
    out: np.ndarray = np.char.add(
        np.char.add("m_", idx.astype(str)),
        np.char.add("_", rng.integers(0, N_MERCHANTS_PER_CAT, len(cats)).astype(str)),
    )
    return out


def generate(cfg: GenConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg.start, tz="UTC")
    span_s = cfg.days * 86400

    # --- customers ---
    n = cfg.n_customers
    cust = pd.DataFrame(
        {
            "customer_id": np.char.add("c_", np.arange(n).astype(str)),
            "home_country": rng.choice(COUNTRIES, n, p=COUNTRY_P),
            "account_created_at": start - pd.to_timedelta(rng.gamma(2.0, 200.0, n), unit="D"),
        }
    )
    typical = rng.lognormal(3.4, 0.6, n)  # typical basket, ~$30 median
    rate = rng.gamma(2.0, 0.6, n)  # purchases per day
    n_cards = rng.integers(1, 3, n)
    n_devices = rng.integers(1, 4, n)

    # --- legitimate purchases ---
    counts = rng.poisson(rate * cfg.days)
    owner = np.repeat(np.arange(n), counts)
    m = len(owner)
    day = rng.integers(0, cfg.days, m)
    hour = rng.normal(15, 4.5, m) % 24  # busier in the evening
    ts = day * 86400 + hour * 3600
    card_slot = rng.integers(0, 2, m) % n_cards[owner]
    dev_slot = (rng.integers(0, 3, m) % n_devices[owner]).astype(str)
    cats = rng.choice(CATEGORIES, m, p=LEGIT_CAT_P)
    amount = typical[owner] * rng.lognormal(0.0, 0.5, m)
    # Legitimate behaviour that looks like fraud, so the classes overlap: big-ticket
    # purchases, a new phone or a borrowed browser, travel, gifts bought abroad.
    big = rng.random(m) < 0.02
    amount[big] *= rng.uniform(3.0, 10.0, big.sum())
    cats[big] = rng.choice(["electronics", "travel", "home"], big.sum())
    fresh = rng.random(m) < 0.015
    dev_slot[fresh] = np.char.add("x", rng.integers(0, 10**6, fresh.sum()).astype(str))
    travelling = rng.random(m) < 0.05
    country = np.where(travelling, rng.choice(COUNTRIES, m, p=COUNTRY_P), cust["home_country"].to_numpy()[owner])
    legit = pd.DataFrame(
        {
            "ts_s": ts,
            "customer_id": cust["customer_id"].to_numpy()[owner],
            "card_id": np.char.add(np.char.add("k_", owner.astype(str)), np.char.add("_", card_slot.astype(str))),
            "device_id": np.char.add(np.char.add("d_", owner.astype(str)), np.char.add("_", dev_slot)),
            "merchant_category": cats,
            "amount": amount,
            "country": country,
            "is_fraud": False,
            "fraud_type": "none",
        }
    )

    frames = [legit]
    drift_s = None if cfg.drift_day is None else cfg.drift_day * 86400

    def quiet(t: float) -> bool:
        """After the drift day fraudsters slow down and stay closer to normal behaviour."""
        return drift_s is not None and t >= drift_s

    def own_device(v: int) -> str:
        return f"d_{v}_{int(rng.integers(0, n_devices[v]))}"

    def card(v: int) -> str:
        return f"k_{v}_{int(rng.integers(0, n_cards[v]))}"

    rows: list[tuple[float, str, str, str, str, float, str, bool, str]] = []

    # --- card testing: many victims' cards from a few devices, small amounts ---
    for b in range(rng.poisson(cfg.card_testing_bursts_per_day * cfg.days)):
        t0 = rng.uniform(0, span_s - 9 * 3600)
        q = quiet(t0)
        k = int(rng.integers(4, 25))
        duration = rng.uniform(4 * 3600, 8 * 3600) if q else rng.uniform(600, 3600)
        rotating = rng.random() < (0.6 if q else 0.3)  # a fresh device per attempt
        times = np.sort(t0 + rng.uniform(0, duration, k))
        cc = str(rng.choice(COUNTRIES, p=COUNTRY_P))
        for i, t in enumerate(times):
            v = int(rng.integers(0, n))
            rows.append(
                (
                    float(t),
                    str(cust.customer_id.iat[v]),
                    card(v),
                    f"fd_ct_{b}_{i}" if rotating else f"fd_ct_{b}",
                    str(rng.choice(["digital", "gift_cards", "books", "grocery"], p=[0.5, 0.2, 0.15, 0.15])),
                    float(rng.uniform(1.0, 25.0)),
                    cc if rng.random() < 0.7 else str(cust.home_country.iat[v]),
                    True,
                    "card_testing",
                )
            )
    # --- account takeover: a victim's account, a new device, a few larger purchases ---
    for a in range(rng.poisson(cfg.ato_per_day * cfg.days)):
        v = int(rng.integers(0, n))
        t0 = rng.uniform(0, span_s - 4 * 3600)
        q = quiet(t0)
        k = int(rng.integers(1, 6))
        times = np.sort(t0 + rng.uniform(0, rng.uniform(1800, 4 * 3600), k))
        home = str(cust.home_country.iat[v])
        abroad = rng.random() < (0.2 if q else 0.6)
        cc = str(rng.choice(COUNTRIES[home != COUNTRIES])) if abroad else home
        lo, hi = (1.2, 3.0) if q else (1.5, 6.0)
        for t in times:
            rows.append(
                (
                    float(t),
                    str(cust.customer_id.iat[v]),
                    card(v),
                    f"fd_ato_{a}",
                    str(rng.choice(["electronics", "gift_cards", "travel", "fashion"])),
                    float(typical[v] * rng.uniform(lo, hi)),
                    cc,
                    True,
                    "account_takeover",
                )
            )
    # --- lone anomaly: one larger purchase, sometimes from the victim's own (stolen) device ---
    for a in range(rng.poisson(cfg.lone_anomaly_per_day * cfg.days)):
        v = int(rng.integers(0, n))
        t = float(rng.uniform(0, span_s - 1))
        q = quiet(t)
        rows.append(
            (
                t,
                str(cust.customer_id.iat[v]),
                card(v),
                own_device(v) if rng.random() < 0.4 else f"fd_one_{a}",
                str(rng.choice(["electronics", "gift_cards", "fashion", "home"])),
                float(typical[v] * (rng.uniform(1.2, 3.0) if q else rng.uniform(2.0, 8.0))),
                str(cust.home_country.iat[v]) if rng.random() < 0.6 else str(rng.choice(COUNTRIES)),
                True,
                "lone_anomaly",
            )
        )
    frames.append(
        pd.DataFrame(
            rows,
            columns=[
                "ts_s",
                "customer_id",
                "card_id",
                "device_id",
                "merchant_category",
                "amount",
                "country",
                "is_fraud",
                "fraud_type",
            ],
        )
    )
    tx = pd.concat(frames, ignore_index=True)

    # seasonal inflation after the drift day, for everyone
    if drift_s is not None:
        tx.loc[tx.ts_s >= drift_s, "amount"] *= cfg.drift_inflation

    tx["merchant_id"] = _merchant(rng, tx["merchant_category"].to_numpy())
    tx["amount"] = tx["amount"].round(2).clip(lower=0.5)
    # Microsecond timestamps, strictly increasing: ties would be counted together by a SQL
    # RANGE window but in arrival order by a stream, and that is not the skew under test.
    tx = tx.sort_values("ts_s", kind="stable").reset_index(drop=True)
    us = np.round(tx["ts_s"].to_numpy() * 1e6).astype("int64")
    seq = np.arange(len(us), dtype="int64")
    us = np.maximum.accumulate(us - seq) + seq
    tx["ts"] = start + pd.to_timedelta(us, unit="us")
    tx.insert(0, "txn_id", np.char.add("t_", np.arange(len(tx)).astype(str)))

    # --- late labels ---
    delay_days = rng.lognormal(np.log(7.0), 0.7, len(tx))
    reported = np.where(tx.is_fraud.to_numpy(), rng.random(len(tx)) < 0.85, rng.random(len(tx)) < 0.0005)
    tx["chargeback_at"] = pd.Series(pd.NaT, index=tx.index, dtype="datetime64[ns, UTC]")
    tx.loc[reported, "chargeback_at"] = tx.loc[reported, "ts"] + pd.to_timedelta(delay_days[reported], unit="D")
    tx = tx.drop(columns=["ts_s"])
    return tx, cust


def write(cfg: GenConfig, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    tx, cust = generate(cfg)
    tx.to_parquet(out / "transactions.parquet", index=False)
    cust.to_parquet(out / "customers.parquet", index=False)
    return tx, cust


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--customers", type=int, default=8000)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--drift-day", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    from rtml.config import DATA

    cfg = GenConfig(seed=a.seed, n_customers=a.customers, days=a.days, drift_day=a.drift_day)
    tx, _ = write(cfg, a.out or DATA)
    print(
        f"{len(tx):,} transactions, fraud {tx.is_fraud.mean():.3%} "
        f"({tx.fraud_type.value_counts().to_dict()}), reported {tx.chargeback_at.notna().sum():,}"
    )


if __name__ == "__main__":
    main()
