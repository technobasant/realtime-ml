"""Streaming feature computation: per-entity event buffers, updated one event at a time.

This is the serving path. It shares no code with the SQL in ``batch.py`` on purpose; the
parity test (``rtml.parity``) holds the two to the same values.

``SKEW_BUG=1`` plants a realistic defect for the parity test to catch: the device's
distinct-card count reads the 24-hour buffer instead of the last hour — a copy-paste
slip that is invisible in unit tests of the stream code alone.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from rtml.features.defs import DAY, HOUR

US = 1_000_000


def skew_bug_enabled() -> bool:
    return os.environ.get("SKEW_BUG", "0") == "1"


@dataclass(frozen=True)
class Txn:
    txn_id: str
    us: int  # event time, microseconds since the epoch
    customer_id: str
    card_id: str
    device_id: str
    amount: float
    country: str

    @classmethod
    def from_record(cls, r: dict[str, Any]) -> Txn:
        ts = pd.Timestamp(r["ts"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return cls(
            txn_id=str(r["txn_id"]),
            us=int(ts.value // 1000),
            customer_id=str(r["customer_id"]),
            card_id=str(r["card_id"]),
            device_id=str(r["device_id"]),
            amount=float(r["amount"]),
            country=str(r["country"]),
        )


@dataclass
class _Buffer:
    horizon_us: int
    events: deque[Txn] = field(default_factory=deque)

    def add(self, t: Txn) -> None:
        self.events.append(t)
        cutoff = t.us - self.horizon_us
        while self.events and self.events[0].us < cutoff:
            self.events.popleft()

    def within(self, now_us: int, window_s: int) -> list[Txn]:
        cutoff = now_us - window_s * US
        return [e for e in self.events if e.us >= cutoff]


@dataclass
class FeatureRows:
    """The three rows one event produces, ready to push to the online store."""

    customer_velocity: dict[str, Any]
    device_velocity: dict[str, Any]
    customer_device: dict[str, Any]


class OnlineFeatures:
    """Window state for every customer, device and customer-device pair seen so far."""

    def __init__(self, skew_bug: bool | None = None) -> None:
        self.skew_bug = skew_bug_enabled() if skew_bug is None else skew_bug
        self._cust: dict[str, _Buffer] = {}
        self._dev: dict[str, _Buffer] = {}
        self._cd: dict[tuple[str, str], _Buffer] = {}
        self.events = 0
        self.last_us = 0

    def update(self, t: Txn) -> FeatureRows:
        self.events += 1
        self.last_us = max(self.last_us, t.us)
        ts = pd.Timestamp(t.us * 1000, tz="UTC")

        cb = self._cust.setdefault(t.customer_id, _Buffer(7 * DAY * US))
        cb.add(t)
        h1, d1, d7 = cb.within(t.us, HOUR), cb.within(t.us, DAY), cb.within(t.us, 7 * DAY)
        cust = {
            "customer_id": t.customer_id,
            "event_timestamp": ts,
            "cust_txn_count_1h": len(h1),
            "cust_txn_count_24h": len(d1),
            "cust_amount_sum_24h": sum(e.amount for e in d1),
            "cust_txn_count_7d": len(d7),
            "cust_amount_mean_7d": sum(e.amount for e in d7) / len(d7),
            "cust_distinct_countries_7d": len({e.country for e in d7}),
            "cust_last_txn_epoch": t.us / US,
        }

        db = self._dev.setdefault(t.device_id, _Buffer(DAY * US))
        db.add(t)
        dh1, dd1 = db.within(t.us, HOUR), db.within(t.us, DAY)
        cards_window = dd1 if self.skew_bug else dh1
        dev = {
            "device_id": t.device_id,
            "event_timestamp": ts,
            "dev_txn_count_1h": len(dh1),
            "dev_distinct_cards_1h": len({e.card_id for e in cards_window}),
            "dev_distinct_customers_24h": len({e.customer_id for e in dd1}),
            "dev_amount_sum_1h": sum(e.amount for e in dh1),
        }

        pb = self._cd.setdefault((t.customer_id, t.device_id), _Buffer(30 * DAY * US))
        pb.add(t)
        cd = {
            "customer_id": t.customer_id,
            "device_id": t.device_id,
            "event_timestamp": ts,
            "cd_txn_count_30d": len(pb.within(t.us, 30 * DAY)),
        }
        return FeatureRows(cust, dev, cd)

    def replay(self, txns: Iterable[Txn]) -> Iterator[tuple[Txn, FeatureRows]]:
        for t in txns:
            yield t, self.update(t)


def txns_from_frame(df: pd.DataFrame) -> Iterator[Txn]:
    us = (pd.to_datetime(df["ts"], utc=True).astype("int64") // 1000).to_numpy()
    amount = df["amount"].to_numpy(dtype=np.float64)
    cols = [df[c].astype(str).tolist() for c in ("txn_id", "customer_id", "card_id", "device_id", "country")]
    for i, (txn_id, customer_id, card_id, device_id, country) in enumerate(zip(*cols, strict=True)):
        yield Txn(txn_id, int(us[i]), customer_id, card_id, device_id, float(amount[i]), country)
