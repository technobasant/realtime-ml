"""The naive training join, kept so its cost can be measured.

It builds a training set the way it is often first built: take the feature tables as
they stand on the day the data is pulled, keep each entity's latest row, and join on the
entity key. Every training row then sees features computed *after* it happened — the
rest of a card-testing burst, the account takeover's later purchases, and chargebacks
that include its own.
"""

from __future__ import annotations

import pandas as pd


def latest_as_of(table: pd.DataFrame, keys: list[str], as_of: pd.Timestamp) -> pd.DataFrame:
    t = table[table["event_timestamp"] < as_of]
    t = t.sort_values("event_timestamp").drop_duplicates(keys, keep="last")
    return t.drop(columns=["event_timestamp"])


def naive_join(rows: pd.DataFrame, tables: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    out = rows
    for name, keys in (
        ("customer_velocity", ["customer_id"]),
        ("device_velocity", ["device_id"]),
        ("customer_device", ["customer_id", "device_id"]),
        ("customer_profile", ["customer_id"]),
    ):
        out = out.merge(latest_as_of(tables[name], keys, as_of), on=keys, how="left")
    return out
