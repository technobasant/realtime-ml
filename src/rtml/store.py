"""The Feast feature store, defined in code.

Offline: Parquet files read by DuckDB. Online: Redis in the stack, SQLite in tests.
The repository is applied programmatically so every environment (host, container, CI)
registers its own paths instead of sharing a registry that points at someone's disk.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureStore, FeatureView, Field, FileSource, PushSource, RepoConfig
from feast.infra.offline_stores.duckdb import DuckDBOfflineStoreConfig
from feast.infra.online_stores.redis import RedisOnlineStoreConfig
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.types import Float64, Int64, String
from feast.value_type import ValueType

from rtml.config import DATA

PROJECT = "rtml"

customer = Entity(name="customer", join_keys=["customer_id"], value_type=ValueType.STRING)
device = Entity(name="device", join_keys=["device_id"], value_type=ValueType.STRING)

# ttl=0: no lookback limit. The windows carry their own time bounds, and Feast's Redis
# store does not enforce a view's ttl at read time, so a ttl here would be applied
# offline but not online — a skew of its own.
NO_TTL = timedelta(0)


def _file(name: str, data: Path) -> FileSource:
    return FileSource(
        name=f"{name}_file", path=str(data / "features" / f"{name}.parquet"), timestamp_field="event_timestamp"
    )


def definitions(data: Path) -> list[object]:
    cust_push = PushSource(name="customer_velocity_push", batch_source=_file("customer_velocity", data))
    dev_push = PushSource(name="device_velocity_push", batch_source=_file("device_velocity", data))
    cd_push = PushSource(name="customer_device_push", batch_source=_file("customer_device", data))
    customer_velocity = FeatureView(
        name="customer_velocity",
        entities=[customer],
        ttl=NO_TTL,
        schema=[
            Field(name="cust_txn_count_1h", dtype=Int64),
            Field(name="cust_txn_count_24h", dtype=Int64),
            Field(name="cust_amount_sum_24h", dtype=Float64),
            Field(name="cust_txn_count_7d", dtype=Int64),
            Field(name="cust_amount_mean_7d", dtype=Float64),
            Field(name="cust_distinct_countries_7d", dtype=Int64),
            Field(name="cust_last_txn_epoch", dtype=Float64),
        ],
        source=cust_push,
        online=True,
    )
    device_velocity = FeatureView(
        name="device_velocity",
        entities=[device],
        ttl=NO_TTL,
        schema=[
            Field(name="dev_txn_count_1h", dtype=Int64),
            Field(name="dev_distinct_cards_1h", dtype=Int64),
            Field(name="dev_distinct_customers_24h", dtype=Int64),
            Field(name="dev_amount_sum_1h", dtype=Float64),
        ],
        source=dev_push,
        online=True,
    )
    customer_device = FeatureView(
        name="customer_device",
        entities=[customer, device],
        ttl=NO_TTL,
        schema=[Field(name="cd_txn_count_30d", dtype=Int64)],
        source=cd_push,
        online=True,
    )
    customer_profile = FeatureView(
        name="customer_profile",
        entities=[customer],
        ttl=NO_TTL,
        schema=[
            Field(name="home_country", dtype=String),
            Field(name="account_age_days", dtype=Float64),
            Field(name="cust_chargebacks_90d", dtype=Int64),
        ],
        source=_file("customer_profile", data),
        online=True,
    )
    return [
        customer,
        device,
        cust_push,
        dev_push,
        cd_push,
        customer_velocity,
        device_velocity,
        customer_device,
        customer_profile,
    ]


FEATURE_REFS = (
    [
        f"customer_velocity:{f}"
        for f in (
            "cust_txn_count_1h",
            "cust_txn_count_24h",
            "cust_amount_sum_24h",
            "cust_txn_count_7d",
            "cust_amount_mean_7d",
            "cust_distinct_countries_7d",
            "cust_last_txn_epoch",
        )
    ]
    + [
        f"device_velocity:{f}"
        for f in ("dev_txn_count_1h", "dev_distinct_cards_1h", "dev_distinct_customers_24h", "dev_amount_sum_1h")
    ]
    + ["customer_device:cd_txn_count_30d"]
    + [f"customer_profile:{f}" for f in ("home_country", "account_age_days", "cust_chargebacks_90d")]
)


def get_store(data: Path | None = None, redis: str | None = None, apply: bool = True) -> FeatureStore:
    """A store over ``data``; online in Redis when ``redis`` (or ``REDIS_URL``) is set, else SQLite."""
    data = data or DATA
    redis = redis if redis is not None else os.environ.get("REDIS_URL")
    online = (
        RedisOnlineStoreConfig(connection_string=redis)
        if redis
        else SqliteOnlineStoreConfig(path=str(data / "online.db"))
    )
    (data / "features").mkdir(parents=True, exist_ok=True)
    cfg = RepoConfig(
        project=PROJECT,
        provider="local",
        registry=str(data / "registry.db"),
        offline_store=DuckDBOfflineStoreConfig(),
        online_store=online,
        entity_key_serialization_version=3,
        repo_path=str(data),
    )
    store = FeatureStore(config=cfg)
    if apply:
        store.apply(definitions(data))  # type: ignore[arg-type]
    return store
