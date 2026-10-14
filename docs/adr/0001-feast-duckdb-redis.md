# 1. Feast, with DuckDB over Parquet offline and Redis online, defined in code

Status: accepted · 2026-10

## Context

Training needs point-in-time joins over months of feature history; serving needs one
entity's latest values in a few milliseconds. The two must agree on definitions and
on what "as of" means, or the model is trained on one thing and served another. All
of it has to fit a Docker VM of 8 CPUs and about 12 GB next to Kafka, Postgres and
MLflow.

## Decision

* Feast 0.66. Offline store: DuckDB reading Parquet files. Online store: Redis 7.4.
* Four feature views. Three are fed by the stream through push sources —
  `customer_velocity`, `device_velocity`, `customer_device` — and backed by Parquet
  written by the batch backfill. One, `customer_profile`, is a daily batch snapshot.
* `ttl=0` on every view: no lookback limit. See ADR 3 for why.
* The repository is applied from code (`rtml.store.get_store`), not from a
  `feature_store.yaml` and `feast apply`. Each environment — host, container, CI —
  registers its own paths, so no registry file points at someone else's disk.
* Training reads through `get_historical_features`; a test holds it equal to a
  `pandas.merge_asof` written independently.

## Why not

* **Postgres as the offline store**: the history is append-only Parquet already, and
  DuckDB joins 485,000 entity rows against four views in about 8 s here. Postgres
  would add a load step and a second copy.
* **A hand-written feature store**: the point of the repository is the failure modes
  of a real one — PIT joins, materialisation, push sources, TTL semantics — and Feast
  has all of them, including one surprise (ADR 3).

## Consequences

* The Feast Python SDK is on the request path: an online read of three views and
  fifteen features took 0.49 ms at the median and 3.4 ms at p99 in the live run — most of
  the scoring time.
* A Feast upgrade can change join semantics; the merge_asof test is the tripwire.
