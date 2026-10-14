# 3. A transaction is scored with the state after the entity's previous event

Status: accepted · 2026-10

## Context

"Card count in the last hour" is ambiguous until three things are pinned down: whether
the window includes the transaction being scored, when the window is evaluated, and
what happens to an entity that has gone quiet. Get any of them different between
training and serving and the model learns on one feature and is served another.

## Decision

* A stored row is the entity's state **after** an event at time `t`, aggregating its
  events in `[t − W, t]`.
* A transaction is scored with the state after the entity's **previous** event. Online
  that is automatic: the payment is authorised before its event reaches the stream.
  Offline the training row asks Feast for features at `t − 1 ms`.
* Windows are not re-expired at read time. A customer whose last purchase was three days
  ago still shows that day's one-hour count. The model is given
  `secs_since_last_cust_txn` to discount stale windows.
* Feature views have `ttl=0`. Feast applies a view's TTL in historical retrieval but its
  Redis store does not enforce it on read, so a TTL would expire features offline that
  are still served online — a skew built into the store. With no TTL both sides return
  the last row whatever its age.
* The daily profile snapshot for day *d* is stamped at midnight and computed from what
  was known before it; a chargeback received during day *d* is visible from *d + 1*.

## Consequences

* Online and offline agree exactly in the replay (100% of rows), and the remaining
  end-to-end disagreement is stream lag, which is measured rather than designed away.
* Re-expiring at read time would be more intuitive and would need the same logic in
  the SQL backfill and the API; it is on the roadmap, not in 0.1.0.
