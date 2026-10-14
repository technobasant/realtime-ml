# 2. Every window feature is implemented twice, and a parity test holds them together

Status: accepted · 2026-10

## Context

Training features are backfilled in bulk; serving features are updated one event at a
time. In most platforms those are two codebases — SQL or Spark for the backfill, a
stream processor for serving — and training/serving skew lives in the gap between them.
Sharing one Python function between both paths hides the gap rather than closing it:
the backfill is then not how the warehouse would compute it, and the test passes by
construction.

## Decision

* The backfill is SQL: DuckDB window functions with `RANGE BETWEEN … PRECEDING AND
  CURRENT ROW`, `count(DISTINCT …)` included (`rtml.features.batch`).
* The stream is a per-entity event buffer updated one event at a time
  (`rtml.features.online`), shared by the stream processor and the parity test.
* `rtml.features.defs` is the contract — names, windows, semantics — and the only code
  both paths share is the request-time transform (amount ratios, time since the last
  transaction, new-device flag), computed from the transaction itself.
* The parity test replays the event log through the stream code and compares every row
  with the SQL output. A planted defect (`SKEW_BUG=1`: the device's distinct-card count
  reads the 24-hour buffer instead of the last hour) proves the test can fail.
* Event timestamps are unique to the microsecond. A SQL `RANGE` window counts ties
  together; a stream counts them in arrival order. The generator never emits ties, and
  this is a known parity gap for data that does.

## Consequences

* Two implementations cost more to change; the parity test is what makes that safe.
* The planted defect changes 17% of one feature's values and leaves the model's test
  AUC-PR unchanged (the model barely uses that feature). A model-level check would have
  passed it; only the feature-level comparison catches it. See MEASUREMENTS.
