# Changelog

## 0.1.0 — 2026-10

First release.

- A seeded generator of payments for Lumen & Loom, a fictional homewares retailer:
  8,000 customers over 90 days, card-testing bursts, account takeovers and lone
  anomalies, legitimate behaviour that overlaps them, chargebacks that arrive a median
  week late (15% never), and a drift switch.
- Window features implemented twice: SQL window functions in DuckDB for the backfill,
  an incremental per-entity state machine for the stream. A parity test holds them equal
  row for row, and `SKEW_BUG=1` plants a defect it must catch.
- Feast 0.66 with a DuckDB/Parquet offline store and a Redis online store, defined in
  code: three streaming feature views fed by push sources, one daily batch view, and
  point-in-time training joins checked against `merge_asof`. A naive latest-snapshot
  join is kept so its cost can be measured.
- LightGBM training on time-based splits with a label-maturity window, a cost-optimal
  threshold, and MLflow 3.17 tracking and registry (Postgres backend) with `champion`
  and `challenger` aliases.
- A FastAPI scoring service that reads Feast online, answers with the champion, scores
  the challenger in shadow on its own thread and logs every score with its features to
  Postgres.
- A stream processor that warm-starts its windows from the event log, consumes Kafka
  4.3.1 (KRaft) and pushes to Feast.
- A drift monitor exporting hour-matched PSI, KS, score PSI, flag-rate z and labelled
  recall to Prometheus, with alert rules.
- A promotion gate on evalproof 0.1.1: paired bootstrap on per-transaction cost, exact
  McNemar on correct-at-threshold, and a 2 ms model latency budget.
- Offline experiments for leakage, label delay, model quality and calibration, skew,
  drift detection and the gate; `make demo` for the live stack. Results in
  MEASUREMENTS.md.
- CI: ruff, mypy strict, pytest, and the whole stack end to end on the runner.
