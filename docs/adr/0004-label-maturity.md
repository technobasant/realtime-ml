# 4. Train on transactions old enough for their chargebacks to have arrived

Status: accepted · 2026-10

## Context

A fraud label is a chargeback, and chargebacks arrive days to weeks after the purchase
(median about a week here; 15% of frauds are never reported). On training day, the most
recent transactions look legitimate because their chargebacks have not landed yet.
Train on them and the model learns that recent fraud is legitimate; validate on them and
the metrics are wrong in a way that looks like model weakness.

## Decision

* The model trained on day 63 uses transactions from days 14–42 and validates on days
  43–49: every row is at least 14 days old. Labels are the chargebacks known on day 63.
* Days 0–13 only build feature history.
* The threshold is chosen on the validation window to minimise a stated cost: a $5
  review for every flag, the full amount for every missed fraud.
* Metrics on future traffic are reported against the true labels, which only a
  simulation has, next to what the same model would appear to score with the labels
  available at the time.

## Consequences

* The newest two weeks of data are never trained on. The label experiment in
  MEASUREMENTS measures what that costs and what the alternatives cost.
* The promotion gate inherits the same delay: a shadow comparison decided the day the
  shadow period ends sees about a quarter of its frauds.
