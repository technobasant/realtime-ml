# 7. Drift monitoring: fixed reference, count window, calendar features excluded

Status: accepted · 2026-10

## Context

The usual recipe — PSI per feature against the training distribution, alert above 0.2 —
was measured against a drift that hurts the model (fraudsters going quiet, prices up 15%)
and against no drift at all. See MEASUREMENTS for the numbers.

## Decision

* Reference: the training validation window (days 43–49), scored by the current champion.
* Window: the champion's last 3,000 scored transactions, re-evaluated every 10 s. A count
  window behaves the same under a replay that compresses a day into minutes.
* Feature PSI alerts exclude `account_age_days` and `cust_chargebacks_90d`. Both trend
  with the calendar; against a fixed reference they reached PSI 0.2 within weeks with
  nothing changed, and they were the only features that ever fired offline.
* PSI is computed against the reference re-weighted to the window's own mix of hours.
  A 3,000-transaction window is about seven hours of traffic. Against the whole-day
  reference, hour-of-day PSI was 5.1 and the first live run's drift alert fired before a
  single drifted transaction had been sent; excluding hour of day only moved the problem
  to `secs_since_last_cust_txn` (PSI 0.71, also a daily cycle), which fired 4 minutes
  early in the second run. Re-weighting removes the cycle from every feature at once.
  Both the matched and the unmatched PSI are exported.
* The flag-rate alert uses |z| > 8, not 4. Fraud arrives in bursts, so the binomial
  variance is too small and z > 4 fired on 17% of drift-free windows.
* Recall on the chargebacks that have arrived is exported too; it is the signal that
  tracks the model's real performance, and it is about two weeks late.

## Consequences

* None of the label-free monitors caught the tactics drift both quickly and quietly;
  this is reported as a limit, not tuned away.
