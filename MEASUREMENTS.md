# Measurements

Every number in the README comes from this file, and every number here comes from a
file in [`results/`](results/) written by the code in this repository. Nothing is
estimated. Where a result is unflattering it stays in.

* [Environment](#environment)
* [The data](#the-data)
* [Leakage: naive join against point-in-time join](#leakage)
* [Label delay](#label-delay)
* [Model quality and calibration](#model-quality-and-calibration)
* [Skew: offline against online](#skew)
* [Drift: time to detect against false alarms](#drift)
* [The promotion gate](#the-promotion-gate)
* [The live stack: latency, parity, gate, alert](#the-live-stack)
* [How to reproduce](#how-to-reproduce)

## Environment

| | |
|---|---|
| Offline experiments | Apple M5, 10 cores, macOS (Darwin 27), Python 3.12.15, run natively with `uv` |
| Live stack | Docker Desktop 29.8.2, a VM of 8 CPUs and 11.7 GiB; every service is a container on that VM |
| Libraries | Feast 0.66.0, LightGBM 4.7.0 (4 threads to train, 1 to serve), DuckDB 1.5.6, MLflow 3.17.0, evalproof 0.1.1, FastAPI, confluent-kafka |
| Services | Apache Kafka 4.3.1 (KRaft, one node), Redis 7.4.6, Postgres 17.6, MLflow server 3.17.0, Prometheus 3.5.0 |

All containers share one host, so the network between them is local: the latency
figures contain no real network hops, and a production deployment would add them.

## The data

A reproducible generator ([`rtml/generate.py`](src/rtml/generate.py)) of payments for Lumen &
Loom, an online homewares retailer (a standard e-commerce scenario). Seed 7, 8,000 customers, 90 days from
2026-07-01: **874,446 transactions, 0.80% fraud.**

* **Legitimate** customers have a home country, one or two cards, one to three devices,
  a typical basket and a purchase rate. 2% of their purchases are big-ticket (3–10× the
  basket), 1.5% come from a device never seen before, 5% from abroad.
* **Card testing**: 4–25 victims' cards tried with $1–25 purchases inside 10–60 minutes,
  from one device or (30% of bursts) a fresh device per attempt.
* **Account takeover**: one to five purchases at 1.5–6× the victim's basket, from a new
  device, 60% from another country.
* **Lone anomaly**: one purchase at 2–8× the basket, 40% from the victim's own device.
* **Labels arrive late.** A fraud becomes a chargeback after a lognormal delay (median 7
  days, a tail past a month); 15% are never reported; 0.05% of legitimate purchases are
  disputed anyway.
* **Drift switch** (off unless stated): from the drift day fraudsters slow down and stay
  closer to normal — card testing spread over 4–8 hours with more device rotation, takeovers
  mostly from the home country at 1.2–3× — and every price rises 15%.

The absolute scores below describe this generator, not real fraud. The comparisons are
the point: the same data, the same model, one thing changed.

**The calendar** ([`rtml/timeline.py`](src/rtml/timeline.py)). Days 0–13 only build feature
history. The model is trained on day 63, on transactions from days 14–42, validated on
days 43–49 (every row at least 14 days old), with the chargebacks known on day 63. Days
63–76 are its future: 136,621 transactions, 1,145 frauds. Days 77–83 are the shadow
period.

**Cost model**, used to pick the threshold and to score models: every flag costs a $5
review; every missed fraud costs its amount. "Cost saved" is against approving everything.

## Leakage

`python -m rtml.experiments.offline leakage` → [`results/leakage.json`](results/leakage.json)

The naive pipeline builds its training set the way it is often first built: take the
feature tables as they stand on the day the data is pulled (day 63), keep each entity's
latest row, join on the entity key. The point-in-time pipeline asks Feast for each
transaction's features one millisecond before it happened. Same rows, labels, model and
hyperparameters; only the join differs. "Future" is days 63–76, scored with point-in-time
features — what the online store would serve — and true labels.

| Features | Join | Validation AUC-PR (labels known day 63) | Validation recall at 80% precision | **Future AUC-PR** | Future recall at 80% precision | Future cost saved |
|---|---|---|---|---|---|---|
| v2 (customer + device) | naive | **0.746** | **0.779** | **0.363** | 0.334 | 35.0% |
| v2 | point-in-time | 0.681 | 0.362 | **0.941** | 0.912 | 82.5% |
| v1 (customer only) | naive | 0.551 | 0.277 | 0.180 | 0.058 | 20.2% |
| v1 | point-in-time | 0.412 | 0.031 | 0.613 | 0.416 | 66.5% |

* **The naive pipeline looks better at validation and is far worse in production.** v2
  naive reports AUC-PR 0.746 and recall 0.78 at 80% precision; on the future it delivers
  0.363 and 0.33. The point-in-time model reports less (0.681) and delivers 0.941.
  Measured on true labels the validation gap is the same shape: 0.852 naive, 0.937 PIT.
* **What it learned instead.** The naive v2's top features by gain are `dev_txn_count_1h`
  (41%), `cd_txn_count_30d` (21%) and `cust_chargebacks_90d` (17%): the rest of the
  card-testing burst, and chargebacks that include the transaction's own. The PIT model's
  are `cd_txn_count_30d` (32%) and `dev_distinct_customers_24h` (30%).
* **The audit that catches it in one line**: 100% of naive training rows took their
  customer features, and 98% their device features, from a row timestamped after the
  transaction. A PIT join can never do that; `tests/test_features.py` checks both.

## Label delay

`python -m rtml.experiments.offline labels` → [`results/labels.json`](results/labels.json)

Four ways to train the v2 model on day 63, all judged on days 63–76 with true labels.

| Training rows | Rows | Frauds labelled / true (train + valid) | Validation AUC-PR as measured | Future AUC-PR | Future precision / recall at its threshold | Future cost saved |
|---|---|---|---|---|---|---|
| Everything to day 62, labels known day 63 | 407,737 | 2,644 / 3,782 | **0.158** | 0.930 | 0.893 / **0.812** | 70.7% |
| Matured 14 days (**the default**) | 281,430 | 2,318 / 2,801 | 0.681 | 0.941 | 0.668 / 0.954 | **82.5%** |
| Matured 30 days | 126,348 | 1,359 / 1,603 | 0.728 | 0.937 | 0.576 / 0.961 | 81.4% |
| Everything, true labels (only a simulation has them) | 407,737 | 3,782 / 3,782 | 0.958 | 0.957 | 0.684 / 0.959 | 84.0% |

* Training on everything with the labels you have barely hurts the ranking (AUC-PR 0.930)
  but **ruins the threshold**: the validation week's frauds are mostly unlabelled, its
  measured AUC-PR is 0.158, and the cost-optimal threshold chosen on it flags too little —
  recall 0.81 and 11.8 points of cost saving lost against the default.
* Waiting 30 days instead of 14 throws away 55% of the rows for no gain.
* The ceiling — every label, instantly — is 1.5 points of cost saving above the default.

**The numbers you can see lag the truth.** Recall of the default model on one day of
traffic (day 63, 71 frauds), as chargebacks arrive:

| Days after | Frauds known | Share of true frauds | Recall as measured |
|---|---|---|---|
| 1 | 0 | 0% | — |
| 3 | 7 | 10% | 0.857 |
| 7 | 37 | 52% | 0.865 |
| 14 | 53 | 75% | 0.906 |
| 30 | 62 | 87% | 0.919 |
| 60 | 63 | 89% | 0.905 |

True recall that day was 0.930. A week later only half the frauds are known; 11% never
will be.

## Model quality and calibration

`python -m rtml.experiments.offline quality` → [`results/quality.json`](results/quality.json)

Days 63–76, true labels, threshold chosen on the validation window.

| | v2 (champion after the gate) | v1 | Hand-written rule |
|---|---|---|---|
| AUC-PR | **0.941** | 0.613 | — |
| ROC AUC | 0.997 | 0.976 | — |
| Recall at 80% precision | 0.912 | 0.416 | — |
| Threshold (chosen on validation / best on the future) | 0.042 / 0.037 | 0.069 / 0.071 | — |
| Precision at threshold | 0.668 | 0.390 | 0.933 |
| Recall at threshold | 0.954 | 0.704 | 0.365 |
| Share flagged for review | 1.20% | 1.51% | 0.33% |
| Cost per 1,000 transactions | $108.55 | $207.82 | — |
| Cost saved against approving everything | **82.5%** | 66.5% | 20.7% |
| Brier score | 0.0018 | 0.0049 | — |
| Expected calibration error (10 equal-count bins) | 0.0021 | 0.0022 | — |

The rule: amount over 4× the 7-day mean on a new device, or three or more cards on one
device within an hour.

**Recall by fraud type** (v2 / v1): card testing 0.990 / 0.681, account takeover
0.975 / 0.789, lone anomaly **0.764** / 0.602. The lone anomaly from the victim's own
device is the hardest case, as it should be.

**Calibration is good on average and wrong where it matters.** ECE is 0.002 because 99% of
scores are near zero and correct. In the top 1% of scores the model is consistently
under-confident:

| v2, top 1% of scores, five equal bins | Mean score | Actual fraud rate |
|---|---|---|
| lowest | 0.151 | 0.270 |
| | 0.422 | 0.667 |
| | 0.743 | 0.930 |
| | 0.847 | 0.993 |
| highest | 0.912 | 1.000 |

The model was trained on chargebacks, not fraud: 17% of frauds in its training window were
unreported, so it learned P(chargeback), which is lower than P(fraud). Any downstream use
of the score as a probability — expected-loss pricing, say — needs recalibrating on matured
labels.

## Skew

`python -m rtml.experiments.offline skew` → [`results/skew.json`](results/skew.json)

The window features are implemented twice (ADR 2): SQL window functions in DuckDB for the
backfill, a per-entity state machine for the stream. Days 63–69 replayed through the stream
code, warm-started with the 30 days before, compared row for row with the SQL output
(relative tolerance 1e-9).

| | Rows | Rows identical | Feature that differs | Future AUC-PR when served |
|---|---|---|---|---|
| Clean | 68,273 | **100.000%** | none of 12 | 0.938 |
| `SKEW_BUG=1` | 68,273 | **83.156%** | `dev_distinct_cards_1h`, 16.84% of rows | 0.938 |

* The planted defect — the distinct-card count reading the device's 24-hour buffer instead
  of the last hour — is caught at the feature level, in the one feature it touches.
* **The model-level check would have passed it.** v2 barely uses that feature, so AUC-PR,
  flag rate (1.13%) and cost saved (83.9%) are unchanged to three decimals. A parity check
  has to compare features, not scores; the next model might lean on that feature.
* The stream code replays 75,343 events/s on one core.

End-to-end parity on the running stack — the features the API actually served — is in
[the live section](#the-live-stack).

## Drift

`python -m rtml.experiments.drift_eval` → [`results/drift.json`](results/drift.json)

Ten datasets: seeds 1–5 with the drift switch thrown on day 70, and the same seeds without
it. A v2 model trained on each as usual; days 63–89 replayed through each monitor every 6
hours over the trailing 24 hours. A false alarm is an alert in a window with no drift in it
(before onset, or anywhere in a no-drift run): 650 such windows in all.

**What the drift costs**: recall on true labels fell from 0.886–0.962 in the week before
onset to 0.848–0.911 in the week after (−3 to −9 points across seeds).

| Monitor | Drift detected | Median time to detect | False alarms (of 650 clean windows) |
|---|---|---|---|
| Feature PSI > 0.2, all features | 4/5 | 414 h | 50 (7.7%) |
| Feature PSI > 0.2, calendar-trending features excluded | 0/5 | — | 0 |
| Feature PSI > 0.1, calendar-trending features excluded | 0/5 | — | 0 |
| Score PSI > 0.2 | 0/5 | — | 0 |
| Score PSI > 0.1 | 0/5 | — | 0 |
| Flag rate, \|z\| > 4 | 5/5 | **48 h** | 112 (**17.2%**) |
| Flag rate, \|z\| > 8 | 2/5 | 84 h | 3 (0.5%) |
| Recall on arrived chargebacks down 0.10 | 3/5 | 300 h | 6 (0.9%) |

* **Feature PSI did not see this drift.** Its four "detections" all came from
  `cust_chargebacks_90d`, which grows with the calendar as chargebacks accumulate, and
  which raised the same alarm in no-drift runs. Without the two calendar-trending features
  the largest PSI after onset was 0.05; the 15% price rise moved `amount_log` by 0.03–0.04.
  Fraud is 0.8% of traffic: when fraudsters change tactics the feature distributions barely
  move.
* **The score distribution did not move either** (max PSI 0.014), for the same reason.
* **The flag rate is the fastest label-free signal**, and at |z| > 4 it is noise: fraud
  arrives in bursts, the binomial variance is too small, and no-drift runs reached z = 11.
  At |z| > 8 it is quiet and misses three drifts in five.
* **Recall on arrived chargebacks** tracks what matters and is about 12.5 days late. One of
  its three detections (6 h after onset, seed 4) was an alarm already raised before onset;
  the two genuine ones came after 300 and 312 h.
* None of these monitors catches a tactics drift both quickly and quietly. That is
  reported as a limit, not tuned away.

## The promotion gate

`python -m rtml.experiments.offline gate` → [`results/gate.json`](results/gate.json)

Champion: v1. Each challenger scored the same 68,016 transactions of the shadow days 77–83;
labels are the chargebacks known 30 days after the window (474 frauds). Rule (ADR 5):
promote only if the mean cost per transaction is lower with paired-bootstrap p < 0.05
(1,000 resamples), decisions are not significantly worse by exact McNemar, and the p99 of
scoring one transaction on the serving path is within 2 ms (3,000 calls, collector paused).

| Challenger | Cost per transaction, champion → challenger | Saving (95% CI) | p (bootstrap) | McNemar p (improved / regressed) | p99 | Decision |
|---|---|---|---|---|---|---|
| v2: adds device features | $0.208 → $0.099 | **$0.109** (0.085 to 0.137) | < 0.001 | 7e-52 (703 / 245) | 0.03 ms | **promote** |
| v1 retrained, another seed | $0.208 → $0.205 | $0.004 (−0.003 to 0.011) | 0.346 | 0.18 (102 / 123) | 0.10 ms | **refuse**: not significantly lower |
| v1 underfit: 4 leaves, 30 rounds | $0.208 → $0.297 | −$0.089 (−0.129 to −0.057) | < 0.001 | 0.40 (327 / 305) | 0.02 ms | **refuse**: significantly higher |
| v2 oversized: 255 leaves, 2,000 rounds | $0.208 → $0.103 | $0.105 (0.075 to 0.137) | < 0.001 | 1e-70 (718 / 196) | 0.74 ms | promote |
| …the same, against v2 as champion | $0.099 → $0.103 | −$0.005 (−0.020 to 0.006) | 0.636 | 1e-7 (106 / 42) | 0.74 ms | **refuse**: not significantly lower |

* The gate compares a challenger with the champion it would replace. Against v1 the
  oversized model passes; against v2, the model it would actually have to beat, it is
  refused. It is 25× slower than v2 and still inside the 2 ms budget: no challenger here
  breached the latency rule, which is exercised by unit tests only.
* **Deciding early understates the gain.** The same v2-against-v1 decision with the labels
  available 0, 3, 7, 14 and 30 days after the shadow window: 113, 210, 334, 433 and 474
  frauds known; measured saving $0.040, $0.064, $0.087, $0.102 and $0.109 per transaction.
  It was significant from day 0 here because the improvement is large; a smaller one
  would need the wait.

## The live stack

`make up train serve demo` → [`results/live.json`](results/live.json). A fresh stack: v1
as champion, v2 as challenger, the online store loaded as of midnight on day 77. The
dataset for the stack has the drift switch on day 78 with prices up 50% (instead of 15%),
so the drift alert has something visible to find.

The replayer sends each transaction to `POST /score`, then publishes it to Kafka from a
separate thread — authorise, then publish. Requests are open-loop: sent on schedule
whether or not earlier ones have returned. Day 77 is replayed from midnight: 500 requests
of warm-up, then 2,000 at 25/s, 3,000 at 50/s and 4,000 at 100/s, so a day of traffic
passes in about seven minutes.

### Latency

*Server* is the API's own timer around the feature fetch and the model, as returned in the
response. *Client* is the replayer's timer around the HTTP call: the server time plus HTTP,
the event loops on both sides, and the network between two containers on the same VM.

| Load | Requests | Errors | Server p50 / p95 / p99 | Client p50 / p95 / p99 | Client max |
|---|---|---|---|---|---|
| warm-up, 25/s | 500 | 0 | 0.76 / 2.13 / 22.40 ms | 2.21 / 13.83 / 132.52 ms | 582 ms |
| 25/s | 2,000 | 0 | **0.73 / 1.90 / 3.94 ms** | 2.04 / 6.81 / 56.61 ms | 286 ms |
| 50/s | 3,000 | 0 | **0.63 / 1.64 / 4.33 ms** | 1.77 / 5.70 / 33.88 ms | 178 ms |
| 100/s | 4,000 | 0 | **0.54 / 1.69 / 3.53 ms** | 1.61 / 19.84 / 256.91 ms | 693 ms |

From the prediction log, for the same 9,500 requests: the Feast online read (three
feature views, fifteen features) took 0.49 ms at the median and 3.43 ms at p99; v1's
request features plus prediction 0.14 / 0.83 ms; v2's, in shadow, 0.10 / 1.23 ms.

* Scoring is sub-millisecond at the median and under 5 ms at p99 at every rate. The median
  falls as load rises (0.73 → 0.54 ms): busier is warmer.
* **The client-side tail is not explained.** At 100/s the client p99 is 257 ms while the
  server's is 3.5 ms. Moving the Kafka producer off the replay's event loop removed the
  largest cause (it had put the client p99 at 826 ms against 57 ms without it); what is
  left sits between the two timers. See Limits in the README.
* The first live runs were far worse (85 ms server p99 at 25/s); the failure log in the
  README says why and what changed.

### End-to-end parity

The features the API actually served, from the prediction log, against Feast's
point-in-time join for the same transactions:

| | Rows | Served features identical to the PIT join |
|---|---|---|
| All | 9,500 | **99.905%** (9 rows differ) |
| Legitimate | 9,380 | 99.957% (4) |
| Fraud | 120 | **95.833%** (5) |
| — card testing | 89 | 94.382% (5) |
| — account takeover, lone anomaly | 31 | 100% |

Every disagreement is stream lag. A day replayed in seven minutes compresses time about
200-fold; a card-testing burst's attempts, minutes apart in event time, arrive a few
hundred milliseconds apart — sometimes before the previous attempt's event has reached the
online store, so the next score sees one attempt fewer. The rows that disagree are the
ones the device features exist for. At real speed the gap would be far smaller; it is
measured here at replay speed and not at real speed.

### The gate, live

| Comparison | Paired transactions | Frauds with chargebacks (30 days on) | Cost per transaction, champion → challenger | Saving (95% CI) | p | Decision |
|---|---|---|---|---|---|---|
| v2 against v1 | 9,500 | 110 | $0.277 → $0.140 | **$0.137** (0.075 to 0.210) | < 0.001; McNemar 1e-14 (101 / 19) | **promote**; the API served v2 6.1 s later |
| v2 retrained with another seed, against v2 | 5,288 | 31 | $0.160 → $0.166 | −$0.006 (−0.013 to −0.001) | 0.036 | **refuse**: significantly higher |

The latency check benchmarks the challenger on the serving path in the gate's own process:
p99 0.04 ms for v2. The reseeded twin came out significantly worse at p = 0.036 — the kind
of result a 5% test produces about one time in twenty. It would have been refused as "not
significantly lower" either way.

### The drift alert

Replaying on past midnight into day 78 (prices up 50%), the `FeatureDrift` rule —
hour-matched PSI over 0.2 on a behavioural feature for 15 s, over the champion's last
3,000 transactions — **fired 49 s after the first drifted transaction**, about 4,900
drifted transactions in. It did not fire on day 77. Most of the 49 s is waiting for the
window to fill: the monitor runs every 10 s, Prometheus evaluates every 5 s and holds the
rule for 15 s.

Earlier runs fired early: before the hour matching, the alert went off on day 77 traffic,
first on hour of day (PSI 5.1) and then on time since the last purchase (PSI 0.71).

## How to reproduce

```bash
make experiments      # everything above the live section, about 10 minutes
make up train serve   # the stack: build, data, online store, models, API, processor, monitor
make demo             # the live section, about 12 minutes; writes results/live.json
make down
```

Seeds are fixed; the offline numbers reproduce exactly on the same library versions
except the latency columns, which depend on the machine.
