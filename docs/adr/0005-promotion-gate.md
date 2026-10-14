# 5. Promote only on a significant paired improvement within the latency budget

Status: accepted · 2026-10

## Context

A challenger scored in shadow sees exactly the traffic the champion sees, so each
transaction is a matched pair. Comparing two aggregate numbers throws that pairing
away and needs far more traffic to tell two models apart. A gate that promotes
whenever the challenger's number is higher will promote noise about half the time.

## Decision

* The challenger is scored after the response is sent. Its score goes to the prediction
  log and nowhere else.
* The comparison is evalproof 0.1.1's `compare_runs`, over the paired transactions:
  a paired bootstrap on the per-transaction cost, and an exact McNemar test on
  correct-at-threshold.
* Promote only if all three hold:
  1. the challenger's mean cost is lower with paired-bootstrap p < 0.05;
  2. its decisions are not significantly worse (McNemar);
  3. its p99 scoring latency — request features plus predict — is within 2 ms, the
     model's share of a 20 ms scoring SLO.
* evalproof's own `gate.decide` checks for regressions; promotion needs evidence of an
  improvement, so this rule is written here on top of its statistics.
* Promotion moves the MLflow `champion` alias and removes `challenger`; the API follows
  aliases and picks up the change within its 10 s poll.

## Consequences

* "No significant difference" is a refusal. A retrained copy of the champion is never
  promoted, however lucky its sample.
* Labels gate the gate: the decision waits for chargebacks. MEASUREMENTS shows how the
  measured saving grows as labels arrive.
