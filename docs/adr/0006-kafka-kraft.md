# 6. Apache Kafka in KRaft mode, one broker

Status: accepted · 2026-10

## Context

Transactions travel from the payment flow to the stream processor through a log. The
guarantees this repository measures are about features, not about the log, so one
broker is enough; the question was which.

## Decision

Apache Kafka 4.3.1 (`apache/kafka`), one combined broker and controller, 512 MB heap,
one partition keyed by customer. The same image the streaming-platform repository runs
as a three-node cluster.

## Why not Redpanda

Lighter, and protocol-compatible. It would have been a fine choice here; Kafka won on
consistency with the rest of the portfolio and on licence (Redpanda's core is BSL).

## Consequences

One partition keeps a total order, so the processor applies events in the order they
were produced. Scaling out would need a partition key that keeps every entity's events
together — customer, device and the customer-device pair cannot all share one key —
which is a real design problem left for the roadmap.
