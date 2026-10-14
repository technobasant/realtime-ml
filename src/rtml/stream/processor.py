"""The stream processor: transactions from Kafka in, window features into the online store.

On start it rebuilds its window state from the last 30 days of the event log, so the
first value it writes is already right; a processor that starts empty writes counts of
zero for every entity until a month has passed — a skew no test of the stream alone
would show. Then it consumes the topic, updates the state one event at a time, and
pushes each poll's rows to Feast in one batch per feature view.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from typing import Any

import pandas as pd
from confluent_kafka import Consumer, KafkaError
from feast.data_source import PushMode
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from rtml.config import DATA, KAFKA_BOOTSTRAP, TXN_TOPIC
from rtml.dataset import day_ts, load
from rtml.features.online import OnlineFeatures, Txn, txns_from_frame
from rtml.store import get_store

log = logging.getLogger("rtml.processor")

EVENTS = Counter("rtml_stream_events_total", "Transactions consumed")
PUSH_S = Histogram(
    "rtml_stream_push_seconds", "Feast push per batch", buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1)
)
LAG_S = Histogram(
    "rtml_stream_lag_seconds",
    "From the event being produced to its features being readable online",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)
LAST_EVENT = Gauge("rtml_stream_last_event_timestamp", "Event time of the newest transaction applied")


def warm_start(state: OnlineFeatures, start: pd.Timestamp, data: Any = DATA) -> int:
    tx = pd.read_parquet(data / "transactions.parquet")
    hist = tx[(tx.ts >= start - pd.Timedelta(days=30)) & (tx.ts < start)]
    for t in txns_from_frame(hist):
        state.update(t)
    return len(hist)


def run(start: pd.Timestamp, group: str, metrics_port: int) -> None:
    start_http_server(metrics_port)
    store = get_store(apply=False)
    state = OnlineFeatures()
    n = warm_start(state, start)
    log.info("warm start: %d events before %s (skew_bug=%s)", n, start, state.skew_bug)

    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([TXN_TOPIC])
    stopping = False

    def stop(*_: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        msgs = consumer.consume(num_messages=500, timeout=0.05)
        if not msgs:
            continue
        cust, dev, cd, produced = [], [], [], []
        for m in msgs:
            err = m.error()
            if err is not None:
                if err.code() != KafkaError._PARTITION_EOF:
                    log.error("kafka: %s", err)
                continue
            rec = json.loads(m.value() or b"{}")
            rows = state.update(Txn.from_record(rec))
            cust.append(rows.customer_velocity)
            dev.append(rows.device_velocity)
            cd.append(rows.customer_device)
            produced.append(m.timestamp()[1] / 1000)
        if not cust:
            continue
        t0 = time.perf_counter()
        store.push("customer_velocity_push", pd.DataFrame(cust), to=PushMode.ONLINE)
        store.push("device_velocity_push", pd.DataFrame(dev), to=PushMode.ONLINE)
        store.push("customer_device_push", pd.DataFrame(cd), to=PushMode.ONLINE)
        PUSH_S.observe(time.perf_counter() - t0)
        now = time.time()
        for p in produced:
            LAG_S.observe(max(0.0, now - p))
        EVENTS.inc(len(cust))
        LAST_EVENT.set(state.last_us / 1e6)
        consumer.commit(asynchronous=False)
    consumer.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--day", type=int, default=int(os.environ.get("RTML_REPLAY_DAY", "77")), help="replay start day")
    p.add_argument("--group", default="rtml-processor")
    p.add_argument("--metrics-port", type=int, default=9101)
    a = p.parse_args()
    _, cfg = load(DATA)
    run(day_ts(cfg, a.day), a.group, a.metrics_port)


if __name__ == "__main__":
    main()
