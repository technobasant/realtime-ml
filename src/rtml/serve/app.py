"""The scoring API: fetch features from Feast, score with the champion, shadow the challenger.

The challenger's score is computed on a separate thread after the response is built and
only ever written to the prediction log. The response carries the champion's decision
and nothing else.

The endpoint is ``async`` and does its few milliseconds of work inline. Run in the
thread pool instead, overlapping requests fought over the GIL and the p99 at 25
requests/s was 85 ms for 4 ms of work; see the failure log in the README.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import queue
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from rtml.config import PG_DSN
from rtml.serve.models import Registry
from rtml.serve.predlog import Prediction, Writer
from rtml.store import FEATURE_REFS, get_store

log = logging.getLogger("rtml.api")

BUCKETS = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0)
REQUEST_S = Histogram("rtml_request_seconds", "Scoring time inside the API, fetch included", buckets=BUCKETS)
FETCH_S = Histogram("rtml_feature_fetch_seconds", "Feast online read", buckets=BUCKETS)
MODEL_S = Histogram("rtml_model_seconds", "Request features plus predict", ["role"], buckets=BUCKETS)
SCORES = Histogram("rtml_score", "Model scores", ["role"], buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1))
FLAGS = Counter("rtml_flagged_total", "Transactions at or over the threshold", ["role"])
SCORED = Counter("rtml_scored_total", "Transactions scored", ["role"])
MISSING = Counter("rtml_missing_features_total", "Stored features absent online (new entity)", ["feature"])
SHADOW_BACKLOG = Gauge("rtml_shadow_backlog", "Transactions waiting for the challenger")
SHADOW_DROPPED = Counter("rtml_shadow_dropped_total", "Shadow work dropped because the queue was full")

FEATURE_NAMES = [r.split(":")[1] for r in FEATURE_REFS]


class Txn(BaseModel):
    txn_id: str
    ts: datetime
    customer_id: str
    card_id: str
    device_id: str
    merchant_category: str
    amount: float
    country: str


class Decision(BaseModel):
    txn_id: str
    score: float
    flagged: bool
    model_version: str
    latency_ms: float


registry = Registry(poll_s=float(os.environ.get("RTML_MODEL_POLL_S", "10")))
writer = Writer(os.environ.get("RTML_PG_DSN", PG_DSN))
shadow_q: queue.Queue[tuple[dict[str, Any], datetime, str, float]] = queue.Queue(maxsize=10_000)
store: Any = None


def _shadow_loop() -> None:
    while True:
        rec, ts, champion_version, fetch_ms = shadow_q.get()
        SHADOW_BACKLOG.set(shadow_q.qsize())
        challenger = registry.models["challenger"]
        if challenger is None or challenger.version == champion_version:
            continue
        t0 = time.perf_counter()
        s = challenger.score_one(rec)
        ms = (time.perf_counter() - t0) * 1000
        MODEL_S.labels("challenger").observe(ms / 1000)
        SCORES.labels("challenger").observe(s)
        SCORED.labels("challenger").inc()
        flagged = s >= challenger.threshold
        if flagged:
            FLAGS.labels("challenger").inc()
        writer.put(
            Prediction(
                rec["txn_id"], ts, "challenger", challenger.version, s, flagged, challenger.threshold, ms, fetch_ms, rec
            )
        )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global store
    store = get_store(apply=False)
    registry.wait_ready()
    registry.start()
    writer.start()
    threading.Thread(target=_shadow_loop, daemon=True, name="shadow").start()
    # Everything loaded so far (Feast, MLflow, pandas, both models) lives for the life of
    # the process; keep the cyclic collector from walking it on every full collection.
    gc.collect()
    gc.freeze()
    yield
    registry.stop()
    writer.stop()


app = FastAPI(title="realtime-ml", lifespan=lifespan)


def fetch(txn: Txn) -> dict[str, Any]:
    got = store.get_online_features(
        features=FEATURE_REFS, entity_rows=[{"customer_id": txn.customer_id, "device_id": txn.device_id}]
    ).to_dict()
    out: dict[str, Any] = {}
    for n in FEATURE_NAMES:
        v = got[n][0]
        if v is None:
            MISSING.labels(n).inc()
        elif isinstance(v, float) and math.isnan(v):
            v = None
        out[n] = v
    return out


@app.post("/score", response_model=Decision)
async def score(txn: Txn) -> Decision:
    champion = registry.models["champion"]
    if champion is None:
        raise HTTPException(503, "no champion model")
    t0 = time.perf_counter()
    stored = fetch(txn)
    t1 = time.perf_counter()
    rec = {**txn.model_dump(mode="json"), **stored}
    s = champion.score_one(rec)
    t2 = time.perf_counter()
    flagged = s >= champion.threshold
    fetch_ms, model_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
    FETCH_S.observe(t1 - t0)
    MODEL_S.labels("champion").observe(t2 - t1)
    REQUEST_S.observe(t2 - t0)
    SCORES.labels("champion").observe(s)
    SCORED.labels("champion").inc()
    if flagged:
        FLAGS.labels("champion").inc()
    writer.put(
        Prediction(
            txn.txn_id, txn.ts, "champion", champion.version, s, flagged, champion.threshold, model_ms, fetch_ms, rec
        )
    )
    try:
        shadow_q.put_nowait((rec, txn.ts, champion.version, fetch_ms))
    except queue.Full:
        SHADOW_DROPPED.inc()
    return Decision(
        txn_id=txn.txn_id, score=s, flagged=flagged, model_version=champion.version, latency_ms=(t2 - t0) * 1000
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "models": {k: (m.version if m else None) for k, m in registry.models.items()},
        "log_written": writer.written,
        "log_dropped": writer.dropped,
        "log_queue": writer.q.qsize(),
        "shadow_backlog": shadow_q.qsize(),
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
