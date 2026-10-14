"""The scoring API: fetch features from Feast, score with the champion, shadow the challenger.

The challenger's score is computed after the response is sent and only ever written to
the prediction log. The response carries the champion's decision and nothing else.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from rtml.config import PG_DSN
from rtml.serve.models import Model, Registry
from rtml.serve.predlog import Prediction, Writer
from rtml.store import FEATURE_REFS, get_store

log = logging.getLogger("rtml.api")

BUCKETS = (0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0)
REQUEST_S = Histogram("rtml_request_seconds", "Scoring time inside the API, fetch included", buckets=BUCKETS)
FETCH_S = Histogram("rtml_feature_fetch_seconds", "Feast online read", buckets=BUCKETS)
MODEL_S = Histogram("rtml_model_seconds", "Request features plus predict", ["role"], buckets=BUCKETS)
SCORES = Histogram("rtml_score", "Model scores", ["role"], buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1))
FLAGS = Counter("rtml_flagged_total", "Transactions at or over the threshold", ["role"])
SCORED = Counter("rtml_scored_total", "Transactions scored", ["role"])
MISSING = Counter("rtml_missing_features_total", "Stored features absent online (new entity)", ["feature"])


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
store: Any = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global store
    store = get_store(apply=False)
    registry.wait_ready()
    registry.start()
    writer.start()
    yield
    registry.stop()
    writer.stop()


app = FastAPI(title="realtime-ml", lifespan=lifespan)


def fetch(txn: Txn) -> dict[str, Any]:
    names = [r.split(":")[1] for r in FEATURE_REFS]
    got = store.get_online_features(
        features=FEATURE_REFS, entity_rows=[{"customer_id": txn.customer_id, "device_id": txn.device_id}]
    ).to_dict()
    out = {n: got[n][0] for n in names}
    for n, v in out.items():
        if v is None:
            MISSING.labels(n).inc()
    return out


def _clean(features: dict[str, Any]) -> dict[str, Any]:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in features.items()}


def shadow(model: Model, frame: pd.DataFrame, txn: Txn, stored: dict[str, Any], fetch_ms: float) -> None:
    t0 = time.perf_counter()
    score = float(model.score(frame)[0])
    ms = (time.perf_counter() - t0) * 1000
    MODEL_S.labels("challenger").observe(ms / 1000)
    SCORES.labels("challenger").observe(score)
    SCORED.labels("challenger").inc()
    flagged = score >= model.threshold
    if flagged:
        FLAGS.labels("challenger").inc()
    writer.put(
        Prediction(
            txn.txn_id, txn.ts, "challenger", model.version, score, flagged, model.threshold, ms, fetch_ms, stored
        )
    )


@app.post("/score", response_model=Decision)
def score(txn: Txn, background: BackgroundTasks) -> Decision:
    champion = registry.models["champion"]
    if champion is None:
        raise HTTPException(503, "no champion model")
    t0 = time.perf_counter()
    stored = fetch(txn)
    t1 = time.perf_counter()
    frame = pd.DataFrame([{**txn.model_dump(), **stored}])
    s = float(champion.score(frame)[0])
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
    logged = {**txn.model_dump(mode="json"), **_clean(stored)}
    writer.put(
        Prediction(
            txn.txn_id, txn.ts, "champion", champion.version, s, flagged, champion.threshold, model_ms, fetch_ms, logged
        )
    )
    challenger = registry.models["challenger"]
    if challenger is not None and challenger.version != champion.version:
        background.add_task(shadow, challenger, frame, txn, logged, fetch_ms)
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
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
