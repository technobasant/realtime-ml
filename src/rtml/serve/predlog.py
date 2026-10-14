"""The prediction log: every score the API produces, champion and shadow, in Postgres.

Writes are queued and flushed in batches by one thread, so logging never sits on the
request path. The features the model saw are stored with each row; the end-to-end
parity check and the drift monitor both read them back.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id           bigserial PRIMARY KEY,
  txn_id       text        NOT NULL,
  event_ts     timestamptz NOT NULL,
  scored_at    timestamptz NOT NULL DEFAULT now(),
  role         text        NOT NULL,   -- champion | challenger
  version      text        NOT NULL,
  score        double precision NOT NULL,
  flagged      boolean     NOT NULL,
  threshold    double precision NOT NULL,
  model_ms     double precision NOT NULL,
  fetch_ms     double precision NOT NULL,
  features     jsonb       NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_txn ON predictions (txn_id);
CREATE INDEX IF NOT EXISTS predictions_role_id ON predictions (role, id);
"""


@dataclass(frozen=True)
class Prediction:
    txn_id: str
    event_ts: datetime
    role: str
    version: str
    score: float
    flagged: bool
    threshold: float
    model_ms: float
    fetch_ms: float
    features: dict[str, Any]


def init(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(SCHEMA)


class Writer:
    def __init__(self, dsn: str, batch: int = 500, flush_s: float = 0.2) -> None:
        self.dsn = dsn
        self.batch = batch
        self.flush_s = flush_s
        self.q: queue.Queue[Prediction] = queue.Queue(maxsize=100_000)
        self.dropped = 0
        self.written = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="predlog")

    def start(self) -> None:
        init(self.dsn)
        self._thread.start()

    def put(self, p: Prediction) -> None:
        try:
            self.q.put_nowait(p)
        except queue.Full:
            self.dropped += 1

    def _drain(self) -> list[Prediction]:
        items: list[Prediction] = []
        try:
            items.append(self.q.get(timeout=self.flush_s))
            while len(items) < self.batch:
                items.append(self.q.get_nowait())
        except queue.Empty:
            pass
        return items

    def _run(self) -> None:
        conn = psycopg.connect(self.dsn, autocommit=True)
        while not (self._stop.is_set() and self.q.empty()):
            items = self._drain()
            if not items:
                continue
            try:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO predictions (txn_id, event_ts, role, version, score, flagged, threshold,"
                        " model_ms, fetch_ms, features) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        [
                            (
                                p.txn_id,
                                p.event_ts,
                                p.role,
                                p.version,
                                p.score,
                                p.flagged,
                                p.threshold,
                                p.model_ms,
                                p.fetch_ms,
                                json.dumps(p.features),
                            )
                            for p in items
                        ],
                    )
                self.written += len(items)
            except psycopg.Error:
                log.exception("prediction log write failed; %d rows lost", len(items))
                self.dropped += len(items)
                conn = psycopg.connect(self.dsn, autocommit=True)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
