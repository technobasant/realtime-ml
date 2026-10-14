"""Replay generated transactions against the running stack, at a fixed request rate.

Each transaction is scored by the API and then produced to Kafka — the order a payment
system works in: authorise, then publish. The processor folds the event into the online
store, ready for the entity's next transaction. Replaying a day in minutes squeezes the
gap between an entity's transactions by the same factor, which is what the end-to-end
parity check measures.

Requests are sent open-loop: on schedule, whether or not earlier ones have returned, so
a slow server shows up as latency rather than as a lower request rate.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

import httpx
import numpy as np
import pandas as pd
from confluent_kafka import Producer

from rtml.config import KAFKA_BOOTSTRAP, TXN_TOPIC
from rtml.dataset import RAW_COLS
from rtml.store import get_store


class Publisher:
    """Produces to Kafka from its own thread, in the order transactions were scored.

    Calling the Kafka producer from the replay's event loop put up to a second of
    stalls into the *client's* latency figures (p99 826 ms against 57 ms without it)
    while the server's own timings did not move: the first numbers measured the load
    generator.
    """

    def __init__(self) -> None:
        self.q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP, "linger.ms": 5})
        self.thread = threading.Thread(target=self._run, daemon=True, name="publisher")
        self.thread.start()

    def _run(self) -> None:
        while (rec := self.q.get()) is not None:
            self.producer.produce(TXN_TOPIC, key=rec["customer_id"], value=json.dumps(rec))
            self.producer.poll(0)
        self.producer.flush(30)

    def put(self, rec: dict[str, Any]) -> None:
        self.q.put(rec)

    def close(self) -> None:
        self.q.put(None)
        self.thread.join()


@dataclass
class Level:
    rps: float
    n: int


def record(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: row[k] for k in RAW_COLS}
    out["ts"] = pd.Timestamp(row["ts"]).isoformat()
    out["amount"] = float(out["amount"])
    return out


def materialize_profile(day_start: pd.Timestamp) -> float:
    """The nightly batch job: load the new day's customer profile snapshot online."""
    t0 = time.perf_counter()
    get_store(apply=False).materialize(
        start_date=day_start - pd.Timedelta(hours=1), end_date=day_start, feature_views=["customer_profile"]
    )
    return time.perf_counter() - t0


async def _run_level(
    client: httpx.AsyncClient, publisher: Publisher, rows: list[dict[str, Any]], rps: float, api: str
) -> dict[str, Any]:
    results: list[tuple[str, float, float, int]] = []
    sem = asyncio.Semaphore(256)

    async def one(rec: dict[str, Any]) -> None:
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.post(f"{api}/score", json=rec)
                status = r.status_code
                server_ms = float(r.json().get("latency_ms", np.nan)) if status == 200 else np.nan
            except httpx.HTTPError:
                status, server_ms = 0, np.nan
            client_ms = (time.perf_counter() - t0) * 1000
            publisher.put(rec)  # authorise, then publish
            results.append((rec["txn_id"], client_ms, server_ms, status))

    started_at = time.time()
    start = time.perf_counter()
    tasks = []
    for i, rec in enumerate(rows):
        delay = start + i / rps - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(rec)))
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - start
    ok = [x for x in results if x[3] == 200]
    client_ms = np.array([x[1] for x in ok])
    server_ms = np.array([x[2] for x in ok])

    def pct(a: np.ndarray) -> dict[str, float]:
        if a.size == 0:
            return {}
        return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in (50, 95, 99)} | {"max": round(float(a.max()), 2)}

    return {
        "started_at": started_at,
        "target_rps": rps,
        "sent": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "achieved_rps": round(len(rows) / wall, 1),
        "client_ms": pct(client_ms),
        "server_ms": pct(server_ms),
        "first_txn": rows[0]["txn_id"],
        "last_txn": rows[-1]["txn_id"],
    }


async def replay(
    tx: pd.DataFrame, levels: list[Level], api: str, start: pd.Timestamp, offset: int = 0
) -> tuple[list[dict[str, Any]], int]:
    """Replay ``tx`` from ``start``; returns per-level results and the next offset."""
    rows = cast(list[dict[str, Any]], tx[tx.ts >= start].sort_values("ts").to_dict(orient="records"))
    publisher = Publisher()
    out = []
    current_day = pd.Timestamp(rows[offset]["ts"]).normalize()
    async with httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_connections=64)) as client:
        for lv in levels:
            batch = rows[offset : offset + lv.n]
            # Split the level at midnight: the profile snapshot changes there.
            parts: list[list[dict[str, Any]]] = [[]]
            for row in batch:
                day = pd.Timestamp(row["ts"]).normalize()
                if day != current_day:
                    parts.append([])
                    current_day = day
                parts[-1].append(row)
            level_parts = []
            for i, part in enumerate(parts):
                if i > 0:
                    await asyncio.sleep(1.0)  # let the stream catch up before the batch job runs
                    materialize_profile(pd.Timestamp(part[0]["ts"]).normalize())
                if part:
                    level_parts.append(await _run_level(client, publisher, [record(r) for r in part], lv.rps, api))
            out.append(_merge(level_parts) if len(level_parts) > 1 else level_parts[0])
            offset += lv.n
            await asyncio.sleep(2.0)
    publisher.close()
    return out, offset


def _merge(parts: list[dict[str, Any]]) -> dict[str, Any]:
    # Percentiles do not merge; keep the largest part's and say so.
    main = max(parts, key=lambda p: p["sent"])
    return {
        **main,
        "sent": sum(p["sent"] for p in parts),
        "ok": sum(p["ok"] for p in parts),
        "errors": sum(p["errors"] for p in parts),
        "parts_started_at": [p["started_at"] for p in parts],
        "parts_sent": [p["sent"] for p in parts],
        "note": "split at midnight; percentiles from the larger part",
    }
