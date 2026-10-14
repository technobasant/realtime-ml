"""The end-to-end run behind ``make demo``, recorded to ``results/live.json``.

1. Replay the start of the shadow day at three request rates; the champion (v1) answers,
   the challenger (v2) is scored in shadow. Latency per level.
2. End-to-end parity: the features the API served against a point-in-time join.
3. Gate v2 against v1 on the shadow log with matured labels; promote if it passes.
4. Register an equal challenger (v2 retrained with another seed) and replay into the
   drifted day; record when the drift alert fires in Prometheus.
5. Gate the equal challenger; it should be refused.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any

import httpx
import pandas as pd
import psycopg

from rtml import timeline as tl
from rtml.config import DATA, MLFLOW_TRACKING_URI, PG_DSN, RESULTS
from rtml.dataset import day_ts, load
from rtml.experiments.common import environment
from rtml.gate import LATENCY_BUDGET_MS
from rtml.serve.live import aliases, parity, run_live_gate, shadow_latency
from rtml.serve.replay import Level, replay
from rtml.train import Spec, fit, pit_frames, register

API = os.environ.get("RTML_API", "http://localhost:8800")
PROM = os.environ.get("RTML_PROMETHEUS", "http://localhost:9095")
LEVELS = [Level(25, 2000), Level(50, 3000), Level(100, 4000)]


def max_id() -> int:
    with psycopg.connect(PG_DSN) as conn:
        row = conn.execute("SELECT coalesce(max(id), 0) FROM predictions").fetchone()
    return int(row[0]) if row else 0


def wait_for(pred: Any, timeout_s: float = 120.0, every: float = 2.0) -> float:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if pred():
            return time.monotonic() - t0
        time.sleep(every)
    raise TimeoutError("condition not met")


def api_models() -> dict[str, Any]:
    models: dict[str, Any] = httpx.get(f"{API}/healthz", timeout=5).json()["models"]
    return models


def firing(name: str) -> bool:
    alerts = httpx.get(f"{PROM}/api/v1/alerts", timeout=5).json()["data"]["alerts"]
    return any(a["labels"].get("alertname") == name and a["state"] == "firing" for a in alerts)


async def watch_alert(name: str, out: dict[str, Any], stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            if firing(name) and "alert_at" not in out:
                out["alert_at"] = time.time()
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1.0)


async def drift_phase(tx: pd.DataFrame, start: pd.Timestamp, offset: int, n: int) -> dict[str, Any]:
    info: dict[str, Any] = {}
    stop = asyncio.Event()
    watcher = asyncio.create_task(watch_alert("FeatureDrift", info, stop))
    t0 = time.time()
    res, _ = await replay(tx, [Level(100, n)], API, start, offset)
    await asyncio.sleep(40)  # monitor interval + Prometheus `for:` window
    stop.set()
    await watcher
    info["replay"] = res[0]
    info["replay_started_at"] = t0
    return info


def quick() -> None:
    """The CI smoke run: a short replay, parity and a gate decision, with invariants checked."""
    tx, cfg = load(DATA)
    first = max_id()
    levels, _ = asyncio.run(replay(tx, [Level(20, 300), Level(50, 500)], API, day_ts(cfg, tl.SHADOW[0])))
    time.sleep(5)
    par = parity(first)
    dec = run_live_gate(label_delay_days=30, latency_budget_ms=LATENCY_BUDGET_MS, apply=False)
    sent = sum(lv["sent"] for lv in levels)
    errors = sum(lv["errors"] for lv in levels)
    print(json.dumps({"latency": levels, "parity": par, "gate": dec.reason}, indent=2, default=str))
    assert errors == 0, f"{errors} failed requests"
    assert par["rows"] == sent, f"logged {par['rows']} of {sent}"
    assert dec.n_paired == sent, f"shadow scored {dec.n_paired} of {sent}"
    assert par["row_parity_pct"] > 90, "served features disagree with the point-in-time join"


def main() -> None:
    if "--quick" in sys.argv:
        quick()
        return
    tx, cfg = load(DATA)
    out: dict[str, Any] = {"environment": environment(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    start = day_ts(cfg, tl.SHADOW[0])
    out["models_before"] = aliases()
    first = max_id()

    # 1. latency at three request rates, shadow scoring on
    levels, offset = asyncio.run(replay(tx, LEVELS, API, start))
    out["latency"] = levels
    time.sleep(5)

    # 2. end-to-end parity of served features
    out["parity"] = parity(first)

    # 3. gate v2 against v1, promote on a pass
    d1 = run_live_gate(label_delay_days=30, latency_budget_ms=LATENCY_BUDGET_MS, apply=True)
    out["gate_better"] = asdict(d1)
    if d1.promote:
        new = aliases()["champion"]
        out["promotion_pickup_s"] = round(wait_for(lambda: api_models()["champion"] == new), 1)

    # 4. an equal challenger, then the drifted day
    _, (train, valid) = pit_frames(DATA, tl.TRAIN, tl.VALID)
    m = fit(Spec("v2-seed1", "v2", seed=1), train, valid, cfg)
    version = register(m, "challenger", MLFLOW_TRACKING_URI)
    wait_for(lambda: api_models()["challenger"] == version)
    drift_day = cfg.drift_day if cfg.drift_day is not None else tl.SHADOW[0] + 1
    rows_left = int(((tx.ts >= start) & (tx.ts < day_ts(cfg, drift_day))).sum()) - offset
    first_drift = int(((tx.ts >= start) & (tx.ts < day_ts(cfg, drift_day))).sum())
    info = asyncio.run(drift_phase(tx, start, offset, rows_left + 5000))
    # The replay splits at midnight; the second part starts with the first drifted transaction.
    starts = info["replay"].get("parts_started_at") or [info["replay_started_at"] + (first_drift - offset) / 100.0]
    drift_sent_at = starts[-1]
    out["drift"] = {
        "drift_day": cfg.drift_day,
        "inflation": cfg.drift_inflation,
        "alert": "FeatureDrift",
        "fired": "alert_at" in info,
        "seconds_after_first_drifted_txn": round(info["alert_at"] - drift_sent_at, 1) if "alert_at" in info else None,
        "replay": info["replay"],
    }

    # 5. gate the equal challenger
    d2 = run_live_gate(label_delay_days=30, latency_budget_ms=LATENCY_BUDGET_MS, apply=True)
    out["gate_equal"] = asdict(d2)
    out["models_after"] = aliases()
    out["shadow_log_latency"] = shadow_latency()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "live.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("latency", "parity", "drift")}, indent=2, default=str))
    print("gate (v2 vs v1):", d1.reason)
    print("gate (v2 seed 1 vs v2):", d2.reason)


if __name__ == "__main__":
    main()
