"""Checks that read the running stack's prediction log: the gate and end-to-end parity."""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import psycopg

from rtml.config import DATA, MLFLOW_TRACKING_URI, MODEL_NAME, PG_DSN
from rtml.dataset import labels_as_of, pit_join
from rtml.gate import Decision, Scored, decide
from rtml.store import FEATURE_REFS, get_store

STORED = [r.split(":")[1] for r in FEATURE_REFS]


def _query(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description or []]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def aliases() -> dict[str, str]:
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient(MLFLOW_TRACKING_URI)
    out = {}
    for alias in ("champion", "challenger"):
        with contextlib.suppress(MlflowException):
            out[alias] = str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
    return out


def run_live_gate(label_delay_days: float, latency_budget_ms: float, apply: bool) -> Decision:
    """Pair the current champion's and challenger's logged scores, wait out the labels, decide."""
    a = aliases()
    if "challenger" not in a:
        raise SystemExit("no challenger registered")
    log = _query(
        "SELECT txn_id, role, score, threshold, model_ms, event_ts FROM predictions"
        " WHERE (role = 'champion' AND version = %s) OR (role = 'challenger' AND version = %s)",
        (a["champion"], a["challenger"]),
    ).drop_duplicates(["txn_id", "role"], keep="last")
    champ = log[log.role == "champion"].set_index("txn_id")
    chall = log[log.role == "challenger"].set_index("txn_id")
    tx = pd.read_parquet(DATA / "transactions.parquet").set_index("txn_id")
    ids = sorted(set(champ.index) & set(chall.index) & set(tx.index))  # ignore hand-sent test requests
    tx = tx.loc[ids]
    as_of = pd.Timestamp(champ.loc[ids, "event_ts"].max()) + pd.Timedelta(days=label_delay_days)
    y = labels_as_of(tx, as_of).to_numpy().astype(np.float64)
    amount = tx["amount"].to_numpy(dtype=np.float64)

    def scored(df: pd.DataFrame, name: str) -> Scored:
        d = df.loc[ids]
        return Scored(
            name,
            ids,
            d["score"].to_numpy(np.float64),
            float(d["threshold"].iloc[0]),
            d["model_ms"].to_numpy(np.float64),
        )

    # Latency is benchmarked, not read from the shadow log: shadow timings share a process
    # with the live path and measure its contention (p99 3.9 ms for a model whose median
    # is 0.16 ms, and the champion's own p99 was 2.5 ms on the same server).
    sample = _query(
        "SELECT features FROM predictions WHERE role = 'challenger' AND version = %s ORDER BY id DESC LIMIT 500",
        (a["challenger"],),
    )["features"].tolist()
    bench = benchmark(a["challenger"], [f if isinstance(f, dict) else json.loads(f) for f in sample])
    chall_scored = scored(chall, f"v{a['challenger']}")
    dec = decide(
        scored(champ, f"v{a['champion']}"),
        Scored(chall_scored.name, ids, chall_scored.scores, chall_scored.threshold, bench),
        y,
        amount,
        latency_budget_ms=latency_budget_ms,
    )
    if apply and dec.promote:
        from mlflow import MlflowClient

        client = MlflowClient(MLFLOW_TRACKING_URI)
        client.set_registered_model_alias(MODEL_NAME, "champion", a["challenger"])
        client.delete_registered_model_alias(MODEL_NAME, "challenger")
    return dec


def benchmark(version: str, recs: list[dict[str, Any]]) -> np.ndarray:
    """Score logged requests one at a time on the serving path (``Model.score_one``)."""
    m = load_version(version)
    out = []
    for rec in recs * 2:
        t0 = time.perf_counter()
        m.score_one(rec)
        out.append((time.perf_counter() - t0) * 1000)
    return np.asarray(out[len(recs) // 4 :])  # drop warm-up


def load_version(version: str) -> Any:
    import mlflow
    from mlflow import MlflowClient

    from rtml.serve.models import Model

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mv = MlflowClient().get_model_version(MODEL_NAME, version)
    return Model(
        alias="",
        version=version,
        spec=mv.tags.get("spec", ""),
        booster=mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/{version}"),
        features=json.loads(mv.tags["features"]),
        threshold=float(mv.tags["threshold"]),
    )


def shadow_latency() -> dict[str, dict[str, float]]:
    """Per-version scoring time as logged by the API: the live path, contention included."""
    df = _query("SELECT role, version, model_ms, fetch_ms FROM predictions")
    out = {}
    for (role, version), g in df.groupby(["role", "version"]):
        out[f"{role} v{version}"] = {
            "n": len(g),
            "model_p50_ms": round(float(g.model_ms.median()), 3),
            "model_p99_ms": round(float(g.model_ms.quantile(0.99)), 3),
            "fetch_p50_ms": round(float(g.fetch_ms.median()), 3),
            "fetch_p99_ms": round(float(g.fetch_ms.quantile(0.99)), 3),
        }
    return out


def _same(a: Any, b: Any) -> bool:
    a_missing = a is None or (isinstance(a, float) and math.isnan(a))
    b_missing = b is None or (isinstance(b, float) and math.isnan(b))
    if a_missing or b_missing:
        return a_missing and b_missing
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def parity(first_id: int = 0) -> dict[str, Any]:
    """Compare the features the API served with what a point-in-time join says it should have."""
    log = _query(
        "SELECT txn_id, features FROM predictions WHERE role = 'champion' AND id > %s ORDER BY id", (first_id,)
    ).drop_duplicates("txn_id", keep="last")
    served = pd.DataFrame([f if isinstance(f, dict) else json.loads(f) for f in log["features"]])
    served["txn_id"] = log["txn_id"].to_numpy()
    tx = pd.read_parquet(DATA / "transactions.parquet")
    rows = tx[tx.txn_id.isin(served.txn_id)]
    pit = pit_join(get_store(apply=False), rows).set_index("txn_id")
    served = served.set_index("txn_id").loc[pit.index]
    bad = pd.DataFrame(
        {f: [not _same(a, b) for a, b in zip(served[f], pit[f], strict=True)] for f in STORED}, index=pit.index
    )
    any_bad = bad.any(axis=1)
    fraud = pit["is_fraud"].astype(bool)
    return {
        "rows": len(pit),
        "row_parity_pct": round(100 * (1 - any_bad.mean()), 3),
        "row_parity_pct_fraud": round(100 * (1 - any_bad[fraud].mean()), 3) if fraud.any() else None,
        "row_parity_pct_legit": round(100 * (1 - any_bad[~fraud].mean()), 3),
        "mismatch_pct_by_feature": {f: round(100 * bad[f].mean(), 3) for f in STORED if bad[f].any()},
        "mismatch_by_fraud_type": {k: round(100 * v, 2) for k, v in any_bad.groupby(pit["fraud_type"]).mean().items()},
    }


def decision_dict(d: Decision) -> dict[str, Any]:
    return asdict(d)
