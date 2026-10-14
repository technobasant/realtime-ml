"""The drift monitor: reads the prediction log on a schedule and exports gauges to Prometheus.

Every ``--every`` seconds it takes the champion's last ``--window`` scored transactions,
rebuilds the model features from what was logged, and compares them with the training
validation window: PSI and KS per feature, PSI on the score, the flag rate as a z-score,
and recall on the transactions whose chargebacks have arrived. Prometheus evaluates the
alert rules in ``ops/prometheus/alerts.yml`` against these gauges.

The window is a count of transactions, not a span of time, so a replay that compresses a
day into minutes is monitored the same way as live traffic.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import psycopg
from prometheus_client import Gauge, start_http_server

from rtml import timeline as tl
from rtml.config import DATA, PG_DSN
from rtml.dataset import day_slice, labels_as_of, load, pit_join
from rtml.features.defs import MODEL_FEATURES_V2, model_matrix
from rtml.monitor.drift import TRENDING, edges, flag_rate_z, ks, psi
from rtml.serve.models import Registry
from rtml.store import get_store

log = logging.getLogger("rtml.monitor")

PSI = Gauge("rtml_feature_psi", "PSI of a feature against the reference window", ["feature", "trending"])
KS = Gauge("rtml_feature_ks", "KS statistic of a feature against the reference window", ["feature"])
SCORE_PSI = Gauge("rtml_score_psi", "PSI of the champion's score against the reference window")
FLAG_RATE = Gauge("rtml_flag_rate", "Share of the window flagged by the champion")
FLAG_Z = Gauge("rtml_flag_rate_z", "Flag rate against the reference, as a z-score")
RECALL = Gauge("rtml_labelled_recall", "Champion recall on frauds whose chargebacks have arrived")
LABELLED = Gauge("rtml_labelled_frauds", "Frauds with a chargeback in the window")
WINDOW = Gauge("rtml_monitor_window", "Transactions in the monitored window")
NEWEST = Gauge("rtml_monitor_newest_event_timestamp", "Event time of the newest monitored transaction")


def reference() -> pd.DataFrame:
    tx, cfg = load(DATA)
    return pit_join(get_store(apply=False), day_slice(tx, cfg, *tl.VALID))


def run(window: int, every: float, port: int) -> None:
    start_http_server(port)
    ref = reference()
    ref_x = model_matrix(ref, MODEL_FEATURES_V2)
    cuts = {f: edges(ref_x[f].to_numpy()) for f in MODEL_FEATURES_V2}
    tx, _ = load(DATA)
    labels = tx.set_index("txn_id")[["chargeback_at"]]
    registry = Registry()
    registry.wait_ready()
    version, ref_score, ref_flag = "", np.array([]), 0.0
    log.info("reference: %d transactions from days %s", len(ref), tl.VALID)
    while True:
        champion = registry.models["champion"]
        if champion is not None and champion.version != version:
            version = champion.version
            ref_score = champion.score(ref)
            ref_flag = float((ref_score >= champion.threshold).mean())
        with psycopg.connect(PG_DSN) as conn:
            rows = conn.execute(
                "SELECT txn_id, score, flagged, features FROM predictions WHERE role = 'champion'"
                " ORDER BY id DESC LIMIT %s",
                (window,),
            ).fetchall()
        if len(rows) >= 500:
            feats = pd.DataFrame([r[3] if isinstance(r[3], dict) else json.loads(r[3]) for r in rows])
            x = model_matrix(feats, MODEL_FEATURES_V2)
            for f in MODEL_FEATURES_V2:
                PSI.labels(f, str(f in TRENDING).lower()).set(psi(ref_x[f].to_numpy(), x[f].to_numpy(), cuts=cuts[f]))
                KS.labels(f).set(ks(ref_x[f].to_numpy(), x[f].to_numpy()))
            scores = np.array([r[1] for r in rows], dtype=np.float64)
            flagged = np.array([r[2] for r in rows], dtype=bool)
            SCORE_PSI.set(psi(ref_score, scores))
            FLAG_RATE.set(float(flagged.mean()))
            FLAG_Z.set(flag_rate_z(ref_flag, int(flagged.sum()), len(rows)))
            # Labels as the chargeback feed has them at the newest event time in the window.
            ts = pd.to_datetime(feats["ts"], utc=True)
            now = ts.max()
            ids = [r[0] for r in rows]
            known = labels_as_of(labels.loc[ids], now).to_numpy().astype(bool)
            LABELLED.set(int(known.sum()))
            if known.sum() >= 10:
                RECALL.set(float(flagged[known].mean()))
            WINDOW.set(len(rows))
            NEWEST.set(now.timestamp())
        time.sleep(every)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--window", type=int, default=3000)
    p.add_argument("--every", type=float, default=10.0)
    p.add_argument("--port", type=int, default=9102)
    a = p.parse_args()
    run(a.window, a.every, a.port)


if __name__ == "__main__":
    main()
