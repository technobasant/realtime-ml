"""Offline experiments: leakage, label delay, model quality, skew, and the promotion gate.

``python -m rtml.experiments.offline all`` regenerates every table in MEASUREMENTS.md
that does not need the running stack. Each writes ``results/<name>.json``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from rtml import timeline as tl
from rtml.dataset import day_slice, day_ts, labels_as_of, load, load_tables, naive, pit_join
from rtml.experiments.common import dataset, r, save
from rtml.features.defs import CUSTOMER_DEVICE, CUSTOMER_VELOCITY, DEVICE_VELOCITY, model_matrix
from rtml.features.online import OnlineFeatures, txns_from_frame
from rtml.gate import LATENCY_BUDGET_MS, Scored, decide
from rtml.metrics import best_threshold, quality, reliability
from rtml.store import get_store
from rtml.train import Spec, Trained, evaluate, fit

QUALITY_KEYS = ["auc_pr", "recall_at_p80", "precision", "recall", "flag_rate", "cost_saved_vs_approve_all"]


def _frames(d: Path, *ranges: tuple[int, int]) -> list[pd.DataFrame]:
    tx, cfg = load(d)
    store = get_store(d, redis="", apply=False)
    return [pit_join(store, day_slice(tx, cfg, a, b)) for a, b in ranges]


def _pick(q: dict[str, float]) -> dict[str, float]:
    return {k: r(q[k]) for k in QUALITY_KEYS}


# --------------------------------------------------------------------------- leakage


def leakage(seed: int = 7) -> dict[str, Any]:
    """Naive join versus point-in-time join, judged on the future the model will meet."""
    d = dataset(seed)
    tx, cfg = load(d)
    tables = load_tables(d)
    train, valid, test = _frames(d, tl.TRAIN, tl.VALID, tl.TEST)
    as_of = day_ts(cfg, tl.TRAINED_AT)
    n_train = naive(day_slice(tx, cfg, *tl.TRAIN), tables, as_of)
    n_valid = naive(day_slice(tx, cfg, *tl.VALID), tables, as_of)

    # The audit that catches it: how many training rows see a feature row from their future?
    future = {}
    for name, keys in (("customer_velocity", ["customer_id"]), ("device_velocity", ["device_id"])):
        t = tables[name][tables[name].event_timestamp < as_of].sort_values("event_timestamp")
        last = t.drop_duplicates(keys, keep="last")[[*keys, "event_timestamp"]]
        j = day_slice(tx, cfg, *tl.TRAIN)[[*keys, "ts"]].merge(last, on=keys, how="left")
        future[name] = r((j.event_timestamp > j.ts).mean())

    y_test = test["is_fraud"].to_numpy()
    rows = []
    for fs in ("v1", "v2"):
        for join, tr, va in (("point-in-time", train, valid), ("naive", n_train, n_valid)):
            m = fit(Spec(f"{join}-{fs}", fs), tr, va, cfg)
            p_va = m.predict(va)
            rows.append(
                {
                    "features": fs,
                    "join": join,
                    "valid_auc_pr_reported": r(m.valid["auc_pr"]),
                    "valid_auc_pr_true_labels": r(average_precision_score(va["is_fraud"], p_va)),
                    "valid_recall_at_p80": r(m.valid["recall_at_p80"]),
                    "test": _pick(evaluate(m, test, y_test)),
                    "top_features": _top(m, 4),
                }
            )
    return {"seed": seed, "train_rows_with_future_features": future, "rows": rows}


def _top(m: Trained, k: int) -> list[str]:
    gain = pd.Series(m.booster.feature_importance("gain"), index=m.features)
    return [f"{n} ({v:.0%})" for n, v in (gain / gain.sum()).sort_values(ascending=False).head(k).items()]


# --------------------------------------------------------------------------- labels


def labels(seed: int = 7) -> dict[str, Any]:
    """What delayed chargebacks do to training and to the numbers you can see."""
    d = dataset(seed)
    _, cfg = load(d)
    test = _frames(d, tl.TEST)[0]
    y_test = test["is_fraud"].to_numpy()
    variants = [
        ("all rows to day 62, labels known on day 63", (14, 55), (56, 62), tl.TRAINED_AT),
        ("rows matured 14 days (default)", tl.TRAIN, tl.VALID, tl.TRAINED_AT),
        ("rows matured 30 days", (14, 26), (27, 33), tl.TRAINED_AT),
        ("all rows to day 62, true labels (simulation only)", (14, 55), (56, 62), None),
    ]
    out = []
    for name, tr_days, va_days, labels_day in variants:
        train, valid = _frames(d, tr_days, va_days)
        known = labels_as_of(pd.concat([train, valid]), day_ts(cfg, tl.TRAINED_AT))
        truth = pd.concat([train, valid])["is_fraud"]
        m = fit(Spec(name, "v2", tr_days, va_days, labels_day), train, valid, cfg)
        out.append(
            {
                "variant": name,
                "train_rows": len(train),
                "frauds_labelled": int(known[truth.to_numpy()].sum()) if labels_day is not None else int(truth.sum()),
                "frauds_true": int(truth.sum()),
                "valid_auc_pr_reported": r(m.valid["auc_pr"]),
                "test": _pick(evaluate(m, test, y_test)),
            }
        )

    # Reported recall for one day of production traffic, as the labels trickle in.
    train, valid = _frames(d, tl.TRAIN, tl.VALID)
    champion = fit(Spec("default", "v2"), train, valid, cfg)
    day = _frames(d, (tl.TEST[0], tl.TEST[0]))[0]
    flagged = champion.predict(day) >= champion.threshold
    true_recall = float(flagged[day["is_fraud"].to_numpy()].mean())
    curve = []
    for k in (1, 3, 7, 14, 30, 60):
        seen = labels_as_of(day, day_ts(cfg, tl.TEST[0] + 1 + k)).to_numpy().astype(bool)
        curve.append(
            {
                "days_after": k,
                "frauds_known": int(seen.sum()),
                "share_of_true_frauds": r(seen.sum() / day["is_fraud"].sum()),
                "measured_recall": r(flagged[seen].mean()) if seen.any() else None,
            }
        )
    return {"seed": seed, "variants": out, "day": tl.TEST[0], "true_recall": r(true_recall), "label_curve": curve}


# --------------------------------------------------------------------------- quality


def model_quality(seed: int = 7) -> dict[str, Any]:
    d = dataset(seed)
    _, cfg = load(d)
    train, valid, test = _frames(d, tl.TRAIN, tl.VALID, tl.TEST)
    y = test["is_fraud"].to_numpy().astype(np.float64)
    amount = test["amount"].to_numpy(dtype=np.float64)
    models = {fs: fit(Spec(fs, fs), train, valid, cfg) for fs in ("v1", "v2")}
    out: dict[str, Any] = {"seed": seed, "test_rows": len(test), "test_frauds": int(y.sum())}
    for fs, m in models.items():
        p = m.predict(test)
        q = asdict(quality(y, p, amount, m.threshold))
        oracle_thr = best_threshold(y, p, amount)
        out[fs] = {
            **{k: r(v, 4) for k, v in q.items()},
            "threshold_if_tuned_on_test": r(oracle_thr, 4),
            "best_iteration": m.booster.best_iteration,
            "train_seconds": r(m.train_seconds, 2),
            "recall_by_type": {
                k: r(v)
                for k, v in test.assign(f=p >= m.threshold).query("is_fraud").groupby("fraud_type").f.mean().items()
            },
            "reliability": reliability(y, p, 10).round(5).to_dict(orient="records"),
            "top_decile_reliability": reliability(y[p >= np.quantile(p, 0.99)], p[p >= np.quantile(p, 0.99)], 5)
            .round(4)
            .to_dict(orient="records"),
        }
    # A hand-written rule, the baseline a fraud team already has.
    feats = model_matrix(test, models["v2"].features)
    rule = ((feats.amount_to_cust_mean_7d > 4) & (feats.is_new_device == 1)) | (feats.dev_distinct_cards_1h >= 3)
    flag = rule.to_numpy()
    out["rule"] = {
        "description": "amount > 4x the 7-day mean on a new device, or 3+ cards on a device within an hour",
        "precision": r(float(y[flag].mean()) if flag.any() else 0.0),
        "recall": r(float(flag[y == 1].mean())),
        "flag_rate": r(float(flag.mean()), 4),
        "cost_saved_vs_approve_all": r(
            1 - (np.where(flag, 5.0, 0) + np.where((y == 1) & ~flag, amount, 0)).sum() / amount[y == 1].sum()
        ),
    }
    return out


# --------------------------------------------------------------------------- skew


def skew(seed: int = 7, days: tuple[int, int] = (63, 69)) -> dict[str, Any]:
    """Offline SQL features versus the streaming state machine, row for row.

    The stream is warm-started with 30 days of history, as the processor is; only the
    rows inside ``days`` are compared.
    """
    d = dataset(seed)
    tx, cfg = load(d)
    tables = load_tables(d)
    lo, hi = day_ts(cfg, days[0]), day_ts(cfg, days[1] + 1)
    history = tx[(tx.ts >= lo - pd.Timedelta(days=30)) & (tx.ts < hi)]
    out: dict[str, Any] = {"seed": seed, "days": list(days)}
    test = _frames(d, days)[0]
    train, valid = _frames(d, tl.TRAIN, tl.VALID)
    m = fit(Spec("v2", "v2"), train, valid, cfg)
    for bug in (False, True):
        t0 = time.perf_counter()
        state = OnlineFeatures(skew_bug=bug)
        replayed = list(state.replay(txns_from_frame(history)))
        rows = [(t, f) for t, f in replayed if t.us >= lo.value // 1000]
        secs = time.perf_counter() - t0
        per_feature: dict[str, float] = {}
        mismatched_rows = np.zeros(len(rows), dtype=bool)
        for name, cols, keys in (
            ("customer_velocity", CUSTOMER_VELOCITY, ["customer_id"]),
            ("device_velocity", DEVICE_VELOCITY, ["device_id"]),
            ("customer_device", CUSTOMER_DEVICE, ["customer_id", "device_id"]),
        ):
            online = pd.DataFrame([getattr(f, name) for _, f in rows])
            off = tables[name]
            off = off[(off.event_timestamp >= lo) & (off.event_timestamp < hi)]
            j = online.merge(
                off, on=[*keys, "event_timestamp"], how="left", suffixes=("_on", "_off"), validate="one_to_one"
            )
            for c in cols:
                bad = ~np.isclose(j[f"{c}_on"].astype(float), j[f"{c}_off"].astype(float), rtol=1e-9, atol=1e-9)
                per_feature[c] = r(100 * bad.mean(), 3)
                mismatched_rows |= bad
        # What the defect costs: score the test days with the features the buggy stream would serve.
        served = test
        if bug:
            # Serve the test days with the features the buggy stream would have written:
            # the device's state after its previous event, as the online store holds it.
            dev = pd.DataFrame([f.device_velocity for _, f in replayed])[
                ["device_id", "event_timestamp", "dev_distinct_cards_1h"]
            ].sort_values("event_timestamp")
            q = test.drop(columns=["dev_distinct_cards_1h"]).assign(_at=test["ts"] - pd.Timedelta(milliseconds=1))
            served = pd.merge_asof(
                q.sort_values("_at"),
                dev.rename(columns={"event_timestamp": "_at"}).astype({"_at": q["_at"].dtype}),
                on="_at",
                by="device_id",
            )
        e = evaluate(m, served, served["is_fraud"].to_numpy())
        out["bug" if bug else "clean"] = {
            "rows": len(rows),
            "replay_seconds": r(secs, 1),
            "events_per_second": int(len(history) / secs),
            "row_parity_pct": r(100 * (1 - mismatched_rows.mean()), 3),
            "feature_mismatch_pct": per_feature,
            "test_auc_pr": r(e["auc_pr"]),
            "test_precision": r(e["precision"]),
            "test_flag_rate": r(e["flag_rate"], 4),
            "test_cost_saved": r(e["cost_saved_vs_approve_all"]),
        }
    return out


# --------------------------------------------------------------------------- gate


def _latency(m: Trained, rows: pd.DataFrame, n: int = 400) -> np.ndarray:
    """Single-transaction scoring latency on the serving path: request features plus predict."""
    sample = rows.sample(n=min(n, len(rows)), random_state=0)
    recs = [{str(k): v for k, v in r.items()} for r in sample.to_dict(orient="records")]
    out = []
    for rec in recs:
        t0 = time.perf_counter()
        m.score_one(rec)
        out.append((time.perf_counter() - t0) * 1000)
    return np.asarray(out[20:])  # drop warm-up


def gate(seed: int = 7, budget_ms: float = LATENCY_BUDGET_MS) -> dict[str, Any]:
    d = dataset(seed)
    _, cfg = load(d)
    train, valid, shadow = _frames(d, tl.TRAIN, tl.VALID, tl.SHADOW)
    y = labels_as_of(shadow, day_ts(cfg, tl.SHADOW[1] + 1 + 30)).to_numpy().astype(np.float64)
    amount = shadow["amount"].to_numpy(dtype=np.float64)
    ids = shadow["txn_id"].tolist()

    champion = fit(Spec("v1 champion", "v1"), train, valid, cfg)
    challengers: list[tuple[str, Callable[[], Trained]]] = [
        ("v2: adds device features", lambda: fit(Spec("v2", "v2"), train, valid, cfg)),
        ("v1 retrained, another seed", lambda: fit(Spec("v1-seed1", "v1", seed=1), train, valid, cfg)),
        (
            "v1 underfit: 4 leaves, 30 rounds",
            lambda: fit(Spec("v1-small", "v1", rounds=30, params={"num_leaves": 4}), train, valid, cfg),
        ),
        (
            "v2 oversized: 255 leaves, 2000 rounds, no early stop",
            lambda: fit(
                Spec(
                    "v2-large",
                    "v2",
                    rounds=2000,
                    early_stop=False,
                    params={"num_leaves": 255, "min_data_in_leaf": 20, "learning_rate": 0.01},
                ),
                train,
                valid,
                cfg,
            ),
        ),
    ]

    def scored(m: Trained, name: str) -> Scored:
        return Scored(name, ids, m.predict(shadow), m.threshold, _latency(m, shadow))

    base = scored(champion, "v1")
    out: dict[str, Any] = {
        "seed": seed,
        "shadow_days": list(tl.SHADOW),
        "labels": "chargebacks known 30 days after the shadow window",
        "budget_ms": budget_ms,
        "champion_p99_ms": r(float(np.percentile(base.latency_ms, 99)), 2),
        "decisions": [],
    }
    v2 = None
    for name, make in challengers:
        m = make()
        s = scored(m, name)
        if name.startswith("v2:"):
            v2 = s
        dec = decide(base, s, y, amount, latency_budget_ms=budget_ms)
        out["decisions"].append({"challenger": name, "best_iteration": m.booster.best_iteration, **_dec(dec)})

    # The same better challenger, decided early: labels that have not arrived count as legit.
    assert v2 is not None
    early = []
    for k in (0, 3, 7, 14, 30):
        yk = labels_as_of(shadow, day_ts(cfg, tl.SHADOW[1] + 1 + k)).to_numpy().astype(np.float64)
        dec = decide(base, v2, yk, amount, latency_budget_ms=budget_ms)
        early.append({"days_after_shadow": k, **_dec(dec)})
    out["v2_decided_early"] = early
    return out


def _dec(dec: Any) -> dict[str, Any]:
    x = asdict(dec)
    return {
        "promote": x["promote"],
        "reason": x["reason"],
        "frauds_labelled": x["frauds"],
        "cost_per_txn": [r(x["champion_cost_per_txn"], 4), r(x["challenger_cost_per_txn"], 4)],
        "saving_per_txn": r(x["cost_delta_per_txn"], 4),
        "ci": [r(v, 4) for v in x["cost_ci"]],
        "p_cost": r(x["cost_p"], 4),
        "p_mcnemar": float(f"{x['mcnemar_p']:.3g}"),
        "improved_regressed": [x["improved"], x["regressed"]],
        "challenger_p99_ms": r(x["challenger_p99_ms"], 2),
    }


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {
    "leakage": leakage,
    "labels": labels,
    "quality": model_quality,
    "skew": skew,
    "gate": gate,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("name", choices=[*EXPERIMENTS, "all"])
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    for name in EXPERIMENTS if a.name == "all" else [a.name]:
        t0 = time.perf_counter()
        res = EXPERIMENTS[name](seed=a.seed)
        res["wall_seconds"] = r(time.perf_counter() - t0, 1)
        print(f"{name}: {save(name, res)} ({res['wall_seconds']} s)")


if __name__ == "__main__":
    main()
