"""How fast each monitor notices drift, and how often it cries wolf.

Ten generated datasets: five seeds with the drift switch thrown on day 70 (fraudsters go
quiet, prices rise 15%) and the same five seeds without it. A v2 model is trained on each
as usual, then days 63-89 are replayed through four monitors, evaluated every 6 hours on
the trailing 24 hours (a 6-hour window compares evenings with the whole-day reference and
alarms on the time of day):

* feature PSI — any model feature over the threshold (with and without the two features
  that trend with the calendar);
* score PSI — the model's own score distribution;
* flag rate — the share of transactions flagged, as a z-score against the reference rate;
* labelled recall — recall on the last 14 days' frauds whose chargebacks have arrived,
  more than 0.10 below the reference, once 30 such frauds exist.

Time to detect is hours from drift onset to the first alert; a false alarm is an alert in
a window with no drift in it (before onset, or any window of a no-drift run).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from rtml import timeline as tl
from rtml.dataset import day_ts, labels_as_of, load
from rtml.experiments.common import dataset, r, save
from rtml.experiments.offline import _frames
from rtml.features.defs import model_matrix
from rtml.monitor.drift import PSI_ALERT, edges, flag_rate_z, psi
from rtml.train import Spec, fit

DRIFT_DAY = 70
MONITOR = (63, 89)
STEP_H = 6
WINDOW_H = 24
Z_ALERT = 4.0
RECALL_DROP = 0.10
MIN_LABELLED = 30
# Features that trend with the calendar rather than with behaviour: accounts age, and the
# 90-day chargeback count fills up as the data's history grows.
TRENDING = {"account_age_days", "cust_chargebacks_90d"}

Rule = Callable[[float, float, float, float, "float | None", float], bool]
MONITORS: dict[str, Rule] = {
    "feature PSI > 0.2, all features": lambda fp, sp, s, z, rec, ref: fp > PSI_ALERT,
    "feature PSI > 0.2, trending features excluded": lambda fp, sp, s, z, rec, ref: sp > PSI_ALERT,
    "feature PSI > 0.1, trending features excluded": lambda fp, sp, s, z, rec, ref: sp > 0.1,
    "score PSI > 0.2": lambda fp, sp, s, z, rec, ref: s > PSI_ALERT,
    "score PSI > 0.1": lambda fp, sp, s, z, rec, ref: s > 0.1,
    "flag rate |z| > 4": lambda fp, sp, s, z, rec, ref: abs(z) > Z_ALERT,
    "flag rate |z| > 8": lambda fp, sp, s, z, rec, ref: abs(z) > 8,
    "labelled recall down 0.10": lambda fp, sp, s, z, rec, ref: rec is not None and rec < ref - RECALL_DROP,
}


def run_one(seed: int, drift: bool) -> dict[str, Any]:
    d = dataset(seed, DRIFT_DAY if drift else None)
    _, cfg = load(d)
    train, valid, mon = _frames(d, tl.TRAIN, tl.VALID, MONITOR)
    m = fit(Spec("v2", "v2"), train, valid, cfg)

    ref_x = model_matrix(valid, m.features)
    ref_score = m.predict(valid)
    ref_flag = float((ref_score >= m.threshold).mean())
    y_valid = labels_as_of(valid, day_ts(cfg, tl.TRAINED_AT)).to_numpy().astype(bool)
    ref_recall = float((ref_score >= m.threshold)[y_valid].mean())
    cuts = {f: edges(ref_x[f].to_numpy()) for f in m.features}
    score_cuts = edges(ref_score)

    mon_x = model_matrix(mon, m.features)
    mon_score = m.predict(mon)
    flagged = mon_score >= m.threshold
    ts = mon["ts"]
    onset = day_ts(cfg, DRIFT_DAY)

    windows = []
    t = day_ts(cfg, MONITOR[0]) + pd.Timedelta(hours=WINDOW_H)
    end = day_ts(cfg, MONITOR[1] + 1)
    while t <= end:
        sel = ((ts >= t - pd.Timedelta(hours=WINDOW_H)) & (ts < t)).to_numpy()
        f_psi = {f: psi(ref_x[f].to_numpy(), mon_x[f].to_numpy()[sel], cuts=cuts[f]) for f in m.features}
        worst = max(f_psi, key=lambda k: f_psi[k])
        s_psi = psi(ref_score, mon_score[sel], cuts=score_cuts)
        z = flag_rate_z(ref_flag, int(flagged[sel].sum()), int(sel.sum()))
        # labelled recall: last 14 days of traffic, frauds whose chargeback has arrived by t
        lab = ((ts >= t - pd.Timedelta(days=14)) & (ts < t)).to_numpy()
        known = lab & labels_as_of(mon, t).to_numpy().astype(bool)
        rec = float(flagged[known].mean()) if known.sum() >= MIN_LABELLED else None
        stationary = {k: v for k, v in f_psi.items() if k not in TRENDING}
        worst_st = max(stationary, key=lambda k: stationary[k])
        windows.append(
            {
                "end": t,
                "drifted": bool(drift and t - pd.Timedelta(hours=WINDOW_H) >= onset),
                "clean": bool(not drift or t <= onset),
                "feature_psi": f_psi[worst],
                "feature": worst,
                "stationary_psi": stationary[worst_st],
                "stationary_feature": worst_st,
                "amount_log_psi": f_psi["amount_log"],
                "score_psi": s_psi,
                "flag_z": z,
                "recall": rec,
                "alerts": {
                    name: rule(f_psi[worst], stationary[worst_st], s_psi, z, rec, ref_recall)
                    for name, rule in MONITORS.items()
                },
            }
        )
        t += pd.Timedelta(hours=STEP_H)

    # What drift costs the model, on true labels: the week before onset against the week after.
    y = mon["is_fraud"].to_numpy().astype(bool)
    before = (ts < onset).to_numpy()
    after = ((ts >= onset) & (ts < onset + pd.Timedelta(days=7))).to_numpy()
    out: dict[str, Any] = {
        "seed": seed,
        "drift": drift,
        "true_recall_before": r(flagged[before & y].mean()),
        "true_recall_week_after": r(flagged[after & y].mean()),
        "reference_recall_labelled": r(ref_recall),
        "monitors": {},
    }
    clean = [w for w in windows if w["clean"]]
    drifted = [w for w in windows if w["drifted"]]
    for name in MONITORS:
        fa = sum(w["alerts"][name] for w in clean)
        first = next((w for w in windows if w["end"] > onset and w["alerts"][name]), None) if drift else None
        out["monitors"][name] = {
            "clean_windows": len(clean),
            "false_alarms": int(fa),
            "detect_hours": r((first["end"] - onset).total_seconds() / 3600, 1) if first else None,
            "first_feature": (first["feature"] if "all" in name else first["stationary_feature"])
            if first and name.startswith("feature")
            else None,
        }
    for key in ("feature_psi", "stationary_psi", "amount_log_psi", "score_psi", "flag_z"):
        out[f"max_{key}_clean"] = r(max(abs(w[key]) for w in clean))
        out[f"max_{key}_drifted"] = r(max((abs(w[key]) for w in drifted), default=0.0))
    out["trending_feature_alarms"] = sorted({w["feature"] for w in clean if w["feature_psi"] > PSI_ALERT})
    return out


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in MONITORS:
        det = [x["monitors"][name]["detect_hours"] for x in runs if x["drift"]]
        caught = [h for h in det if h is not None]
        fa = sum(x["monitors"][name]["false_alarms"] for x in runs)
        clean = sum(x["monitors"][name]["clean_windows"] for x in runs)
        out[name] = {
            "detected": f"{len(caught)}/{len(det)}",
            "median_detect_hours": r(float(np.median(caught)), 1) if caught else None,
            "detect_hours": det,
            "false_alarms": fa,
            "clean_windows": clean,
            "false_alarm_rate": r(fa / clean, 4) if clean else None,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = p.parse_args()
    t0 = time.perf_counter()
    runs = [run_one(s, drift) for s in a.seeds for drift in (True, False)]
    res = {
        "drift_day": DRIFT_DAY,
        "monitor_days": list(MONITOR),
        "window_hours": WINDOW_H,
        "step_hours": STEP_H,
        "seeds": a.seeds,
        "summary": summarise(runs),
        "runs": runs,
        "wall_seconds": r(time.perf_counter() - t0, 1),
    }
    print(f"drift: {save('drift', res)} ({res['wall_seconds']} s)")


if __name__ == "__main__":
    main()
