"""Drift statistics: PSI and KS per feature, on the model score, and on the flag rate.

PSI uses ten quantile bins fixed on the reference window, so a feature that piles up on
one value (most counts are 0 or 1) gets fewer, wider bins rather than empty ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp

FloatArray = NDArray[np.float64]
EPS = 1e-4
PSI_ALERT = 0.2
# Features that trend with the calendar rather than with behaviour: accounts age, and the
# 90-day chargeback count fills up as history accumulates. PSI against a fixed reference
# fires on them whether or not anything has changed.
TRENDING = frozenset({"account_age_days", "cust_chargebacks_90d"})
# Features with a daily cycle: any window shorter than a day differs from a whole-day
# reference by construction (hour_of_day reached PSI 5 on a 3,000-transaction window).
SEASONAL = frozenset({"hour_of_day"})


def kind(feature: str) -> str:
    return "trending" if feature in TRENDING else "seasonal" if feature in SEASONAL else "behavioural"


def edges(ref: FloatArray, bins: int = 10) -> FloatArray:
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.array([-np.inf, np.inf])
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def psi(ref: FloatArray, cur: FloatArray, bins: int = 10, cuts: FloatArray | None = None) -> float:
    """Population stability index; missing values get their own bin."""
    e = edges(ref, bins) if cuts is None else cuts

    def dist(x: FloatArray) -> FloatArray:
        finite = np.isfinite(x)
        h = np.histogram(x[finite], bins=e)[0].astype(np.float64)
        h = np.append(h, float((~finite).sum()))
        return np.maximum(h / max(1.0, float(x.size)), EPS)

    p, q = dist(ref), dist(cur)
    return float(np.sum((q - p) * np.log(q / p)))


def _bins(x: FloatArray, cuts: FloatArray) -> NDArray[np.int64]:
    """Bin index per value; missing values go to an extra last bin."""
    idx = np.full(x.size, len(cuts) - 1, dtype=np.int64)
    finite = np.isfinite(x)
    idx[finite] = np.clip(np.searchsorted(cuts, x[finite], side="right") - 1, 0, len(cuts) - 2)
    return idx


def hourly_reference(ref: FloatArray, hours: FloatArray, cuts: FloatArray) -> FloatArray:
    """Reference bin counts by hour of day, shape (24, bins)."""
    counts = np.zeros((24, len(cuts)), dtype=np.float64)
    np.add.at(counts, (hours.astype(np.int64) % 24, _bins(ref, cuts)), 1.0)
    return counts


def psi_hour_matched(ref_hourly: FloatArray, cur: FloatArray, hours: FloatArray, cuts: FloatArray) -> float:
    """PSI against a reference re-weighted to the window's own mix of hours.

    A window shorter than a day is all evenings or all mornings; compared with a
    whole-day reference, every feature with a daily cycle "drifts". Weighting the
    reference's hourly distributions by the window's hour mix removes the cycle and
    leaves the change.
    """
    total = ref_hourly.sum(axis=1, keepdims=True)
    overall = ref_hourly.sum(axis=0) / max(1.0, float(ref_hourly.sum()))
    by_hour = np.where(total > 0, ref_hourly / np.maximum(total, 1.0), overall)
    w = np.bincount(hours.astype(np.int64) % 24, minlength=24) / max(1, hours.size)
    p = np.maximum(w @ by_hour, EPS)
    q = np.maximum(np.bincount(_bins(cur, cuts), minlength=len(cuts)) / max(1, cur.size), EPS)
    return float(np.sum((q - p) * np.log(q / p)))


def ks(ref: FloatArray, cur: FloatArray) -> float:
    a, b = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(ks_2samp(a, b).statistic)


def flag_rate_z(ref_rate: float, flagged: int, n: int) -> float:
    """z-score of a window's flag rate against the reference rate (normal approximation)."""
    if n == 0 or not 0 < ref_rate < 1:
        return 0.0
    return (flagged / n - ref_rate) / math.sqrt(ref_rate * (1 - ref_rate) / n)


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    ks: float


def feature_drift(ref: dict[str, FloatArray], cur: dict[str, FloatArray]) -> list[FeatureDrift]:
    out = [FeatureDrift(f, psi(ref[f], cur[f]), ks(ref[f], cur[f])) for f in ref]
    return sorted(out, key=lambda d: d.psi, reverse=True)
