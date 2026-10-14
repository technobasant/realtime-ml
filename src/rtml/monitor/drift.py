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
