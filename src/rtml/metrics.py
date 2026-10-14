"""Model-quality metrics, including the cost model that picks the operating threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from rtml.config import REVIEW_COST

Array = NDArray[np.float64]


def case_cost(y: Array, flagged: NDArray[np.bool_], amount: Array, review_cost: float = REVIEW_COST) -> Array:
    """Per-transaction cost: a review for every flag, the full amount for every missed fraud."""
    return np.where(flagged, review_cost, 0.0) + np.where((y == 1) & ~flagged, amount, 0.0)


def best_threshold(y: Array, p: Array, amount: Array, review_cost: float = REVIEW_COST) -> float:
    """The score threshold that minimises total cost on (validation) data."""
    grid = np.unique(np.quantile(p, np.linspace(0.90, 0.99999, 400)))
    costs = [case_cost(y, p >= t, amount, review_cost).sum() for t in grid]
    return float(grid[int(np.argmin(costs))])


def recall_at_precision(y: Array, p: Array, target: float) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


def reliability(y: Array, p: Array, bins: int = 10) -> pd.DataFrame:
    """Equal-count bins over the scores, so the rare high-score region gets its own rows."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(n=("y", "size"), mean_score=("p", "mean"), fraud_rate=("y", "mean")).reset_index()


def ece(y: Array, p: Array, bins: int = 10) -> float:
    r = reliability(y, p, bins)
    return float((r["n"] * (r["mean_score"] - r["fraud_rate"]).abs()).sum() / r["n"].sum())


@dataclass(frozen=True)
class Quality:
    n: int
    frauds: int
    auc_pr: float
    auc_roc: float
    recall_at_p80: float
    threshold: float
    precision: float
    recall: float
    flag_rate: float
    cost_per_1k: float
    cost_saved_vs_approve_all: float
    brier: float
    ece: float


def quality(y: Array, p: Array, amount: Array, threshold: float) -> Quality:
    flagged = p >= threshold
    tp = float(((y == 1) & flagged).sum())
    cost = case_cost(y, flagged, amount).sum()
    approve_all = float(amount[y == 1].sum())
    return Quality(
        n=len(y),
        frauds=int(y.sum()),
        auc_pr=float(average_precision_score(y, p)),
        auc_roc=float(roc_auc_score(y, p)),
        recall_at_p80=recall_at_precision(y, p, 0.8),
        threshold=threshold,
        precision=tp / max(1.0, float(flagged.sum())),
        recall=tp / max(1.0, float(y.sum())),
        flag_rate=float(flagged.mean()),
        cost_per_1k=1000.0 * cost / len(y),
        cost_saved_vs_approve_all=1.0 - cost / approve_all if approve_all else 0.0,
        brier=float(np.mean((p - y) ** 2)),
        ece=ece(y, p),
    )
