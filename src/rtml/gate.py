"""The promotion gate: a paired comparison of challenger and champion on the same traffic.

Both models scored the same transactions (the challenger in shadow), so every case is a
pair. The comparison is evalproof's: a paired bootstrap on the per-transaction cost and
an exact McNemar test on correct-at-threshold. The rule is deliberately one-sided —
evalproof's own gate guards against regressions; promotion needs evidence of an
improvement:

* the challenger's mean cost per transaction is lower, with paired-bootstrap p < alpha;
* its decisions are not significantly worse (McNemar on correctness, if it is lower);
* its scoring p99 latency stays inside the budget.

Anything else keeps the champion. "Not significantly different" is a refusal, not a tie.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from evalproof import __version__ as evalproof_version
from evalproof.compare import MetricComparison, compare_runs
from evalproof.models import CaseResult, Run, RunInfo, Score, SystemOutput
from numpy.typing import NDArray

from rtml.config import REVIEW_COST
from rtml.metrics import case_cost

FloatArray = NDArray[np.float64]

# The model's share of a 20 ms scoring SLO at p99; the feature fetch and the network
# take the rest. Measured on the serving path: request features plus predict.
LATENCY_BUDGET_MS = 2.0


@dataclass(frozen=True)
class Scored:
    """One model's scores on a shared set of transactions."""

    name: str
    txn_ids: list[str]
    scores: FloatArray
    threshold: float
    latency_ms: FloatArray  # per-request scoring latency, from the shadow log or a benchmark


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str
    n_paired: int
    frauds: int
    champion_cost_per_txn: float
    challenger_cost_per_txn: float
    cost_delta_per_txn: float  # champion minus challenger: positive means the challenger saves money
    cost_ci: tuple[float, float]
    cost_p: float
    champion_accuracy: float
    challenger_accuracy: float
    mcnemar_p: float
    improved: int
    regressed: int
    challenger_p99_ms: float
    latency_budget_ms: float


def benchmark_ms(score: Callable[[dict[str, Any]], float], recs: list[dict[str, Any]], n: int = 3000) -> FloatArray:
    """Per-call latency of ``score`` over logged requests, one at a time.

    The p99 of a few hundred calls was dominated by whatever else the machine was doing
    (one run put a model with a 0.04 ms median at 2.8 ms). So: a warm-up, a full
    collection, the collector paused while timing, and 3,000 calls.
    """
    for rec in recs[:200]:
        score(rec)
    gc.collect()
    gc.disable()
    try:
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            rec = recs[i % len(recs)]
            t0 = time.perf_counter()
            score(rec)
            out[i] = (time.perf_counter() - t0) * 1000
    finally:
        gc.enable()
    return out


def to_run(m: Scored, y: FloatArray, amount: FloatArray, dataset_hash: str, review_cost: float = REVIEW_COST) -> Run:
    flagged = m.scores >= m.threshold
    cost = case_cost(y, flagged, amount, review_cost)
    correct = flagged == (y == 1)
    # Latency is per case when it comes from the shadow log, a benchmark sample otherwise.
    lat = m.latency_ms if len(m.latency_ms) == len(m.txn_ids) else np.zeros(len(m.txn_ids))
    now = datetime.now(UTC).isoformat()
    results = [
        CaseResult(
            case_id=cid,
            output=SystemOutput(output=f"{s:.6f}", latency_ms=float(ms)),
            scores={
                "neg_cost": Score(value=-float(c)),
                "correct": Score(value=float(ok), passed=bool(ok)),
            },
        )
        for cid, s, ms, c, ok in zip(m.txn_ids, m.scores, lat, cost, correct, strict=True)
    ]
    info = RunInfo(
        suite="fraud-shadow",
        system=m.name,
        dataset_path="shadow-window",
        dataset_hash=dataset_hash,
        started_at=now,
        finished_at=now,
        wall_time_s=0.0,
        evalproof_version=evalproof_version,
        seed=0,
    )
    return Run(info=info, results=results)


def decide(
    champion: Scored,
    challenger: Scored,
    y: FloatArray,
    amount: FloatArray,
    *,
    alpha: float = 0.05,
    latency_budget_ms: float = LATENCY_BUDGET_MS,
    n_boot: int = 1000,
    seed: int = 0,
) -> Decision:
    if champion.txn_ids != challenger.txn_ids:
        raise ValueError("champion and challenger must be scored on the same transactions, in the same order")
    key = hashlib.sha256("\n".join(champion.txn_ids).encode()).hexdigest()[:16]
    base = to_run(champion, y, amount, key)
    cand = to_run(challenger, y, amount, key)
    cmp = compare_runs(base, cand, alpha=alpha, n_boot=n_boot, seed=seed)
    cost: MetricComparison = cmp.metrics["neg_cost"]
    acc: MetricComparison = cmp.metrics["correct"]
    p99 = float(np.percentile(challenger.latency_ms, 99))

    def fmt(p: float) -> str:
        # a bootstrap p-value of 0 means "below the resolution of n_boot replicates"
        return f"p<{1 / n_boot:.3g}" if p == 0 else f"p={p:.3g}"

    reasons = []
    if cost.delta < 0 and cost.p_value < alpha:
        reasons.append(f"cost significantly higher ({-cost.delta:.4f}/txn more, {fmt(cost.p_value)})")
    elif not (cost.delta > 0 and cost.p_value < alpha):
        reasons.append(f"cost not significantly lower (saves {cost.delta:+.4f}/txn, {fmt(cost.p_value)})")
    if acc.delta < 0 and acc.p_value < alpha:
        reasons.append(f"decisions significantly worse (McNemar {fmt(acc.p_value)})")
    if p99 > latency_budget_ms:
        reasons.append(f"p99 {p99:.2f} ms over the {latency_budget_ms:.1f} ms budget")
    promote = not reasons
    reason = (
        f"promote: saves {cost.delta:.4f}/txn (95% CI {cost.ci_low:.4f} to {cost.ci_high:.4f}, {fmt(cost.p_value)}), "
        f"p99 {p99:.2f} ms"
        if promote
        else "refuse: " + "; ".join(reasons)
    )
    return Decision(
        promote=promote,
        reason=reason,
        n_paired=cost.n_paired,
        frauds=int(y.sum()),
        champion_cost_per_txn=-cost.baseline,
        challenger_cost_per_txn=-cost.candidate,
        cost_delta_per_txn=cost.delta,
        cost_ci=(cost.ci_low, cost.ci_high),
        cost_p=cost.p_value,
        champion_accuracy=acc.baseline,
        challenger_accuracy=acc.candidate,
        mcnemar_p=acc.p_value,
        improved=len(acc.improved),
        regressed=len(acc.regressed),
        challenger_p99_ms=p99,
        latency_budget_ms=latency_budget_ms,
    )


def main() -> None:
    """Gate the live stack: read the shadow log, join matured labels, decide, maybe promote."""
    from rtml.serve.live import run_live_gate

    p = argparse.ArgumentParser(description="Compare challenger and champion on the shadow log.")
    p.add_argument("--label-delay-days", type=float, default=30.0)
    p.add_argument("--latency-budget-ms", type=float, default=LATENCY_BUDGET_MS)
    p.add_argument("--apply", action="store_true", help="move the champion alias if the gate passes")
    a = p.parse_args()
    d = run_live_gate(a.label_delay_days, a.latency_budget_ms, a.apply)
    print(json.dumps(asdict(d), indent=2))


if __name__ == "__main__":
    main()
