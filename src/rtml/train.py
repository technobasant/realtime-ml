"""Train LightGBM fraud models on point-in-time features and register them in MLflow.

``python -m rtml.train`` trains the two models the stack starts with — v1 (customer
features only) as ``champion`` and v2 (adds device features) as ``challenger`` — and
leaves promotion to the gate.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from rtml import timeline as tl
from rtml.config import DATA, MLFLOW_TRACKING_URI, MODEL_NAME
from rtml.dataset import day_slice, day_ts, labels_as_of, load, pit_join
from rtml.features.defs import FEATURE_SETS, model_matrix
from rtml.generate import GenConfig
from rtml.metrics import best_threshold, quality
from rtml.store import get_store

BASE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 4,
}


@dataclass
class Spec:
    name: str
    feature_set: str = "v2"
    train_days: tuple[int, int] = tl.TRAIN
    valid_days: tuple[int, int] = tl.VALID
    labels_at_day: int | None = tl.TRAINED_AT  # None: true labels (only a simulation has them)
    rounds: int = 600
    early_stop: bool = True
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trained:
    spec: Spec
    booster: lgb.Booster
    features: list[str]
    threshold: float
    valid: dict[str, float]
    train_rows: int
    train_seconds: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = model_matrix(frame, self.features)
        return np.asarray(self.booster.predict(x, num_iteration=self.booster.best_iteration), dtype=np.float64)


def labelled(rows: pd.DataFrame, cfg: GenConfig, labels_at_day: int | None) -> np.ndarray:
    if labels_at_day is None:
        return rows["is_fraud"].astype("int8").to_numpy()
    return labels_as_of(rows, day_ts(cfg, labels_at_day)).to_numpy()


def fit(spec: Spec, train: pd.DataFrame, valid: pd.DataFrame, cfg: GenConfig) -> Trained:
    """Fit on ``train``, early-stop and pick the cost-optimal threshold on ``valid``.

    Both frames already carry their features — point-in-time or naive, the caller decides.
    """
    feats = FEATURE_SETS[spec.feature_set]
    y_tr, y_va = labelled(train, cfg, spec.labels_at_day), labelled(valid, cfg, spec.labels_at_day)
    x_tr, x_va = model_matrix(train, feats), model_matrix(valid, feats)
    params = {**BASE_PARAMS, **spec.params, "seed": spec.seed}
    t0 = time.perf_counter()
    booster = lgb.train(
        params,
        lgb.Dataset(x_tr, y_tr, free_raw_data=False),
        num_boost_round=spec.rounds,
        valid_sets=[lgb.Dataset(x_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)] if spec.early_stop else [],
    )
    secs = time.perf_counter() - t0
    if not spec.early_stop:
        booster.best_iteration = spec.rounds
    p_va = np.asarray(booster.predict(x_va, num_iteration=booster.best_iteration), dtype=np.float64)
    amount = valid["amount"].to_numpy(dtype=np.float64)
    thr = best_threshold(y_va.astype(np.float64), p_va, amount)
    q = quality(y_va.astype(np.float64), p_va, amount, thr)
    return Trained(spec, booster, feats, thr, {k: float(v) for k, v in asdict(q).items()}, len(train), secs)


def evaluate(model: Trained, frame: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    p = model.predict(frame)
    q = quality(y.astype(np.float64), p, frame["amount"].to_numpy(dtype=np.float64), model.threshold)
    return {k: float(v) for k, v in asdict(q).items()}


def pit_frames(data: Path, *ranges: tuple[int, int]) -> tuple[GenConfig, list[pd.DataFrame]]:
    tx, cfg = load(data)
    store = get_store(data, redis="", apply=True)
    return cfg, [pit_join(store, day_slice(tx, cfg, a, b)) for a, b in ranges]


def register(
    model: Trained, alias: str, tracking_uri: str = MLFLOW_TRACKING_URI, extra: dict[str, float] | None = None
) -> str:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name=model.spec.name) as run:
        mlflow.log_params(
            {
                "feature_set": model.spec.feature_set,
                "train_days": f"{model.spec.train_days[0]}-{model.spec.train_days[1]}",
                "valid_days": f"{model.spec.valid_days[0]}-{model.spec.valid_days[1]}",
                "labels_at_day": model.spec.labels_at_day,
                "best_iteration": model.booster.best_iteration,
                "threshold": model.threshold,
                "seed": model.spec.seed,
                **{f"lgb_{k}": v for k, v in {**BASE_PARAMS, **model.spec.params}.items()},
            }
        )
        mlflow.log_metrics({f"valid_{k}": v for k, v in model.valid.items()})
        mlflow.log_metrics(extra or {})
        mlflow.lightgbm.log_model(model.booster, name="model")
        uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(uri, MODEL_NAME)
    client = MlflowClient()
    client.set_model_version_tag(MODEL_NAME, mv.version, "features", json.dumps(model.features))
    client.set_model_version_tag(MODEL_NAME, mv.version, "threshold", repr(model.threshold))
    client.set_model_version_tag(MODEL_NAME, mv.version, "spec", model.spec.name)
    client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
    return str(mv.version)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--tracking-uri", default=MLFLOW_TRACKING_URI)
    a = p.parse_args()
    cfg, (train, valid) = pit_frames(a.data, tl.TRAIN, tl.VALID)
    for spec, alias in ((Spec("v1-customer", "v1"), "champion"), (Spec("v2-device", "v2"), "challenger")):
        m = fit(spec, train, valid, cfg)
        version = register(m, alias, a.tracking_uri)
        auc, thr = m.valid["auc_pr"], m.threshold
        print(f"{spec.name}: version {version} as {alias}; valid AUC-PR {auc:.3f}, threshold {thr:.4f}")


if __name__ == "__main__":
    main()
