"""Models by alias from the MLflow registry, reloaded when an alias moves."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from rtml.config import MLFLOW_TRACKING_URI, MODEL_NAME
from rtml.features.defs import model_matrix

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Model:
    alias: str
    version: str
    spec: str
    booster: lgb.Booster
    features: list[str]
    threshold: float

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(model_matrix(frame, self.features)), dtype=np.float64)


def load(alias: str, tracking_uri: str = MLFLOW_TRACKING_URI, name: str = MODEL_NAME) -> Model | None:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(tracking_uri)
    try:
        mv = MlflowClient().get_model_version_by_alias(name, alias)
    except MlflowException:
        return None
    booster = mlflow.lightgbm.load_model(f"models:/{name}/{mv.version}")
    return Model(
        alias=alias,
        version=str(mv.version),
        spec=mv.tags.get("spec", ""),
        booster=booster,
        features=json.loads(mv.tags["features"]),
        threshold=float(mv.tags["threshold"]),
    )


class Registry:
    """Holds the champion and the challenger; a background thread follows alias moves."""

    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI, poll_s: float = 10.0) -> None:
        self.tracking_uri = tracking_uri
        self.poll_s = poll_s
        self.models: dict[str, Model | None] = {"champion": None, "challenger": None}
        self._stop = threading.Event()

    def refresh(self) -> None:
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException

        client = MlflowClient(self.tracking_uri)
        for alias in self.models:
            try:
                version = str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
            except MlflowException:
                self.models[alias] = None
                continue
            current = self.models[alias]
            if current is None or current.version != version:
                self.models[alias] = load(alias, self.tracking_uri)
                log.info("loaded %s = version %s", alias, version)

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.wait(self.poll_s):
                try:
                    self.refresh()
                except Exception:  # keep serving the models we have
                    log.exception("model refresh failed")

        threading.Thread(target=loop, daemon=True, name="model-refresh").start()

    def stop(self) -> None:
        self._stop.set()

    def wait_ready(self, timeout_s: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_s
        while self.models["champion"] is None and time.monotonic() < deadline:
            try:
                self.refresh()
            except Exception:
                log.warning("registry not ready yet")
            if self.models["champion"] is None:
                time.sleep(2)
