"""Paths and settings shared by every component. Everything is overridable by environment."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("RTML_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("RTML_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("RTML_RESULTS", ROOT / "results"))
FEATURE_REPO = Path(os.environ.get("RTML_FEATURE_REPO", ROOT / "feature_repo"))

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
MODEL_NAME = os.environ.get("RTML_MODEL_NAME", "fraud-detector")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:19092")
TXN_TOPIC = os.environ.get("RTML_TXN_TOPIC", "transactions")
PG_DSN = os.environ.get("RTML_PG_DSN", "postgresql://rtml:rtml-dev-only@localhost:5433/rtml")

# Business costs used to pick the operating threshold and to score a model.
# A missed fraud costs the transaction amount; a false alarm costs a manual review
# plus customer friction. Both are assumptions, stated once, used everywhere.
REVIEW_COST = float(os.environ.get("RTML_REVIEW_COST", "5.0"))
