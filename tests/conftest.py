from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rtml.dataset import build
from rtml.features.batch import velocity_tables
from rtml.generate import GenConfig
from rtml.store import get_store

SMALL = GenConfig(seed=3, n_customers=600, days=20)


@pytest.fixture(scope="session")
def small(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pd.DataFrame, dict[str, pd.DataFrame]]:
    """A three-week dataset, its feature tables and a Feast repository over them."""
    d = tmp_path_factory.mktemp("data")
    tx, tables = build(SMALL, d)
    get_store(d, redis="")
    return d, tx, tables


@pytest.fixture(scope="session")
def velocity(small: tuple[Path, pd.DataFrame, dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    return velocity_tables(small[1])
