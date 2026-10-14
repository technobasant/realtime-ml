"""Prepare the stack for a replay that starts at the beginning of ``--day``.

Builds the dataset if it is missing, registers the Feast repository against Redis, and
loads the online store with every entity's state as of that midnight — the same rows a
point-in-time join would pick for a transaction at that moment.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rtml import timeline as tl
from rtml.config import DATA
from rtml.dataset import build, day_ts, load
from rtml.generate import GenConfig
from rtml.store import get_store


def bootstrap(data: Path, day: int, seed: int, drift_day: int | None, inflation: float) -> None:
    if not (data / "dataset.json").exists():
        build(GenConfig(seed=seed, days=tl.DAYS, drift_day=drift_day, drift_inflation=inflation), data)
    _, cfg = load(data)
    store = get_store(data)
    t0 = time.perf_counter()
    store.materialize(start_date=day_ts(cfg, 0) - (day_ts(cfg, 1) - day_ts(cfg, 0)), end_date=day_ts(cfg, day))
    print(f"online store loaded as of day {day} ({day_ts(cfg, day)}) in {time.perf_counter() - t0:.1f} s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--day", type=int, default=tl.SHADOW[0])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--drift-day", type=int, default=None)
    p.add_argument("--drift-inflation", type=float, default=1.15)
    a = p.parse_args()
    bootstrap(a.data, a.day, a.seed, a.drift_day, a.drift_inflation)


if __name__ == "__main__":
    main()
