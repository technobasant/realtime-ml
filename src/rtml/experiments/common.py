"""Shared helpers for the experiments: datasets on disk, result files, environment."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from rtml.config import DATA, RESULTS
from rtml.dataset import build
from rtml.generate import GenConfig
from rtml.store import get_store


def dataset(seed: int = 7, drift_day: int | None = None, root: Path | None = None, **kw: Any) -> Path:
    """Build (or reuse) a generated dataset under ``data/exp-...``."""
    tag = (
        f"exp-s{seed}"
        + (f"-drift{drift_day}" if drift_day is not None else "")
        + "".join(f"-{k}{v}" for k, v in sorted(kw.items()))
    )
    d = (root or DATA) / tag
    if not (d / "dataset.json").exists():
        build(GenConfig(seed=seed, days=90, drift_day=drift_day, **kw), d)
        get_store(d, redis="")
    return d


def environment() -> dict[str, str]:
    def cmd(*a: str) -> str:
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=5).stdout.strip()
        except OSError:
            return ""

    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpus": str(os.cpu_count()),
        "cpu": cmd("sysctl", "-n", "machdep.cpu.brand_string")
        or cmd("sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
    }


def save(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps({"environment": environment(), **payload}, indent=2, default=str))
    return path


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)
