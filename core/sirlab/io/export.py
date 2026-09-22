"""Writes results/<exp_id>.json (small artifacts) and downsamples curves
for the web dashboard (PLAN.md section 4.7): the browser does not need
20,000-point trajectories, so we decimate to ~600 points via uniform
index-stride selection that always keeps the first and last point.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sirlab.io.schema import dump_json

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
WEB_DATA_DIR = Path(__file__).resolve().parents[3] / "web" / "public" / "data"


def decimate(t: NDArray[np.float64], y: NDArray[np.float64], *, target_points: int = 600):
    """Uniform-stride decimation to ~target_points, always keeping the
    endpoints. Simpler than Douglas-Peucker and sufficient for smooth SIR
    trajectories at the display resolutions the dashboard uses."""
    n = len(t)
    if n <= target_points:
        return t, y
    idx = np.unique(np.linspace(0, n - 1, target_points).astype(int))
    return t[idx], y[idx]


def export_result(experiment_id: str, payload: dict, *, also_to_web: bool = True) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{experiment_id}.json"
    dump_json(payload, str(out_path))
    if also_to_web:
        WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
        dump_json(payload, str(WEB_DATA_DIR / f"{experiment_id}.json"))
    return out_path
