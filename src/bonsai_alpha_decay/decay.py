from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def estimate_exponential_decay(
    ic_by_horizon: pd.Series | dict[float, float],
) -> dict[str, Any]:
    series = pd.Series(ic_by_horizon, dtype=float).sort_index()
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[valid > 0]
    if len(valid) < 2 or valid.index.nunique() < 2:
        return {
            "ic0": np.nan,
            "lambda": np.nan,
            "half_life": np.nan,
            "r2": np.nan,
            "n_points": int(len(valid)),
            "status": "insufficient_positive_points",
        }
    x = valid.index.to_numpy(dtype=float)
    y = np.log(valid.to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ np.array([intercept, slope])
    residual = y - fitted
    total = y - y.mean()
    r2 = (
        float(1.0 - np.dot(residual, residual) / np.dot(total, total))
        if np.dot(total, total) > 0
        else np.nan
    )
    decay_rate = float(-slope)
    return {
        "ic0": float(np.exp(intercept)),
        "lambda": decay_rate,
        "half_life": float(np.log(2.0) / decay_rate) if decay_rate > 0 else np.inf,
        "r2": r2,
        "n_points": int(len(valid)),
        "status": "ok" if decay_rate > 0 else "non_decay_or_growth",
    }


def bootstrap_decay(
    ic_by_horizon: pd.Series, samples: int = 250, seed: int = 0
) -> pd.DataFrame:
    if samples < 1:
        raise ValueError("samples must be positive")
    series = pd.Series(ic_by_horizon, dtype=float).sort_index()
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(samples):
        noise = rng.normal(0.0, max(float(series.std(ddof=1)) * 0.1, 1e-8), len(series))
        draw = pd.Series(series.to_numpy() + noise, index=series.index)
        rows.append(estimate_exponential_decay(draw))
    return pd.DataFrame(rows)
