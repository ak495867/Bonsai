from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .decay import estimate_exponential_decay


def rolling_decay_profile(
    ic_table: pd.DataFrame,
    window: int = 63,
    step: int = 5,
    method: str = "auto",
) -> pd.DataFrame:
    if len(ic_table) < window:
        raise ValueError(f"ic_table length ({len(ic_table)}) must be at least window ({window})")
    rows = []
    n_rows = len(ic_table)
    for end_idx in range(window, n_rows + 1, step):
        sub_table = ic_table.iloc[end_idx - window : end_idx]
        as_of_date = ic_table.index[end_idx - 1]
        mean_ic = sub_table.mean(axis=0)
        fit = estimate_exponential_decay(mean_ic, method=method)
        row = {
            "date": as_of_date,
            "lambda": fit["lambda"],
            "half_life": fit["half_life"],
            "ic0": fit["ic0"],
            "r2": fit["r2"],
            "status": fit["status"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def decay_compression_rate(
    rolling_profile: pd.DataFrame,
) -> dict[str, float]:
    valid = rolling_profile.dropna(subset=["lambda"]).copy()
    valid = valid[valid["status"] == "ok"]
    if len(valid) < 3:
        return {
            "lambda_trend": np.nan,
            "half_life_compression": np.nan,
            "initial_half_life": np.nan,
            "final_half_life": np.nan,
            "is_crowding": np.nan,
        }
    y = valid["lambda"].to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    h_initial = float(valid["half_life"].iloc[0])
    h_final = float(valid["half_life"].iloc[-1])
    compression = float(h_final / h_initial) if h_initial > 0.0 else np.nan
    is_crowding = float(1.0 if (slope > 0.0 and compression < 1.0) else 0.0)
    return {
        "lambda_trend": slope,
        "half_life_compression": compression,
        "initial_half_life": h_initial,
        "final_half_life": h_final,
        "is_crowding": is_crowding,
    }
