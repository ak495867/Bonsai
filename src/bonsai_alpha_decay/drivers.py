from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models import MarketPanel

DRIVER_NAMES = (
    "crowding",
    "volatility",
    "turnover",
    "liquidity",
    "market_cap",
    "regime_changes",
)


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    left = pair.iloc[:, 0].rank(method="average").to_numpy(dtype=float)
    right = pair.iloc[:, 1].rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(left, right)[0, 1])


def _regime_change_series(panel: MarketPanel) -> pd.Series:
    market_return = panel.returns.mean(axis=1)
    market_vol = market_return.rolling(20, min_periods=10).std()
    return market_vol.pct_change().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def strategy_driver_exposure(
    panel: MarketPanel, signal: pd.DataFrame
) -> dict[str, float]:
    metadata = panel.metadata.reindex(signal.columns)
    absolute_signal = signal.abs()
    exposures = {}
    for name in DRIVER_NAMES[:-1]:
        values = []
        driver = metadata[name]
        if name == "liquidity":
            driver = np.log1p(driver)
        if name == "market_cap":
            driver = np.log1p(driver)
        for date in signal.index:
            values.append(_safe_corr(absolute_signal.loc[date], driver))
        exposures[name] = (
            float(np.nanmean(np.abs(values))) if np.isfinite(values).any() else np.nan
        )
    regime_changes = _regime_change_series(panel)
    signal_dispersion = signal.rank(axis=1, pct=True).std(axis=1)
    exposures["regime_changes"] = _safe_corr(
        signal_dispersion, regime_changes.reindex(signal.index).fillna(0.0)
    )
    return exposures


def fit_driver_model(records: pd.DataFrame, target: str = "lambda") -> dict[str, Any]:
    required = set(DRIVER_NAMES).union({target})
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"records are missing columns: {sorted(missing)}")
    clean = (
        records[list(DRIVER_NAMES) + [target]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(clean) < len(DRIVER_NAMES) + 2:
        return {
            "target": target,
            "n_observations": int(len(clean)),
            "status": "insufficient_observations",
            "intercept": np.nan,
            "coefficients": {},
            "r2": np.nan,
        }
    x = clean[list(DRIVER_NAMES)].to_numpy(dtype=float)
    y = clean[target].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coefficients
    residual = y - fitted
    total = y - y.mean()
    r2 = (
        float(1.0 - np.dot(residual, residual) / np.dot(total, total))
        if np.dot(total, total) > 0
        else np.nan
    )
    return {
        "target": target,
        "n_observations": int(len(clean)),
        "status": "ok",
        "intercept": float(coefficients[0]),
        "coefficients": {
            name: float(value) for name, value in zip(DRIVER_NAMES, coefficients[1:])
        },
        "r2": r2,
    }
