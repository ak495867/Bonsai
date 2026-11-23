from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .decay import estimate_exponential_decay
from .models import MarketPanel


def detect_market_regimes(
    panel: MarketPanel,
    method: str = "volatility",
    n_regimes: int = 2,
    window: int = 20,
) -> pd.Series:
    """
    Classifies market dates into distinct macroeconomic / liquidity regimes.

    Methods:
        - "volatility": Splits dates by rolling market volatility (low_vol vs high_vol)
        - "trend": Splits dates into bull, bear, or sideways regimes
        - "liquidity": Splits dates by market-wide dollar liquidity volume
    """
    dates = panel.returns.index
    mkt_return = panel.returns.mean(axis=1)

    if method == "volatility":
        rolling_vol = mkt_return.rolling(window, min_periods=max(5, window // 2)).std().bfill()
        if n_regimes == 2:
            median_vol = float(rolling_vol.median())
            labels = np.where(rolling_vol >= median_vol, "high_vol", "low_vol")
        elif n_regimes == 3:
            q33 = float(rolling_vol.quantile(0.33))
            q66 = float(rolling_vol.quantile(0.66))
            labels = np.where(
                rolling_vol >= q66,
                "crisis_vol",
                np.where(rolling_vol <= q33, "low_vol", "normal_vol"),
            )
        else:
            raise ValueError("n_regimes must be 2 or 3 for volatility method")
        return pd.Series(labels, index=dates, name="regime")

    if method == "trend":
        cum_ret = (1.0 + mkt_return).cumprod()
        fast_ma = cum_ret.rolling(window, min_periods=5).mean()
        slow_ma = cum_ret.rolling(window * 3, min_periods=10).mean()
        ratio = (fast_ma / slow_ma.replace(0.0, np.nan)).bfill()
        labels = np.where(ratio > 1.01, "bull", np.where(ratio < 0.99, "bear", "sideways"))
        return pd.Series(labels, index=dates, name="regime")

    if method == "liquidity":
        if "liquidity" in panel.metadata.columns:
            mkt_vol = panel.returns.rolling(window, min_periods=5).std().mean(axis=1)
            med = float(mkt_vol.median())
            labels = np.where(mkt_vol >= med, "illiquid_stressed", "liquid_normal")
            return pd.Series(labels, index=dates, name="regime")

    raise ValueError(f"Unsupported regime detection method: {method}")


def estimate_regime_conditional_decay(
    ic_table: pd.DataFrame,
    regimes: pd.Series,
) -> dict[str, Any]:
    """
    Computes alpha decay parameters conditionally across market regimes to diagnose
    alpha fragility under stress.
    """
    aligned_regimes = regimes.reindex(ic_table.index).dropna()
    unique_regimes = sorted(aligned_regimes.unique())

    regime_results = {}
    for r in unique_regimes:
        mask = aligned_regimes == r
        sub_ic = ic_table.loc[mask]
        n_dates = len(sub_ic)
        if n_dates < 3:
            regime_results[r] = {
                "n_dates": n_dates,
                "status": "insufficient_dates",
                "mean_ic_h1": np.nan,
                "lambda": np.nan,
                "half_life": np.nan,
            }
            continue

        mean_ic = sub_ic.mean(axis=0)
        decay_fit = estimate_exponential_decay(mean_ic)
        h1 = sub_ic.columns[0]
        regime_results[r] = {
            "n_dates": n_dates,
            "status": decay_fit["status"],
            "mean_ic_h1": float(mean_ic[h1]) if len(mean_ic) > 0 else np.nan,
            "lambda": decay_fit["lambda"],
            "half_life": decay_fit["half_life"],
            "ic0": decay_fit["ic0"],
            "r2": decay_fit["r2"],
        }

    # Evaluate Fragility: compare high_vol / crisis_vol to low_vol
    stress_key = "high_vol" if "high_vol" in regime_results else ("crisis_vol" if "crisis_vol" in regime_results else None)
    calm_key = "low_vol" if "low_vol" in regime_results else None

    hl_ratio = np.nan
    is_fair_weather = False
    is_all_weather = False
    classification = "unknown"

    if stress_key and calm_key:
        hl_stress = regime_results[stress_key].get("half_life", np.nan)
        hl_calm = regime_results[calm_key].get("half_life", np.nan)
        ic_stress = regime_results[stress_key].get("mean_ic_h1", np.nan)
        ic_calm = regime_results[calm_key].get("mean_ic_h1", np.nan)

        if np.isfinite(hl_stress) and np.isfinite(hl_calm) and hl_calm > 0:
            hl_ratio = float(hl_stress / hl_calm)

        if ic_calm > 0 and (ic_stress <= 0 or (np.isfinite(hl_ratio) and hl_ratio < 0.5)):
            is_fair_weather = True
            classification = "fair_weather"
        elif ic_calm > 0 and ic_stress > 0 and (np.isfinite(hl_ratio) and hl_ratio >= 0.7):
            is_all_weather = True
            classification = "all_weather"
        elif ic_stress > ic_calm and ic_stress > 0:
            classification = "crisis_alpha"
        else:
            classification = "fragile"

    return {
        "regimes": regime_results,
        "stress_regime": stress_key,
        "calm_regime": calm_key,
        "half_life_stress_to_calm_ratio": hl_ratio,
        "is_fair_weather": is_fair_weather,
        "is_all_weather": is_all_weather,
        "classification": classification,
    }
