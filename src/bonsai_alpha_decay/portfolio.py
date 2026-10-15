from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def simulate_long_short_pnl(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    quantiles: int = 5,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    if not signal.index.equals(returns.index) or not signal.columns.equals(returns.columns):
        raise ValueError("signal and returns must share identical index and columns")
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    ranks = signal.rank(axis=1, pct=True)
    long_thresh = 1.0 - (1.0 / quantiles)
    short_thresh = 1.0 / quantiles
    longs = (ranks >= long_thresh).astype(float)
    shorts = (ranks <= short_thresh).astype(float)
    long_denom = longs.sum(axis=1).replace(0.0, np.nan)
    short_denom = shorts.sum(axis=1).replace(0.0, np.nan)
    long_w = longs.div(long_denom, axis=0).fillna(0.0)
    short_w = shorts.div(short_denom, axis=0).fillna(0.0)
    net_w = long_w - short_w
    active_weights = net_w.shift(1).fillna(0.0)
    gross_pnl = (active_weights * returns).sum(axis=1)
    turnover = net_w.diff().abs().sum(axis=1) * 0.5
    cost = turnover * (cost_bps / 10000.0)
    net_pnl = gross_pnl - cost
    cum_gross = (1.0 + gross_pnl).cumprod() - 1.0
    cum_net = (1.0 + net_pnl).cumprod() - 1.0
    return pd.DataFrame(
        {
            "gross_pnl": gross_pnl,
            "turnover": turnover,
            "cost": cost,
            "net_pnl": net_pnl,
            "cum_gross": cum_gross,
            "cum_net": cum_net,
        },
        index=signal.index,
    )


def portfolio_performance_stats(
    pnl: pd.Series,
    annualization_factor: int = 252,
) -> dict[str, float]:
    clean = pnl.dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < 2:
        return {
            "annualized_return": np.nan,
            "annualized_vol": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "max_drawdown": np.nan,
            "calmar_ratio": np.nan,
            "win_rate": np.nan,
        }
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))
    ann_return = mean * annualization_factor
    ann_vol = std * np.sqrt(annualization_factor)
    sharpe = float(ann_return / ann_vol) if ann_vol > 0.0 else np.nan
    downside = clean[clean < 0]
    downside_std = float(np.std(downside, ddof=1)) * np.sqrt(annualization_factor) if len(downside) > 1 else 0.0
    sortino = float(ann_return / downside_std) if downside_std > 0.0 else np.nan
    cum = np.cumprod(1.0 + clean)
    peaks = np.maximum.accumulate(cum)
    drawdowns = (cum - peaks) / peaks
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    calmar = float(ann_return / abs(max_dd)) if abs(max_dd) > 0.0 else np.nan
    win_rate = float(np.sum(clean > 0) / n)
    return {
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
    }


def optimal_holding_horizon(
    ic0: float,
    decay_rate: float,
    daily_turnover: float,
    cost_bps: float = 5.0,
    annual_vol: float = 0.15,
    max_horizon: int = 30,
) -> dict[str, Any]:
    if decay_rate <= 0.0 or not np.isfinite(decay_rate) or not np.isfinite(ic0) or ic0 <= 0.0:
        return {
            "optimal_horizon": 1,
            "max_net_sharpe": np.nan,
            "profile": [],
        }
    horizons = np.arange(1, max_horizon + 1)
    profile = []
    best_h = 1
    best_net_sharpe = -np.inf
    cost_rate = (cost_bps / 10000.0) * daily_turnover
    for h in horizons:
        expected_ic = ic0 * (1.0 - np.exp(-decay_rate * h)) / (decay_rate * h)
        annual_ic = expected_ic * np.sqrt(252.0 / h)
        gross_return = annual_ic * annual_vol * np.sqrt(252.0)
        annual_cost = cost_rate * (252.0 / h)
        net_return = gross_return - annual_cost
        net_sharpe = net_return / annual_vol
        entry = {
            "horizon": int(h),
            "expected_ic": float(expected_ic),
            "net_sharpe": float(net_sharpe),
        }
        profile.append(entry)
        if net_sharpe > best_net_sharpe:
            best_net_sharpe = net_sharpe
            best_h = int(h)
    return {
        "optimal_horizon": best_h,
        "max_net_sharpe": float(best_net_sharpe),
        "profile": profile,
    }
