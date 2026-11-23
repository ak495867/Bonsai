from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd


def build_portfolio_weights(
    signal: pd.DataFrame,
    weighting_scheme: str = "quantile",
    quantiles: int = 5,
    volatility: pd.DataFrame | None = None,
    target_leverage: float = 1.0,
) -> pd.DataFrame:
    valid_schemes = {"quantile", "zscore", "vol_parity", "inverse_vol", "linear_rank"}
    if weighting_scheme not in valid_schemes:
        raise ValueError(
            f"weighting_scheme '{weighting_scheme}' must be one of {sorted(valid_schemes)}"
        )
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")

    if weighting_scheme == "quantile":
        ranks = signal.rank(axis=1, pct=True)
        long_thresh = 1.0 - (1.0 / quantiles)
        short_thresh = 1.0 / quantiles
        longs = (ranks >= long_thresh).astype(float)
        shorts = (ranks <= short_thresh).astype(float)
        long_denom = longs.sum(axis=1).replace(0.0, np.nan)
        short_denom = shorts.sum(axis=1).replace(0.0, np.nan)
        long_w = longs.div(long_denom, axis=0).fillna(0.0) * target_leverage
        short_w = shorts.div(short_denom, axis=0).fillna(0.0) * target_leverage
        return long_w - short_w

    if weighting_scheme == "zscore":
        mean_s = signal.mean(axis=1)
        std_s = signal.std(axis=1).replace(0.0, np.nan)
        z = signal.sub(mean_s, axis=0).div(std_s, axis=0).fillna(0.0)
        pos = z.clip(lower=0.0)
        neg = (-z).clip(lower=0.0)
        pos_sum = pos.sum(axis=1).replace(0.0, np.nan)
        neg_sum = neg.sum(axis=1).replace(0.0, np.nan)
        long_w = pos.div(pos_sum, axis=0).fillna(0.0) * target_leverage
        short_w = neg.div(neg_sum, axis=0).fillna(0.0) * target_leverage
        return long_w - short_w

    if weighting_scheme in {"vol_parity", "inverse_vol"}:
        vol_clean = (
            volatility.reindex(index=signal.index, columns=signal.columns)
            .replace(0.0, np.nan)
            .fillna(0.02)
            if volatility is not None
            else pd.DataFrame(0.02, index=signal.index, columns=signal.columns)
        )
        vol_adj_sig = signal.div(vol_clean, axis=0)
        mean_s = vol_adj_sig.mean(axis=1)
        centered = vol_adj_sig.sub(mean_s, axis=0).fillna(0.0)
        pos = centered.clip(lower=0.0)
        neg = (-centered).clip(lower=0.0)
        pos_sum = pos.sum(axis=1).replace(0.0, np.nan)
        neg_sum = neg.sum(axis=1).replace(0.0, np.nan)
        long_w = pos.div(pos_sum, axis=0).fillna(0.0) * target_leverage
        short_w = neg.div(neg_sum, axis=0).fillna(0.0) * target_leverage
        return long_w - short_w

    if weighting_scheme == "linear_rank":
        ranks = signal.rank(axis=1, pct=True)
        centered = ranks.sub(ranks.mean(axis=1), axis=0).fillna(0.0)
        pos = centered.clip(lower=0.0)
        neg = (-centered).clip(lower=0.0)
        pos_sum = pos.sum(axis=1).replace(0.0, np.nan)
        neg_sum = neg.sum(axis=1).replace(0.0, np.nan)
        long_w = pos.div(pos_sum, axis=0).fillna(0.0) * target_leverage
        short_w = neg.div(neg_sum, axis=0).fillna(0.0) * target_leverage
        return long_w - short_w

    raise ValueError(f"unsupported weighting_scheme: {weighting_scheme}")


def compute_dynamic_borrow_rates(
    crowding: pd.Series | pd.DataFrame,
    market_cap: pd.Series | pd.DataFrame,
    base_fee_bps: float = 50.0,
    max_fee_bps: float = 3000.0,
) -> pd.Series | pd.DataFrame:
    if isinstance(crowding, pd.DataFrame) and isinstance(market_cap, pd.DataFrame):
        log_mcap = np.log(market_cap.replace(0.0, np.nan).fillna(1e8))
        cap_penalty = np.clip(12.0 - log_mcap, 0.0, 10.0) * 0.15
        crowd_penalty = (crowding.fillna(0.0) ** 2) * 4.0
        multiplier = 1.0 + crowd_penalty + cap_penalty
        rates = (base_fee_bps * multiplier).clip(base_fee_bps, max_fee_bps)
        return rates
    c_s = crowding.to_numpy(dtype=float) if isinstance(crowding, pd.Series) else np.asarray(crowding, dtype=float)
    m_s = market_cap.to_numpy(dtype=float) if isinstance(market_cap, pd.Series) else np.asarray(market_cap, dtype=float)
    log_m = np.log(np.where(m_s > 0, m_s, 1e8))
    cap_pen = np.clip(12.0 - log_m, 0.0, 10.0) * 0.15
    crowd_pen = (np.nan_to_num(c_s) ** 2) * 4.0
    mult = 1.0 + crowd_pen + cap_pen
    rates = np.clip(base_fee_bps * mult, base_fee_bps, max_fee_bps)
    if isinstance(crowding, pd.Series):
        return pd.Series(rates, index=crowding.index)
    return rates


def simulate_long_short_pnl(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    quantiles: int = 5,
    cost_bps: float = 5.0,
    aum: float = 1e7,
    adv: pd.DataFrame | None = None,
    volatility: pd.DataFrame | None = None,
    impact_gamma: float = 0.0,
    impact_eta: float = 0.0,
    borrow_fee_bps: float | pd.Series | pd.DataFrame = 50.0,
    weighting_scheme: str = "quantile",
    target_leverage: float = 1.0,
) -> pd.DataFrame:
    if not signal.index.equals(returns.index) or not signal.columns.equals(returns.columns):
        raise ValueError("signal and returns must share identical index and columns")
    net_w = build_portfolio_weights(
        signal=signal,
        weighting_scheme=weighting_scheme,
        quantiles=quantiles,
        volatility=volatility,
        target_leverage=target_leverage,
    )
    active_weights = net_w.shift(1).fillna(0.0)
    gross_pnl = (active_weights * returns).sum(axis=1)

    delta_w = net_w.diff().abs()
    delta_w.iloc[0] = net_w.iloc[0].abs()
    turnover = delta_w.sum(axis=1) * 0.5
    linear_fee = turnover * (cost_bps / 10000.0)

    trade_dollar = delta_w * aum
    if (impact_gamma > 0.0 or impact_eta > 0.0) and adv is not None and volatility is not None:
        adv_clean = adv.reindex(index=signal.index, columns=signal.columns).replace(0.0, np.nan).fillna(1e6)
        vol_clean = volatility.reindex(index=signal.index, columns=signal.columns).fillna(0.02)
        participation = trade_dollar / adv_clean
        sqrt_impact = impact_gamma * vol_clean * np.sqrt(np.clip(participation, 0.0, 1.0))
        temp_cost = (delta_w * sqrt_impact).sum(axis=1)
        perm_impact = impact_eta * vol_clean * np.clip(participation, 0.0, 1.0)
        perm_cost = (delta_w * perm_impact).sum(axis=1)
        impact_cost = temp_cost + perm_cost
    else:
        temp_cost = pd.Series(0.0, index=signal.index)
        perm_cost = pd.Series(0.0, index=signal.index)
        impact_cost = pd.Series(0.0, index=signal.index)

    short_weights_active = active_weights.clip(upper=0.0).abs()
    if isinstance(borrow_fee_bps, pd.DataFrame):
        rates = borrow_fee_bps.reindex(index=signal.index, columns=signal.columns).ffill().fillna(50.0)
        daily_rate = rates / (10000.0 * 252.0)
        borrow_cost = (short_weights_active * daily_rate).sum(axis=1)
    elif isinstance(borrow_fee_bps, pd.Series):
        rates = borrow_fee_bps.reindex(signal.columns).fillna(50.0)
        daily_rate = rates / (10000.0 * 252.0)
        borrow_cost = (short_weights_active.mul(daily_rate, axis=1)).sum(axis=1)
    else:
        borrow_cost = short_weights_active.sum(axis=1) * (borrow_fee_bps / (10000.0 * 252.0))

    total_cost = linear_fee + impact_cost + borrow_cost
    net_pnl = gross_pnl - total_cost
    cum_gross = (1.0 + gross_pnl).cumprod() - 1.0
    cum_net = (1.0 + net_pnl).cumprod() - 1.0

    return pd.DataFrame(
        {
            "gross_pnl": gross_pnl,
            "turnover": turnover,
            "linear_cost": linear_fee,
            "impact_cost": impact_cost,
            "temp_impact_cost": temp_cost,
            "perm_impact_cost": perm_cost,
            "borrow_cost": borrow_cost,
            "cost": total_cost,
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


def strategy_capacity_curve(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    adv: pd.DataFrame,
    volatility: pd.DataFrame,
    aum_levels: Sequence[float] = (1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8),
    cost_bps: float = 5.0,
    impact_gamma: float = 0.2,
    impact_eta: float = 0.05,
    borrow_fee_bps: float | pd.Series | pd.DataFrame = 50.0,
    weighting_scheme: str = "quantile",
) -> dict[str, Any]:
    curve = []
    max_capacity_aum = float(aum_levels[0])
    for cap in aum_levels:
        sim = simulate_long_short_pnl(
            signal,
            returns,
            cost_bps=cost_bps,
            aum=cap,
            adv=adv,
            volatility=volatility,
            impact_gamma=impact_gamma,
            impact_eta=impact_eta,
            borrow_fee_bps=borrow_fee_bps,
            weighting_scheme=weighting_scheme,
        )
        perf = portfolio_performance_stats(sim["net_pnl"])
        net_sr = perf["sharpe_ratio"]
        curve.append(
            {
                "aum": float(cap),
                "net_sharpe": float(net_sr) if np.isfinite(net_sr) else np.nan,
                "annualized_return": float(perf["annualized_return"]),
                "impact_drag_annualized": float(sim["impact_cost"].mean() * 252.0),
            }
        )
        if np.isfinite(net_sr) and net_sr >= 1.0:
            max_capacity_aum = float(cap)
    return {
        "capacity_threshold_aum": max_capacity_aum,
        "curve": curve,
    }
