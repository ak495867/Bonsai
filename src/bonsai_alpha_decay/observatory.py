from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Iterable
from scipy.stats import norm
import numpy as np
import pandas as pd

from .decay import estimate_exponential_decay
from .drivers import strategy_driver_exposure
from .engine import (
    build_ic_table,
    build_signal,
    calculate_turnover,
    forward_returns,
    newey_west_ic_stats,
)
from .models import MarketPanel, ObservedStrategy, StrategySpec
from .portfolio import portfolio_performance_stats, simulate_long_short_pnl
from .validation import benjamini_hochberg_fdr, deflated_sharpe_ratio


def evaluate_strategy(
    panel: MarketPanel,
    spec: StrategySpec,
    horizons: Iterable[int],
    method: str = "spearman",
    min_obs: int = 10,
    precomputed_forward: dict[int, pd.DataFrame] | None = None,
    simulate_pnl: bool = False,
    cost_bps: float = 5.0,
) -> ObservedStrategy:
    signal = build_signal(panel, spec)
    forward = (
        precomputed_forward
        if precomputed_forward is not None
        else forward_returns(panel.returns, horizons)
    )
    ic_table = build_ic_table(signal, forward, method, min_obs)
    mean_ic = ic_table.mean(axis=0)
    decay = estimate_exponential_decay(mean_ic)
    drivers = strategy_driver_exposure(panel, signal)
    turnover_series = calculate_turnover(signal)
    mean_turnover = float(turnover_series.mean()) if len(turnover_series) > 0 else np.nan
    h1_col = ic_table.iloc[:, 0]
    nw_stats = newey_west_ic_stats(h1_col)
    stats_dict = {
        "strategy_turnover": mean_turnover,
        "ic_tstat": nw_stats["t_stat"],
        "ic_hac_tstat": nw_stats["hac_t_stat"],
        "ic_ir": nw_stats["ir"],
    }
    pnl_df = None
    if simulate_pnl:
        pnl_df = simulate_long_short_pnl(signal, panel.returns, cost_bps=cost_bps)
        gross_stats = portfolio_performance_stats(pnl_df["gross_pnl"])
        net_stats = portfolio_performance_stats(pnl_df["net_pnl"])
        stats_dict["gross_sharpe"] = gross_stats["sharpe_ratio"]
        stats_dict["net_sharpe"] = net_stats["sharpe_ratio"]
        stats_dict["net_annualized_return"] = net_stats["annualized_return"]
        stats_dict["net_max_drawdown"] = net_stats["max_drawdown"]
        stats_dict["net_calmar"] = net_stats["calmar_ratio"]
        stats_dict["net_win_rate"] = net_stats["win_rate"]

    drivers_with_stats = dict(drivers)
    drivers_with_stats.update(stats_dict)
    return ObservedStrategy(
        spec=spec,
        signal=signal,
        forward_returns=forward,
        ic_by_horizon=ic_table,
        decay=decay,
        drivers=drivers_with_stats,
        pnl=pnl_df,
    )


def _summary_row(result: ObservedStrategy) -> dict[str, object]:
    h1_col = result.ic_by_horizon.iloc[:, 0]
    row = {
        "strategy_id": result.spec.strategy_id,
        "transform": result.spec.transform,
        "holding_period": result.spec.holding_period,
        "signal_lag": result.spec.signal_lag,
        "feature_count": len(result.spec.feature_columns),
        "features": "|".join(result.spec.feature_columns),
        "weights": "|".join(f"{value:.8g}" for value in result.spec.weights),
        "mean_ic": float(h1_col.mean()),
        "median_ic": float(h1_col.median()),
        "ic_volatility": float(h1_col.std()),
    }
    row.update(result.decay)
    row.update(result.drivers)
    return row


def run_observatory(
    panel: MarketPanel,
    strategies: Iterable[StrategySpec],
    horizons: Iterable[int] = (1, 2, 3, 5, 10, 20),
    method: str = "spearman",
    min_obs: int = 10,
    workers: int = 1,
    simulate_pnl: bool = False,
    cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, ObservedStrategy]]:
    panel.validate()
    specs = list(strategies)
    if not specs:
        raise ValueError("strategies must not be empty")
    if workers < 1:
        raise ValueError("workers must be positive")
    unique_ids = [spec.strategy_id for spec in specs]
    if len(set(unique_ids)) != len(unique_ids):
        raise ValueError("strategy_id values must be unique")
    forward = forward_returns(panel.returns, horizons)
    evaluate = lambda spec: evaluate_strategy(
        panel,
        spec,
        horizons,
        method,
        min_obs,
        precomputed_forward=forward,
        simulate_pnl=simulate_pnl,
        cost_bps=cost_bps,
    )
    if workers == 1:
        evaluated = [evaluate(spec) for spec in specs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            evaluated = list(executor.map(evaluate, specs))
    evaluated_map = {result.spec.strategy_id: result for result in evaluated}
    summary = pd.DataFrame([_summary_row(result) for result in evaluated])

    if "ic_hac_tstat" in summary.columns and len(summary) > 0:
        t_vals = np.abs(summary["ic_hac_tstat"].to_numpy(dtype=float))
        p_vals = 2.0 * (1.0 - norm.cdf(np.where(np.isfinite(t_vals), t_vals, 0.0)))
        summary["is_fdr_significant"] = benjamini_hochberg_fdr(p_vals)

    if simulate_pnl and "net_sharpe" in summary.columns and len(summary) > 1:
        net_sharpes = summary["net_sharpe"].to_numpy(dtype=float)
        n_periods = len(panel.returns)
        dsr_vals = [
            deflated_sharpe_ratio(sr, net_sharpes, n_periods)
            for sr in net_sharpes
        ]
        summary["deflated_sharpe_ratio"] = dsr_vals

    return summary, evaluated_map


def ic_long_table(results: dict[str, ObservedStrategy]) -> pd.DataFrame:
    frames = []
    for strategy_id, result in results.items():
        if result.ic_by_horizon.empty:
            continue
        melted = result.ic_by_horizon.reset_index().melt(
            id_vars=["index"], var_name="horizon", value_name="ic"
        )
        melted.rename(columns={"index": "date"}, inplace=True)
        melted["horizon"] = melted["horizon"].astype(int)
        melted.insert(0, "strategy_id", strategy_id)
        frames.append(melted)
    if not frames:
        return pd.DataFrame(columns=["strategy_id", "date", "horizon", "ic"])
    return pd.concat(frames, ignore_index=True)


def strategy_specs_to_frame(strategies: Iterable[StrategySpec]) -> pd.DataFrame:
    rows = []
    for spec in strategies:
        row = asdict(spec)
        row["feature_columns"] = "|".join(spec.feature_columns)
        row["weights"] = "|".join(str(value) for value in spec.weights)
        row["tags"] = "|".join(spec.tags)
        rows.append(row)
    return pd.DataFrame(rows)
