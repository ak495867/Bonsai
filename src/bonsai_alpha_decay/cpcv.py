from __future__ import annotations

from itertools import combinations
from typing import Any, Iterator, Sequence
import numpy as np
import pandas as pd

from .engine import build_signal, forward_returns
from .models import MarketPanel, StrategySpec
from .portfolio import portfolio_performance_stats, simulate_long_short_pnl


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV) based on Marcos Lopez de Prado.

    Splits the timeline into N temporal groups and evaluates all C(N, k) combinatorial
    combinations of k groups as test sets. Purges training data that overlaps with
    test forward-looking horizons and embargos post-test observations to prevent
    autoregressive leakage.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_splits < 3 or n_test_splits < 1 or n_test_splits >= n_splits:
            raise ValueError(
                f"Invalid split configuration: n_splits={n_splits}, n_test_splits={n_test_splits}"
            )
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_window = max(0, int(purge_window))
        self.embargo_pct = max(0.0, float(embargo_pct))

    def split(
        self, dates: Sequence[pd.Timestamp] | pd.DatetimeIndex
    ) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        n_samples = len(dates)
        if n_samples < self.n_splits * 2:
            raise ValueError(f"Insufficient samples ({n_samples}) for {self.n_splits} splits")

        indices = np.arange(n_samples)
        block_bounds = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)
        blocks = [
            indices[block_bounds[i] : block_bounds[i + 1]]
            for i in range(self.n_splits)
        ]
        embargo_samples = int(np.ceil(self.embargo_pct * n_samples))

        comb_id = 0
        for test_block_ids in combinations(range(self.n_splits), self.n_test_splits):
            test_indices_list = [blocks[b_id] for b_id in test_block_ids]
            test_idx = np.concatenate(test_indices_list)
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[test_idx] = True

            train_mask = ~test_mask

            # Apply Purging and Embargoing for each contiguous test segment
            for b_id in test_block_ids:
                b_start = block_bounds[b_id]
                b_end = block_bounds[b_id + 1]

                # Purge training samples immediately preceding test segment
                purge_start = max(0, b_start - self.purge_window)
                train_mask[purge_start:b_start] = False

                # Embargo training samples immediately following test segment
                embargo_end = min(n_samples, b_end + embargo_samples)
                train_mask[b_end:embargo_end] = False

            train_idx = np.where(train_mask)[0]
            yield train_idx, test_idx, comb_id
            comb_id += 1


def evaluate_strategy_cpcv(
    panel: MarketPanel,
    spec: StrategySpec,
    cpcv: CombinatorialPurgedCV | None = None,
    cost_bps: float = 5.0,
    weighting_scheme: str = "quantile",
) -> dict[str, Any]:
    panel.validate()
    if cpcv is None:
        cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, purge_window=5)

    signal = build_signal(panel, spec)
    dates = panel.returns.index
    pnl_df = simulate_long_short_pnl(
        signal,
        panel.returns,
        cost_bps=cost_bps,
        weighting_scheme=weighting_scheme,
    )
    net_pnl = pnl_df["net_pnl"]

    path_results = []
    is_sharpes = []
    oos_sharpes = []

    for train_idx, test_idx, path_id in cpcv.split(dates):
        train_pnl = net_pnl.iloc[train_idx]
        test_pnl = net_pnl.iloc[test_idx]

        is_perf = portfolio_performance_stats(train_pnl)
        oos_perf = portfolio_performance_stats(test_pnl)

        is_sr = float(is_perf["sharpe_ratio"]) if np.isfinite(is_perf["sharpe_ratio"]) else np.nan
        oos_sr = float(oos_perf["sharpe_ratio"]) if np.isfinite(oos_perf["sharpe_ratio"]) else np.nan

        is_sharpes.append(is_sr)
        oos_sharpes.append(oos_sr)

        path_results.append(
            {
                "path_id": path_id,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "is_sharpe": is_sr,
                "oos_sharpe": oos_sr,
                "oos_annualized_return": float(oos_perf["annualized_return"]),
                "oos_max_drawdown": float(oos_perf["max_drawdown"]),
            }
        )

    valid_oos = np.array([sr for sr in oos_sharpes if np.isfinite(sr)])
    valid_is = np.array([sr for sr in is_sharpes if np.isfinite(sr)])

    mean_oos = float(np.mean(valid_oos)) if len(valid_oos) > 0 else np.nan
    median_oos = float(np.median(valid_oos)) if len(valid_oos) > 0 else np.nan
    std_oos = float(np.std(valid_oos, ddof=1)) if len(valid_oos) > 1 else np.nan
    mean_is = float(np.mean(valid_is)) if len(valid_is) > 0 else np.nan

    # Probability of Backtest Overfitting (PBO): proportion of OOS Sharpe ratios <= 0
    pbo = float(np.mean(valid_oos <= 0.0)) if len(valid_oos) > 0 else np.nan
    degradation = float(mean_oos / mean_is) if (mean_is and np.isfinite(mean_is) and mean_is != 0) else np.nan

    return {
        "strategy_id": spec.strategy_id,
        "n_paths": len(path_results),
        "mean_is_sharpe": mean_is,
        "mean_oos_sharpe": mean_oos,
        "median_oos_sharpe": median_oos,
        "std_oos_sharpe": std_oos,
        "pbo": pbo,
        "degradation_ratio": degradation,
        "paths": path_results,
    }
