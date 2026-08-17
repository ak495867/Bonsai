from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .decay import estimate_exponential_decay
from .drivers import strategy_driver_exposure
from .engine import build_ic_table, build_signal, forward_returns
from .models import MarketPanel, ObservedStrategy, StrategySpec


def evaluate_strategy(
    panel: MarketPanel,
    spec: StrategySpec,
    horizons: Iterable[int],
    method: str = "spearman",
    min_obs: int = 10,
) -> ObservedStrategy:
    signal = build_signal(panel, spec)
    forward = forward_returns(panel.returns, horizons)
    ic_table = build_ic_table(signal, forward, method, min_obs)
    mean_ic = ic_table.mean(axis=0)
    decay = estimate_exponential_decay(mean_ic)
    drivers = strategy_driver_exposure(panel, signal)
    return ObservedStrategy(
        spec=spec,
        signal=signal,
        forward_returns=forward,
        ic_by_horizon=ic_table,
        decay=decay,
        drivers=drivers,
    )


def _summary_row(result: ObservedStrategy) -> dict[str, object]:
    row = {
        "strategy_id": result.spec.strategy_id,
        "transform": result.spec.transform,
        "holding_period": result.spec.holding_period,
        "signal_lag": result.spec.signal_lag,
        "feature_count": len(result.spec.feature_columns),
        "features": "|".join(result.spec.feature_columns),
        "weights": "|".join(f"{value:.8g}" for value in result.spec.weights),
        "mean_ic": float(result.ic_by_horizon.iloc[:, 0].mean()),
        "median_ic": float(result.ic_by_horizon.iloc[:, 0].median()),
        "ic_volatility": float(result.ic_by_horizon.iloc[:, 0].std()),
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
    evaluate = lambda spec: evaluate_strategy(panel, spec, horizons, method, min_obs)
    if workers == 1:
        evaluated = [evaluate(spec) for spec in specs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            evaluated = list(executor.map(evaluate, specs))
    evaluated_map = {result.spec.strategy_id: result for result in evaluated}
    summary = pd.DataFrame([_summary_row(result) for result in evaluated])
    return summary, evaluated_map


def ic_long_table(results: dict[str, ObservedStrategy]) -> pd.DataFrame:
    rows = []
    for strategy_id, result in results.items():
        for date, values in result.ic_by_horizon.iterrows():
            for horizon, ic in values.items():
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "date": date,
                        "horizon": int(horizon),
                        "ic": float(ic) if pd.notna(ic) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def strategy_specs_to_frame(strategies: Iterable[StrategySpec]) -> pd.DataFrame:
    rows = []
    for spec in strategies:
        row = asdict(spec)
        row["feature_columns"] = "|".join(spec.feature_columns)
        row["weights"] = "|".join(str(value) for value in spec.weights)
        row["tags"] = "|".join(spec.tags)
        rows.append(row)
    return pd.DataFrame(rows)
