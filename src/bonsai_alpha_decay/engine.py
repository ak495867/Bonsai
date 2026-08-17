from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .models import MarketPanel, StrategySpec


def _feature_frame(panel: MarketPanel, feature: str) -> pd.DataFrame:
    prefix = "__" + feature
    columns = [column for column in panel.features.columns if column.endswith(prefix)]
    if not columns:
        raise ValueError(f"feature not found: {feature}")
    assets = [column.split("__", 1)[0] for column in columns]
    result = panel.features.loc[:, columns].copy()
    result.columns = assets
    result = result.reindex(columns=panel.returns.columns)
    return result


def build_signal(panel: MarketPanel, spec: StrategySpec) -> pd.DataFrame:
    panel.validate()
    spec.validate([column.split("__", 1)[1] for column in panel.features.columns])
    score = pd.DataFrame(0.0, index=panel.returns.index, columns=panel.returns.columns)
    for feature, weight in zip(spec.feature_columns, spec.weights):
        score = score.add(_feature_frame(panel, feature).mul(weight), fill_value=0.0)
    if spec.transform == "rank":
        score = score.rank(axis=1, pct=True)
    elif spec.transform == "zscore":
        mean = score.mean(axis=1)
        std = score.std(axis=1).replace(0.0, np.nan)
        score = score.sub(mean, axis=0).div(std, axis=0)
    elif spec.transform != "linear":
        raise ValueError(f"unsupported transform: {spec.transform}")
    if spec.universe_quantile is not None:
        ranks = score.rank(axis=1, pct=True)
        q = spec.universe_quantile
        score = score.where((ranks <= q) | (ranks >= 1.0 - q), 0.0)
    score = score.replace([np.inf, -np.inf], np.nan)
    return score.shift(spec.signal_lag)


def forward_returns(
    returns: pd.DataFrame, horizons: Iterable[int]
) -> dict[int, pd.DataFrame]:
    if (
        not isinstance(returns.index, pd.DatetimeIndex)
        or not returns.index.is_monotonic_increasing
    ):
        raise ValueError("returns must have a sorted DatetimeIndex")
    output: dict[int, pd.DataFrame] = {}
    for horizon in sorted(set(int(value) for value in horizons)):
        if horizon < 1:
            raise ValueError("horizons must be positive")
        compounded = (1.0 + returns).rolling(horizon, min_periods=horizon).apply(
            np.prod, raw=True
        ) - 1.0
        output[horizon] = compounded.shift(-horizon)
    return output


def cross_sectional_ic(
    signal: pd.DataFrame,
    target: pd.DataFrame,
    method: str = "spearman",
    min_obs: int = 10,
) -> pd.Series:
    if not signal.index.equals(target.index) or not signal.columns.equals(
        target.columns
    ):
        raise ValueError("signal and target must have identical indexes and columns")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    values = []
    for date in signal.index:
        pair = pd.concat([signal.loc[date], target.loc[date]], axis=1).dropna()
        if (
            len(pair) < min_obs
            or pair.iloc[:, 0].nunique() < 2
            or pair.iloc[:, 1].nunique() < 2
        ):
            values.append(np.nan)
            continue
        left = (
            pair.iloc[:, 0].rank(method="average").to_numpy(dtype=float)
            if method == "spearman"
            else pair.iloc[:, 0].to_numpy(dtype=float)
        )
        right = (
            pair.iloc[:, 1].rank(method="average").to_numpy(dtype=float)
            if method == "spearman"
            else pair.iloc[:, 1].to_numpy(dtype=float)
        )
        values.append(float(np.corrcoef(left, right)[0, 1]))
    return pd.Series(values, index=signal.index, name="ic")


def build_ic_table(
    signal: pd.DataFrame,
    forward: dict[int, pd.DataFrame],
    method: str = "spearman",
    min_obs: int = 10,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            horizon: cross_sectional_ic(signal, target, method, min_obs)
            for horizon, target in forward.items()
        }
    )
