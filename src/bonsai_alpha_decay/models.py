from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketPanel:
    prices: pd.DataFrame
    features: pd.DataFrame
    returns: pd.DataFrame
    metadata: pd.DataFrame

    def validate(self) -> None:
        frames = {
            "prices": self.prices,
            "features": self.features,
            "returns": self.returns,
            "metadata": self.metadata,
        }
        for name, frame in frames.items():
            if name != "metadata" and not isinstance(frame.index, pd.DatetimeIndex):
                raise TypeError(f"{name} must use a DatetimeIndex")
            if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
                raise ValueError(f"{name} index must be sorted and unique")
        if not self.prices.index.equals(
            self.features.index
        ) or not self.prices.index.equals(self.returns.index):
            raise ValueError(
                "prices, features, and returns must share an identical date index"
            )
        if not self.prices.columns.equals(self.returns.columns):
            raise ValueError("prices and returns must share identical asset columns")
        if not self.features.columns.is_unique or not self.prices.columns.is_unique:
            raise ValueError("asset and feature columns must be unique")
        if (self.prices <= 0).any().any():
            raise ValueError("prices must be strictly positive")
        if not self.metadata.index.is_unique:
            raise ValueError("metadata index must be unique")
        required = {
            "market_cap",
            "volatility",
            "turnover",
            "liquidity",
            "crowding",
            "regime",
        }
        missing = required.difference(self.metadata.columns)
        if missing:
            raise ValueError(f"metadata is missing required columns: {sorted(missing)}")

    def cutoff(self, as_of: pd.Timestamp) -> MarketPanel:
        cutoff = pd.Timestamp(as_of)
        return MarketPanel(
            prices=self.prices.loc[:cutoff].copy(),
            features=self.features.loc[:cutoff].copy(),
            returns=self.returns.loc[:cutoff].copy(),
            metadata=self.metadata.copy(),
        )


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    feature_columns: tuple[str, ...]
    weights: tuple[float, ...]
    transform: str = "linear"
    holding_period: int = 1
    signal_lag: int = 1
    universe_quantile: float | None = None
    seed: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self, available_features: Sequence[str] | None = None) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id must not be empty")
        if len(self.feature_columns) == 0:
            raise ValueError("feature_columns must not be empty")
        if len(self.feature_columns) != len(self.weights):
            raise ValueError("feature_columns and weights must have equal length")
        if not all(np.isfinite(self.weights)):
            raise ValueError("weights must be finite")
        if self.holding_period < 1 or self.signal_lag < 0:
            raise ValueError(
                "holding_period must be positive and signal_lag must be nonnegative"
            )
        if self.universe_quantile is not None and not 0 < self.universe_quantile <= 0.5:
            raise ValueError("universe_quantile must be in (0, 0.5]")
        if available_features is not None:
            missing = set(self.feature_columns).difference(available_features)
            if missing:
                raise ValueError(
                    f"strategy uses unavailable features: {sorted(missing)}"
                )


@dataclass(frozen=True)
class ObservedStrategy:
    spec: StrategySpec
    signal: pd.DataFrame
    forward_returns: Mapping[int, pd.DataFrame]
    ic_by_horizon: pd.DataFrame
    decay: Mapping[str, Any]
    drivers: Mapping[str, Any]


SignalFunction = Callable[[pd.DataFrame, StrategySpec], pd.DataFrame]
