from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import MarketPanel


@dataclass(frozen=True)
class SyntheticConfig:
    n_days: int = 756
    n_assets: int = 250
    n_features: int = 24
    seed: int = 7
    start: str = "2018-01-02"
    signal_strength: float = 0.08
    decay_rate: float = 0.012
    regime_count: int = 4

    def validate(self) -> None:
        if min(self.n_days, self.n_assets, self.n_features, self.regime_count) < 2:
            raise ValueError("dimensions must be at least 2")
        if self.signal_strength <= 0 or self.decay_rate < 0:
            raise ValueError(
                "signal_strength must be positive and decay_rate must be nonnegative"
            )


def make_synthetic_panel(config: SyntheticConfig = SyntheticConfig()) -> MarketPanel:
    config.validate()
    rng = np.random.default_rng(config.seed)
    dates = pd.bdate_range(config.start, periods=config.n_days)
    assets = pd.Index([f"A{i:04d}" for i in range(config.n_assets)], name="asset")
    feature_names = [f"feature_{i:02d}" for i in range(config.n_features)]
    regime = np.repeat(
        np.arange(config.regime_count),
        int(np.ceil(config.n_days / config.regime_count)),
    )[: config.n_days]
    regimes = pd.Series(regime, index=dates, name="regime")
    latent = rng.normal(size=(config.n_days, config.n_assets))
    market = rng.normal(size=config.n_days)
    innovations = latent * 0.8 + market[:, None] * 0.2
    returns = innovations * 0.01
    feature_matrix = {}
    for feature_index, name in enumerate(feature_names):
        base = rng.normal(size=(config.n_days, config.n_assets))
        cross_section = np.roll(returns, feature_index + 1, axis=0)
        cross_section[: feature_index + 1] = 0.0
        persistence = np.zeros_like(base)
        persistence[0] = base[0]
        for day in range(1, config.n_days):
            persistence[day] = 0.92 * persistence[day - 1] + 0.39 * base[day]
        lag = feature_index % 5 + 1
        informative = np.roll(returns, lag, axis=0)
        informative[:lag] = 0.0
        loading = config.signal_strength * np.exp(-config.decay_rate * feature_index)
        feature_matrix[name] = (
            persistence + loading * informative + 0.05 * cross_section
        )
    features = pd.concat(
        {
            name: pd.DataFrame(value, index=dates, columns=assets)
            for name, value in feature_matrix.items()
        },
        axis=1,
    )
    features.columns = pd.MultiIndex.from_tuples(
        features.columns, names=["feature", "asset"]
    )
    features = features.swaplevel(axis=1).sort_index(axis=1)
    features.columns = pd.Index(
        [f"{asset}__{feature}" for asset, feature in features.columns]
    )
    noise = rng.normal(scale=0.006, size=returns.shape)
    prices = 100.0 * np.exp(np.cumsum(returns + noise, axis=0))
    market_caps = np.exp(rng.normal(9.5, 1.3, size=config.n_assets))
    market_cap_frame = pd.DataFrame(
        np.broadcast_to(market_caps, (config.n_days, config.n_assets)),
        index=dates,
        columns=assets,
    )
    rolling_vol = (
        pd.DataFrame(returns, index=dates, columns=assets)
        .rolling(20, min_periods=5)
        .std()
        .bfill()
    )
    dollar_volume = market_cap_frame * (0.004 + rolling_vol.fillna(0.02))
    metadata = pd.DataFrame(
        {
            "market_cap": market_cap_frame.mean(axis=0),
            "volatility": rolling_vol.mean(axis=0),
            "turnover": (0.03 + rolling_vol.mean(axis=0) * 0.8).clip(0.01, 1.0),
            "liquidity": dollar_volume.mean(axis=0),
            "crowding": pd.Series(rng.uniform(0, 1, config.n_assets), index=assets),
            "regime": pd.Series(regime[-1], index=assets),
        },
        index=assets,
    )
    return MarketPanel(
        prices=pd.DataFrame(prices, index=dates, columns=assets),
        features=features,
        returns=pd.DataFrame(returns, index=dates, columns=assets),
        metadata=metadata,
    )
