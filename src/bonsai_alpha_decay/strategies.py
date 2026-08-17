from __future__ import annotations

from dataclasses import replace

import numpy as np

from .models import StrategySpec


def generate_strategies(
    feature_names: list[str], count: int = 500, seed: int = 11, max_features: int = 4
) -> list[StrategySpec]:
    if not feature_names:
        raise ValueError("feature_names must not be empty")
    if count < 1 or max_features < 1:
        raise ValueError("count and max_features must be positive")
    rng = np.random.default_rng(seed)
    transforms = ("linear", "rank", "zscore")
    output = []
    seen = set()
    index = 0
    while len(output) < count:
        feature_count = int(rng.integers(1, min(max_features, len(feature_names)) + 1))
        selected = tuple(
            sorted(
                rng.choice(feature_names, size=feature_count, replace=False).tolist()
            )
        )
        weights = rng.normal(0.0, 1.0, feature_count)
        weights = weights / max(float(np.abs(weights).sum()), 1e-12)
        transform = transforms[int(rng.integers(0, len(transforms)))]
        holding_period = int(rng.choice([1, 2, 3, 5, 10]))
        signal_lag = int(rng.choice([0, 1, 1, 1, 2]))
        universe_quantile = (
            None if rng.random() < 0.55 else float(rng.choice([0.05, 0.1, 0.2, 0.3]))
        )
        key = (
            selected,
            tuple(np.round(weights, 8)),
            transform,
            holding_period,
            signal_lag,
            universe_quantile,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(
            StrategySpec(
                strategy_id=f"synthetic_{index:05d}",
                feature_columns=selected,
                weights=tuple(float(value) for value in weights),
                transform=transform,
                holding_period=holding_period,
                signal_lag=signal_lag,
                universe_quantile=universe_quantile,
                seed=seed + index,
                tags=("synthetic", transform),
            )
        )
        index += 1
    return output


def with_decay_profile(spec: StrategySpec, decay_rate: float) -> StrategySpec:
    if decay_rate < 0:
        raise ValueError("decay_rate must be nonnegative")
    weights = tuple(
        float(weight * np.exp(-decay_rate * position))
        for position, weight in enumerate(spec.weights)
    )
    return replace(spec, weights=weights)
