from __future__ import annotations

from dataclasses import replace
import numpy as np

from .models import StrategySpec


def generate_strategies(
    feature_names: list[str],
    count: int = 500,
    seed: int = 11,
    max_features: int = 4,
    mode: str = "random",
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

    if mode == "archetype":
        archetypes = [
            ("momentum", (1.0, 0.5), "rank", 5, 1, 0.2, 4.0),
            ("mean_reversion", (-1.0, -0.5), "zscore", 2, 1, 0.1, 2.0),
            ("multi_factor", (0.5, 0.3, 0.2), "rank", 5, 1, None, 5.0),
            ("low_turnover_core", (0.4, 0.3, 0.3), "linear", 10, 1, 0.25, 8.0),
            ("pure_alpha", (1.0,), "rank", 3, 1, 0.2, 3.0),
        ]
        while len(output) < count:
            arch_name, base_w, transform, hp, lag, q, smooth = archetypes[index % len(archetypes)]
            n_f = min(len(base_w), len(feature_names))
            chosen = tuple(sorted(rng.choice(feature_names, size=n_f, replace=False).tolist()))
            weights = base_w[:n_f]
            key = (chosen, weights, transform, hp, lag, q, smooth)
            if key in seen:
                index += 1
                continue
            seen.add(key)
            output.append(
                StrategySpec(
                    strategy_id=f"archetype_{index:05d}",
                    feature_columns=chosen,
                    weights=weights,
                    transform=transform,
                    holding_period=hp,
                    signal_lag=lag,
                    universe_quantile=q,
                    seed=seed + index,
                    tags=("archetype", arch_name, transform),
                    smoothing=smooth,
                )
            )
            index += 1
        return output

    if mode == "informed":
        top_features = feature_names[: min(len(feature_names), 8)]
        while len(output) < count:
            n_f = int(rng.integers(1, min(max_features, len(top_features)) + 1))
            selected = tuple(sorted(rng.choice(top_features, size=n_f, replace=False).tolist()))
            weights = tuple(float(w) for w in rng.uniform(0.2, 1.0, size=n_f))
            transform = transforms[int(rng.integers(0, len(transforms)))]
            hp = int(rng.choice([2, 3, 5, 8]))
            smooth = float(rng.choice([2.0, 3.0, 5.0]))
            q = float(rng.choice([0.1, 0.2, 0.3]))
            key = (selected, weights, transform, hp, 1, q, smooth)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                StrategySpec(
                    strategy_id=f"informed_{index:05d}",
                    feature_columns=selected,
                    weights=weights,
                    transform=transform,
                    holding_period=hp,
                    signal_lag=1,
                    universe_quantile=q,
                    seed=seed + index,
                    tags=("informed", transform),
                    smoothing=smooth,
                )
            )
            index += 1
        return output

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
