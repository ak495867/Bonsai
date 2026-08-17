import numpy as np
import pandas as pd

from bonsai_alpha_decay import (
    StrategySpec,
    SyntheticConfig,
    build_signal,
    cross_sectional_ic,
    forward_returns,
    generate_strategies,
    make_synthetic_panel,
)


def test_forward_returns_are_future_aligned():
    dates = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.DataFrame(
        {"A": [0.1, 0.2, 0.3, 0.4, 0.5], "B": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=dates
    )
    output = forward_returns(returns, [1, 2])
    assert np.isclose(output[1].loc[dates[0], "A"], 0.2)
    assert np.isclose(output[2].loc[dates[0], "A"], 1.2 * 1.3 - 1.0)
    assert np.isnan(output[2].loc[dates[-2], "A"])
    assert np.isnan(output[2].loc[dates[-1], "A"])


def test_signal_lag_blocks_current_feature_changes():
    panel = make_synthetic_panel(
        SyntheticConfig(n_days=40, n_assets=20, n_features=2, seed=2)
    )
    spec = StrategySpec("x", ("feature_00",), (1.0,), signal_lag=1)
    original = build_signal(panel, spec)
    changed_features = panel.features.copy()
    columns = [
        column for column in changed_features.columns if column.endswith("__feature_00")
    ]
    changed_features.loc[panel.features.index[10], columns] = (
        changed_features.loc[panel.features.index[10], columns] + 1000.0
    )
    changed_panel = type(panel)(
        panel.prices, changed_features, panel.returns, panel.metadata
    )
    changed = build_signal(changed_panel, spec)
    assert original.loc[panel.features.index[10]].equals(
        changed.loc[panel.features.index[10]]
    )
    assert not original.loc[panel.features.index[11]].equals(
        changed.loc[panel.features.index[11]]
    )


def test_cross_sectional_ic_detects_order():
    dates = pd.bdate_range("2024-01-01", periods=2)
    signal = pd.DataFrame([[1, 2, 3], [3, 2, 1]], index=dates, columns=list("ABC"))
    target = pd.DataFrame(
        [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], index=dates, columns=list("ABC")
    )
    result = cross_sectional_ic(signal, target, min_obs=3)
    assert np.allclose(result.to_numpy(), [1.0, 1.0])


def test_strategy_factory_is_reproducible_and_unique():
    names = ["feature_00", "feature_01", "feature_02"]
    left = generate_strategies(names, count=50, seed=4)
    right = generate_strategies(names, count=50, seed=4)
    assert left == right
    assert len({spec.strategy_id for spec in left}) == 50
