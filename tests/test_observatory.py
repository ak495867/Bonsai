import numpy as np
import pandas as pd

from bonsai_alpha_decay import (
    StrategySpec,
    SyntheticConfig,
    estimate_exponential_decay,
    make_synthetic_panel,
    run_observatory,
)


def test_half_life_matches_known_decay():
    rate = 0.125
    horizons = np.arange(1, 8)
    ic = pd.Series(0.4 * np.exp(-rate * horizons), index=horizons)
    result = estimate_exponential_decay(ic)
    assert np.isclose(result["lambda"], rate, atol=1e-10)
    assert np.isclose(result["half_life"], np.log(2.0) / rate, atol=1e-10)
    assert result["status"] == "ok"


def test_observatory_returns_requested_driver_fields():
    panel = make_synthetic_panel(
        SyntheticConfig(n_days=90, n_assets=25, n_features=3, seed=6)
    )
    strategies = [
        StrategySpec("a", ("feature_00",), (1.0,), signal_lag=1),
        StrategySpec(
            "b",
            ("feature_01", "feature_02"),
            (0.5, -0.5),
            transform="rank",
            signal_lag=1,
        ),
    ]
    summary, results = run_observatory(
        panel, strategies, horizons=(1, 2, 3, 5), min_obs=5, workers=1
    )
    assert set(summary["strategy_id"]) == {"a", "b"}
    assert {
        "crowding",
        "volatility",
        "turnover",
        "liquidity",
        "market_cap",
        "regime_changes",
    }.issubset(summary.columns)
    assert set(results) == {"a", "b"}


def test_parallel_and_serial_results_match():
    panel = make_synthetic_panel(
        SyntheticConfig(n_days=70, n_assets=20, n_features=2, seed=8)
    )
    strategies = [
        StrategySpec(f"s{i}", ("feature_00",), (1.0,), signal_lag=1) for i in range(4)
    ]
    serial, _ = run_observatory(
        panel, strategies, horizons=(1, 2, 3), min_obs=5, workers=1
    )
    parallel, _ = run_observatory(
        panel, strategies, horizons=(1, 2, 3), min_obs=5, workers=2
    )
    serial = serial.sort_values("strategy_id").reset_index(drop=True)
    parallel = parallel.sort_values("strategy_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(serial, parallel)
