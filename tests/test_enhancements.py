from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bonsai_alpha_decay import (
    DRIVER_NAMES,
    MarketPanel,
    StrategySpec,
    SyntheticConfig,
    build_signal,
    calculate_turnover,
    estimate_exponential_decay,
    fit_driver_model,
    ic_long_table,
    make_synthetic_panel,
    newey_west_ic_stats,
    run_observatory,
)
from bonsai_alpha_decay.persistence import read_frame, write_frame


def test_universe_masking():
    panel = make_synthetic_panel(
        SyntheticConfig(n_days=30, n_assets=10, n_features=2, seed=3)
    )
    mask = pd.DataFrame(True, index=panel.prices.index, columns=panel.returns.columns)
    mask.iloc[:, 0] = False
    masked_panel = MarketPanel(
        prices=panel.prices,
        features=panel.features,
        returns=panel.returns,
        metadata=panel.metadata,
        mask=mask,
    )
    masked_panel.validate()
    spec = StrategySpec("m1", ("feature_00",), (1.0,), signal_lag=0)
    sig = build_signal(masked_panel, spec)
    assert sig.iloc[:, 0].isna().all()
    assert not sig.iloc[:, 1].isna().all()


def test_turnover_calculation():
    dates = pd.bdate_range("2024-01-01", periods=3)
    signal = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        index=dates,
        columns=["A", "B"],
    )
    turnover = calculate_turnover(signal)
    assert len(turnover) == 2
    assert np.isclose(turnover.iloc[0], 1.0)
    assert np.isclose(turnover.iloc[1], 0.0)


def test_newey_west_ic_stats():
    ic = pd.Series([0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06])
    stats = newey_west_ic_stats(ic, max_lag=2)
    assert np.isclose(stats["mean"], 0.05)
    assert stats["se_hac"] > 0
    assert stats["hac_t_stat"] > 0
    assert stats["ir"] > 0


def test_nlls_decay_fitting():
    horizons = np.array([1, 2, 3, 5, 10, 20])
    true_rate = 0.1
    ic_vals = 0.2 * np.exp(-true_rate * horizons)
    res_nlls = estimate_exponential_decay(pd.Series(ic_vals, index=horizons), method="nlls")
    assert np.isclose(res_nlls["lambda"], true_rate, atol=1e-3)
    assert res_nlls["status"] == "ok"


def test_driver_model_with_ridge_and_vif():
    records = pd.DataFrame(
        {
            name: np.random.normal(size=20) for name in DRIVER_NAMES
        }
    )
    records["lambda"] = records["crowding"] * 0.5 + 0.1
    model = fit_driver_model(records, "lambda", alpha=0.1)
    assert model["status"] == "ok"
    assert "vif" in model
    assert len(model["vif"]) == len(DRIVER_NAMES)
    assert np.isfinite(model["condition_number"])


def test_parquet_persistence(tmp_path: Path):
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
    pq_path = tmp_path / "test.parquet"
    write_frame(df, pq_path)
    loaded = read_frame(pq_path)
    pd.testing.assert_frame_equal(df, loaded, check_freq=False)


def test_ic_long_table_vectorized():
    panel = make_synthetic_panel(
        SyntheticConfig(n_days=40, n_assets=15, n_features=2, seed=9)
    )
    spec = StrategySpec("s_vec", ("feature_00",), (1.0,), signal_lag=1)
    _, results = run_observatory(panel, [spec], horizons=(1, 2), min_obs=5)
    long_df = ic_long_table(results)
    assert set(long_df.columns) == {"strategy_id", "date", "horizon", "ic"}
    assert len(long_df) == 40 * 2
    assert (long_df["strategy_id"] == "s_vec").all()
