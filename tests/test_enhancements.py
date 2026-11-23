from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bonsai_alpha_decay import (
    DRIVER_NAMES,
    MarketPanel,
    StrategySpec,
    SyntheticConfig,
    build_portfolio_weights,
    build_signal,
    calculate_turnover,
    compute_dynamic_borrow_rates,
    estimate_bi_exponential_decay,
    estimate_decay_family,
    estimate_exponential_decay,
    fit_driver_model,
    ic_hac_term_structure,
    ic_long_table,
    make_synthetic_panel,
    newey_west_ic_stats,
    run_observatory,
    simulate_long_short_pnl,
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


def test_ic_hac_term_structure():
    dates = pd.bdate_range("2024-01-01", periods=50)
    ic_table = pd.DataFrame(
        {
            1: np.full(50, 0.05) + np.random.normal(0, 0.005, 50),
            2: np.full(50, 0.04) + np.random.normal(0, 0.005, 50),
            5: np.full(50, 0.02) + np.random.normal(0, 0.005, 50),
        },
        index=dates,
    )
    term_struct = ic_hac_term_structure(ic_table)
    assert list(term_struct.index) == [1, 2, 5]
    assert "se_hac" in term_struct.columns
    assert "hac_t_stat" in term_struct.columns
    assert "lag" in term_struct.columns
    # Horizon 5 should have a lag of at least 4 due to MA(h-1)
    assert term_struct.loc[5, "lag"] >= 4


def test_portfolio_weighting_schemes():
    dates = pd.bdate_range("2024-01-01", periods=10)
    assets = ["A", "B", "C", "D", "E"]
    signal = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]] * 10,
        index=dates,
        columns=assets,
    )
    returns = pd.DataFrame(0.01, index=dates, columns=assets)
    returns["E"] = 0.04
    returns["A"] = -0.02

    # Z-score weighting
    w_z = build_portfolio_weights(signal, weighting_scheme="zscore", target_leverage=1.0)
    assert np.allclose(w_z.sum(axis=1), 0.0)
    assert np.allclose(w_z.abs().sum(axis=1), 2.0)
    sim_z = simulate_long_short_pnl(signal, returns, weighting_scheme="zscore")
    assert "net_pnl" in sim_z.columns
    assert sim_z["gross_pnl"].iloc[-1] > 0

    # Vol-parity weighting
    vol = pd.DataFrame(
        [[0.01, 0.02, 0.015, 0.03, 0.01]] * 10,
        index=dates,
        columns=assets,
    )
    w_vol = build_portfolio_weights(signal, weighting_scheme="vol_parity", volatility=vol)
    assert np.allclose(w_vol.sum(axis=1), 0.0)
    assert np.allclose(w_vol.abs().sum(axis=1), 2.0)
    sim_vol = simulate_long_short_pnl(signal, returns, weighting_scheme="vol_parity", volatility=vol)
    assert "net_pnl" in sim_vol.columns


def test_permanent_impact_and_dynamic_borrow():
    dates = pd.bdate_range("2024-01-01", periods=10)
    assets = ["A", "B", "C", "D", "E"]
    signal = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]] * 10,
        index=dates,
        columns=assets,
    )
    returns = pd.DataFrame(0.01, index=dates, columns=assets)
    adv = pd.DataFrame(1e6, index=dates, columns=assets)
    vol = pd.DataFrame(0.02, index=dates, columns=assets)
    crowding = pd.Series([0.1, 0.2, 0.5, 0.8, 0.95], index=assets)
    market_cap = pd.Series([1e10, 5e9, 1e9, 5e8, 1e7], index=assets)

    # Dynamic borrow rates
    dynamic_borrow = compute_dynamic_borrow_rates(crowding, market_cap, base_fee_bps=50.0)
    assert dynamic_borrow["E"] > dynamic_borrow["A"]
    assert dynamic_borrow["E"] >= 50.0

    # Permanent impact simulation
    sim = simulate_long_short_pnl(
        signal,
        returns,
        adv=adv,
        volatility=vol,
        impact_gamma=0.2,
        impact_eta=0.1,
        borrow_fee_bps=dynamic_borrow,
    )
    assert "temp_impact_cost" in sim.columns
    assert "perm_impact_cost" in sim.columns
    assert sim["temp_impact_cost"].sum() > 0
    assert sim["perm_impact_cost"].sum() > 0
    assert np.isclose(sim["impact_cost"], sim["temp_impact_cost"] + sim["perm_impact_cost"]).all()
    assert sim["borrow_cost"].sum() > 0


def test_bi_exponential_decay():
    horizons = np.array([1, 2, 3, 5, 10, 20, 30])
    # Fast: a=0.08, lambda=0.693 (half-life=1)
    # Slow: a=0.04, lambda=0.0693 (half-life=10)
    ic_vals = 0.08 * np.exp(-0.693 * horizons) + 0.04 * np.exp(-0.0693 * horizons)
    res = estimate_bi_exponential_decay(pd.Series(ic_vals, index=horizons))
    assert res["status"] == "ok"
    assert res["r2"] > 0.99
    assert res["half_life_fast"] < res["half_life_slow"]
    assert np.isclose(res["half_life_fast"], 1.0, atol=0.2)
    assert np.isclose(res["half_life_slow"], 10.0, atol=1.5)

    # In decay family
    fam = estimate_decay_family(pd.Series(ic_vals, index=horizons))
    assert "bi_exponential" in fam["models"]
    assert "bi_exponential" in fam["aic"]




