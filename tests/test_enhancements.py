from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bonsai_alpha_decay import (
    CombinatorialPurgedCV,
    DRIVER_NAMES,
    MarketPanel,
    StrategySpec,
    SyntheticConfig,
    build_panel_from_daily_bars,
    build_portfolio_weights,
    build_signal,
    calculate_turnover,
    compute_dynamic_borrow_rates,
    compute_strategy_correlation_matrix,
    estimate_bi_exponential_decay,
    estimate_decay_family,
    estimate_exponential_decay,
    evaluate_formulaic_alpha,
    evaluate_strategy_cpcv,
    fit_driver_model,
    generate_tearsheet_html,
    hierarchical_alpha_clustering,
    ic_hac_term_structure,
    ic_long_table,
    ledoit_wolf_shrinkage,
    make_synthetic_panel,
    neutralize_signal,
    newey_west_ic_stats,
    optimize_portfolio_weights,
    run_observatory,
    simulate_long_short_pnl,
    simulate_optimized_portfolio_pnl,
    synthesize_master_ensemble,
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


def test_vectorized_wls_factor_neutralization():
    dates = pd.bdate_range("2024-01-01", periods=20)
    assets = [f"A{i}" for i in range(10)]
    np.random.seed(42)
    sig_data = np.random.normal(size=(20, 10))
    fac_data = np.random.normal(size=(20, 10))
    weights_data = np.random.uniform(1.0, 10.0, size=(20, 10))

    signal = pd.DataFrame(sig_data, index=dates, columns=assets)
    factors = {"market": pd.DataFrame(fac_data, index=dates, columns=assets)}
    weights = pd.DataFrame(weights_data, index=dates, columns=assets)

    # Vectorized OLS (equal weights)
    res_ols = neutralize_signal(signal, factors)
    assert res_ols.shape == signal.shape
    # Check that residual is orthogonal to factor
    for t in range(len(dates)):
        dot_prod = np.dot(res_ols.iloc[t], factors["market"].iloc[t])
        assert np.isclose(dot_prod, 0.0, atol=1e-10)

    # Vectorized WLS (market-cap weights)
    res_wls = neutralize_signal(signal, factors, weights=weights)
    assert res_wls.shape == signal.shape
    for t in range(len(dates)):
        w = weights.iloc[t].to_numpy()
        w_dot = np.dot(res_wls.iloc[t] * w, factors["market"].iloc[t])
        assert np.isclose(w_dot, 0.0, atol=1e-10)


def test_combinatorial_purged_cv():
    dates = pd.bdate_range("2024-01-01", periods=60)
    cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_window=3, embargo_pct=0.02)
    splits = list(cpcv.split(dates))
    # C(5, 2) = 10 paths
    assert len(splits) == 10
    for train_idx, test_idx, path_id in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # Check no overlap between train and test
        assert len(set(train_idx).intersection(set(test_idx))) == 0

    # Evaluate strategy with CPCV
    panel = make_synthetic_panel(
        SyntheticConfig(n_days=60, n_assets=15, n_features=2, seed=10)
    )
    spec = StrategySpec("cpcv_test", ("feature_00",), (1.0,), signal_lag=1)
    res = evaluate_strategy_cpcv(panel, spec, cpcv=cpcv)
    assert res["n_paths"] == 10
    assert np.isfinite(res["mean_oos_sharpe"])
    assert 0.0 <= res["pbo"] <= 1.0


def test_build_panel_from_daily_bars_with_delisting():
    dates = pd.bdate_range("2024-01-01", periods=10)
    rows = []
    for d in dates:
        rows.append({"date": d, "asset": "AAA", "close": 100.0, "volume": 10000})
        # BBB trades only on first 5 days then delists
        if d <= dates[4]:
            rows.append({"date": d, "asset": "BBB", "close": 50.0, "volume": 5000})
    bars = pd.DataFrame(rows)

    delist = pd.DataFrame([{"date": dates[5], "asset": "BBB", "delisting_return": -0.50}])
    panel = build_panel_from_daily_bars(bars, delisting_df=delist)
    panel.validate()

    assert "AAA" in panel.returns.columns
    assert "BBB" in panel.returns.columns
    assert panel.mask is not None
    # BBB should be tradeable on day 4, have delisting return on day 5, and False thereafter
    assert panel.mask.loc[dates[4], "BBB"]
    assert np.isclose(panel.returns.loc[dates[5], "BBB"], -0.50)
    assert not panel.mask.loc[dates[6], "BBB"]


def test_multiprocess_observatory_execution():
    panel = make_synthetic_panel(
        SyntheticConfig(n_days=40, n_assets=10, n_features=2, seed=14)
    )
    strategies = [
        StrategySpec("s_p1", ("feature_00",), (1.0,), signal_lag=1),
        StrategySpec("s_p2", ("feature_01",), (-1.0,), signal_lag=1),
    ]
    summary, results = run_observatory(
        panel,
        strategies,
        horizons=(1, 2),
        min_obs=5,
        workers=2,
        backend="process",
        simulate_pnl=True,
    )
    assert len(summary) == 2
    assert set(results.keys()) == {"s_p1", "s_p2"}
    assert "net_sharpe" in summary.columns


def test_multichart_tearsheet_svg(tmp_path: Path):
    h_ic = pd.Series({1: 0.06, 2: 0.045, 3: 0.035, 5: 0.02, 10: 0.01})
    decay_info = {"ic0": 0.06, "lambda": 0.15, "half_life": 4.62, "r2": 0.98}
    decay_family = {
        "best_model": "exponential",
        "models": {
            "power_law": {"ic0": 0.06, "alpha": 0.6},
            "bi_exponential": {"a_fast": 0.04, "lambda_fast": 0.5, "a_slow": 0.02, "lambda_slow": 0.05},
        },
    }
    pnl_df = pd.DataFrame(
        {
            "cum_gross": np.linspace(0, 0.25, 40),
            "cum_net": np.linspace(0, 0.18, 40) - np.array([0.0 if i < 30 else 0.02 for i in range(40)]),
        }
    )
    cap_info = {
        "capacity_threshold_aum": 5e7,
        "curve": [
            {"aum": 1e6, "net_sharpe": 2.2},
            {"aum": 1e7, "net_sharpe": 1.8},
            {"aum": 1e8, "net_sharpe": 0.5},
        ],
    }
    summary = {
        "strategy_id": "test_multi",
        "mean_ic": 0.06,
        "net_sharpe": 1.8,
        "crowding": 0.45,
        "volatility": 0.02,
        "liquidity": 1e7,
    }
    html = generate_tearsheet_html(
        strategy_id="test_multi",
        summary=summary,
        ic_by_horizon=h_ic,
        decay_info=decay_info,
        pnl_df=pnl_df,
        capacity_info=cap_info,
        decay_family=decay_family,
    )
    assert "test_multi" in html
    assert "Drawdown" in html
    assert "Capital Capacity Curve" in html
    assert "Systematic Driver Exposures" in html
    assert html.count("<svg") >= 4
    out_file = tmp_path / "multi_tearsheet.html"
    out_file.write_text(html, encoding="utf-8")
    assert out_file.exists()


def test_formulaic_alpha_expressions():
    panel = make_synthetic_panel(
        SyntheticConfig(n_days=40, n_assets=10, n_features=3, seed=20)
    )

    # 1. Linear combination with rank
    expr1 = "cs_rank(feature_00) - cs_rank(feature_01)"
    sig1 = evaluate_formulaic_alpha(expr1, panel, signal_lag=1)
    assert sig1.shape == panel.returns.shape
    assert sig1.iloc[0].isna().all()  # signal_lag=1
    assert not sig1.iloc[1].isna().all()

    # 2. Time-series decay linear and cross-sectional z-score
    expr2 = "cs_zscore(ts_decay_linear(feature_00, 5))"
    sig2 = evaluate_formulaic_alpha(expr2, panel, signal_lag=1)
    assert sig2.shape == panel.returns.shape

    # 3. Time-series delta and power
    expr3 = "signed_power(ts_delta(close, 2), 2.0)"
    sig3 = evaluate_formulaic_alpha(expr3, panel, signal_lag=1)
    assert sig3.shape == panel.returns.shape

    # 4. Unknown variable raises NameError
    with pytest.raises(NameError):
        evaluate_formulaic_alpha("cs_rank(nonexistent_feature)", panel)


def test_portfolio_optimization_and_shrinkage():
    # 1. Ledoit-Wolf Shrinkage
    ret = np.random.normal(size=(50, 10))
    shrunk = ledoit_wolf_shrinkage(ret)
    assert shrunk.shape == (10, 10)
    assert np.all(np.linalg.eigvalsh(shrunk) > 0)  # Positive definite

    # 2. Portfolio Optimization with Dollar Neutrality and Factor Neutrality
    alpha = np.array([0.05, 0.04, 0.03, 0.01, 0.0, -0.01, -0.02, -0.03, -0.04, -0.05])
    # Single market factor
    factor = np.ones((10, 1))
    w_opt = optimize_portfolio_weights(
        alpha=alpha,
        factors=factor,
        cov=shrunk,
        risk_aversion=1.0,
        max_weight=0.25,
        max_leverage=2.0,
    )
    # Check dollar neutrality
    assert np.isclose(np.sum(w_opt), 0.0, atol=1e-5)
    # Check factor neutrality
    assert np.isclose(np.dot(w_opt, factor[:, 0]), 0.0, atol=1e-5)
    # Check gross leverage cap
    assert np.sum(np.abs(w_opt)) <= 2.0001
    # Check high alpha names are long and low alpha names are short
    assert w_opt[0] > 0
    assert w_opt[-1] < 0

    # 3. Walk-Forward Optimized PnL Simulation
    dates = pd.bdate_range("2024-01-01", periods=15)
    assets = [f"A{i}" for i in range(10)]
    signal = pd.DataFrame(np.tile(alpha, (15, 1)), index=dates, columns=assets)
    returns = pd.DataFrame(ret[:15], index=dates, columns=assets)
    sim_opt = simulate_optimized_portfolio_pnl(signal, returns, max_weight=0.25)
    assert "net_pnl" in sim_opt.columns
    assert "turnover" in sim_opt.columns
    assert "cum_net" in sim_opt.columns


def test_hrp_master_ensemble():
    dates = pd.bdate_range("2024-01-01", periods=100)
    # Generate 4 synthetic strategy PnL series: 2 momentum (correlated) and 2 reversion (correlated)
    np.random.seed(42)
    base_mom = np.random.normal(0.001, 0.01, size=100)
    base_rev = np.random.normal(0.001, 0.01, size=100)

    pnl_df = pd.DataFrame(
        {
            "mom_1": base_mom + np.random.normal(0, 0.002, 100),
            "mom_2": base_mom + np.random.normal(0, 0.002, 100),
            "rev_1": base_rev + np.random.normal(0, 0.002, 100),
            "rev_2": base_rev + np.random.normal(0, 0.002, 100),
        },
        index=dates,
    )

    corr = compute_strategy_correlation_matrix(pnl_df)
    assert corr.shape == (4, 4)
    # Mom_1 and Mom_2 correlation should be high
    assert corr.loc["mom_1", "mom_2"] > 0.8
    # Clustering
    clust = hierarchical_alpha_clustering(corr)
    assert len(clust["quasi_diag_order"]) == 4

    # HRP Ensemble
    res_hrp = synthesize_master_ensemble(pnl_df, ensemble_method="hrp")
    weights = res_hrp["weights"]
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)
    # HRP should allocate ~50% to momentum cluster and ~50% to reversion cluster
    mom_weight = weights["mom_1"] + weights["mom_2"]
    rev_weight = weights["rev_1"] + weights["rev_2"]
    assert np.isclose(mom_weight, 0.5, atol=0.1)
    assert np.isclose(rev_weight, 0.5, atol=0.1)
    assert res_hrp["diversification_ratio"] > 1.0












