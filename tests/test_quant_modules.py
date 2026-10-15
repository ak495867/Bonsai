from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bonsai_alpha_decay import (
    StrategySpec,
    SyntheticConfig,
    benjamini_hochberg_fdr,
    decay_compression_rate,
    deflated_sharpe_ratio,
    generate_tearsheet_html,
    haircut_sharpe_ratio,
    make_synthetic_panel,
    neutralize_signal,
    optimal_holding_horizon,
    portfolio_performance_stats,
    rolling_decay_profile,
    run_observatory,
    save_tearsheet,
    simulate_long_short_pnl,
)


def test_simulate_long_short_pnl():
    dates = pd.bdate_range("2024-01-01", periods=10)
    assets = ["A", "B", "C", "D", "E"]
    signal = pd.DataFrame(
        [[1, 2, 3, 4, 5]] * 10,
        index=dates,
        columns=assets,
    )
    returns = pd.DataFrame(
        0.01,
        index=dates,
        columns=assets,
    )
    returns["E"] = 0.05
    returns["A"] = -0.05
    pnl_df = simulate_long_short_pnl(signal, returns, quantiles=5, cost_bps=5.0)
    assert "gross_pnl" in pnl_df.columns
    assert "net_pnl" in pnl_df.columns
    assert "turnover" in pnl_df.columns
    stats = portfolio_performance_stats(pnl_df["net_pnl"])
    assert np.isfinite(stats["sharpe_ratio"])
    assert stats["annualized_return"] > 0


def test_optimal_holding_horizon():
    opt = optimal_holding_horizon(ic0=0.08, decay_rate=0.05, daily_turnover=0.3, cost_bps=5.0)
    assert opt["optimal_horizon"] >= 1
    assert len(opt["profile"]) > 0
    assert np.isfinite(opt["max_net_sharpe"])


def test_factor_neutralization():
    dates = pd.bdate_range("2024-01-01", periods=5)
    assets = ["A", "B", "C", "D", "E"]
    mkt = pd.DataFrame(np.ones((5, 5)), index=dates, columns=assets)
    signal = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]] * 5,
        index=dates,
        columns=assets,
    )
    factors = {"market": mkt}
    neutral = neutralize_signal(signal, factors)
    assert neutral.shape == signal.shape
    assert np.isclose(neutral.mean(axis=1).to_numpy(), 0.0).all()


def test_validation_dsr_and_fdr():
    trials = [0.2, 0.5, 0.8, 1.2, 1.5, 1.8, 2.1]
    dsr = deflated_sharpe_ratio(1.5, trials, n_periods=252)
    assert 0.0 <= dsr <= 1.0
    haircut = haircut_sharpe_ratio(1.5, trial_count=100, n_periods=252)
    assert 0.0 <= haircut <= 1.5
    p_vals = [0.001, 0.004, 0.015, 0.045, 0.12, 0.35]
    fdr_mask = benjamini_hochberg_fdr(p_vals, alpha=0.05)
    assert fdr_mask[0]
    assert not fdr_mask[-1]


def test_rolling_decay_profile_and_compression():
    dates = pd.bdate_range("2024-01-01", periods=100)
    horizons = [1, 2, 3, 5, 10]
    data = {}
    for h in horizons:
        data[h] = 0.1 * np.exp(-0.05 * h) + np.random.normal(0, 0.001, size=100)
    ic_table = pd.DataFrame(data, index=dates)
    profile = rolling_decay_profile(ic_table, window=30, step=10)
    assert len(profile) > 0
    comp = decay_compression_rate(profile)
    assert "lambda_trend" in comp
    assert "half_life_compression" in comp


def test_tearsheet_generation(tmp_path: Path):
    h_ic = pd.Series({1: 0.05, 2: 0.035, 3: 0.025, 5: 0.015, 10: 0.008})
    decay_info = {"ic0": 0.05, "lambda": 0.15, "half_life": 4.62, "r2": 0.98}
    pnl_df = pd.DataFrame(
        {
            "cum_gross": np.linspace(0, 0.2, 50),
            "cum_net": np.linspace(0, 0.15, 50),
        }
    )
    summary = {"mean_ic": 0.05, "net_sharpe": 1.8, "strategy_turnover": 0.12}
    html = generate_tearsheet_html("strat_test", summary, h_ic, decay_info, pnl_df=pnl_df)
    assert "strat_test" in html
    assert "<svg" in html
    out_file = tmp_path / "tearsheet.html"
    save_tearsheet(html, out_file)
    assert out_file.exists()


def test_observatory_with_pnl_simulation():
    panel = make_synthetic_panel(
        SyntheticConfig(n_days=50, n_assets=20, n_features=2, seed=12)
    )
    strategies = [
        StrategySpec("s1", ("feature_00",), (1.0,), signal_lag=1),
        StrategySpec("s2", ("feature_01",), (-1.0,), signal_lag=1),
    ]
    summary, results = run_observatory(
        panel,
        strategies,
        horizons=(1, 2, 3),
        min_obs=5,
        simulate_pnl=True,
        cost_bps=3.0,
    )
    assert "net_sharpe" in summary.columns
    assert "deflated_sharpe_ratio" in summary.columns
    assert "is_fdr_significant" in summary.columns
    assert results["s1"].pnl is not None
