from pathlib import Path
from bonsai_alpha_decay import (
    SyntheticConfig,
    generate_strategies,
    generate_tearsheet_html,
    make_synthetic_panel,
    run_observatory,
    save_tearsheet,
)

config = SyntheticConfig(
    n_days=756,
    n_assets=250,
    n_features=24,
    seed=7,
    fat_tails=True,
    garch_vol=True,
)
panel = make_synthetic_panel(config)
feature_names = [f"feature_{index:02d}" for index in range(config.n_features)]
strategies = generate_strategies(feature_names, count=100, seed=8)

summary, results = run_observatory(
    panel,
    strategies,
    horizons=(1, 2, 3, 5, 10, 20),
    workers=4,
    simulate_pnl=True,
    cost_bps=5.0,
)

cols = [
    col
    for col in [
        "strategy_id",
        "mean_ic",
        "ic_hac_tstat",
        "half_life",
        "strategy_turnover",
        "net_sharpe",
        "deflated_sharpe_ratio",
        "status",
    ]
    if col in summary.columns
]
print(summary.sort_values("net_sharpe", ascending=False).head(10)[cols].to_string(index=False))

top_id = summary.sort_values("net_sharpe", ascending=False).iloc[0]["strategy_id"]
top_obs = results[top_id]
top_row = summary[summary["strategy_id"] == top_id].iloc[0].to_dict()

html = generate_tearsheet_html(
    top_id,
    top_row,
    top_obs.ic_by_horizon.mean(axis=0),
    top_obs.decay,
    pnl_df=top_obs.pnl,
)
out_path = Path("reports/example_top_alpha.html")
save_tearsheet(html, out_path)
print(f"Saved tearsheet to {out_path}")
