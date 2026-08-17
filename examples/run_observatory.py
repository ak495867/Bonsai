from bonsai_alpha_decay import (
    SyntheticConfig,
    generate_strategies,
    make_synthetic_panel,
    run_observatory,
)

config = SyntheticConfig(n_days=756, n_assets=250, n_features=24, seed=7)
panel = make_synthetic_panel(config)
feature_names = [f"feature_{index:02d}" for index in range(config.n_features)]
strategies = generate_strategies(feature_names, count=500, seed=8)
summary, detailed = run_observatory(
    panel, strategies, horizons=(1, 2, 3, 5, 10, 20), workers=4
)
print(
    summary.sort_values("half_life", na_position="last").head(20).to_string(index=False)
)
