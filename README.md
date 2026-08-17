# Bonsai: Alpha Decay Observatory

Bonsai studies how alpha dies rather than searching only for alpha. It generates reproducible synthetic strategies, computes point-in-time cross-sectional information coefficients, fits exponential decay curves, estimates half-lives, and models whether crowding, volatility, turnover, liquidity, market capitalization, or regime changes are associated with faster decay.

## Research definition

For strategy signal `signal_t` and forward return `return_{t+h}`, Bonsai computes the cross-sectional information coefficient on each date:

`IC_h(t) = corr(signal_t, return_{t+h})`

The default correlation is Spearman rank correlation. The forward return is compounded over the selected horizon and aligned with `shift(-h)`, so the observation at date `t` contains only returns from `t+1` through `t+h` and never uses returns after the evaluated horizon.

For the mean IC curve across dates, Bonsai fits:

`IC(h) = IC0 * exp(-lambda * h)`

The fitted half-life is:

`t_half = ln(2) / lambda`

Only positive finite IC observations enter the log-linear fit. A status field distinguishes valid decay estimates from insufficient data and non-decaying or growing curves.

## Bias controls

The engine requires sorted, unique business-date indexes, positive prices, aligned returns and features, explicit signal lags, and explicit future-return construction. Signals are created from the feature snapshot at or before the signal date. The supplied synthetic generator creates informative features from lagged returns only. Strategy IDs are deterministic and the random strategy factory is seed-controlled.

The repository is a research engine rather than an execution system. It does not infer transaction costs, borrow availability, corporate actions, delisted assets, or point-in-time universe membership from generic files. A production dataset adapter should supply those fields explicitly and preserve historical membership. The `metadata` contract includes the requested driver variables at the asset level.

## Layout

| Path | Purpose |
| --- | --- |
| `src/bonsai_alpha_decay/models.py` | Validated panel and strategy data models |
| `src/bonsai_alpha_decay/synthetic.py` | Reproducible synthetic panel generation |
| `src/bonsai_alpha_decay/strategies.py` | Seeded strategy specification generation |
| `src/bonsai_alpha_decay/engine.py` | Signal construction, forward returns, and IC calculations |
| `src/bonsai_alpha_decay/decay.py` | Exponential decay and half-life estimation |
| `src/bonsai_alpha_decay/drivers.py` | Driver exposures and cross-strategy regression |
| `src/bonsai_alpha_decay/observatory.py` | End-to-end multi-strategy orchestration |
| `src/bonsai_alpha_decay/persistence.py` | CSV and JSON persistence |
| `src/bonsai_alpha_decay/cli.py` | Command-line entry points |
| `tests/` | Leakage, mathematical, schema, and reproducibility tests |

## Install

Use Python 3.10 or newer.

```text
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Run tests

```text
pytest
```

## Generate strategy specifications

```text
bonsai-generate --days 756 --assets 250 --features 24 --count 500 --seed 7 --output data/strategies.json
```

## Run the full synthetic observatory

```text
bonsai-run --days 756 --assets 250 --features 24 --count 500 --workers 4 --seed 7 --output reports/run
```

The run writes `strategy_summary.csv`, `ic_long.csv`, `strategies.csv`, and `run_manifest.json`. The summary contains the mean IC, IC volatility, `ic0`, `lambda`, half-life, fit quality, status, and all requested driver exposures. The manifest contains the synthetic configuration and a standardized driver regression for `lambda`.

## Python API

```python
from bonsai_alpha_decay import SyntheticConfig, generate_strategies, make_synthetic_panel, run_observatory

config = SyntheticConfig(n_days=756, n_assets=250, n_features=24, seed=7)
panel = make_synthetic_panel(config)
strategies = generate_strategies([f"feature_{index:02d}" for index in range(config.n_features)], count=500, seed=8)
summary, detailed = run_observatory(panel, strategies, horizons=(1, 2, 3, 5, 10, 20), workers=4)
```

## Adding real data

Build a `MarketPanel` with a date-indexed price frame, a date-indexed return frame, a date-indexed feature frame with columns named `ASSET__FEATURE`, and asset-indexed metadata containing `market_cap`, `volatility`, `turnover`, `liquidity`, `crowding`, and `regime`. Call `panel.validate()` before evaluation. Feature values must be timestamped according to their actual information availability, and the adapter must apply any reporting lag before creating the feature frame.

## License

This project is released under the MIT License. See `LICENSE`.
