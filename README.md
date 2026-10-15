# Bonsai: Alpha Decay Observatory

Bonsai is an institutional-grade alpha research engine and execution simulator. Rather than mining for overfitted in-sample Sharpe ratios, Bonsai studies **how alpha dies**: measuring the term structure of information coefficients, fitting generalized decay curves (Exponential, Power-Law, and Delayed-Diffusion Hump), simulating portfolio execution with square-root market impact and borrow costs, analyzing capital capacity curves, and applying strict econometric corrections (Newey-West HAC, Deflated Sharpe Ratio, and False Discovery Rate).

## Core Capabilities

- **2D Vectorized Cross-Sectional IC Engine**: Point-in-time cross-sectional Spearman rank and Pearson correlations evaluated simultaneously across all dates and assets.
- **Newey-West (HAC) Standard Errors**: Automatic Moving Average $\text{MA}(h-1)$ autocorrelation adjustments for overlapping forward return horizons.
- **Generalized Decay Model Family**: Automated model selection via AIC across:
  - *Exponential*: $\text{IC}(h) = \text{IC}_0 \cdot e^{-\lambda h}$
  - *Power-Law / Hyperbolic*: $\text{IC}(h) = \text{IC}_0 \cdot h^{-\alpha}$
  - *Delayed Diffusion (Hump)*: $\text{IC}(h) = A \cdot h^\beta \cdot e^{-\lambda h}$ with peak horizon $h^* = \beta / \lambda$
- **Portfolio & Execution Drag Simulation**: Long/short quantile portfolios ($Q_5 - Q_1$) with turnover tracking, Almgren-Chriss square-root market impact ($\gamma \sigma \sqrt{\text{Trade}/\text{ADV}}$), and short-leg borrow fees.
- **Capital Capacity Curves**: Multi-scale AUM stress tests (\$1M to \$500M) reporting the maximum capital capacity before market impact eliminates Net Sharpe.
- **P-Hacking & Multiple Testing Defenses**:
  - *Deflated Sharpe Ratio (DSR)*: Adjusts for trial count, strategy variance, skewness, and kurtosis (López de Prado).
  - *Haircut Sharpe Ratio*: Penalty discounting for data snooping (Harvey & Liu).
  - *Benjamini-Hochberg False Discovery Rate (FDR)*: False positive control across strategy universes.
- **Factor & Style Neutralization**: Cross-sectional OLS projection stripping out Market Beta, Size, Volatility, and Momentum to measure pure idiosyncratic residual alpha.
- **Time-Varying Decay & Crowding Detection**: Rolling decay curves $\lambda(t)$ measuring alpha compression ($d\lambda / dt > 0$) to detect arbitrage crowding and alpha extinction in real time.
- **Publication-Ready Tear-Sheets**: Standalone interactive HTML reports with embedded SVG charts for decay profiles and cumulative Net PnL.
- **High-Performance Storage**: Native Apache Parquet (`.parquet`) serialization.

## Layout

| Path | Purpose |
| --- | --- |
| `src/bonsai_alpha_decay/models.py` | Validated panel, tradeable mask, and strategy data models |
| `src/bonsai_alpha_decay/synthetic.py` | Synthetic panel generation with GARCH volatility, Student-t fat tails, and liquidity cascades |
| `src/bonsai_alpha_decay/strategies.py` | Seeded strategy specification factory |
| `src/bonsai_alpha_decay/engine.py` | Vectorized signal construction, forward returns, 2D IC, turnover, and Newey-West HAC |
| `src/bonsai_alpha_decay/decay.py` | Exponential, Power-law, Hump-shaped decay fitting, and AIC model selection |
| `src/bonsai_alpha_decay/portfolio.py` | Long/short quantile simulation, square-root market impact, borrow fees, and capacity curves |
| `src/bonsai_alpha_decay/neutralization.py` | Cross-sectional factor orthogonalization |
| `src/bonsai_alpha_decay/validation.py` | Deflated Sharpe Ratio (DSR), Haircut Sharpe, and Benjamini-Hochberg FDR |
| `src/bonsai_alpha_decay/rolling.py` | Rolling decay profiles and alpha crowding compression detection |
| `src/bonsai_alpha_decay/reporting.py` | Standalone HTML tear-sheet generator with inline SVG charts |
| `src/bonsai_alpha_decay/drivers.py` | Driver exposures, VIF collinearity checks, and Ridge regression |
| `src/bonsai_alpha_decay/observatory.py` | Multi-strategy orchestration with cached returns and statistical summaries |
| `src/bonsai_alpha_decay/persistence.py` | Parquet, CSV, and JSON persistence |
| `src/bonsai_alpha_decay/cli.py` | Command-line entry points |
| `tests/` | Leakage, econometric, portfolio, capacity, and reproducibility test suites |

## Install

Requires Python 3.10 or newer.

```text
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Run Tests

```text
pytest
```

## CLI Usage

### 1. Generate strategy specifications

```text
bonsai-generate --days 756 --assets 250 --features 24 --count 500 --seed 7 --output data/strategies.json
```

### 2. Run the Observatory with Execution Simulation and Tear-Sheets

```text
bonsai-run --days 756 --assets 250 --features 24 --count 500 --workers 4 --seed 7 --format parquet --simulate-pnl --cost-bps 5.0 --tearsheet --output reports/run
```

Outputs written to `reports/run/`:
- `strategy_summary.parquet`: Mean IC, HAC $t$-stat, half-life, decay model, turnover, Gross/Net Sharpe, DSR, FDR significance, and driver exposures.
- `ic_long.parquet`: Point-in-time melted IC series across all dates, horizons, and strategies.
- `strategies.parquet`: Serialized strategy configurations.
- `run_manifest.json`: Configuration manifest and driver regression diagnostics (VIF and condition number).
- `tearsheets/*.html`: Publication-ready visual diagnostic tear-sheets for top-performing alphas.

## Python API

```python
from bonsai_alpha_decay import (
    SyntheticConfig,
    estimate_decay_family,
    generate_strategies,
    generate_tearsheet_html,
    make_synthetic_panel,
    run_observatory,
    save_tearsheet,
    strategy_capacity_curve,
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

top_strat = summary.sort_values("net_sharpe", ascending=False).iloc[0]
obs = results[top_strat["strategy_id"]]

decay_family = estimate_decay_family(obs.ic_by_horizon.mean(axis=0))

adv_frame = panel.metadata["liquidity"].to_frame().T.reindex(panel.returns.index).ffill()
vol_frame = panel.metadata["volatility"].to_frame().T.reindex(panel.returns.index).ffill()
capacity = strategy_capacity_curve(
    obs.signal,
    panel.returns,
    adv=adv_frame,
    volatility=vol_frame,
)

html = generate_tearsheet_html(
    top_strat["strategy_id"],
    top_strat.to_dict(),
    obs.ic_by_horizon.mean(axis=0),
    obs.decay,
    pnl_df=obs.pnl,
)
save_tearsheet(html, "reports/top_alpha_tearsheet.html")
```

## Real-World Data Ingestion

To evaluate live production data, construct a `MarketPanel` with:
- `prices`: Date-indexed positive asset price frame.
- `returns`: Date-indexed asset return frame.
- `features`: Date-indexed feature matrix with columns formatted as `{asset}__{feature}`.
- `metadata`: Asset-indexed metadata containing `market_cap`, `volatility`, `turnover`, `liquidity`, `crowding`, and `regime`.
- `mask` (optional): Date-indexed boolean DataFrame flagging point-in-time tradeable universe membership to eliminate survivorship bias.

Call `panel.validate()` before evaluation.

## License

This project is released under the MIT License. See `LICENSE`.
