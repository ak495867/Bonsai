# Bonsai: Quantitative Alpha Decay Observatory & Execution Platform

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-39%20passed%20%7C%20100%25-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-institutional--quant-darkblue.svg)]()
[![Storage](https://img.shields.io/badge/format-Apache%20Parquet-orange.svg)](https://arrow.apache.org/)

Bonsai is an institutional-grade quantitative alpha research engine, execution simulator, and decay observatory. Rather than data-snooping for overfitted in-sample Sharpe ratios, Bonsai studies **how alpha decays over time**: measuring the term structure of information coefficients, fitting generalized decay curves (Exponential, Power-Law, Delayed-Diffusion Hump, and Bi-Exponential), simulating real-world execution drag with Almgren-Chriss market impact and dynamic borrow costs, constructing factor-neutral convex portfolios, clustering strategies into Hierarchical Risk Parity (HRP) ensembles, evaluating regime-conditional decay, and enforcing strict econometric corrections (Newey-West HAC, Deflated Sharpe Ratio, and Combinatorial Purged Cross-Validation).

---

## Key Pillars & Capabilities

### 1. Vectorized Econometrics & Alpha Decay Term Structure
- **Vectorized 2D Cross-Sectional IC Engine**: Point-in-time cross-sectional Spearman rank and Pearson correlation term structures evaluated across all dates, assets, and forward horizons simultaneously without Python loops.
- **Multi-Horizon Newey-West HAC Statistics**: Multi-lag autocorrelation- and heteroskedasticity-consistent covariance adjustments ($L = h - 1$ truncation) for overlapping forward returns, providing robust $t$-statistics and $p$-values across horizons $h \in \{1, 2, 3, 5, 10, 20\}$.
- **Four Generalized Decay Families with Automated Model Selection**:
  - **Exponential**: $\text{IC}(h) = \text{IC}_0 \cdot e^{-\lambda h}$ with half-life $t_{1/2} = \ln(2)/\lambda$.
  - **Power-Law / Hyperbolic**: $\text{IC}(h) = \text{IC}_0 \cdot h^{-\alpha}$ capturing long-memory decay.
  - **Delayed Diffusion (Hump)**: $\text{IC}(h) = A \cdot h^\beta \cdot e^{-\lambda h}$ with optimal peak horizon $h^* = \beta / \lambda$.
  - **Bi-Exponential**: $\text{IC}(h) = A_{\text{fast}} e^{-\lambda_{\text{fast}} h} + A_{\text{slow}} e^{-\lambda_{\text{slow}} h}$ isolating fast microstructure alpha dissipation from slow macro diffusion.
  - Model selection automatically arbitrated via Akaike Information Criterion ($\text{AIC}$).

### 2. Symbolic Formulaic Alpha Engine
- Safe, non-`eval` Abstract Syntax Tree (AST) evaluator parsing symbolic expressions (WorldQuant Alpha101 / 101 Alphas format).
- Full vectorized operator library:
  - **Cross-Sectional**: `cs_rank(x)`, `cs_zscore(x)`, `scale(x, target=1.0)`.
  - **Time-Series**: `ts_decay_linear(x, d)`, `ts_rank(x, d)`, `ts_delta(x, d)`, `ts_mean(x, d)`, `ts_std(x, d)`, `ts_corr(x, y, d)`.
  - **Algebraic**: `signed_power(x, p)`, `+`, `-`, `*`, `/`.

### 3. Factor Neutralization & Convex Optimization
- **Tensor WLS Factor Orthogonalization**: Cross-sectional weighted least squares solved via batch Einstein summation (`np.einsum`) tensor contractions (~100x acceleration over iterative regressions) with square-root market-cap weighting.
- **Analytical Ledoit-Wolf Covariance Shrinkage**: Optimal linear combination $\mathbf{\Sigma}^* = \delta^* \mathbf{F} + (1 - \delta^*) \mathbf{S}$ guaranteeing well-conditioned, invertible covariance matrices.
- **Factor-Constrained Portfolio Optimizer**: Quadratic programming solver maximizing expected alpha while enforcing dollar neutrality ($\mathbf{1}^\top \mathbf{w} = 0$), factor neutrality ($\mathbf{B}^\top \mathbf{w} = \mathbf{0}$), leverage bounds ($\sum |w_i| \le L$), and quadratic turnover penalties.

### 4. Microstructure & Market Impact Simulation
- **Almgren-Chriss Execution Drag**:
  - Temporary square-root price impact: $I_{\text{temp}} = \gamma \cdot \sigma_t \cdot \sqrt{\frac{|\Delta w_{i,t}| \cdot \text{AUM}}{\text{ADV}_{i,t}}}$.
  - Permanent linear price impact: $I_{\text{perm}} = \kappa \cdot \sigma_t \cdot \frac{|\Delta w_{i,t}| \cdot \text{AUM}}{\text{ADV}_{i,t}}$.
- **Dynamic Endogenous Borrow Rates**: Asset-specific short-leg borrow cost modeling scaling quadratically with short crowding and market utilization.
- **Capital Capacity Stress Curves**: Multi-scale AUM stress tests (\$1M to \$500M) computing the capital threshold where execution impact erodes Net Sharpe to zero.

### 5. Multi-Alpha Clustering & HRP Master Ensemble
- **Hierarchical Alpha Clustering**: Strategy correlation distance metric $d_{ij} = \sqrt{0.5(1 - \rho_{ij})}$ and optimal leaf seriation to group redundant alphas into uncorrelated clusters.
- **Hierarchical Risk Parity (HRP)**: Recursive bisection inverse-variance allocation delivering maximum out-of-sample diversification without matrix inversion.
- **Portfolio Diversification Ratio**: Real-time ratio of weighted asset volatilities to total ensemble portfolio volatility: $\text{DR} = \frac{\mathbf{w}^\top \mathbf{\sigma}}{\sqrt{\mathbf{w}^\top \mathbf{\Sigma} \mathbf{w}}}$.

### 6. Regime-Conditional Alpha Decay Engine
- Unsupervised point-in-time market regime detection across:
  - Realized cross-sectional return volatility (`low_vol`, `normal_vol`, `high_vol`).
  - Cross-sectional trend & momentum (`bull_trend`, `bear_trend`, `sideways`).
  - Liquidity & volume surges (`low_volume`, `surge_volume`).
- Piecewise decay calibration ($\lambda_{\text{regime}}$) with robustness classification:
  - `all_weather`: Consistent positive IC and stable decay across every regime.
  - `fair_weather`: Profitable in benign regimes but decays rapidly under stress.
  - `fragile`: Inversion of alpha or catastrophic decay under high volatility.

### 7. Overfitting Defenses & Cross-Validation
- **Combinatorial Purged Cross-Validation (CPCV)**: Generates $N$ groups choosing $k$ test splits with training label purging and post-test embargoes to completely prevent lookahead and autocorrelation leakage.
- **Probability of Backtest Overfitting (PBO)**: Non-parametric estimation of the probability that the in-sample optimal alpha underperforms the median out-of-sample strategy.
- **Deflated Sharpe Ratio (DSR)**: Adjusts for trial variance, non-normality (skewness and kurtosis), and multiple testing count (Bailey & López de Prado).
- **Haircut Sharpe Ratio & Benjamini-Hochberg FDR**: Multiple hypothesis correction controlling false positive discovery rates.

### 8. Interactive Visual Reporting
- **Multi-Chart Strategy Tear-Sheets**: Self-contained SVG/HTML reports featuring IC decay curves, cumulative gross/net PnL with drawdown underwater charts, capital capacity curves, and factor exposure bars.
- **Executive Master Observatory Dashboard**: Universe-wide scatter plot (Mean IC vs. $\lambda$ decay rate colored by decay family), cross-alpha PnL correlation heatmap matrix, and summary ranking tables.

---

## Repository Layout

```
d:\Bonsai\
├── src\bonsai_alpha_decay\
│   ├── __init__.py            # Central package exports
│   ├── cli.py                 # CLI entry points: bonsai-run, bonsai-generate, bonsai-ingest
│   ├── cpcv.py                # Combinatorial Purged Cross-Validation & PBO estimation
│   ├── decay.py               # Exponential, Power-law, Hump, and Bi-Exponential decay fitting
│   ├── drivers.py             # Alpha driver attribution, VIF checks, and Ridge regression
│   ├── engine.py              # 2D vectorized IC, Newey-West HAC term structure, turnover
│   ├── ensemble.py            # Hierarchical clustering, quasi-diagonal seriation, HRP ensemble
│   ├── expressions.py         # Symbolic Formulaic Alpha AST parser and vectorized operators
│   ├── ingestion.py           # Production daily OHLCV bar loader with delisting returns
│   ├── models.py              # MarketPanel, StrategySpec, ObservedStrategy data models
│   ├── neutralization.py      # Vectorized WLS factor neutralization via batch einsum
│   ├── observatory.py         # Multi-strategy orchestration with parallel execution engines
│   ├── observatory_report.py  # Executive Master Observatory HTML/SVG dashboard generator
│   ├── optimization.py        # Factor-constrained convex portfolio optimizer & Ledoit-Wolf
│   ├── persistence.py         # Parquet, JSON, and CSV serialization + MarketPanel disk I/O
│   ├── portfolio.py           # Long/short simulation, Almgren-Chriss impact, dynamic borrow
│   ├── regimes.py             # Regime detection, piecewise decay, fragility classification
│   ├── reporting.py           # Multi-chart SVG tearsheet generation
│   ├── rolling.py             # Rolling decay profiles and crowding compression detection
│   ├── strategies.py          # Strategy specification factory (random, archetype, informed)
│   ├── synthetic.py           # Synthetic panel generator with GARCH volatility and fat tails
│   └── validation.py          # DSR, Haircut Sharpe, and Benjamini-Hochberg FDR corrections
├── docs\                      # In-depth architectural & theoretical documentation
├── examples\                  # Runnable end-to-end Python workflows
├── tests\                     # Comprehensive pytest test suite (39 tests)
└── pyproject.toml             # Build system and console script specifications
```

---

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/your-org/bonsai-alpha-decay.git
cd bonsai-alpha-decay

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies in editable mode
pip install --upgrade pip
pip install -e .
```

To run the complete test suite:
```bash
pytest -v
```

---

## Command-Line Interface (CLI)

Bonsai installs three command-line tools:

### 1. `bonsai-generate`
Generates seeded synthetic strategy specifications.

```bash
bonsai-generate --days 756 --assets 250 --features 24 --count 500 --seed 7 --output data/strategies.json
```

### 2. `bonsai-run`
Runs the alpha decay observatory across the strategy universe.

```bash
bonsai-run \
  --days 756 \
  --assets 250 \
  --features 24 \
  --count 100 \
  --workers 8 \
  --backend process \
  --simulate-pnl \
  --cost-bps 5.0 \
  --tearsheet \
  --format parquet \
  --output reports/run
```

**Key CLI Arguments:**
- `--backend [thread|process]`: Switch between thread pool and multi-core `ProcessPoolExecutor` for parallel execution.
- `--panel-dir <path>`: Load a pre-ingested, real-world `MarketPanel` from disk instead of generating synthetic data.
- `--simulate-pnl`: Simulate dynamic execution drag (Almgren-Chriss impact, borrow fees).
- `--tearsheet`: Generate visual SVG tear-sheets for top strategies and the Executive Master Observatory dashboard.
- `--strategy-mode [random|archetype|informed]`: Select strategy generation archetype.

### 3. `bonsai-ingest`
Ingests production daily OHLCV bar files (Parquet or CSV) into a validated point-in-time `MarketPanel` with delisting penalty returns to prevent survivorship bias.

```bash
bonsai-ingest \
  --input-dir data/bars/ \
  --output-dir data/panel/ \
  --format parquet \
  --delisting-penalty -0.35
```

---

## Python API Tutorials

### 1. Symbolic Formulaic Alpha Mining

Evaluate WorldQuant-style alpha expressions directly against a `MarketPanel`:

```python
from bonsai_alpha_decay import (
    SyntheticConfig,
    evaluate_formulaic_alpha,
    make_synthetic_panel,
)

panel = make_synthetic_panel(SyntheticConfig(n_days=252, n_assets=100, seed=42))

# Vectorized formulaic alpha expression
expr = "scale(cs_rank(ts_delta(close, 5)) - cs_rank(ts_mean(volume, 20)))"
alpha_signal = evaluate_formulaic_alpha(expr, panel=panel)

print(alpha_signal.head())
```

### 2. Factor-Constrained Convex Portfolio Optimization

Construct an optimal long/short portfolio with strict dollar neutrality and factor neutrality:

```python
from bonsai_alpha_decay import ledoit_wolf_shrinkage, optimize_portfolio_weights

# Market beta loading from panel features
beta_cols = [c for c in panel.features.columns if c.endswith("__feature_00")]
beta_loadings = panel.features.loc[:, beta_cols].iloc[-1].to_numpy().reshape(-1, 1)
shrunk_cov = ledoit_wolf_shrinkage(panel.returns.iloc[-60:].to_numpy())

# Optimize weights at time t
w = optimize_portfolio_weights(
    alpha=alpha_signal.iloc[-1].to_numpy(),
    factors=beta_loadings,
    cov=shrunk_cov,
    risk_aversion=1.0,
    max_weight=0.04,
)

assert abs(w.sum()) < 1e-5  # Dollar neutral
print(f"Active long positions: {(w > 0.001).sum()}, short positions: {(w < -0.001).sum()}")
```

### 3. Hierarchical Risk Parity (HRP) Master Ensemble

Cluster redundant alphas and allocate capital via inverse-variance tree seriation:

```python
from bonsai_alpha_decay import (
    compute_strategy_correlation_matrix,
    hierarchical_alpha_clustering,
    synthesize_master_ensemble,
)

# strategy_returns is a DataFrame of Net PnL series across strategies
corr = compute_strategy_correlation_matrix(strategy_returns)
clustering = hierarchical_alpha_clustering(corr, max_clusters=5)

ensemble = synthesize_master_ensemble(
    strategy_pnl_or_results=strategy_returns,
    ensemble_method="hrp",
)

print(f"Master Ensemble Sharpe: {ensemble['performance']['sharpe_ratio']:.2f}")
print(f"Diversification Ratio: {ensemble['diversification_ratio']:.2f}x")
```

### 4. Regime-Conditional Alpha Decay Calibration

Detect macro volatility and trend regimes and calibrate piecewise decay:

```python
from bonsai_alpha_decay import (
    detect_market_regimes,
    estimate_regime_conditional_decay,
)

# Detect regimes across market history
regimes = detect_market_regimes(panel, method="volatility", n_regimes=2)

# Calibrate decay across low_vol vs high_vol environments
regime_decay = estimate_regime_conditional_decay(ic_table, regimes)

print(f"Alpha Fragility Profile: {regime_decay['classification']}")
print(f"Low Vol Half-Life: {regime_decay['regimes']['low_vol']['half_life']:.1f} days")
print(f"High Vol Half-Life: {regime_decay['regimes']['high_vol']['half_life']:.1f} days")
```

### 5. Combinatorial Purged Cross-Validation (CPCV)

Estimate the Probability of Backtest Overfitting (PBO):

```python
from bonsai_alpha_decay import CombinatorialPurgedCV, evaluate_strategy_cpcv

cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, purge_window=5, embargo_pct=0.02)
pbo_results = evaluate_strategy_cpcv(strategy_returns_matrix, cv)

print(f"Probability of Backtest Overfitting (PBO): {pbo_results['pbo']:.2%}")
```

### 6. Executive Master Observatory Report Generation

Generate the standalone HTML/SVG master dashboard:

```python
from bonsai_alpha_decay import generate_master_observatory_html, save_master_report

html = generate_master_observatory_html(summary_df, ensemble_info=ensemble)
save_master_report(html, "reports/master_observatory_report.html")
```

---

## Econometric Formulations

### 1. Newey-West HAC Variance Estimator
For forward return horizons $h > 1$, overlapping observations induce an $\text{MA}(h-1)$ autocorrelation structure in the IC series:
$$\widehat{\text{Var}}(\overline{\text{IC}}) = \frac{1}{T} \left( \widehat{\gamma}_0 + 2 \sum_{l=1}^{h-1} \left(1 - \frac{l}{h}\right) \widehat{\gamma}_l \right)$$
The Newey-West adjusted $t$-statistic is:
$$t_{\text{HAC}} = \frac{\overline{\text{IC}}}{\sqrt{\widehat{\text{Var}}(\overline{\text{IC}})}}$$

### 2. Bi-Exponential Decay Dynamics
Distinguishes immediate order-flow / microstructure dissipation from fundamental information diffusion:
$$\text{IC}(h) = A_{\text{fast}} e^{-\lambda_{\text{fast}} h} + A_{\text{slow}} e^{-\lambda_{\text{slow}} h}, \quad \text{with } \lambda_{\text{fast}} \gg \lambda_{\text{slow}}$$

### 3. Almgren-Chriss Execution Drag
Total transaction cost per asset per rebalance:
$$\Delta \text{Cost}_{i,t} = \text{AUM} \cdot |\Delta w_{i,t}| \left( \text{Fee}_{\text{fixed}} + \gamma \sigma_{i,t} \sqrt{\frac{|\Delta w_{i,t}| \text{AUM}}{\text{ADV}_{i,t}}} + \kappa \sigma_{i,t} \frac{|\Delta w_{i,t}| \text{AUM}}{\text{ADV}_{i,t}} \right)$$

### 4. Ledoit-Wolf Covariance Shrinkage
Optimal linear combination of the empirical sample covariance $\mathbf{S}$ and constant-correlation target $\mathbf{F}$:
$$\mathbf{\Sigma}^* = \delta^* \mathbf{F} + (1 - \delta^*) \mathbf{S}, \quad \delta^* \in [0, 1]$$

---

## License

Proprietary quantitative research engine. All rights reserved.
