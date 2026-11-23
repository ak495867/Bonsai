# Symbolic Formulaic Alpha Engine

Bonsai incorporates an institutional-grade domain-specific language (DSL) for defining and testing quantitative signals. The engine uses Python's standard `ast` module to execute mathematical expressions in vector space without lookahead bias or security risks associated with `eval()`.

---

## 1. Syntax & Operator Reference

### 1.1 Cross-Sectional Operators
Cross-sectional operators act across all assets $N$ at a single point in time $t$.

| Operator | Syntax | Description |
| :--- | :--- | :--- |
| `cs_rank(x)` | `cs_rank(x)` | Scales cross-sectional values to the unit interval $[0, 1]$. Missing values are ignored. |
| `cs_zscore(x)` | `cs_zscore(x)` | Demeans and standardizes values to zero mean and unit cross-sectional variance: $(x_i - \mu) / \sigma$. |
| `scale(x, target=1.0)` | `scale(x, 1.0)` | Normalizes weights such that the sum of absolute values equals `target`: $\sum_i \|w_i\| = \text{target}$. |

### 1.2 Time-Series Operators
Time-series operators act along the date dimension $T$ independently for each asset $N$.

| Operator | Syntax | Description |
| :--- | :--- | :--- |
| `ts_delta(x, d)` | `ts_delta(close, 5)` | Time-series difference over $d$ periods: $x_t - x_{t-d}$. |
| `ts_mean(x, d)` | `ts_mean(volume, 20)` | Rolling simple moving average over window $d$. |
| `ts_std(x, d)` | `ts_std(returns, 20)` | Rolling sample standard deviation over window $d$. |
| `ts_rank(x, d)` | `ts_rank(close, 10)` | Rolling percentile rank of the current value within the past $d$ observations. |
| `ts_decay_linear(x, d)` | `ts_decay_linear(close, 10)` | Linearly weighted moving average with weights $w_k = d - k$. |
| `ts_corr(x, y, d)` | `ts_corr(close, volume, 20)` | Rolling Pearson correlation between series $x$ and $y$ over window $d$. |

### 1.3 Algebraic & Non-Linear Functions

| Operator | Syntax | Description |
| :--- | :--- | :--- |
| `signed_power(x, p)` | `signed_power(x, 2.0)` | Preserves sign: $\text{sign}(x) \cdot \|x\|^p$. |
| Standard Arithmetic | `x + y`, `x - y`, `x * y`, `x / y` | Element-wise vectorized algebraic operations. |
| Negation | `-x` | Element-wise sign flip. |

---

## 2. Market Panel Data Variables

Expressions can reference standard price/volume fields and features present in the `MarketPanel`:
- `close`: Daily closing prices.
- `returns`: Daily asset returns.
- `volume`: Daily trading volume or dollar liquidity.
- `feature_XX`: Domain features present in `panel.features` (e.g., `feature_00`, `feature_01`, etc.).

---

## 3. Alpha Expression Library (WorldQuant Alpha101 Examples)

### 3.1 Short-Term Reversal
Captures mean-reversion over 5 trading days:
```python
alpha_rev = "scale(-1.0 * ts_delta(close, 5))"
```

### 3.2 Volume-Weighted Price Momentum
Blends 20-day momentum with linear decay weighting:
```python
alpha_mom = "scale(ts_decay_linear(ts_delta(close, 20), 10))"
```

### 3.3 Price-Volume Divergence
Exploits correlation between price changes and volume surges:
```python
alpha_divergence = "-1.0 * ts_corr(ts_delta(close, 1), ts_delta(volume, 1), 20)"
```

### 3.4 Multi-Factor Composite Alpha
Combines cross-sectional rank of momentum with inverse volatility:
```python
alpha_composite = "scale(cs_rank(ts_delta(close, 10)) - cs_rank(ts_std(returns, 20)))"
```

---

## 4. Python API Usage

```python
from bonsai_alpha_decay import (
    SyntheticConfig,
    evaluate_formulaic_alpha,
    make_synthetic_panel,
)

panel = make_synthetic_panel(SyntheticConfig(n_days=252, n_assets=100, seed=7))

# Evaluate expression safely
signal = evaluate_formulaic_alpha(
    "scale(cs_rank(ts_delta(close, 5)) - cs_rank(ts_mean(volume, 20)))",
    panel=panel,
)

print("Signal shape:", signal.shape)
print("Latest signal vector:\n", signal.iloc[-1].head())
```
