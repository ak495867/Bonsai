# Bonsai Architecture & Quantitative Design

This document details the architectural principles, computational graph, tensor designs, and econometric frameworks underlying the **Bonsai Alpha Decay Observatory**.

---

## 1. System Topology & Dataflow

Bonsai operates on a zero-lookahead, strictly point-in-time pipelined architecture:

```
[Production Daily OHLCV Bars / Synthetic Panel]
                   │
                   ▼
       [MarketPanel Validation]
  (prices, returns, features, metadata, mask)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
[Strategy Specifications] [Symbolic Formulaic Alpha Engine]
(feature combinations)   (AST parser: ts_*, cs_*, scale)
        └──────────┬──────────┘
                   │
                   ▼
     [Signal Neutralization Engine]
  (Vectorized WLS via np.einsum tensor contractions)
                   │
                   ▼
     [Vectorized 2D Engine & HAC]
   - Forward returns: R_{t,t+h}
   - Cross-sectional Spearman & Pearson IC
   - Multi-horizon Newey-West HAC covariance
                   │
                   ▼
     [Decay Curve Calibration]
   - Exponential: e^{-λ h}
   - Power-Law: h^{-α}
   - Delayed Diffusion (Hump): h^β e^{-λ h}
   - Bi-Exponential: A_f e^{-λ_f h} + A_s e^{-λ_s h}
   - Automated AIC Model Selection
                   │
                   ▼
    [Microstructure & Portfolio Engine]
   - Continuous weighting: zscore, vol_parity, linear_rank
   - Convex Optimizer: min w^T Σ w - α^T w + γ ||Δw|| s.t. B^T w = 0
   - Almgren-Chriss temporary & permanent impact
   - Dynamic short-leg borrow rates
   - Multi-scale AUM capacity stress curve
                   │
                   ▼
    [Overfitting & Regime Defense]
   - Combinatorial Purged Cross-Validation (CPCV) & PBO
   - Deflated Sharpe Ratio (DSR) & Benjamini-Hochberg FDR
   - Unsupervised Regime Detection (volatility, trend, volume)
   - Fragility classification (all_weather, fair_weather, fragile)
                   │
                   ▼
   [Hierarchical Clustering & HRP Ensemble]
   - Distance metric: d_{ij} = sqrt(0.5 * (1 - ρ_{ij}))
   - Quasi-diagonal seriation & recursive bisection
                   │
                   ▼
  [Executive Master Observatory Dashboard & Visual SVG Tear-Sheets]
```

---

## 2. Core Modules & Engine Internals

### 2.1 The MarketPanel Data Model (`models.py`)
A `MarketPanel` encapsulates the multi-asset universe with strict point-in-time constraints:
- `prices` $(T \times N)$: Positive closing prices.
- `returns` $(T \times N)$: Daily log/arithmetic asset returns.
- `features` $(T \times (N \cdot K))$: Point-in-time feature matrix with columns indexed as `{asset}__{feature}`.
- `metadata` $(N \times M)$: Static and rolling asset properties (`market_cap`, `volatility`, `liquidity`, `borrow_utilization`).
- `mask` $(T \times N)$: Optional boolean tradeable universe indicator preventing survivorship bias.

### 2.2 Tensor WLS Neutralization (`neutralization.py`)
To isolate idiosyncratic residual alpha $\mathbf{\epsilon}_t$ from common risk drivers $\mathbf{X}_t$ (Market Beta, Size, Volatility, Momentum), Bonsai avoids looping over time steps.
Given weights $\mathbf{w}_t = \sqrt{\text{market\_cap}_{t}}$, the weighted OLS estimator is:
$$\mathbf{\beta}_t = \left(\mathbf{X}_t^\top \mathbf{W}_t \mathbf{X}_t\right)^{-1} \mathbf{X}_t^\top \mathbf{W}_t \mathbf{y}_t$$
In Bonsai, this is vectorized across all dates $T$ using batch tensor contractions:
```python
# Transform to weighted space
X_w = X * sqrt_w[:, :, None]
y_w = y * sqrt_w

# Batch X^T X and X^T y via Einstein summation
XtX = np.einsum('tni,tnj->tij', X_w, X_w)  # (T, K, K)
Xty = np.einsum('tni,tn->ti', X_w, y_w)     # (T, K)

# Batch linear solve
betas = np.linalg.solve(XtX + reg * np.eye(K), Xty)
y_pred = np.einsum('tni,ti->tn', X, betas)
residuals = y - y_pred
```
This achieves a **~100x speedup** over iterative regression loops.

### 2.3 Symbolic Formulaic Alpha Engine (`expressions.py`)
Expressions are evaluated using Python's standard `ast` library in an isolated execution tree without unsafe `eval()` or string formatting:
- **`ts_decay_linear(x, d)`**: Linearly weighted rolling sum:
  $$w_i = \frac{d - i}{\sum_{k=0}^{d-1} (d - k)}, \quad i \in [0, d-1]$$
- **`ts_rank(x, d)`**: Rolling percentile rank over $d$ days.
- **`cs_rank(x)`**: Cross-sectional percentile rank across universe $N$ at each date $t$:
  $$\text{cs\_rank}(x)_{i,t} = \frac{\text{rank}(x_{i,t}) - 1}{N_t - 1}$$
- **`cs_zscore(x)`**: Cross-sectional standardization $(\mathbf{x}_t - \mu_t) / \sigma_t$.
- **`scale(x, target=1.0)`**: Normalizes cross-sectional weights so $\sum_i |w_i| = \text{target}$.

### 2.4 Factor-Constrained Convex Optimization (`optimization.py`)
Portfolio construction is framed as a quadratic program (QP):
$$\min_{\mathbf{w}} \frac{1}{2} \mathbf{w}^\top \mathbf{\Sigma} \mathbf{w} - \lambda_{\text{alpha}} \mathbf{\alpha}^\top \mathbf{w} + \frac{\gamma}{2} \|\mathbf{w} - \mathbf{w}_{t-1}\|_2^2$$
Subject to:
1. **Dollar Neutrality**: $\mathbf{1}^\top \mathbf{w} = 0$
2. **Factor Neutrality**: $\mathbf{B}^\top \mathbf{w} = \mathbf{0}$
3. **Gross Leverage**: $\|\mathbf{w}\|_1 \le L_{\max}$
4. **Position Bounds**: $-w_{\max} \le w_i \le w_{\max}$

Covariance matrix $\mathbf{\Sigma}$ is conditioned using analytical **Ledoit-Wolf shrinkage**, preventing singularity when $N > T$.

### 2.5 Hierarchical Risk Parity (HRP) Multi-Alpha Ensemble (`ensemble.py`)
Unlike classical Markowitz optimization which suffers from error maximization when inverting covariance matrices, HRP operates in three robust stages:
1. **Tree Clustering**: Converts strategy correlation matrix $\mathbf{R}$ to distance metric $d_{ij} = \sqrt{0.5 (1 - \rho_{ij})}$ and builds a hierarchical tree using Ward's minimum variance or single linkage.
2. **Quasi-Diagonalization**: Reorders the correlation matrix so that correlated strategies sit adjacently along the diagonal.
3. **Recursive Bisection**: Recursively bisects clusters, allocating capital based on inverse cluster variance:
   $$w_1 = 1 - \frac{V_1}{V_1 + V_2}, \quad w_2 = \frac{V_1}{V_1 + V_2}$$
   where $V_k$ is the cluster variance under inverse-variance weights.

### 2.6 Combinatorial Purged Cross-Validation (`cpcv.py`)
Standard $k$-fold cross-validation leaks information in financial time series due to serial correlation and overlapping labels. CPCV partitions $T$ observations into $N$ contiguous blocks, generating $\binom{N}{k}$ combinations of test splits:
- **Purging**: Removes training samples whose forward return horizon overlaps with test samples.
- **Embargoing**: Discards training samples immediately following test sets to prevent autoregressive post-test leakage.
- **PBO Estimation**: Tracks the rank of the in-sample optimal strategy within out-of-sample distributions, fitting a kernel density estimate of the probability of backtest overfitting.

---

## 3. Parallelization & Scaling

Bonsai supports two concurrent execution backends in `run_observatory`:
- `backend="thread"`: Ideal for fast in-memory runs with minimal IPC overhead.
- `backend="process"`: Leverages `ProcessPoolExecutor` across multiple physical CPU cores. Worker tasks (`_evaluate_worker`) are serialized at top module scope, fully compliant with Windows multiprocessing spawn semantics.
