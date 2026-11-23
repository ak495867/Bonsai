# Alpha Decay Term Structure & Market Regimes

Alpha is non-stationary. A quantitative signal that produces high information coefficients ($\text{IC}$) in low-volatility trending markets may rapidly decay, invert, or incur catastrophic execution drag during volatile liquidity shocks. Bonsai provides an end-to-end econometric framework to measure alpha decay term structures and evaluate regime-conditional fragility.

---

## 1. The Information Coefficient Term Structure

Given an alpha signal $\mathbf{s}_t = (s_{1,t}, \dots, s_{N,t})^\top$ and forward asset returns $\mathbf{R}_{t, t+h}$, the cross-sectional Information Coefficient at horizon $h$ is:
$$\text{IC}_t(h) = \text{Corr}_{\text{Spearman}}\left(\mathbf{s}_t, \mathbf{R}_{t, t+h}\right)$$

Evaluating across multiple horizons $h \in \{1, 2, 3, 5, 10, 20\}$ constructs the **Alpha Term Structure**:
$$\overline{\text{IC}}(h) = \frac{1}{T} \sum_{t=1}^T \text{IC}_t(h)$$

### 1.1 Newey-West HAC Econometric Inference
Because forward returns overlap for horizons $h > 1$, the time series of information coefficients $\text{IC}_t(h)$ exhibits serial correlation up to lag $h - 1$. Applying standard ordinary least squares (OLS) variance formulas inflates $t$-statistics by a factor of $\sqrt{h}$, creating false illusions of statistical significance.

Bonsai enforces Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) variance:
$$\widehat{\text{Var}}_{\text{HAC}}(\overline{\text{IC}}) = \frac{1}{T} \left( \widehat{\gamma}_0 + 2 \sum_{l=1}^{h-1} \left(1 - \frac{l}{h}\right) \widehat{\gamma}_l \right)$$
where $\widehat{\gamma}_l = \frac{1}{T} \sum_{t=l+1}^T (\text{IC}_t - \overline{\text{IC}})(\text{IC}_{t-l} - \overline{\text{IC}})$.

The robust $t$-statistic is computed as:
$$t_{\text{HAC}} = \frac{\overline{\text{IC}}(h)}{\sqrt{\widehat{\text{Var}}_{\text{HAC}}(\overline{\text{IC}})}}$$

---

## 2. Generalized Alpha Decay Families

Bonsai fits four functional forms to the empirical $\overline{\text{IC}}(h)$ term structure using non-linear least squares (Levenberg-Marquardt):

### 2.1 Exponential Decay
$$\text{IC}(h) = \text{IC}_0 \cdot e^{-\lambda h}$$
- **Decay Rate**: $\lambda > 0$
- **Half-Life**: $t_{1/2} = \frac{\ln(2)}{\lambda}$
- **Interpretation**: Standard memoryless diffusion of information into asset prices.

### 2.2 Power-Law (Hyperbolic) Decay
$$\text{IC}(h) = \text{IC}_0 \cdot h^{-\alpha}$$
- **Long-Memory Parameter**: $\alpha > 0$
- **Interpretation**: Pervasive in fundamental signals (earnings, value) where information incorporates slowly across multi-quarter investor horizons.

### 2.3 Delayed Diffusion (Hump-Shaped) Decay
$$\text{IC}(h) = A \cdot h^\beta \cdot e^{-\lambda h}$$
- **Optimal Peak Horizon**: $h^* = \frac{\beta}{\lambda}$
- **Interpretation**: Arises when signals require gradual investor attention or execution cascades before reaching full price impact.

### 2.4 Bi-Exponential Decay
$$\text{IC}(h) = A_{\text{fast}} \cdot e^{-\lambda_{\text{fast}} h} + A_{\text{slow}} \cdot e^{-\lambda_{\text{slow}} h}$$
- **Fast Microstructure Component**: $\lambda_{\text{fast}} \gg \lambda_{\text{slow}}$ captures order-flow toxicity and transient liquidity consumption.
- **Slow Macro Component**: Captures sustained economic drift.

### 2.5 Automated AIC Model Selection
The optimal model family is determined via the Akaike Information Criterion:
$$\text{AIC} = n \cdot \ln\left(\frac{\text{RSS}}{n}\right) + 2k + \frac{2k(k+1)}{n - k - 1}$$
where $n$ is the number of horizons tested, $\text{RSS}$ is the residual sum of squares, and $k$ is the parameter count.

---

## 3. Unsupervised Market Regime Detection

Market dynamics are partitioned into discrete point-in-time states using rolling historical metrics:

### 3.1 Volatility Regimes (`method="volatility"`)
- Measures 20-day annualized cross-sectional market return volatility $\sigma_t$.
- Discretized into `low_vol`, `normal_vol`, and `high_vol` based on historical percentile cutoffs.

### 3.2 Trend Regimes (`method="trend"`)
- Measures 20-day cumulative benchmark return $R_{\text{mkt}, t-20:t}$.
- Discretized into `bull_trend`, `bear_trend`, and `sideways`.

### 3.3 Volume Surge Regimes (`method="volume"`)
- Measures the ratio of current market volume to its 50-day moving average: $\text{VolRatio}_t = \frac{V_t}{\text{SMA}(V, 50)}$.
- Discretized into `normal_volume` and `surge_volume`.

---

## 4. Regime-Conditional Decay & Fragility

Bonsai segments the empirical IC matrix by regime and fits independent decay curves:
$$\text{IC}_{\text{regime}}(h) = \text{IC}_{0, \text{regime}} \cdot e^{-\lambda_{\text{regime}} h}$$

### Robustness Classification Taxonomy

| Profile | Behavior | Econometric Criteria |
| :--- | :--- | :--- |
| **All-Weather** | Alpha retains positive IC and stable half-life across all market conditions. | $\min_{\text{regimes}} \text{IC}_0 > 0.01$ and $\frac{\max \lambda}{\min \lambda} \le 2.5$. |
| **Fair-Weather** | Highly profitable in calm markets, but decay accelerates drastically under stress. | $\text{IC}_{0, \text{low}} > 0.03$, but $\lambda_{\text{high}} > 2.5 \cdot \lambda_{\text{low}}$. |
| **Fragile** | Inverts into negative alpha or suffers complete decay collapse during turbulent periods. | $\text{IC}_{0, \text{high}} \le 0$ or half-life collapses by $> 80\%$. |

---

## 5. Python API Example

```python
from bonsai_alpha_decay import (
    build_ic_table,
    detect_market_regimes,
    estimate_regime_conditional_decay,
)

# 1. Build empirical IC table
ic_table = build_ic_table(signal, panel.returns, horizons=(1, 2, 3, 5, 10))

# 2. Detect regimes
regimes = detect_market_regimes(panel, method="volatility", n_regimes=2)

# 3. Fit regime-conditional decay
decay_res = estimate_regime_conditional_decay(ic_table, regimes)

print(f"Alpha Robustness Profile: {decay_res['classification']}")
for regime_name, stats in decay_res["regimes"].items():
    print(f"[{regime_name}] IC0: {stats['ic0']:.4f}, Half-Life: {stats['half_life']:.2f} days")
```
