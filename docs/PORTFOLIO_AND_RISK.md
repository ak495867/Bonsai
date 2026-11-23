# Portfolio Construction, Execution Drag & Risk Modeling

In production quantitative trading, gross theoretical paper returns diverge significantly from realized net returns due to market friction. Bonsai models market microstructure, execution drag, factor neutrality, and portfolio risk.

---

## 1. Portfolio Weighting Schemes

Bonsai supports four portfolio allocation methods in `build_portfolio_weights`:

### 1.1 Quantile Long/Short (`method="quantile"`)
Assigns equal weights to the top quantile (long $Q_5$) and bottom quantile (short $Q_1$), normalized such that gross leverage $\sum |w_i| = 1.0$.

### 1.2 Cross-Sectional Z-Score (`method="zscore"`)
Constructs continuous weights proportional to cross-sectional standard deviations:
$$w_{i,t} = \frac{s_{i,t} - \mu_t}{\sum_j |s_{j,t} - \mu_t|}$$
Guarantees exact dollar neutrality $\sum_i w_{i,t} = 0$.

### 1.3 Volatility Parity (`method="vol_parity"`)
Scales continuous signals inversely with realized asset volatility $\sigma_{i,t}$:
$$w_{i,t} \propto \frac{s_{i,t} - \mu_t}{\sigma_{i,t}}$$
Ensures high-volatility names do not dominate portfolio variance.

### 1.4 Linear Rank Weighting (`method="linear_rank"`)
Centered rank weighting mapping percentile ranks linearly to $[-0.5, +0.5]$ before gross leverage scaling.

---

## 2. Microstructure & Execution Drag

### 2.1 Almgren-Chriss Market Impact Model
Trading an asset incurs both temporary and permanent price dislocations:
- **Temporary Impact** (liquidity consumption across the order book):
  $$I_{\text{temp}, i, t} = \gamma \cdot \sigma_{i,t} \cdot \sqrt{\frac{|\Delta w_{i,t}| \cdot \text{AUM}}{\text{ADV}_{i,t}}}$$
  where $\gamma \approx 0.10$ is the Kyle-Almgren price impact coefficient, $\sigma_{i,t}$ is daily return volatility, and $\text{ADV}_{i,t}$ is Average Daily Volume in dollars.
- **Permanent Impact** (adverse selection and persistent price shift):
  $$I_{\text{perm}, i, t} = \kappa \cdot \sigma_{i,t} \cdot \left(\frac{|\Delta w_{i,t}| \cdot \text{AUM}}{\text{ADV}_{i,t}}\right)$$

### 2.2 Endogenous Dynamic Borrow Fees
Borrowing shares to maintain short positions incurs an annualized financing rate that scales with utilization and crowding:
$$\text{BorrowRate}_{i,t} = \text{BaseRate} + \alpha_{\text{util}} \cdot \left(\frac{\text{ShortInterest}_{i,t}}{\text{Float}_{i,t}}\right)^2$$
Assets under hard-to-borrow (HTB) pressure incur borrow fees up to several hundred basis points annually, modeled directly in the net PnL series.

### 2.3 Capital Capacity Stress Testing
Bonsai sweeps across AUM levels (\$1M, \$5M, \$10M, \$50M, \$100M, \$500M) and simulates net performance:
- As AUM increases, transaction impact expands at rate $\mathcal{O}(\sqrt{\text{AUM}})$.
- **Capital Capacity Threshold**: The exact AUM level at which Net Sharpe decays to zero.

---

## 3. Factor-Constrained Convex Portfolio Optimizer

When unconstrained long/short allocations carry unwanted factor exposures (e.g., unintended market beta or sector tilts), Bonsai provides a convex quadratic programming solver in `optimize_portfolio_weights`.

### 3.1 Objective Function
$$\max_{\mathbf{w}} \mathbf{\alpha}^\top \mathbf{w} - \frac{\lambda_{\text{risk}}}{2} \mathbf{w}^\top \mathbf{\Sigma}^* \mathbf{w} - \gamma_{\text{turnover}} \|\mathbf{w} - \mathbf{w}_{t-1}\|_1$$

### 3.2 Constraints
$$\begin{aligned}
\mathbf{1}^\top \mathbf{w} &= 0 \quad &&\text{(Dollar Neutrality)} \\
\mathbf{B}^\top \mathbf{w} &= \mathbf{0} \quad &&\text{(Factor Neutrality across } K \text{ risk factors)} \\
\|\mathbf{w}\|_1 &\le L_{\max} \quad &&\text{(Gross Leverage Bound)} \\
-w_{\max} \le w_i &\le w_{\max} \quad &&\text{(Single-Name Concentration Limits)}
\end{aligned}$$

### 3.3 Analytical Ledoit-Wolf Covariance Shrinkage
Sample covariance matrices $\mathbf{S}$ computed over short histories $T < N$ are singular and invert with extreme numerical noise. Bonsai conditions the covariance matrix via Ledoit-Wolf shrinkage towards a constant-correlation target $\mathbf{F}$:
$$\mathbf{\Sigma}^* = \delta^* \mathbf{F} + (1 - \delta^*) \mathbf{S}$$
The optimal shrinkage intensity $\delta^* \in [0, 1]$ is derived analytically from the asymptotic distribution of second-order sample moments without cross-validation.

---

## 4. Multi-Alpha Clustering & Hierarchical Risk Parity (HRP)

When ensembling dozens of alpha strategies, equal weighting or Markowitz mean-variance optimization produces suboptimal out-of-sample allocations due to correlation clustering and estimation error.

Bonsai synthesizes a Master Ensemble via **Hierarchical Risk Parity (HRP)**:

```
                  ┌───────── Master Ensemble ─────────┐
                  │                                   │
         ┌── Cluster A (Momentum) ──┐        ┌── Cluster B (Reversal) ──┐
         │                          │        │                          │
      Alpha 1                    Alpha 2  Alpha 3                    Alpha 4
```

1. **Correlation Distance**: Strategies are mapped to metric space $d_{ij} = \sqrt{0.5 (1 - \rho_{ij})}$.
2. **Seriation**: Leaf nodes are ordered so strategy covariance entries decay smoothly away from the main diagonal.
3. **Recursive Bisection**: Capital is split top-down between left and right sub-trees according to inverse variance weighting, preventing highly correlated alphas from receiving excess total risk budget.
4. **Diversification Ratio Monitoring**:
   $$\text{DR} = \frac{\sum_i w_i \sigma_i}{\sqrt{\mathbf{w}^\top \mathbf{\Sigma} \mathbf{w}}}$$
   Measures how many effective uncorrelated bets the ensemble achieves ($> 1.5\text{x}$ target).
