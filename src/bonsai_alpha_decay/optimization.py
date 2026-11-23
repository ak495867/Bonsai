from __future__ import annotations

from typing import Any, Mapping
from scipy.optimize import minimize
import numpy as np
import pandas as pd

from .portfolio import portfolio_performance_stats


def ledoit_wolf_shrinkage(returns_matrix: np.ndarray) -> np.ndarray:
    """
    Computes Ledoit-Wolf analytical covariance shrinkage towards diagonal target.
    Guarantees positive definiteness even when N >= T.
    """
    X = np.asarray(returns_matrix, dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    t, n = X.shape
    if t < 3:
        return np.eye(n) * 0.04

    # Demean returns
    mean = np.mean(X, axis=0)
    X_c = X - mean

    # Sample covariance S
    sample_cov = (X_c.T @ X_c) / (t - 1)

    # Shrinkage target: diagonal with equal variance
    var = np.diag(sample_cov)
    avg_var = float(np.mean(var))
    target = np.eye(n) * avg_var

    # Estimate optimal shrinkage intensity delta
    # Formula based on Ledoit & Wolf (2004)
    y = X_c**2
    phi_mat = (y.T @ y) / t - 2.0 * (X_c.T @ X_c) * sample_cov / t + sample_cov**2
    phi = float(np.sum(phi_mat))

    gamma = float(np.linalg.norm(sample_cov - target, "fro") ** 2)
    kappa = phi / gamma if gamma > 0 else 0.0
    shrinkage = max(0.0, min(1.0, kappa / t))

    shrunk = (1.0 - shrinkage) * sample_cov + shrinkage * target
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(shrunk))
    if min_eig <= 1e-8:
        shrunk += (abs(min_eig) + 1e-6) * np.eye(n)

    return shrunk


def optimize_portfolio_weights(
    alpha: Sequence[float] | np.ndarray,
    factors: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    prev_weights: np.ndarray | None = None,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.0,
    max_weight: float = 0.10,
    max_leverage: float = 2.0,
) -> np.ndarray:
    """
    Solves convex mean-variance portfolio construction with factor neutrality.

    min_w -w^T alpha + 0.5 * lambda_risk * w^T Sigma w + lambda_turnover * ||w - w0||_1
    s.t.
        sum(w) = 0                  (dollar neutrality)
        F^T w = 0                   (factor neutrality)
        sum(|w|) <= max_leverage    (gross leverage cap)
        -max_weight <= w_i <= max_weight
    """
    a = np.asarray(alpha, dtype=float)
    n = len(a)
    if n == 0:
        return np.array([], dtype=float)

    sigma = cov if cov is not None else np.eye(n) * 0.04
    w0 = prev_weights if prev_weights is not None else np.zeros(n, dtype=float)

    # Objective function
    def objective(w: np.ndarray) -> float:
        ret = float(np.dot(w, a))
        risk = 0.5 * risk_aversion * float(w.T @ sigma @ w)
        t_cost = turnover_penalty * float(np.sum(np.abs(w - w0)))
        return -ret + risk + t_cost

    def obj_grad(w: np.ndarray) -> np.ndarray:
        grad = -a + risk_aversion * (sigma @ w)
        if turnover_penalty > 0.0:
            grad += turnover_penalty * np.sign(w - w0)
        return grad

    # Constraints
    constraints: list[dict[str, Any]] = []

    # 1. Dollar neutrality: sum(w) = 0
    constraints.append({"type": "eq", "fun": lambda w: float(np.sum(w))})

    # 2. Factor neutrality: F^T w = 0
    if factors is not None and factors.shape[1] > 0:
        f_mat = np.asarray(factors, dtype=float)
        for k in range(f_mat.shape[1]):
            f_col = f_mat[:, k]
            constraints.append({"type": "eq", "fun": lambda w, col=f_col: float(np.dot(w, col))})

    # 3. Gross leverage: max_leverage - sum(|w|) >= 0
    constraints.append({"type": "ineq", "fun": lambda w: float(max_leverage - np.sum(np.abs(w)))})

    # Bounds: [-max_weight, max_weight]
    bounds = [(-abs(max_weight), abs(max_weight)) for _ in range(n)]

    # Initial guess: dollar-neutral projection of alpha
    w_init = a - np.mean(a)
    sum_abs = np.sum(np.abs(w_init))
    if sum_abs > 0:
        w_init = (w_init / sum_abs) * min(max_leverage, 1.0)
    else:
        w_init = np.zeros(n)

    res = minimize(
        objective,
        w_init,
        jac=obj_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-7},
    )

    if res.success:
        return res.x
    return w_init


def simulate_optimized_portfolio_pnl(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    factors: Mapping[str, pd.DataFrame] | None = None,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.0002,
    max_weight: float = 0.10,
    max_leverage: float = 2.0,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Runs walk-forward convex portfolio optimization over time.
    """
    if not signal.index.equals(returns.index) or not signal.columns.equals(returns.columns):
        raise ValueError("signal and returns must share identical index and columns")

    dates = signal.index
    assets = signal.columns
    n_dates = len(dates)
    n_assets = len(assets)

    weights_mat = np.zeros((n_dates, n_assets), dtype=float)
    ret_mat = returns.to_numpy(dtype=float)

    # Estimate rolling shrinkage covariance
    shrunk_cov = ledoit_wolf_shrinkage(ret_mat)

    fac_names = list(factors.keys()) if factors else []
    fac_mats = [factors[k].to_numpy(dtype=float) for k in fac_names] if factors else []

    w_prev = np.zeros(n_assets, dtype=float)
    for t in range(n_dates):
        a_row = signal.iloc[t].to_numpy(dtype=float)
        valid = np.isfinite(a_row)
        if np.sum(valid) < 3:
            weights_mat[t] = 0.0
            continue

        a_clean = np.where(valid, a_row, 0.0)

        f_t = None
        if fac_mats:
            f_cols = [fm[t] for fm in fac_mats]
            f_t = np.column_stack(f_cols)

        # Optimize for date t
        w_opt = optimize_portfolio_weights(
            alpha=a_clean,
            factors=f_t,
            cov=shrunk_cov,
            prev_weights=w_prev,
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
            max_weight=max_weight,
            max_leverage=max_leverage,
        )
        weights_mat[t] = w_opt
        w_prev = w_opt

    weights_df = pd.DataFrame(weights_mat, index=dates, columns=assets)
    active_weights = weights_df.shift(1).fillna(0.0)
    gross_pnl = (active_weights * returns).sum(axis=1)

    delta_w = weights_df.diff().abs()
    delta_w.iloc[0] = weights_df.iloc[0].abs()
    turnover = delta_w.sum(axis=1) * 0.5
    cost = turnover * (cost_bps / 10000.0)
    net_pnl = gross_pnl - cost

    cum_gross = (1.0 + gross_pnl).cumprod() - 1.0
    cum_net = (1.0 + net_pnl).cumprod() - 1.0

    return pd.DataFrame(
        {
            "gross_pnl": gross_pnl,
            "turnover": turnover,
            "cost": cost,
            "net_pnl": net_pnl,
            "cum_gross": cum_gross,
            "cum_net": cum_net,
        },
        index=dates,
    )
