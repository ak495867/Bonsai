from __future__ import annotations

from typing import Any, Mapping
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd

from .models import ObservedStrategy
from .portfolio import portfolio_performance_stats


def compute_strategy_pnl_matrix(
    results: Mapping[str, ObservedStrategy] | pd.DataFrame,
    pnl_column: str = "net_pnl",
) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results.copy()

    pnl_dict = {}
    for sid, obs in results.items():
        if obs.pnl is not None and pnl_column in obs.pnl.columns:
            pnl_dict[sid] = obs.pnl[pnl_column]

    if not pnl_dict:
        raise ValueError("No valid PnL series found in strategy results")

    df = pd.DataFrame(pnl_dict).fillna(0.0)
    return df


def compute_strategy_correlation_matrix(
    pnl_matrix: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    corr = pnl_matrix.corr(method=method)
    return corr.fillna(0.0)


def hierarchical_alpha_clustering(
    corr_matrix: pd.DataFrame,
    linkage_method: str = "ward",
) -> dict[str, Any]:
    """
    Performs Hierarchical Agglomerative Clustering on the alpha correlation matrix.
    Distance metric: d_{i, j} = sqrt(0.5 * (1 - rho_{i, j}))
    """
    corr_mat = corr_matrix.to_numpy(dtype=float)
    corr_mat = np.clip(corr_mat, -1.0, 1.0)
    # Angular / correlation distance metric
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr_mat), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)

    condensed_dist = squareform(dist, checks=False)
    # Ward linkage requires Euclidean distance; if not valid, fallback to average
    try:
        z = linkage(condensed_dist, method=linkage_method)
    except Exception:
        z = linkage(condensed_dist, method="average")

    order = leaves_list(z)
    ordered_sids = [corr_matrix.columns[i] for i in order]

    return {
        "linkage": z,
        "quasi_diag_order": order,
        "ordered_strategy_ids": ordered_sids,
    }


def _get_cluster_var(cov: np.ndarray, items: list[int]) -> float:
    cov_sub = cov[np.ix_(items, items)]
    inv_diag = 1.0 / np.clip(np.diag(cov_sub), 1e-8, np.inf)
    w = inv_diag / np.sum(inv_diag)
    return float(w.T @ cov_sub @ w)


def _hrp_recursive_bisection(
    cov: np.ndarray, items: list[int], weights: np.ndarray
) -> None:
    if len(items) <= 1:
        return

    split = len(items) // 2
    c1 = items[:split]
    c2 = items[split:]

    v1 = _get_cluster_var(cov, c1)
    v2 = _get_cluster_var(cov, c2)

    alpha1 = 1.0 - (v1 / (v1 + v2)) if (v1 + v2) > 0 else 0.5
    alpha2 = 1.0 - alpha1

    weights[c1] *= alpha1
    weights[c2] *= alpha2

    _hrp_recursive_bisection(cov, c1, weights)
    _hrp_recursive_bisection(cov, c2, weights)


def hierarchical_risk_parity_weights(
    cov_matrix: pd.DataFrame | np.ndarray,
    quasi_diag_order: list[int] | np.ndarray,
) -> pd.Series:
    """
    Computes optimal portfolio allocation weights using Hierarchical Risk Parity (HRP).
    """
    cov = cov_matrix.to_numpy(dtype=float) if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix, dtype=float)
    n = len(cov)
    weights = np.ones(n, dtype=float)

    ordered_items = list(quasi_diag_order)
    _hrp_recursive_bisection(cov, ordered_items, weights)

    weights /= np.sum(weights)
    if isinstance(cov_matrix, pd.DataFrame):
        return pd.Series(weights, index=cov_matrix.columns)
    return pd.Series(weights)


def synthesize_master_ensemble(
    results: Mapping[str, ObservedStrategy] | pd.DataFrame,
    ensemble_method: str = "hrp",
    max_strategies: int | None = None,
) -> dict[str, Any]:
    """
    Combines individual alpha strategies into an orthogonal Master Ensemble Portfolio.

    Parameters:
        results: Mapping of strategy_id to ObservedStrategy, or PnL DataFrame
        ensemble_method: "hrp" (Hierarchical Risk Parity), "equal_weight", or "inv_vol"
        max_strategies: Optional cap on strategy count (picks highest Sharpe alphas)
    """
    pnl_df = compute_strategy_pnl_matrix(results, pnl_column="net_pnl")
    if pnl_df.shape[1] < 2:
        sid = pnl_df.columns[0]
        perf = portfolio_performance_stats(pnl_df[sid])
        return {
            "weights": pd.Series([1.0], index=[sid]),
            "combined_pnl": pnl_df[sid],
            "performance": perf,
            "correlation_matrix": pd.DataFrame([[1.0]], index=[sid], columns=[sid]),
            "diversification_ratio": 1.0,
        }

    if max_strategies is not None and max_strategies < pnl_df.shape[1]:
        # Filter to top N strategies by Sharpe
        srs = {
            col: portfolio_performance_stats(pnl_df[col])["sharpe_ratio"]
            for col in pnl_df.columns
        }
        sorted_cols = sorted(srs.keys(), key=lambda k: srs.get(k, -999.0), reverse=True)
        pnl_df = pnl_df[sorted_cols[:max_strategies]]

    corr_df = compute_strategy_correlation_matrix(pnl_df)
    cov_df = pnl_df.cov() * 252.0

    clustering = hierarchical_alpha_clustering(corr_df)
    order = clustering["quasi_diag_order"]

    if ensemble_method == "hrp":
        weights = hierarchical_risk_parity_weights(cov_df, order)
    elif ensemble_method == "inv_vol":
        vols = pnl_df.std().replace(0.0, np.nan).fillna(0.02)
        inv = 1.0 / vols
        weights = inv / inv.sum()
    elif ensemble_method == "equal_weight":
        weights = pd.Series(1.0 / len(pnl_df.columns), index=pnl_df.columns)
    else:
        raise ValueError(f"Unknown ensemble_method: {ensemble_method}")

    combined_pnl = (pnl_df * weights).sum(axis=1)
    perf = portfolio_performance_stats(combined_pnl)

    # Compute Diversification Ratio: weighted average volatility / ensemble volatility
    individual_ann_vols = pnl_df.std() * np.sqrt(252.0)
    weighted_vol = float(np.dot(weights, individual_ann_vols))
    ensemble_vol = float(perf["annualized_vol"])
    div_ratio = float(weighted_vol / ensemble_vol) if ensemble_vol > 0 else 1.0

    return {
        "weights": weights,
        "combined_pnl": combined_pnl,
        "performance": perf,
        "correlation_matrix": corr_df,
        "clustering": clustering,
        "diversification_ratio": div_ratio,
    }
