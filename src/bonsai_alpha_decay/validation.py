from __future__ import annotations

from typing import Sequence
from scipy.stats import norm
import numpy as np


def deflated_sharpe_ratio(
    strategy_sharpe: float,
    trial_sharpes: Sequence[float] | np.ndarray,
    n_periods: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[np.isfinite(trials)]
    k = len(trials)
    if k < 2 or n_periods < 3 or not np.isfinite(strategy_sharpe):
        return np.nan
    var_trials = float(np.var(trials, ddof=1))
    std_trials = float(np.sqrt(var_trials))
    if std_trials <= 0.0:
        return np.nan
    euler = 0.57721566490153286
    z1 = norm.ppf(1.0 - (1.0 / k))
    z2 = norm.ppf(1.0 - (1.0 / (k * np.e)))
    expected_max_sr = std_trials * ((1.0 - euler) * z1 + euler * z2)
    denom = 1.0 - skewness * strategy_sharpe + ((kurtosis - 1.0) / 4.0) * (strategy_sharpe**2)
    denom = max(denom, 1e-12)
    se_sr = np.sqrt(denom / (n_periods - 1.0))
    if se_sr <= 0.0:
        return np.nan
    z_stat = (strategy_sharpe - expected_max_sr) / se_sr
    return float(norm.cdf(z_stat))


def haircut_sharpe_ratio(
    strategy_sharpe: float,
    trial_count: int,
    n_periods: int,
) -> float:
    if trial_count < 2 or n_periods < 3 or not np.isfinite(strategy_sharpe):
        return np.nan
    penalty = np.log(float(trial_count)) / (2.0 * max(abs(strategy_sharpe), 0.1) * np.sqrt(float(n_periods)))
    haircut = max(0.0, 1.0 - penalty)
    return float(strategy_sharpe * haircut)


def benjamini_hochberg_fdr(
    p_values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    if n == 0:
        return np.array([], dtype=bool)
    valid_mask = np.isfinite(p_arr)
    valid_indices = np.where(valid_mask)[0]
    valid_p = p_arr[valid_mask]
    m = len(valid_p)
    if m == 0:
        return np.zeros(n, dtype=bool)
    sort_order = np.argsort(valid_p)
    sorted_p = valid_p[sort_order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    significant_sorted = sorted_p <= thresholds
    if not np.any(significant_sorted):
        return np.zeros(n, dtype=bool)
    max_idx = np.max(np.where(significant_sorted)[0])
    selected_indices = sort_order[: max_idx + 1]
    result = np.zeros(n, dtype=bool)
    result[valid_indices[selected_indices]] = True
    return result
