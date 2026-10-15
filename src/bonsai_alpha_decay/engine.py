from __future__ import annotations

from typing import Iterable
from scipy.stats import rankdata
import numpy as np
import pandas as pd

from .models import MarketPanel, StrategySpec


def _feature_frame(panel: MarketPanel, feature: str) -> pd.DataFrame:
    direct_cols = [f"{asset}__{feature}" for asset in panel.returns.columns]
    feature_cols = panel.features.columns
    if all(col in feature_cols for col in direct_cols):
        result = panel.features.loc[:, direct_cols].copy()
        result.columns = panel.returns.columns
        return result
    prefix = "__" + feature
    columns = [column for column in feature_cols if column.endswith(prefix)]
    if not columns:
        raise ValueError(f"feature not found: {feature}")
    assets = [column.split("__", 1)[0] for column in columns]
    result = panel.features.loc[:, columns].copy()
    result.columns = assets
    return result.reindex(columns=panel.returns.columns)


def build_signal(panel: MarketPanel, spec: StrategySpec) -> pd.DataFrame:
    panel.validate()
    spec.validate([column.split("__", 1)[1] for column in panel.features.columns])
    score = pd.DataFrame(0.0, index=panel.returns.index, columns=panel.returns.columns)
    for feature, weight in zip(spec.feature_columns, spec.weights):
        score = score.add(_feature_frame(panel, feature).mul(weight), fill_value=0.0)
    if spec.transform == "rank":
        score = score.rank(axis=1, pct=True)
    elif spec.transform == "zscore":
        mean = score.mean(axis=1)
        std = score.std(axis=1).replace(0.0, np.nan)
        score = score.sub(mean, axis=0).div(std, axis=0)
    elif spec.transform != "linear":
        raise ValueError(f"unsupported transform: {spec.transform}")
    if spec.universe_quantile is not None:
        ranks = score.rank(axis=1, pct=True)
        q = spec.universe_quantile
        score = score.where((ranks <= q) | (ranks >= 1.0 - q), 0.0)
    if panel.mask is not None:
        score = score.where(panel.mask, np.nan)
    score = score.replace([np.inf, -np.inf], np.nan)
    return score.shift(spec.signal_lag)


def forward_returns(
    returns: pd.DataFrame, horizons: Iterable[int]
) -> dict[int, pd.DataFrame]:
    if (
        not isinstance(returns.index, pd.DatetimeIndex)
        or not returns.index.is_monotonic_increasing
    ):
        raise ValueError("returns must have a sorted DatetimeIndex")
    output: dict[int, pd.DataFrame] = {}
    sorted_horizons = sorted(set(int(value) for value in horizons))
    if any(h < 1 for h in sorted_horizons):
        raise ValueError("horizons must be positive")
    if (returns > -1.0).all().all():
        log_ret = np.log1p(returns)
        for horizon in sorted_horizons:
            compounded = np.exp(log_ret.rolling(horizon, min_periods=horizon).sum()) - 1.0
            output[horizon] = compounded.shift(-horizon)
    else:
        cum = (1.0 + returns).cumprod()
        for horizon in sorted_horizons:
            compounded = cum.shift(-horizon) / cum - 1.0
            output[horizon] = compounded
    return output


def cross_sectional_ic(
    signal: pd.DataFrame,
    target: pd.DataFrame,
    method: str = "spearman",
    min_obs: int = 10,
) -> pd.Series:
    if not signal.index.equals(target.index) or not signal.columns.equals(
        target.columns
    ):
        raise ValueError("signal and target must have identical indexes and columns")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    s_mat = signal.to_numpy(dtype=float)
    t_mat = target.to_numpy(dtype=float)
    valid = np.isfinite(s_mat) & np.isfinite(t_mat)
    counts = valid.sum(axis=1)
    values = np.full(len(signal), np.nan, dtype=float)
    all_valid_mask = valid.all(axis=1)
    if all_valid_mask.any():
        idx_all = np.where(all_valid_mask)[0]
        s_sub = s_mat[idx_all]
        t_sub = t_mat[idx_all]
        if method == "spearman":
            r_s = rankdata(s_sub, axis=1)
            r_t = rankdata(t_sub, axis=1)
        else:
            r_s = s_sub
            r_t = t_sub
        r_s_c = r_s - r_s.mean(axis=1, keepdims=True)
        r_t_c = r_t - r_t.mean(axis=1, keepdims=True)
        dots = (r_s_c * r_t_c).sum(axis=1)
        n_s = np.linalg.norm(r_s_c, axis=1)
        n_t = np.linalg.norm(r_t_c, axis=1)
        denom = n_s * n_t
        valid_denom = denom > 0.0
        calculated = np.where(valid_denom, dots / np.where(valid_denom, denom, 1.0), np.nan)
        values[idx_all] = calculated
    partial_idx = np.where((~all_valid_mask) & (counts >= min_obs))[0]
    for i in partial_idx:
        v_idx = valid[i]
        s_row = s_mat[i, v_idx]
        t_row = t_mat[i, v_idx]
        if np.ptp(s_row) == 0.0 or np.ptp(t_row) == 0.0:
            continue
        if method == "spearman":
            r_s = rankdata(s_row)
            r_t = rankdata(t_row)
        else:
            r_s = s_row
            r_t = t_row
        r_s_centered = r_s - np.mean(r_s)
        r_t_centered = r_t - np.mean(r_t)
        norm_s = np.linalg.norm(r_s_centered)
        norm_t = np.linalg.norm(r_t_centered)
        if norm_s > 0.0 and norm_t > 0.0:
            values[i] = float(np.dot(r_s_centered, r_t_centered) / (norm_s * norm_t))
    return pd.Series(values, index=signal.index, name="ic")


def build_ic_table(
    signal: pd.DataFrame,
    forward: dict[int, pd.DataFrame],
    method: str = "spearman",
    min_obs: int = 10,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            horizon: cross_sectional_ic(signal, target, method, min_obs)
            for horizon, target in forward.items()
        }
    )


def calculate_turnover(signal: pd.DataFrame) -> pd.Series:
    abs_sum = signal.abs().sum(axis=1).replace(0.0, np.nan)
    norm_weights = signal.div(abs_sum, axis=0).fillna(0.0)
    delta = norm_weights.diff().abs().sum(axis=1) * 0.5
    return delta.iloc[1:]


def newey_west_ic_stats(ic_series: pd.Series, max_lag: int | None = None) -> dict[str, float]:
    clean = ic_series.dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < 2:
        return {
            "mean": np.nan,
            "std": np.nan,
            "se_hac": np.nan,
            "t_stat": np.nan,
            "hac_t_stat": np.nan,
            "ir": np.nan,
        }
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))
    ir = float(mean / std) if std > 0.0 else np.nan
    se_iid = float(std / np.sqrt(n)) if std > 0.0 else np.nan
    t_stat = float(mean / se_iid) if se_iid and se_iid > 0.0 else np.nan
    lag = max_lag if max_lag is not None else int(np.ceil(4.0 * ((n / 100.0) ** (2.0 / 9.0))))
    lag = min(lag, n - 1)
    demeaned = clean - mean
    gamma0 = float(np.dot(demeaned, demeaned) / n)
    v_hac = gamma0
    for j in range(1, lag + 1):
        weight = 1.0 - (j / (lag + 1.0))
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j]) / n)
        v_hac += 2.0 * weight * gamma_j
    v_hac = max(v_hac, 1e-16)
    se_hac = float(np.sqrt(v_hac / n))
    hac_t_stat = float(mean / se_hac) if se_hac > 0.0 else np.nan
    return {
        "mean": mean,
        "std": std,
        "se_hac": se_hac,
        "t_stat": t_stat,
        "hac_t_stat": hac_t_stat,
        "ir": ir,
    }
