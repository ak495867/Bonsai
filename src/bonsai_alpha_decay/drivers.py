from __future__ import annotations

from typing import Any
from scipy.stats import rankdata
import numpy as np
import pandas as pd

from .models import MarketPanel

DRIVER_NAMES = (
    "crowding",
    "volatility",
    "turnover",
    "liquidity",
    "market_cap",
    "regime_changes",
)


def _safe_corr(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    l_arr = left.to_numpy(dtype=float) if isinstance(left, pd.Series) else np.asarray(left, dtype=float)
    r_arr = right.to_numpy(dtype=float) if isinstance(right, pd.Series) else np.asarray(right, dtype=float)
    valid = np.isfinite(l_arr) & np.isfinite(r_arr)
    if np.sum(valid) < 3:
        return np.nan
    l_sub = l_arr[valid]
    r_sub = r_arr[valid]
    if np.ptp(l_sub) == 0.0 or np.ptp(r_sub) == 0.0:
        return np.nan
    r_l = rankdata(l_sub)
    r_r = rankdata(r_sub)
    l_c = r_l - np.mean(r_l)
    r_c = r_r - np.mean(r_r)
    n_l = np.linalg.norm(l_c)
    n_r = np.linalg.norm(r_c)
    if n_l > 0.0 and n_r > 0.0:
        return float(np.dot(l_c, r_c) / (n_l * n_r))
    return np.nan


def _regime_change_series(panel: MarketPanel) -> pd.Series:
    market_return = panel.returns.mean(axis=1)
    market_vol = market_return.rolling(20, min_periods=10).std()
    return market_vol.pct_change().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def strategy_driver_exposure(
    panel: MarketPanel, signal: pd.DataFrame
) -> dict[str, float]:
    metadata = panel.metadata.reindex(signal.columns)
    sig_mat = np.abs(signal.to_numpy(dtype=float))
    exposures = {}
    valid_sig = np.isfinite(sig_mat)
    full_row_mask = valid_sig.all(axis=1)
    has_full_rows = bool(full_row_mask.any())

    if has_full_rows:
        idx_full = np.where(full_row_mask)[0]
        sub_sig = sig_mat[idx_full]
        ranked_sig = rankdata(sub_sig, axis=1)
        sig_c = ranked_sig - ranked_sig.mean(axis=1, keepdims=True)
        sig_norm = np.linalg.norm(sig_c, axis=1)
        rem_idx = np.where(~full_row_mask)[0]
    else:
        idx_full = np.array([], dtype=int)
        sub_sig = None
        sig_c = None
        sig_norm = None
        rem_idx = np.arange(len(sig_mat))

    for name in DRIVER_NAMES[:-1]:
        raw_driver = metadata[name].to_numpy(dtype=float)
        if name in {"liquidity", "market_cap"}:
            raw_driver = np.log1p(raw_driver)
        valid_d = np.isfinite(raw_driver)
        if np.sum(valid_d) < 3 or np.ptp(raw_driver[valid_d]) == 0.0:
            exposures[name] = np.nan
            continue
        vals = []
        if valid_d.all() and has_full_rows:
            r_drv = rankdata(raw_driver)
            drv_c = r_drv - np.mean(r_drv)
            drv_norm = np.linalg.norm(drv_c)
            if drv_norm > 0.0:
                dots = sig_c @ drv_c
                denom = sig_norm * drv_norm
                valid_m = (denom > 0.0) & np.isfinite(dots)
                corrs = np.abs(dots[valid_m] / denom[valid_m])
                vals.extend(corrs.tolist())
            scan_idx = rem_idx
        else:
            scan_idx = np.arange(len(sig_mat))

        for i in scan_idx:
            row = sig_mat[i]
            valid = np.isfinite(row) & valid_d
            if np.sum(valid) < 3:
                continue
            r_sub = row[valid]
            d_sub = raw_driver[valid]
            if np.ptp(r_sub) == 0.0 or np.ptp(d_sub) == 0.0:
                continue
            r_row = rankdata(r_sub)
            r_drv = rankdata(d_sub)
            r_row_c = r_row - np.mean(r_row)
            r_drv_c = r_drv - np.mean(r_drv)
            n_row = np.linalg.norm(r_row_c)
            n_drv = np.linalg.norm(r_drv_c)
            if n_row > 0.0 and n_drv > 0.0:
                vals.append(abs(float(np.dot(r_row_c, r_drv_c) / (n_row * n_drv))))
        exposures[name] = float(np.mean(vals)) if vals else np.nan

    regime_changes = _regime_change_series(panel)
    signal_dispersion = signal.rank(axis=1, pct=True).std(axis=1)
    exposures["regime_changes"] = _safe_corr(
        signal_dispersion, regime_changes.reindex(signal.index).fillna(0.0)
    )
    return exposures


def fit_driver_model(
    records: pd.DataFrame,
    target: str = "lambda",
    alpha: float = 0.0,
) -> dict[str, Any]:
    required = set(DRIVER_NAMES).union({target})
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"records are missing columns: {sorted(missing)}")
    clean = (
        records[list(DRIVER_NAMES) + [target]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(clean) < len(DRIVER_NAMES) + 2:
        return {
            "target": target,
            "n_observations": int(len(clean)),
            "status": "insufficient_observations",
            "intercept": np.nan,
            "coefficients": {},
            "r2": np.nan,
            "vif": {},
            "condition_number": np.nan,
        }
    x = clean[list(DRIVER_NAMES)].to_numpy(dtype=float)
    y = clean[target].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    if alpha > 0.0:
        reg = alpha * np.eye(design.shape[1])
        reg[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + reg, design.T @ y)
    else:
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coefficients
    residual = y - fitted
    total = y - y.mean()
    r2 = (
        float(1.0 - np.dot(residual, residual) / np.dot(total, total))
        if np.dot(total, total) > 0
        else np.nan
    )
    try:
        corr_matrix = np.corrcoef(standardized, rowvar=False)
        inv_corr = np.linalg.pinv(corr_matrix)
        vifs = {
            name: float(inv_corr[i, i])
            for i, name in enumerate(DRIVER_NAMES)
        }
        cond_num = float(np.linalg.cond(design))
    except Exception:
        vifs = {}
        cond_num = np.nan
    return {
        "target": target,
        "n_observations": int(len(clean)),
        "status": "ok",
        "intercept": float(coefficients[0]),
        "coefficients": {
            name: float(value) for name, value in zip(DRIVER_NAMES, coefficients[1:])
        },
        "r2": r2,
        "vif": vifs,
        "condition_number": cond_num,
    }
