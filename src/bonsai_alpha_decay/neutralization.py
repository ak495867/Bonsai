from __future__ import annotations

from typing import Mapping
import numpy as np
import pandas as pd


def neutralize_signal(
    signal: pd.DataFrame,
    factors: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    if not factors:
        return signal.copy()
    for name, factor_df in factors.items():
        if not signal.index.equals(factor_df.index) or not signal.columns.equals(factor_df.columns):
            raise ValueError(f"factor {name} does not match signal index or columns")
    factor_names = list(factors.keys())
    sig_mat = signal.to_numpy(dtype=float)
    fac_mats = [factors[name].to_numpy(dtype=float) for name in factor_names]
    residual_mat = np.full_like(sig_mat, np.nan)
    n_dates, n_assets = sig_mat.shape

    for t in range(n_dates):
        y = sig_mat[t]
        x_cols = [fac[t] for fac in fac_mats]
        valid = np.isfinite(y)
        for col in x_cols:
            valid = valid & np.isfinite(col)
        n_valid = int(np.sum(valid))
        if n_valid < len(factor_names) + 2:
            residual_mat[t, valid] = y[valid]
            continue
        y_clean = y[valid]
        stacked = [np.ones(n_valid)] + [col[valid] for col in x_cols]
        X = np.column_stack(stacked)
        try:
            beta = np.linalg.lstsq(X, y_clean, rcond=None)[0]
            fitted = X @ beta
            residual_mat[t, valid] = y_clean - fitted
        except Exception:
            residual_mat[t, valid] = y_clean

    return pd.DataFrame(residual_mat, index=signal.index, columns=signal.columns)
