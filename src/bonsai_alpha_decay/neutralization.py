from __future__ import annotations

from typing import Mapping
import numpy as np
import pandas as pd


def neutralize_signal(
    signal: pd.DataFrame,
    factors: Mapping[str, pd.DataFrame] | pd.DataFrame,
    weights: pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame:
    if isinstance(factors, pd.DataFrame):
        factors = {"factor_0": factors}
    if not len(factors):
        return signal.copy()
    for name, factor_df in factors.items():
        if not signal.index.equals(factor_df.index) or not signal.columns.equals(factor_df.columns):
            raise ValueError(f"factor {name} does not match signal index or columns")
    factor_names = list(factors.keys())
    sig_mat = signal.to_numpy(dtype=float)
    fac_mats = [factors[name].to_numpy(dtype=float) for name in factor_names]
    residual_mat = np.full_like(sig_mat, np.nan)
    n_dates, n_assets = sig_mat.shape
    k_features = len(factor_names) + 1

    if weights is not None:
        if isinstance(weights, pd.DataFrame):
            w_mat = weights.reindex(index=signal.index, columns=signal.columns).to_numpy(dtype=float)
        elif isinstance(weights, pd.Series):
            w_arr = weights.reindex(signal.columns).to_numpy(dtype=float)
            w_mat = np.broadcast_to(w_arr, (n_dates, n_assets)).copy()
        else:
            w_mat = np.broadcast_to(np.asarray(weights, dtype=float), (n_dates, n_assets)).copy()
        w_mat = np.where(np.isfinite(w_mat) & (w_mat > 0), w_mat, 1.0)
    else:
        w_mat = np.ones((n_dates, n_assets), dtype=float)

    sqrt_w = np.sqrt(w_mat)
    valid_mask = np.isfinite(sig_mat) & np.isfinite(w_mat)
    for fm in fac_mats:
        valid_mask = valid_mask & np.isfinite(fm)

    all_valid_dates = valid_mask.all(axis=1)
    if all_valid_dates.any():
        idx_all = np.where(all_valid_dates)[0]
        t_all = len(idx_all)
        x_all = np.ones((t_all, n_assets, k_features), dtype=float)
        for k_idx, fm in enumerate(fac_mats):
            x_all[:, :, k_idx + 1] = fm[idx_all]

        y_all = sig_mat[idx_all]
        sw_all = sqrt_w[idx_all]

        x_w = x_all * sw_all[:, :, None]
        y_w = y_all * sw_all

        xtx = np.einsum("tni,tnj->tij", x_w, x_w)
        reg = 1e-12 * np.eye(k_features)[None, :, :]
        xtx_reg = xtx + reg
        xty = np.einsum("tni,tn->ti", x_w, y_w)

        try:
            betas = np.linalg.solve(xtx_reg, xty)
            fitted = np.einsum("tni,ti->tn", x_all, betas)
            residual_mat[idx_all] = y_all - fitted
        except Exception:
            all_valid_dates[idx_all] = False

    rem_dates = np.where(~all_valid_dates)[0]
    for t in rem_dates:
        valid = valid_mask[t]
        n_valid = int(np.sum(valid))
        if n_valid < k_features + 1:
            residual_mat[t, valid] = sig_mat[t, valid]
            continue
        y_clean = sig_mat[t, valid]
        sw_clean = sqrt_w[t, valid]
        stacked = [np.ones(n_valid)] + [fac[t, valid] for fac in fac_mats]
        x_single = np.column_stack(stacked)
        x_w_single = x_single * sw_clean[:, None]
        y_w_single = y_clean * sw_clean
        try:
            beta = np.linalg.lstsq(x_w_single, y_w_single, rcond=None)[0]
            fitted_single = x_single @ beta
            residual_mat[t, valid] = y_clean - fitted_single
        except Exception:
            residual_mat[t, valid] = y_clean

    return pd.DataFrame(residual_mat, index=signal.index, columns=signal.columns)
