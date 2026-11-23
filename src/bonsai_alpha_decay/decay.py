from __future__ import annotations

from typing import Any
import warnings
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd


def _fit_nlls(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    p0 = [float(y[0]) if len(y) > 0 else 0.01, 0.05]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                lambda h, a, b: a * np.exp(-b * h),
                x,
                y,
                p0=p0,
                maxfev=2000,
                bounds=([-np.inf, -10.0], [np.inf, 10.0]),
            )
        fitted = popt[0] * np.exp(-popt[1] * x)
        res = y - fitted
        tot = y - np.mean(y)
        r2 = float(1.0 - np.dot(res, res) / np.dot(tot, tot)) if np.dot(tot, tot) > 0 else np.nan
        return float(popt[0]), float(popt[1]), r2
    except Exception:
        return np.nan, np.nan, np.nan


def estimate_exponential_decay(
    ic_by_horizon: pd.Series | dict[float, float],
    method: str = "auto",
) -> dict[str, Any]:
    series = pd.Series(ic_by_horizon, dtype=float).sort_index()
    clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
    valid_positive = clean_series[clean_series > 0]
    if method == "nlls" and len(clean_series) >= 2 and clean_series.index.nunique() >= 2:
        x_all = clean_series.index.to_numpy(dtype=float)
        y_all = clean_series.to_numpy(dtype=float)
        ic0, decay_rate, r2 = _fit_nlls(x_all, y_all)
        status = "ok" if (np.isfinite(decay_rate) and decay_rate > 0) else "non_decay_or_growth"
        return {
            "ic0": ic0,
            "lambda": decay_rate,
            "half_life": float(np.log(2.0) / decay_rate) if (np.isfinite(decay_rate) and decay_rate > 0) else np.inf,
            "r2": r2,
            "r2_ic": r2,
            "n_points": int(len(clean_series)),
            "status": status,
        }
    if len(valid_positive) < 2 or valid_positive.index.nunique() < 2:
        if method == "auto" and len(clean_series) >= 2 and clean_series.index.nunique() >= 2:
            x_all = clean_series.index.to_numpy(dtype=float)
            y_all = clean_series.to_numpy(dtype=float)
            ic0, decay_rate, r2 = _fit_nlls(x_all, y_all)
            if np.isfinite(decay_rate):
                status = "ok" if decay_rate > 0 else "non_decay_or_growth"
                return {
                    "ic0": ic0,
                    "lambda": decay_rate,
                    "half_life": float(np.log(2.0) / decay_rate) if decay_rate > 0 else np.inf,
                    "r2": r2,
                    "r2_ic": r2,
                    "n_points": int(len(clean_series)),
                    "status": status,
                }
        return {
            "ic0": np.nan,
            "lambda": np.nan,
            "half_life": np.nan,
            "r2": np.nan,
            "r2_ic": np.nan,
            "n_points": int(len(valid_positive)),
            "status": "insufficient_positive_points",
        }
    x = valid_positive.index.to_numpy(dtype=float)
    y = np.log(valid_positive.to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ np.array([intercept, slope])
    residual = y - fitted
    total = y - y.mean()
    r2 = (
        float(1.0 - np.dot(residual, residual) / np.dot(total, total))
        if np.dot(total, total) > 0
        else np.nan
    )
    decay_rate = float(-slope)
    ic0 = float(np.exp(intercept))
    ic_fitted = ic0 * np.exp(-decay_rate * x)
    ic_actual = valid_positive.to_numpy(dtype=float)
    ic_res = ic_actual - ic_fitted
    ic_tot = ic_actual - np.mean(ic_actual)
    r2_ic = (
        float(1.0 - np.dot(ic_res, ic_res) / np.dot(ic_tot, ic_tot))
        if np.dot(ic_tot, ic_tot) > 0
        else np.nan
    )
    return {
        "ic0": ic0,
        "lambda": decay_rate,
        "half_life": float(np.log(2.0) / decay_rate) if decay_rate > 0 else np.inf,
        "r2": r2,
        "r2_ic": r2_ic,
        "n_points": int(len(valid_positive)),
        "status": "ok" if decay_rate > 0 else "non_decay_or_growth",
    }


def estimate_power_law_decay(
    ic_by_horizon: pd.Series | dict[float, float],
) -> dict[str, Any]:
    series = pd.Series(ic_by_horizon, dtype=float).sort_index()
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid > 0) & (valid.index > 0)]
    if len(valid) < 2 or valid.index.nunique() < 2:
        return {
            "ic0": np.nan,
            "alpha": np.nan,
            "r2": np.nan,
            "status": "insufficient_points",
        }
    x = np.log(valid.index.to_numpy(dtype=float))
    y = np.log(valid.to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ np.array([intercept, slope])
    res = y - fitted
    tot = y - y.mean()
    r2 = float(1.0 - np.dot(res, res) / np.dot(tot, tot)) if np.dot(tot, tot) > 0 else np.nan
    alpha = float(-slope)
    return {
        "ic0": float(np.exp(intercept)),
        "alpha": alpha,
        "r2": r2,
        "status": "ok" if alpha > 0 else "non_decay_or_growth",
    }


def estimate_hump_decay(
    ic_by_horizon: pd.Series | dict[float, float],
) -> dict[str, Any]:
    series = pd.Series(ic_by_horizon, dtype=float).sort_index()
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid > 0) & (valid.index > 0)]
    if len(valid) < 3 or valid.index.nunique() < 3:
        return {
            "scale": np.nan,
            "beta": np.nan,
            "lambda": np.nan,
            "peak_horizon": np.nan,
            "r2": np.nan,
            "status": "insufficient_points",
        }
    h = valid.index.to_numpy(dtype=float)
    y = np.log(valid.to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(h)), np.log(h), h])
    coeffs = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coeffs
    res = y - fitted
    tot = y - y.mean()
    r2 = float(1.0 - np.dot(res, res) / np.dot(tot, tot)) if np.dot(tot, tot) > 0 else np.nan
    beta = float(coeffs[1])
    decay_rate = float(-coeffs[2])
    peak_h = float(beta / decay_rate) if (decay_rate > 0 and beta > 0) else 1.0
    return {
        "scale": float(np.exp(coeffs[0])),
        "beta": beta,
        "lambda": decay_rate,
        "peak_horizon": peak_h,
        "r2": r2,
        "status": "ok" if (decay_rate > 0) else "non_decay_or_growth",
    }


def estimate_bi_exponential_decay(
    ic_by_horizon: pd.Series | dict[float, float],
) -> dict[str, Any]:
    series = pd.Series(ic_by_horizon, dtype=float).sort_index()
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid > 0) & (valid.index > 0)]
    if len(valid) < 4 or valid.index.nunique() < 4:
        return {
            "a_fast": np.nan,
            "lambda_fast": np.nan,
            "half_life_fast": np.nan,
            "a_slow": np.nan,
            "lambda_slow": np.nan,
            "half_life_slow": np.nan,
            "fast_ratio": np.nan,
            "r2": np.nan,
            "status": "insufficient_points",
        }
    h = valid.index.to_numpy(dtype=float)
    y = valid.to_numpy(dtype=float)
    y0 = float(y[0])
    p0 = [y0 * 0.5, 0.5, y0 * 0.5, 0.05]
    bounds = (
        [0.0, 0.0001, 0.0, 0.0001],
        [np.inf, 10.0, np.inf, 10.0],
    )

    def bi_exp(x: np.ndarray, a_f: float, l_f: float, a_s: float, l_s: float) -> np.ndarray:
        return a_f * np.exp(-l_f * x) + a_s * np.exp(-l_s * x)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(bi_exp, h, y, p0=p0, bounds=bounds, maxfev=4000)
        a_f, l_f, a_s, l_s = popt
        if l_f < l_s:
            a_f, a_s = a_s, a_f
            l_f, l_s = l_s, l_f
        fitted = bi_exp(h, a_f, l_f, a_s, l_s)
        res = y - fitted
        tot = y - np.mean(y)
        r2 = float(1.0 - np.dot(res, res) / np.dot(tot, tot)) if np.dot(tot, tot) > 0 else np.nan
        hl_f = float(np.log(2.0) / l_f) if l_f > 0 else np.inf
        hl_s = float(np.log(2.0) / l_s) if l_s > 0 else np.inf
        fast_ratio = float(a_f / (a_f + a_s)) if (a_f + a_s) > 0 else 0.5
        return {
            "a_fast": float(a_f),
            "lambda_fast": float(l_f),
            "half_life_fast": hl_f,
            "a_slow": float(a_s),
            "lambda_slow": float(l_s),
            "half_life_slow": hl_s,
            "fast_ratio": fast_ratio,
            "r2": r2,
            "status": "ok" if (l_f > 0 and l_s > 0) else "non_decay_or_growth",
        }
    except Exception:
        return {
            "a_fast": np.nan,
            "lambda_fast": np.nan,
            "half_life_fast": np.nan,
            "a_slow": np.nan,
            "lambda_slow": np.nan,
            "half_life_slow": np.nan,
            "fast_ratio": np.nan,
            "r2": np.nan,
            "status": "fit_failed",
        }


def estimate_decay_family(
    ic_by_horizon: pd.Series | dict[float, float],
) -> dict[str, Any]:
    series = pd.Series(ic_by_horizon, dtype=float).sort_index()
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid > 0) & (valid.index > 0)]
    n = len(valid)
    if n < 3:
        exp_res = estimate_exponential_decay(series)
        return {
            "best_model": "exponential",
            "models": {"exponential": exp_res},
            "aic": {},
        }
    h = valid.index.to_numpy(dtype=float)
    y = valid.to_numpy(dtype=float)
    exp_fit = estimate_exponential_decay(valid)
    pow_fit = estimate_power_law_decay(valid)
    hump_fit = estimate_hump_decay(valid)
    bi_exp_fit = estimate_bi_exponential_decay(valid) if n >= 4 else None

    def _aic(fitted_y: np.ndarray, k: int) -> float:
        res = y - fitted_y
        rss = float(np.dot(res, res))
        if rss <= 1e-12:
            return -np.inf
        return float(n * np.log(rss / n) + 2 * k)

    exp_pred = exp_fit["ic0"] * np.exp(-exp_fit["lambda"] * h) if np.isfinite(exp_fit["ic0"]) else np.full(n, np.nan)
    pow_pred = pow_fit["ic0"] * (h ** -pow_fit["alpha"]) if np.isfinite(pow_fit["ic0"]) else np.full(n, np.nan)
    hump_pred = hump_fit["scale"] * (h ** hump_fit["beta"]) * np.exp(-hump_fit["lambda"] * h) if np.isfinite(hump_fit["scale"]) else np.full(n, np.nan)
    bi_exp_pred = (
        bi_exp_fit["a_fast"] * np.exp(-bi_exp_fit["lambda_fast"] * h)
        + bi_exp_fit["a_slow"] * np.exp(-bi_exp_fit["lambda_slow"] * h)
        if (bi_exp_fit and np.isfinite(bi_exp_fit["a_fast"]))
        else np.full(n, np.nan)
    )

    aic_dict = {}
    if np.isfinite(exp_pred).all():
        aic_dict["exponential"] = _aic(exp_pred, 2)
    if np.isfinite(pow_pred).all():
        aic_dict["power_law"] = _aic(pow_pred, 2)
    if np.isfinite(hump_pred).all():
        aic_dict["hump"] = _aic(hump_pred, 3)
    if np.isfinite(bi_exp_pred).all():
        aic_dict["bi_exponential"] = _aic(bi_exp_pred, 4)

    models_dict = {
        "exponential": exp_fit,
        "power_law": pow_fit,
        "hump": hump_fit,
    }
    if bi_exp_fit is not None:
        models_dict["bi_exponential"] = bi_exp_fit

    best_model = min(aic_dict, key=aic_dict.get) if aic_dict else "exponential"
    return {
        "best_model": best_model,
        "models": models_dict,
        "aic": aic_dict,
    }


def bootstrap_decay(
    ic_by_horizon: pd.Series, samples: int = 250, seed: int = 0
) -> pd.DataFrame:
    if samples < 1:
        raise ValueError("samples must be positive")
    series = pd.Series(ic_by_horizon, dtype=float).sort_index()
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(samples):
        noise = rng.normal(0.0, max(float(series.std(ddof=1)) * 0.1, 1e-8), len(series))
        draw = pd.Series(series.to_numpy() + noise, index=series.index)
        rows.append(estimate_exponential_decay(draw))
    return pd.DataFrame(rows)
