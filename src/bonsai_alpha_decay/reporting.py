from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd


def _svg_decay_comparison(
    horizons: Sequence[float],
    ic_values: Sequence[float],
    decay_info: Mapping[str, Any],
    decay_family: Mapping[str, Any] | None = None,
) -> str:
    h_arr = np.asarray(horizons, dtype=float)
    y_arr = np.asarray(ic_values, dtype=float)
    if len(h_arr) == 0:
        return "<p>No decay curve data</p>"
    width, height = 500, 260
    pad_l, pad_r, pad_t, pad_b = 60, 40, 30, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    max_h = max(float(np.max(h_arr)), 1.0)
    min_y = min(float(np.min(y_arr)), 0.0)
    max_y = max(float(np.max(y_arr)), 0.05) * 1.15
    y_range = max(max_y - min_y, 1e-6)

    def tx(h: float) -> float:
        return pad_l + (h / max_h) * plot_w

    def ty(y: float) -> float:
        return pad_t + plot_h - ((y - min_y) / y_range) * plot_h

    dense_h = np.linspace(0.5, max_h, 60)

    # 1. Exponential fit line
    ic0 = float(decay_info.get("ic0", np.nan))
    decay_rate = float(decay_info.get("lambda", np.nan))
    exp_pts = []
    if np.isfinite(ic0) and np.isfinite(decay_rate):
        for h in dense_h:
            y_fit = ic0 * np.exp(-decay_rate * h)
            exp_pts.append(f"{tx(h):.1f},{ty(y_fit):.1f}")
    exp_line = f'<polyline fill="none" stroke="#dc2626" stroke-width="2" points="{" ".join(exp_pts)}" />' if exp_pts else ""

    # 2. Power-Law fit line
    models = decay_family.get("models", {}) if decay_family else {}
    pow_info = models.get("power_law", {})
    pow_pts = []
    if np.isfinite(pow_info.get("ic0", np.nan)) and np.isfinite(pow_info.get("alpha", np.nan)):
        p_ic0 = float(pow_info["ic0"])
        p_alpha = float(pow_info["alpha"])
        for h in dense_h:
            y_pow = p_ic0 * (h ** -p_alpha)
            pow_pts.append(f"{tx(h):.1f},{ty(y_pow):.1f}")
    pow_line = f'<polyline fill="none" stroke="#9333ea" stroke-width="2" stroke-dasharray="4,3" points="{" ".join(pow_pts)}" />' if pow_pts else ""

    # 3. Bi-exponential or Hump line
    bi_info = models.get("bi_exponential", {})
    bi_pts = []
    if np.isfinite(bi_info.get("a_fast", np.nan)):
        af, lf, as_, ls = bi_info["a_fast"], bi_info["lambda_fast"], bi_info["a_slow"], bi_info["lambda_slow"]
        for h in dense_h:
            y_bi = af * np.exp(-lf * h) + as_ * np.exp(-ls * h)
            bi_pts.append(f"{tx(h):.1f},{ty(y_bi):.1f}")
    bi_line = f'<polyline fill="none" stroke="#059669" stroke-width="2" stroke-dasharray="2,2" points="{" ".join(bi_pts)}" />' if bi_pts else ""

    # Points for empirical IC
    points_svg = []
    for h, val in zip(h_arr, y_arr):
        if np.isfinite(val):
            points_svg.append(f'<circle cx="{tx(h):.1f}" cy="{ty(val):.1f}" r="4" fill="#2563eb" />')

    zero_y = ty(0.0)
    return f"""
    <svg width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0;">
        <line x1="{pad_l}" y1="{zero_y:.1f}" x2="{width - pad_r}" y2="{zero_y:.1f}" stroke="#94a3b8" stroke-dasharray="3,3" />
        {exp_line}
        {pow_line}
        {bi_line}
        {"".join(points_svg)}
        <text x="{pad_l}" y="{height - 10}" fill="#64748b" font-size="11">Horizon (days)</text>
        <text x="15" y="{pad_t + 10}" fill="#64748b" font-size="11" transform="rotate(-90 15,{pad_t + 10})">Mean IC</text>
    </svg>
    """


def _svg_cumulative_pnl(cum_gross: Sequence[float], cum_net: Sequence[float]) -> str:
    g_arr = np.asarray(cum_gross, dtype=float)
    n_arr = np.asarray(cum_net, dtype=float)
    if len(g_arr) == 0:
        return "<p>No PnL data</p>"
    width, height = 500, 260
    pad_l, pad_r, pad_t, pad_b = 60, 40, 30, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_vals = np.concatenate([g_arr, n_arr, [0.0]])
    min_y = float(np.min(all_vals))
    max_y = float(np.max(all_vals))
    y_range = max(max_y - min_y, 1e-6)
    n_points = len(g_arr)

    def tx(i: int) -> float:
        return pad_l + (i / max(n_points - 1, 1)) * plot_w

    def ty(y: float) -> float:
        return pad_t + plot_h - ((y - min_y) / y_range) * plot_h

    gross_pts = " ".join([f"{tx(i):.1f},{ty(g_arr[i]):.1f}" for i in range(n_points)])
    net_pts = " ".join([f"{tx(i):.1f},{ty(n_arr[i]):.1f}" for i in range(n_points)])
    zero_y = ty(0.0)

    return f"""
    <svg width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0;">
        <line x1="{pad_l}" y1="{zero_y:.1f}" x2="{width - pad_r}" y2="{zero_y:.1f}" stroke="#94a3b8" stroke-dasharray="3,3" />
        <polyline fill="none" stroke="#16a34a" stroke-width="2" points="{gross_pts}" />
        <polyline fill="none" stroke="#2563eb" stroke-width="2" points="{net_pts}" />
        <text x="{pad_l}" y="{height - 10}" fill="#64748b" font-size="11">Time (Steps)</text>
        <text x="15" y="{pad_t + 10}" fill="#64748b" font-size="11" transform="rotate(-90 15,{pad_t + 10})">Cumulative Return</text>
    </svg>
    """


def _svg_drawdown(cum_net: Sequence[float]) -> str:
    n_arr = np.asarray(cum_net, dtype=float)
    if len(n_arr) == 0:
        return "<p>No Drawdown data</p>"
    wealth = 1.0 + n_arr
    peaks = np.maximum.accumulate(wealth)
    dd = (wealth - peaks) / np.maximum(peaks, 1e-8)

    width, height = 500, 260
    pad_l, pad_r, pad_t, pad_b = 60, 40, 30, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    min_dd = min(float(np.min(dd)), -0.05)
    max_dd = 0.0
    y_range = max(max_dd - min_dd, 1e-4)
    n_points = len(dd)

    def tx(i: int) -> float:
        return pad_l + (i / max(n_points - 1, 1)) * plot_w

    def ty(y: float) -> float:
        return pad_t + ((max_dd - y) / y_range) * plot_h

    zero_y = ty(0.0)
    pts = [f"{tx(i):.1f},{ty(dd[i]):.1f}" for i in range(n_points)]
    poly_pts = [f"{pad_l:.1f},{zero_y:.1f}"] + pts + [f"{tx(n_points - 1):.1f},{zero_y:.1f}"]

    return f"""
    <svg width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0;">
        <line x1="{pad_l}" y1="{zero_y:.1f}" x2="{width - pad_r}" y2="{zero_y:.1f}" stroke="#94a3b8" />
        <polygon fill="rgba(239, 68, 68, 0.2)" points="{" ".join(poly_pts)}" />
        <polyline fill="none" stroke="#dc2626" stroke-width="2" points="{" ".join(pts)}" />
        <text x="{pad_l}" y="{height - 10}" fill="#64748b" font-size="11">Time (Steps)</text>
        <text x="15" y="{pad_t + 10}" fill="#64748b" font-size="11" transform="rotate(-90 15,{pad_t + 10})">Drawdown (%)</text>
    </svg>
    """


def _svg_capacity_curve(capacity_curve: list[dict[str, Any]]) -> str:
    if not capacity_curve:
        return "<p>No capacity curve data</p>"
    width, height = 500, 260
    pad_l, pad_r, pad_t, pad_b = 60, 40, 30, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    aums = [float(item["aum"]) for item in capacity_curve]
    sharpes = [float(item["net_sharpe"]) for item in capacity_curve if np.isfinite(item["net_sharpe"])]
    if not sharpes:
        return "<p>No valid Sharpe points in capacity curve</p>"

    min_sr = min(min(sharpes), 0.0)
    max_sr = max(max(sharpes), 2.0) * 1.15
    sr_range = max(max_sr - min_sr, 1e-4)

    min_log_aum = np.log10(max(min(aums), 1e5))
    max_log_aum = np.log10(max(max(aums), 1e6))
    aum_range = max(max_log_aum - min_log_aum, 1.0)

    def tx(aum: float) -> float:
        log_a = np.log10(max(aum, 1e5))
        return pad_l + ((log_a - min_log_aum) / aum_range) * plot_w

    def ty(sr: float) -> float:
        return pad_t + plot_h - ((sr - min_sr) / sr_range) * plot_h

    sr_1_y = ty(1.0)
    pts = []
    points_svg = []
    for item in capacity_curve:
        a = float(item["aum"])
        sr = float(item["net_sharpe"])
        if np.isfinite(sr):
            cx, cy = tx(a), ty(sr)
            pts.append(f"{cx:.1f},{cy:.1f}")
            points_svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#0284c7" />')

    return f"""
    <svg width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0;">
        <line x1="{pad_l}" y1="{sr_1_y:.1f}" x2="{width - pad_r}" y2="{sr_1_y:.1f}" stroke="#f59e0b" stroke-dasharray="4,4" />
        <polyline fill="none" stroke="#0284c7" stroke-width="2" points="{" ".join(pts)}" />
        {"".join(points_svg)}
        <text x="{pad_l}" y="{height - 10}" fill="#64748b" font-size="11">AUM ($ log scale)</text>
        <text x="15" y="{pad_t + 10}" fill="#64748b" font-size="11" transform="rotate(-90 15,{pad_t + 10})">Net Sharpe</text>
        <text x="{width - pad_r - 90}" y="{sr_1_y - 5:.1f}" fill="#f59e0b" font-size="10">Target Sharpe = 1.0</text>
    </svg>
    """


def _svg_factor_exposure_bar(drivers: Mapping[str, float]) -> str:
    clean_drivers = {k: float(v) for k, v in drivers.items() if isinstance(v, (int, float)) and np.isfinite(v)}
    if not clean_drivers:
        return "<p>No driver data</p>"
    width, height = 500, 260
    pad_l, pad_r, pad_t, pad_b = 90, 40, 30, 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    names = list(clean_drivers.keys())
    n_bars = len(names)
    bar_h = min(20.0, (plot_h / max(n_bars, 1)) - 4.0)

    center_x = pad_l + plot_w * 0.5
    scale_w = (plot_w * 0.5) / 1.0

    bars_svg = []
    for i, name in enumerate(names):
        val = clean_drivers[name]
        y = pad_t + i * (plot_h / n_bars) + 4.0
        bw = abs(val) * scale_w
        bx = center_x if val >= 0 else (center_x - bw)
        color = "#2563eb" if val >= 0 else "#dc2626"
        bars_svg.append(f'<text x="{pad_l - 8}" y="{y + bar_h * 0.8:.1f}" text-anchor="end" fill="#334155" font-size="10">{name}</text>')
        bars_svg.append(f'<rect x="{bx:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3" />')
        bars_svg.append(f'<text x="{bx + bw + 4 if val >= 0 else bx - 4:.1f}" y="{y + bar_h * 0.8:.1f}" text-anchor="{"start" if val >= 0 else "end"}" fill="#64748b" font-size="9">{val:.2f}</text>')

    return f"""
    <svg width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0;">
        <line x1="{center_x:.1f}" y1="{pad_t}" x2="{center_x:.1f}" y2="{height - pad_b}" stroke="#94a3b8" />
        {"".join(bars_svg)}
        <text x="{center_x - 40}" y="{height - 8}" fill="#64748b" font-size="11">Driver Correlation</text>
    </svg>
    """


def generate_tearsheet_html(
    strategy_id: str,
    summary: Mapping[str, Any],
    ic_by_horizon: pd.Series | Mapping[float, float],
    decay_info: Mapping[str, Any],
    pnl_df: pd.DataFrame | None = None,
    rolling_profile: pd.DataFrame | None = None,
    capacity_info: dict[str, Any] | None = None,
    decay_family: Mapping[str, Any] | None = None,
) -> str:
    h_keys = list(ic_by_horizon.keys()) if isinstance(ic_by_horizon, dict) else list(ic_by_horizon.index)
    ic_vals = list(ic_by_horizon.values()) if isinstance(ic_by_horizon, dict) else list(ic_by_horizon.values)
    decay_chart = _svg_decay_comparison(h_keys, ic_vals, decay_info, decay_family=decay_family)

    pnl_chart = ""
    dd_chart = ""
    if pnl_df is not None and not pnl_df.empty and "cum_gross" in pnl_df.columns:
        pnl_chart = _svg_cumulative_pnl(pnl_df["cum_gross"].to_numpy(), pnl_df["cum_net"].to_numpy())
        if "cum_net" in pnl_df.columns:
            dd_chart = _svg_drawdown(pnl_df["cum_net"].to_numpy())

    capacity_chart = ""
    if capacity_info and "curve" in capacity_info:
        capacity_chart = _svg_capacity_curve(capacity_info["curve"])

    driver_keys = ["crowding", "volatility", "turnover", "liquidity", "market_cap", "regime_changes"]
    drivers_dict = {k: summary[k] for k in driver_keys if k in summary}
    driver_chart = _svg_factor_exposure_bar(drivers_dict) if drivers_dict else ""

    def _fmt(val: Any) -> str:
        if isinstance(val, (int, float)):
            return f"{val:.4f}" if np.isfinite(val) else "N/A"
        return str(val)

    metric_cards = []
    for k, v in summary.items():
        if isinstance(v, (int, float, str)) and k not in {"features", "weights"}:
            metric_cards.append(
                f'<div class="card"><div class="label">{k}</div><div class="val">{_fmt(v)}</div></div>'
            )

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Bonsai Tear-Sheet: {strategy_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ margin: 0 0 8px 0; font-size: 24px; color: #0f172a; }}
        .subtitle {{ color: #64748b; margin-bottom: 24px; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
        .card {{ background: #ffffff; padding: 14px; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .label {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; margin-bottom: 4px; }}
        .val {{ font-size: 18px; font-weight: 700; color: #1e293b; }}
        .chart-section {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-bottom: 24px; }}
        .chart-box {{ background: #ffffff; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .chart-title {{ font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 12px; }}
        .legend {{ font-size: 12px; color: #475569; margin-top: 8px; }}
        .tag-green {{ color: #16a34a; font-weight: 600; }}
        .tag-blue {{ color: #2563eb; font-weight: 600; }}
        .tag-red {{ color: #dc2626; font-weight: 600; }}
        .tag-purple {{ color: #9333ea; font-weight: 600; }}
        .tag-amber {{ color: #f59e0b; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Strategy Tear-Sheet: {strategy_id}</h1>
        <div class="subtitle">Bonsai Alpha Decay Observatory — Bias-Controlled Quantitative Diagnostic</div>
        <div class="grid">
            {"".join(metric_cards)}
        </div>
        <div class="chart-section">
            <div class="chart-box">
                <div class="chart-title">Information Coefficient Decay Profile</div>
                {decay_chart}
                <div class="legend"><span class="tag-blue">• Empirical IC</span> &nbsp;|&nbsp; <span class="tag-red">— Exponential</span> &nbsp;|&nbsp; <span class="tag-purple">-- Power-Law</span> &nbsp;|&nbsp; <span class="tag-green">·· Bi-Exponential</span></div>
            </div>
            {"<div class='chart-box'><div class='chart-title'>Cumulative Long/Short Return (Gross vs Net of Costs)</div>" + pnl_chart + "<div class='legend'><span class='tag-green'>— Gross Spread PnL</span> &nbsp;|&nbsp; <span class='tag-blue'>— Net of Transaction Costs</span></div></div>" if pnl_chart else ""}
            {"<div class='chart-box'><div class='chart-title'>Underwater Drawdown Profile</div>" + dd_chart + "<div class='legend'><span class='tag-red'>— Peak-to-Trough Drawdown</span></div></div>" if dd_chart else ""}
            {"<div class='chart-box'><div class='chart-title'>Capital Capacity Curve (AUM Stress Test)</div>" + capacity_chart + "<div class='legend'><span class='tag-blue'>— Net Sharpe vs AUM</span> &nbsp;|&nbsp; <span class='tag-amber'>-- Capacity Threshold</span></div></div>" if capacity_chart else ""}
            {"<div class='chart-box'><div class='chart-title'>Systematic Driver Exposures</div>" + driver_chart + "<div class='legend'><span>Rank correlation to market metadata drivers</span></div></div>" if driver_chart else ""}
        </div>
    </div>
</body>
</html>
"""


def save_tearsheet(html_content: str, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html_content, encoding="utf-8")
    return destination
