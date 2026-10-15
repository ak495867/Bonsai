from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd


def _svg_decay_curve(horizons: Sequence[float], ic_values: Sequence[float], ic0: float, decay_rate: float) -> str:
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

    fit_pts = []
    dense_h = np.linspace(0, max_h, 60)
    for h in dense_h:
        y_fit = ic0 * np.exp(-decay_rate * h) if (np.isfinite(ic0) and np.isfinite(decay_rate)) else 0.0
        fit_pts.append(f"{tx(h):.1f},{ty(y_fit):.1f}")
    fit_line = " ".join(fit_pts)

    points_svg = []
    for h, val in zip(h_arr, y_arr):
        if np.isfinite(val):
            points_svg.append(f'<circle cx="{tx(h):.1f}" cy="{ty(val):.1f}" r="4" fill="#2563eb" />')

    zero_y = ty(0.0)
    return f"""
    <svg width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0;">
        <line x1="{pad_l}" y1="{zero_y:.1f}" x2="{width - pad_r}" y2="{zero_y:.1f}" stroke="#94a3b8" stroke-dasharray="3,3" />
        <polyline fill="none" stroke="#dc2626" stroke-width="2" points="{fit_line}" />
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


def generate_tearsheet_html(
    strategy_id: str,
    summary: Mapping[str, Any],
    ic_by_horizon: pd.Series | Mapping[float, float],
    decay_info: Mapping[str, Any],
    pnl_df: pd.DataFrame | None = None,
    rolling_profile: pd.DataFrame | None = None,
) -> str:
    h_keys = list(ic_by_horizon.keys()) if isinstance(ic_by_horizon, dict) else list(ic_by_horizon.index)
    ic_vals = list(ic_by_horizon.values()) if isinstance(ic_by_horizon, dict) else list(ic_by_horizon.values)
    ic0 = float(decay_info.get("ic0", np.nan))
    decay_rate = float(decay_info.get("lambda", np.nan))
    decay_chart = _svg_decay_curve(h_keys, ic_vals, ic0, decay_rate)

    pnl_chart = ""
    if pnl_df is not None and not pnl_df.empty and "cum_gross" in pnl_df.columns:
        pnl_chart = _svg_cumulative_pnl(pnl_df["cum_gross"].to_numpy(), pnl_df["cum_net"].to_numpy())

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
        .chart-section {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 24px; }}
        .chart-box {{ background: #ffffff; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .chart-title {{ font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 12px; }}
        .legend {{ font-size: 12px; color: #475569; margin-top: 8px; }}
        .tag-green {{ color: #16a34a; font-weight: 600; }}
        .tag-blue {{ color: #2563eb; font-weight: 600; }}
        .tag-red {{ color: #dc2626; font-weight: 600; }}
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
                <div class="legend"><span class="tag-blue">• Empirical IC</span> &nbsp;|&nbsp; <span class="tag-red">— Fitted Exponential Curve</span></div>
            </div>
            {"<div class='chart-box'><div class='chart-title'>Cumulative Long/Short Return (Gross vs Net of Costs)</div>" + pnl_chart + "<div class='legend'><span class='tag-green'>— Gross Spread PnL</span> &nbsp;|&nbsp; <span class='tag-blue'>— Net of Transaction Costs</span></div></div>" if pnl_chart else ""}
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
