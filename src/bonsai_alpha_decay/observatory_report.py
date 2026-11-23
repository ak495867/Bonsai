from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd


def _svg_alpha_scatter(summary_df: pd.DataFrame) -> str:
    valid = summary_df.dropna(subset=["mean_ic", "lambda"]).copy()
    if len(valid) == 0:
        return "<p>No strategy data for scatter plot</p>"
    width, height = 540, 300
    pad_l, pad_r, pad_t, pad_b = 60, 40, 30, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    x_vals = valid["mean_ic"].to_numpy(dtype=float)
    y_vals = valid["lambda"].to_numpy(dtype=float)

    min_x, max_x = min(float(np.min(x_vals)), -0.01), max(float(np.max(x_vals)), 0.08)
    min_y, max_y = 0.0, max(float(np.max(y_vals)), 0.3)
    x_range = max(max_x - min_x, 1e-4)
    y_range = max(max_y - min_y, 1e-4)

    def tx(x: float) -> float:
        return pad_l + ((x - min_x) / x_range) * plot_w

    def ty(y: float) -> float:
        return pad_t + plot_h - ((y - min_y) / y_range) * plot_h

    model_colors = {
        "exponential": "#dc2626",
        "power_law": "#9333ea",
        "hump": "#f59e0b",
        "bi_exponential": "#059669",
    }

    dots = []
    zero_x = tx(0.0)
    for _, row in valid.iterrows():
        cx = tx(float(row["mean_ic"]))
        cy = ty(float(row["lambda"]))
        model = str(row.get("best_decay_model", "exponential"))
        color = model_colors.get(model, "#2563eb")
        sr = float(row.get("net_sharpe", 1.0)) if np.isfinite(row.get("net_sharpe", np.nan)) else 1.0
        radius = max(3.0, min(10.0, 3.0 + max(0.0, sr) * 1.5))
        dots.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{color}" opacity="0.75" />')

    return f"""
    <svg width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0;">
        <line x1="{zero_x:.1f}" y1="{pad_t}" x2="{zero_x:.1f}" y2="{height - pad_b}" stroke="#cbd5e1" stroke-dasharray="3,3" />
        <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#94a3b8" />
        <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="#94a3b8" />
        {"".join(dots)}
        <text x="{pad_l}" y="{height - 10}" fill="#64748b" font-size="11">Mean IC</text>
        <text x="15" y="{pad_t + 10}" fill="#64748b" font-size="11" transform="rotate(-90 15,{pad_t + 10})">Decay Rate (λ)</text>
        <text x="{width - pad_r - 120}" y="{pad_t + 15}" fill="#16a34a" font-size="10" font-weight="600">Persistent Alphas ↗</text>
    </svg>
    """


def _svg_correlation_heatmap(corr_matrix: pd.DataFrame, max_cols: int = 8) -> str:
    sub_corr = corr_matrix.iloc[:max_cols, :max_cols]
    n = len(sub_corr)
    if n == 0:
        return "<p>No correlation data</p>"

    cell_size = min(36.0, 240.0 / n)
    pad_l, pad_t = 70.0, 40.0
    width = pad_l + n * cell_size + 30.0
    height = pad_t + n * cell_size + 30.0

    cells = []
    labels = []

    for i, col_name in enumerate(sub_corr.columns):
        # Shorten label
        short_label = col_name if len(col_name) <= 8 else col_name[:7] + "…"
        labels.append(f'<text x="{pad_l - 6}" y="{pad_t + i * cell_size + cell_size * 0.7:.1f}" text-anchor="end" fill="#334155" font-size="9">{short_label}</text>')
        labels.append(f'<text x="{pad_l + i * cell_size + cell_size * 0.5:.1f}" y="{pad_t - 6}" text-anchor="middle" fill="#334155" font-size="9">{short_label}</text>')

        for j, row_name in enumerate(sub_corr.index):
            val = float(sub_corr.iloc[i, j])
            # Color scale: -1 (blue) to 0 (white) to 1 (red)
            if val >= 0:
                r = 255
                g = int(255 * (1.0 - val * 0.7))
                b = int(255 * (1.0 - val * 0.7))
            else:
                r = int(255 * (1.0 - abs(val) * 0.7))
                g = int(255 * (1.0 - abs(val) * 0.7))
                b = 255
            x = pad_l + j * cell_size
            y = pad_t + i * cell_size
            cells.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_size - 1:.1f}" height="{cell_size - 1:.1f}" fill="rgb({r},{g},{b})" rx="2" />')
            cells.append(f'<text x="{x + cell_size * 0.5:.1f}" y="{y + cell_size * 0.65:.1f}" text-anchor="middle" fill="#0f172a" font-size="8">{val:.2f}</text>')

    return f"""
    <svg width="{width:.0f}" height="{height:.0f}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0;">
        {"".join(labels)}
        {"".join(cells)}
    </svg>
    """


def generate_master_observatory_html(
    summary_df: pd.DataFrame,
    ensemble_info: Mapping[str, Any] | None = None,
    title: str = "Bonsai Master Observatory Executive Report",
) -> str:
    n_strats = len(summary_df)
    mean_ic = float(summary_df["mean_ic"].mean()) if "mean_ic" in summary_df else np.nan
    top_ic = float(summary_df["mean_ic"].max()) if "mean_ic" in summary_df else np.nan
    med_hl = float(summary_df["half_life"].median()) if "half_life" in summary_df else np.nan

    fdr_sig_count = int(summary_df["is_fdr_significant"].sum()) if "is_fdr_significant" in summary_df else 0
    fdr_sig_pct = (fdr_sig_count / max(n_strats, 1)) * 100.0

    med_dsr = float(summary_df["deflated_sharpe_ratio"].median()) if "deflated_sharpe_ratio" in summary_df else np.nan

    ens_sharpe = float(ensemble_info["performance"]["sharpe_ratio"]) if (ensemble_info and "performance" in ensemble_info) else np.nan
    div_ratio = float(ensemble_info.get("diversification_ratio", np.nan)) if ensemble_info else np.nan

    scatter_svg = _svg_alpha_scatter(summary_df)

    corr_svg = ""
    if ensemble_info and "correlation_matrix" in ensemble_info:
        corr_svg = _svg_correlation_heatmap(ensemble_info["correlation_matrix"])

    def _fmt(val: Any) -> str:
        if isinstance(val, (int, float)):
            return f"{val:.4f}" if np.isfinite(val) else "N/A"
        return str(val)

    top_table_rows = []
    sort_col = "net_sharpe" if "net_sharpe" in summary_df.columns else "mean_ic"
    top_strats = summary_df.sort_values(sort_col, ascending=False).head(10)
    for _, row in top_strats.iterrows():
        sid = str(row["strategy_id"])
        ic = _fmt(row.get("mean_ic"))
        hac_t = _fmt(row.get("ic_hac_tstat"))
        hl = _fmt(row.get("half_life"))
        model = str(row.get("best_decay_model", "exponential"))
        sr = _fmt(row.get("net_sharpe"))
        dsr = _fmt(row.get("deflated_sharpe_ratio"))
        fdr = "YES" if bool(row.get("is_fdr_significant", False)) else "NO"
        fdr_badge = f'<span style="color:#16a34a; font-weight:700;">{fdr}</span>' if fdr == "YES" else f'<span style="color:#94a3b8;">{fdr}</span>'

        top_table_rows.append(
            f"<tr><td><b>{sid}</b></td><td>{ic}</td><td>{hac_t}</td><td>{hl}</td><td>{model}</td><td>{sr}</td><td>{dsr}</td><td>{fdr_badge}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ margin: 0 0 8px 0; font-size: 26px; color: #0f172a; }}
        .subtitle {{ color: #64748b; margin-bottom: 24px; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }}
        .card {{ background: #ffffff; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .label {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; margin-bottom: 4px; }}
        .val {{ font-size: 20px; font-weight: 700; color: #1e293b; }}
        .chart-section {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 28px; }}
        .chart-box {{ background: #ffffff; padding: 18px; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .chart-title {{ font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 12px; }}
        .legend {{ font-size: 12px; color: #475569; margin-top: 8px; }}
        .tag-red {{ color: #dc2626; font-weight: 600; }}
        .tag-purple {{ color: #9333ea; font-weight: 600; }}
        .tag-amber {{ color: #f59e0b; font-weight: 600; }}
        .tag-green {{ color: #059669; font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; font-size: 13px; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #f1f5f9; }}
        th {{ background: #f8fafc; color: #475569; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
        tr:hover {{ background: #f8fafc; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="subtitle">Institutional Quantitative Research & Alpha Term Structure Observatory</div>
        
        <div class="grid">
            <div class="card"><div class="label">Strategies Tested</div><div class="val">{n_strats}</div></div>
            <div class="card"><div class="label">Top Mean IC</div><div class="val">{_fmt(top_ic)}</div></div>
            <div class="card"><div class="label">Median Half-Life</div><div class="val">{_fmt(med_hl)} d</div></div>
            <div class="card"><div class="label">FDR Significant</div><div class="val">{fdr_sig_count} ({fdr_sig_pct:.1f}%)</div></div>
            <div class="card"><div class="label">Median DSR</div><div class="val">{_fmt(med_dsr)}</div></div>
            {"<div class='card'><div class='label'>Master Ensemble Sharpe</div><div class='val' style='color:#16a34a;'>" + _fmt(ens_sharpe) + "</div></div>" if np.isfinite(ens_sharpe) else ""}
            {"<div class='card'><div class='label'>Diversification Ratio</div><div class='val'>" + _fmt(div_ratio) + "x</div></div>" if np.isfinite(div_ratio) else ""}
        </div>

        <div class="chart-section">
            <div class="chart-box">
                <div class="chart-title">Alpha Universe Dispersion: Mean IC vs Decay Rate (λ)</div>
                {scatter_svg}
                <div class="legend"><span class="tag-red">• Exponential</span> &nbsp;|&nbsp; <span class="tag-purple">• Power-Law</span> &nbsp;|&nbsp; <span class="tag-amber">• Hump</span> &nbsp;|&nbsp; <span class="tag-green">• Bi-Exponential</span></div>
            </div>
            {"<div class='chart-box'><div class='chart-title'>Top Alphas Cross-Correlation Heatmap Matrix</div>" + corr_svg + "<div class='legend'><span>Cross-sectional strategy PnL correlation</span></div></div>" if corr_svg else ""}
        </div>

        <div style="background:#ffffff; padding:18px; border-radius:8px; border:1px solid #e2e8f0;">
            <div class="chart-title" style="margin-bottom:14px;">Top Performing Alphas Ranking Table</div>
            <table>
                <thead>
                    <tr>
                        <th>Strategy ID</th>
                        <th>Mean IC</th>
                        <th>HAC t-stat</th>
                        <th>Half-Life</th>
                        <th>Decay Model</th>
                        <th>Net Sharpe</th>
                        <th>DSR</th>
                        <th>FDR Valid</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(top_table_rows)}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""


def save_master_report(html_content: str, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html_content, encoding="utf-8")
    return destination
