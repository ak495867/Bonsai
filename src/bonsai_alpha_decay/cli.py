from __future__ import annotations

import argparse
import json
from pathlib import Path

from .drivers import fit_driver_model
from .observatory import ic_long_table, run_observatory, strategy_specs_to_frame
from .persistence import write_frame, write_json
from .reporting import generate_tearsheet_html, save_tearsheet
from .strategies import generate_strategies
from .synthetic import SyntheticConfig, make_synthetic_panel


def _panel_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--days", type=int, default=756)
    parser.add_argument("--assets", type=int, default=250)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)


def generate_main() -> None:
    parser = argparse.ArgumentParser(prog="bonsai-generate")
    _panel_config(parser)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", default="data/strategies.json")
    args = parser.parse_args()
    config = SyntheticConfig(
        n_days=args.days, n_assets=args.assets, n_features=args.features, seed=args.seed
    )
    feature_names = [f"feature_{index:02d}" for index in range(args.features)]
    strategies = generate_strategies(feature_names, args.count, args.seed + 1)
    write_frame(strategy_specs_to_frame(strategies), args.output)
    payload = {
        "panel": config.__dict__,
        "strategy_count": len(strategies),
        "output": str(args.output),
    }
    print(json.dumps(payload, indent=2))


def run_main() -> None:
    parser = argparse.ArgumentParser(prog="bonsai-run")
    _panel_config(parser)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default="reports/run")
    parser.add_argument("--min-obs", type=int, default=10)
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--ridge-alpha", type=float, default=0.0)
    parser.add_argument("--simulate-pnl", action="store_true")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--tearsheet", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    config = SyntheticConfig(
        n_days=args.days, n_assets=args.assets, n_features=args.features, seed=args.seed
    )
    panel = make_synthetic_panel(config)
    feature_names = [f"feature_{index:02d}" for index in range(args.features)]
    strategies = generate_strategies(feature_names, args.count, args.seed + 1)
    summary, results = run_observatory(
        panel,
        strategies,
        workers=args.workers,
        min_obs=args.min_obs,
        simulate_pnl=args.simulate_pnl,
        cost_bps=args.cost_bps,
    )
    ext = f".{args.format}"
    write_frame(summary, output / f"strategy_summary{ext}")
    write_frame(ic_long_table(results), output / f"ic_long{ext}")
    write_frame(strategy_specs_to_frame(strategies), output / f"strategies{ext}")
    driver_model = fit_driver_model(summary, "lambda", alpha=args.ridge_alpha)
    write_json(
        {
            "panel": config.__dict__,
            "strategy_count": len(strategies),
            "driver_model": driver_model,
        },
        output / "run_manifest.json",
    )
    if args.tearsheet:
        tearsheet_dir = output / "tearsheets"
        top_ids = summary.sort_values("mean_ic", ascending=False).head(5)["strategy_id"]
        for sid in top_ids:
            obs = results[sid]
            row_dict = summary[summary["strategy_id"] == sid].iloc[0].to_dict()
            h_ic = obs.ic_by_horizon.mean(axis=0)
            html = generate_tearsheet_html(
                sid,
                row_dict,
                h_ic,
                obs.decay,
                pnl_df=obs.pnl,
            )
            save_tearsheet(html, tearsheet_dir / f"{sid}.html")

    display_cols = [
        col
        for col in [
            "strategy_id",
            "mean_ic",
            "ic_hac_tstat",
            "half_life",
            "lambda",
            "strategy_turnover",
            "net_sharpe",
            "deflated_sharpe_ratio",
            "status",
        ]
        if col in summary.columns
    ]
    top = summary.sort_values("mean_ic", ascending=False).head(10)[display_cols]
    print(top.to_string(index=False))


if __name__ == "__main__":
    run_main()
