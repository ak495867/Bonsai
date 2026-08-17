from __future__ import annotations

import argparse
import json
from pathlib import Path

from .drivers import fit_driver_model
from .observatory import ic_long_table, run_observatory, strategy_specs_to_frame
from .persistence import write_frame, write_json
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
    args = parser.parse_args()
    output = Path(args.output)
    config = SyntheticConfig(
        n_days=args.days, n_assets=args.assets, n_features=args.features, seed=args.seed
    )
    panel = make_synthetic_panel(config)
    feature_names = [f"feature_{index:02d}" for index in range(args.features)]
    strategies = generate_strategies(feature_names, args.count, args.seed + 1)
    summary, results = run_observatory(
        panel, strategies, workers=args.workers, min_obs=args.min_obs
    )
    write_frame(summary, output / "strategy_summary.csv")
    write_frame(ic_long_table(results), output / "ic_long.csv")
    write_frame(strategy_specs_to_frame(strategies), output / "strategies.csv")
    driver_model = fit_driver_model(summary, "lambda")
    write_json(
        {
            "panel": config.__dict__,
            "strategy_count": len(strategies),
            "driver_model": driver_model,
        },
        output / "run_manifest.json",
    )
    top = summary.sort_values("mean_ic", ascending=False).head(10)[
        ["strategy_id", "mean_ic", "half_life", "lambda", "status"]
    ]
    print(top.to_string(index=False))


if __name__ == "__main__":
    run_main()
