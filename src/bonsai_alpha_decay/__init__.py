from .decay import (
    bootstrap_decay,
    estimate_bi_exponential_decay,
    estimate_decay_family,
    estimate_exponential_decay,
    estimate_hump_decay,
    estimate_power_law_decay,
)
from .cpcv import CombinatorialPurgedCV, evaluate_strategy_cpcv
from .drivers import DRIVER_NAMES, fit_driver_model, strategy_driver_exposure
from .engine import (
    build_ic_table,
    build_signal,
    calculate_turnover,
    cross_sectional_ic,
    forward_returns,
    ic_hac_term_structure,
    newey_west_ic_stats,
)
from .models import MarketPanel, ObservedStrategy, StrategySpec
from .neutralization import neutralize_signal
from .observatory import (
    evaluate_strategy,
    ic_long_table,
    run_observatory,
    strategy_specs_to_frame,
)
from .portfolio import (
    build_portfolio_weights,
    compute_dynamic_borrow_rates,
    optimal_holding_horizon,
    portfolio_performance_stats,
    simulate_long_short_pnl,
    strategy_capacity_curve,
)
from .reporting import generate_tearsheet_html, save_tearsheet
from .rolling import decay_compression_rate, rolling_decay_profile
from .strategies import generate_strategies, with_decay_profile
from .synthetic import SyntheticConfig, make_synthetic_panel
from .validation import (
    benjamini_hochberg_fdr,
    deflated_sharpe_ratio,
    haircut_sharpe_ratio,
)

__all__ = [
    "DRIVER_NAMES",
    "CombinatorialPurgedCV",
    "MarketPanel",
    "ObservedStrategy",
    "StrategySpec",
    "SyntheticConfig",
    "benjamini_hochberg_fdr",
    "bootstrap_decay",
    "build_ic_table",
    "build_portfolio_weights",
    "build_signal",
    "calculate_turnover",
    "compute_dynamic_borrow_rates",
    "cross_sectional_ic",
    "decay_compression_rate",
    "deflated_sharpe_ratio",
    "estimate_bi_exponential_decay",
    "estimate_decay_family",
    "estimate_exponential_decay",
    "estimate_hump_decay",
    "estimate_power_law_decay",
    "evaluate_strategy",
    "evaluate_strategy_cpcv",
    "fit_driver_model",
    "forward_returns",
    "generate_strategies",
    "generate_tearsheet_html",
    "haircut_sharpe_ratio",
    "ic_hac_term_structure",
    "ic_long_table",
    "make_synthetic_panel",
    "neutralize_signal",
    "newey_west_ic_stats",
    "optimal_holding_horizon",
    "portfolio_performance_stats",
    "rolling_decay_profile",
    "run_observatory",
    "save_tearsheet",
    "simulate_long_short_pnl",
    "strategy_capacity_curve",
    "strategy_driver_exposure",
    "strategy_specs_to_frame",
    "with_decay_profile",
]
