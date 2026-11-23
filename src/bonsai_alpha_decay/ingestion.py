from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

from .models import MarketPanel


def read_daily_bars_from_directory(
    dir_path: str | Path,
    file_format: str = "parquet",
) -> pd.DataFrame:
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pattern = f"*.{file_format}"
    files = sorted(path.rglob(pattern))
    if not files:
        raise ValueError(f"No {file_format} files found under {dir_path}")

    frames = []
    for f in files:
        if file_format == "parquet":
            df = pd.read_parquet(f)
        elif file_format == "csv":
            df = pd.read_csv(f)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def build_panel_from_daily_bars(
    bars: pd.DataFrame,
    delisting_df: pd.DataFrame | None = None,
    default_delisting_return: float = -0.35,
    features_to_generate: Sequence[str] = (
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "volume_surge",
    ),
) -> MarketPanel:
    required = {"date", "asset", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars DataFrame missing required columns: {sorted(missing)}")

    clean_bars = bars.copy()
    clean_bars["date"] = pd.to_datetime(clean_bars["date"])
    clean_bars.sort_values(["date", "asset"], inplace=True)

    # Pivot prices
    prices = clean_bars.pivot(index="date", columns="asset", values="close")
    prices.sort_index(inplace=True)

    # Compute daily close-to-close returns
    returns = prices.pct_change(fill_method=None)

    # Create point-in-time mask: True on dates when asset has active prices
    mask = prices.notna() & (prices > 0)

    # Apply delisting returns
    if delisting_df is not None and not delisting_df.empty:
        delist_clean = delisting_df.copy()
        delist_clean["date"] = pd.to_datetime(delist_clean["date"])
        for _, row in delist_clean.iterrows():
            d_date = row["date"]
            asset = row["asset"]
            d_ret = float(row.get("delisting_return", default_delisting_return))
            if asset in returns.columns and d_date in returns.index:
                returns.loc[d_date, asset] = d_ret
                # Mark as non-tradeable following delisting date
                post_dates = returns.index[returns.index > d_date]
                if len(post_dates) > 0:
                    mask.loc[post_dates, asset] = False
                    returns.loc[post_dates, asset] = np.nan
    else:
        # Detect dropoffs: if asset vanishes before end of panel, apply default delisting penalty
        for asset in prices.columns:
            valid_dates = prices.index[prices[asset].notna()]
            if len(valid_dates) > 0 and valid_dates[-1] < prices.index[-1]:
                last_active = valid_dates[-1]
                idx_pos = prices.index.get_loc(last_active)
                if idx_pos + 1 < len(prices.index):
                    delist_date = prices.index[idx_pos + 1]
                    returns.loc[delist_date, asset] = default_delisting_return
                    post_dates = prices.index[prices.index > delist_date]
                    if len(post_dates) > 0:
                        mask.loc[post_dates, asset] = False

    # Forward-fill prices for index consistency
    prices = prices.ffill().bfill().fillna(100.0)
    returns = returns.fillna(0.0)

    # Generate canonical features
    feature_dict = {}
    if "return_1d" in features_to_generate:
        feature_dict["return_1d"] = returns.shift(1).fillna(0.0)
    if "return_5d" in features_to_generate:
        feature_dict["return_5d"] = returns.rolling(5, min_periods=1).sum().shift(1).fillna(0.0)
    if "return_20d" in features_to_generate:
        feature_dict["return_20d"] = returns.rolling(20, min_periods=1).sum().shift(1).fillna(0.0)
    if "volatility_20d" in features_to_generate:
        feature_dict["volatility_20d"] = (
            returns.rolling(20, min_periods=5).std().shift(1).fillna(0.02)
        )
    if "volume_surge" in features_to_generate and "volume" in clean_bars.columns:
        volume = clean_bars.pivot(index="date", columns="asset", values="volume")
        adv_20 = volume.rolling(20, min_periods=5).mean()
        surge = (volume / adv_20.replace(0.0, np.nan)).shift(1).fillna(1.0)
        feature_dict["volume_surge"] = surge

    # Format multi-index features as {asset}__{feature}
    features_concat = pd.concat(
        {feat: df for feat, df in feature_dict.items()},
        axis=1,
    )
    features_concat.columns = pd.MultiIndex.from_tuples(
        features_concat.columns, names=["feature", "asset"]
    )
    features_concat = features_concat.swaplevel(axis=1).sort_index(axis=1)
    features_concat.columns = pd.Index(
        [f"{asset}__{feature}" for asset, feature in features_concat.columns]
    )

    # Build metadata
    assets = prices.columns
    rolling_vol = returns.rolling(20, min_periods=5).std().mean(axis=0).fillna(0.02)
    turnover = (0.02 + rolling_vol * 0.8).clip(0.01, 1.0)
    liquidity = pd.Series(1e7, index=assets)
    if "volume" in clean_bars.columns:
        vol_df = clean_bars.pivot(index="date", columns="asset", values="volume").fillna(1e5)
        liquidity = (prices * vol_df).mean(axis=0).fillna(1e7)

    metadata = pd.DataFrame(
        {
            "market_cap": pd.Series(1e9, index=assets),
            "volatility": rolling_vol,
            "turnover": turnover,
            "liquidity": liquidity,
            "crowding": pd.Series(0.5, index=assets),
            "regime": pd.Series(0, index=assets),
        },
        index=assets,
    )

    panel = MarketPanel(
        prices=prices,
        features=features_concat,
        returns=returns,
        metadata=metadata,
        mask=mask,
    )
    panel.validate()
    return panel
