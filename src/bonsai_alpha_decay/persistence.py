from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ext = destination.suffix.lower()
    if ext == ".csv":
        frame.to_csv(destination, index=True)
    elif ext == ".json":
        frame.to_json(destination, orient="records", date_format="iso")
    elif ext == ".parquet":
        frame.to_parquet(destination)
    else:
        raise ValueError("supported frame formats are csv, json, and parquet")
    return destination


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    ext = source.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(source, index_col=0, parse_dates=True)
    if ext == ".json":
        return pd.read_json(source)
    if ext == ".parquet":
        return pd.read_parquet(source)
    raise ValueError("supported frame formats are csv, json, and parquet")
