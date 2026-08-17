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
    if destination.suffix.lower() == ".csv":
        frame.to_csv(destination, index=True)
    elif destination.suffix.lower() == ".json":
        frame.to_json(destination, orient="records", date_format="iso")
    else:
        raise ValueError("supported frame formats are csv and json")
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
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, index_col=0, parse_dates=True)
    if source.suffix.lower() == ".json":
        return pd.read_json(source)
    raise ValueError("supported frame formats are csv and json")
