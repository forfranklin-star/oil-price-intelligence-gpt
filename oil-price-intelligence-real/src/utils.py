from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for rel in ["data/raw", "data/processed", "data/features", "reports", "models"]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def atomic_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def dataframe_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict[str, Any]]:
    if max_rows:
        df = df.tail(max_rows)
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict("records")


def polite_get(session, url: str, *, timeout: int, delay: float, headers: dict | None = None, params: dict | None = None):
    time.sleep(max(0.0, delay) + random.random() * 0.15)
    return session.get(url, timeout=timeout, headers=headers, params=params)


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)
