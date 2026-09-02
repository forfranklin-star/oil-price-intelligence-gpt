from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from src.utils import polite_get


class FredProvider:
    """FRED observations. Macro series are allowed to be publication-lagged;
    the report records their actual observation date rather than pretending they
    are today's values."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session = requests.Session()
        self.headers = {"User-Agent": cfg["data"]["user_agent"]}

    def series(self, series_id: str, days: int = 900) -> tuple[pd.DataFrame, str]:
        r = polite_get(self.session, self.cfg["data"]["fred_csv_base"],
                       timeout=self.cfg["data"]["request_timeout_seconds"],
                       delay=self.cfg["data"]["request_delay_seconds"],
                       headers=self.headers, params={"id": series_id})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if len(df.columns) < 2:
            raise RuntimeError(f"FRED {series_id}: malformed response")
        df = df.iloc[:, :2]
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
        df = df[df["date"] >= cutoff].dropna().sort_values("date")
        if df.empty:
            raise RuntimeError(f"FRED {series_id}: empty observations")
        return df, "fred_csv"
