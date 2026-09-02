from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import requests

from src.utils import polite_get


class FredProvider:
    """FRED graph CSV endpoint; no API key required for these public series."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session = requests.Session()
        self.headers = {"User-Agent": cfg["data"]["user_agent"]}

    def series(self, series_id: str, days: int = 900) -> tuple[pd.DataFrame, str]:
        try:
            r = polite_get(
                self.session,
                self.cfg["data"]["fred_csv_base"],
                timeout=self.cfg["data"]["request_timeout_seconds"],
                delay=self.cfg["data"]["request_delay_seconds"],
                headers=self.headers,
                params={"id": series_id},
            )
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
            df = df[df["date"] >= cutoff].dropna().sort_values("date")
            if df.empty:
                raise ValueError("empty FRED result")
            return df, "fred_csv"
        except Exception:
            dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
            rng = np.random.default_rng(abs(hash(series_id)) % (2**32))
            base = {"DGS10": 4.0, "CPIAUCSL": 320, "PAYEMS": 159000, "DTWEXBGS": 121}.get(series_id, 100)
            scale = {"DGS10": 0.03, "CPIAUCSL": 0.08, "PAYEMS": 25, "DTWEXBGS": 0.12}.get(series_id, 0.1)
            vals = base + np.cumsum(rng.normal(0, scale, len(dates)))
            return pd.DataFrame({"date": dates, "value": vals}), "synthetic_fallback"
