from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from src.utils import polite_get


class MarketProvider:
    """Yahoo Chart API-compatible provider with deterministic synthetic fallback."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session = requests.Session()
        self.headers = {"User-Agent": cfg["data"]["user_agent"]}

    def _fetch_yahoo(self, symbol: str, days: int) -> pd.DataFrame:
        end = int(time.time())
        start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        url = f'{self.cfg["data"]["yahoo_chart_base"]}/{symbol}'
        params = {"period1": start, "period2": end, "interval": "1d", "events": "history", "includeAdjustedClose": "true"}
        r = polite_get(
            self.session,
            url,
            timeout=self.cfg["data"]["request_timeout_seconds"],
            delay=self.cfg["data"]["request_delay_seconds"],
            headers=self.headers,
            params=params,
        )
        r.raise_for_status()
        result = r.json()["chart"]["result"][0]
        ts = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
        q = result["indicators"]["quote"][0]
        close = pd.to_numeric(pd.Series(q["close"]), errors="coerce")
        return pd.DataFrame({"date": ts, "price": close}).dropna().drop_duplicates("date").sort_values("date")

    @staticmethod
    def _synthetic(symbol: str, days: int) -> pd.DataFrame:
        seed = abs(hash(symbol)) % (2**32)
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=max(260, int(days * 5 / 7)))
        bases = {"CL=F": 72.0, "BZ=F": 76.0, "DX-Y.NYB": 101.0, "CNY=X": 7.15}
        vols = {"CL=F": 0.018, "BZ=F": 0.016, "DX-Y.NYB": 0.004, "CNY=X": 0.002}
        ret = rng.normal(0.0001, vols.get(symbol, 0.01), len(dates))
        price = bases.get(symbol, 100.0) * np.exp(np.cumsum(ret))
        return pd.DataFrame({"date": dates, "price": price})

    def history(self, symbol: str, days: int | None = None) -> tuple[pd.DataFrame, str]:
        days = days or self.cfg["data"]["market_history_days"]
        try:
            df = self._fetch_yahoo(symbol, days)
            if len(df) < 60:
                raise ValueError("insufficient rows")
            return df, "yahoo_chart"
        except Exception:
            return self._synthetic(symbol, days), "synthetic_fallback"
