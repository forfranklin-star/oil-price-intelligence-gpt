from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src.utils import polite_get


class MarketDataError(RuntimeError):
    pass


class MarketProvider:
    """Real market-data provider with strict freshness/quality validation.

    Yahoo Finance is used only as a transport-compatible delayed market source.
    Synthetic data is NEVER silently substituted. EIA can be enabled with
    EIA_API_KEY for official petroleum spot data.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session = requests.Session()
        self.headers = {"User-Agent": cfg["data"]["user_agent"]}

    def _validate(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df.empty:
            raise MarketDataError(f"{symbol}: empty market response")
        df = df.dropna(subset=["date", "price"]).copy()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna().drop_duplicates("date").sort_values("date")
        if len(df) < self.cfg["data"]["min_market_rows"]:
            raise MarketDataError(f"{symbol}: only {len(df)} usable rows")
        latest = pd.Timestamp(df["date"].max()).normalize()
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        age = int((today - latest).days)
        max_age = int(self.cfg["data"]["max_market_staleness_days"])
        if age > max_age:
            raise MarketDataError(f"{symbol}: latest observation is {age} days old; limit={max_age}")
        if (df["price"] <= 0).any():
            raise MarketDataError(f"{symbol}: non-positive price detected")
        return df

    def _fetch_yahoo(self, symbol: str, days: int) -> pd.DataFrame:
        end = int(time.time())
        start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        url = f'{self.cfg["data"]["yahoo_chart_base"]}/{symbol}'
        params = {"period1": start, "period2": end, "interval": "1d", "events": "history"}
        r = polite_get(self.session, url, timeout=self.cfg["data"]["request_timeout_seconds"],
                       delay=self.cfg["data"]["request_delay_seconds"], headers=self.headers, params=params)
        r.raise_for_status()
        result = (r.json().get("chart", {}).get("result") or [None])[0]
        if not result:
            raise MarketDataError(f"{symbol}: Yahoo returned no result")
        ts = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
        q = result["indicators"]["quote"][0]
        close = pd.to_numeric(pd.Series(q["close"]), errors="coerce")
        return self._validate(pd.DataFrame({"date": ts, "price": close}), symbol)

    def _fetch_eia(self, series_id: str, days: int) -> pd.DataFrame:
        key = ((self.cfg["data"].get("eia_api_key") or "").strip() or __import__("os").getenv("EIA_API_KEY", "").strip())
        if not key:
            raise MarketDataError("EIA_API_KEY is not configured")
        url = self.cfg["data"]["eia_api_base"]
        params = {
            "api_key": key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": series_id,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": str(max(days, 400)),
        }
        r = polite_get(self.session, url, timeout=self.cfg["data"]["request_timeout_seconds"],
                       delay=self.cfg["data"]["request_delay_seconds"], headers=self.headers, params=params)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
        if not rows:
            raise MarketDataError(f"EIA {series_id}: empty response")
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["period"], errors="coerce")
        df["price"] = pd.to_numeric(df["value"], errors="coerce")
        return self._validate(df[["date", "price"]], series_id)

    def history(self, symbol: str, days: int | None = None) -> tuple[pd.DataFrame, str]:
        days = days or self.cfg["data"]["market_history_days"]
        # For WTI/Brent futures the daily Yahoo chart is the primary current
        # market series. EIA spot data is a secondary historical/validation
        # source; it is not allowed to silently replace a fresh futures series.
        eia_map = self.cfg["symbols"].get("eia", {})
        logical = next((k for k, v in self.cfg["symbols"].items() if k != "eia" and v == symbol), None)
        eia_id = eia_map.get(symbol) or (eia_map.get(logical) if logical else None)
        errors = []
        try:
            return self._fetch_yahoo(symbol, days), "yahoo_chart_delayed_real_market"
        except Exception as exc:
            errors.append(str(exc))
        if eia_id and self.cfg["data"].get("allow_eia_fallback", True):
            try:
                return self._fetch_eia(eia_id, days), f"eia_spot:{eia_id}"
            except Exception as exc:
                errors.append(str(exc))
        raise MarketDataError(f"No valid real-time-enough data for {symbol}: {' | '.join(errors)}")
