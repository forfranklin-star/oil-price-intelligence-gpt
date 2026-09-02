from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import requests


class ChinaDieselProvider:
    """China diesel adapter. Uses user-provided CSV URL if configured; otherwise a transparent Brent/CNY proxy."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def history(self, brent: pd.DataFrame, cnyusd: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        url = (self.cfg["data"].get("china_diesel_csv_url") or "").strip()
        if url:
            try:
                r = requests.get(url, timeout=self.cfg["data"]["request_timeout_seconds"], headers={"User-Agent": self.cfg["data"]["user_agent"]})
                r.raise_for_status()
                df = pd.read_csv(StringIO(r.text))
                df["date"] = pd.to_datetime(df["date"])
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                return df[["date", "price"]].dropna().sort_values("date"), "configured_csv"
            except Exception:
                pass

        x = brent.rename(columns={"price": "brent"}).merge(cnyusd.rename(columns={"price": "cny"}), on="date", how="outer").sort_values("date")
        x[["brent", "cny"]] = x[["brent", "cny"]].ffill().bfill()
        # 仅作占位代理：将国际原油成本经汇率和调价滞后映射为人民币/吨量级。
        raw = 4300 + 42 * x["brent"].rolling(10, min_periods=1).mean() * (x["cny"] / 7.0)
        smooth = raw.ewm(span=12, adjust=False).mean()
        return pd.DataFrame({"date": x["date"], "price": smooth}), "brent_cny_proxy"
