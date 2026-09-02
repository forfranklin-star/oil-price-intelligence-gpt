from __future__ import annotations

from io import StringIO
import pandas as pd
import requests


class DieselDataError(RuntimeError):
    pass


class ChinaDieselProvider:
    """Only real domestic diesel observations are accepted.

    Production requires a real CSV/API endpoint. No Brent/CNY proxy and no
    generated series are allowed in this provider.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def history(self, *_args) -> tuple[pd.DataFrame, str]:
        urls = self.cfg["data"].get("china_diesel_csv_urls", [])
        if isinstance(urls, str):
            urls = [urls] if urls.strip() else []
        urls = [u.strip() for u in urls if str(u).strip()]
        errors = []
        for url in urls:
            try:
                r = requests.get(url, timeout=self.cfg["data"]["request_timeout_seconds"],
                                 headers={"User-Agent": self.cfg["data"]["user_agent"]})
                r.raise_for_status()
                df = pd.read_csv(StringIO(r.text))
                required = {"date", "price"}
                if not required.issubset(df.columns):
                    raise DieselDataError(f"{url}: CSV must contain date,price")
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                df = df[["date", "price"]].dropna().drop_duplicates("date").sort_values("date")
                if len(df) < int(self.cfg["data"]["min_diesel_rows"]):
                    raise DieselDataError(f"{url}: only {len(df)} valid rows")
                age = (pd.Timestamp.now().normalize() - df["date"].max().normalize()).days
                if age > int(self.cfg["data"]["max_diesel_staleness_days"]):
                    raise DieselDataError(f"{url}: latest observation is {age} days old")
                if (df["price"] <= 0).any():
                    raise DieselDataError(f"{url}: non-positive price detected")
                return df, f"configured_real_csv:{url}"
            except Exception as exc:
                errors.append(str(exc))
        raise DieselDataError(
            "No verified domestic diesel series is configured. Refusing to synthesize/proxy diesel data. "
            + " | ".join(errors)
        )
