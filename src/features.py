from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_macro(df: pd.DataFrame, name: str) -> pd.DataFrame:
    x = df.rename(columns={"value": name}).copy().sort_values("date")
    return x.set_index("date").resample("D").last().ffill().reset_index()


def build_feature_frame(
    wti: pd.DataFrame,
    brent: pd.DataFrame,
    dxy: pd.DataFrame,
    us10y: pd.DataFrame,
    cpi: pd.DataFrame,
    payroll: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    base = wti.rename(columns={"price": "wti"}).copy()
    for df, col in [(brent, "brent"), (dxy, "dxy")]:
        base = base.merge(df.rename(columns={"price": col}), on="date", how="outer")
    base = base.sort_values("date").set_index("date").resample("D").last().ffill().reset_index()
    base = base.merge(_daily_macro(us10y, "us10y"), on="date", how="left")
    base = base.merge(_daily_macro(cpi, "cpi"), on="date", how="left")
    base = base.merge(_daily_macro(payroll, "payroll"), on="date", how="left")
    for c in ["us10y", "cpi", "payroll"]:
        base[c] = base[c].ffill().bfill()

    if not events.empty:
        e = events.copy()
        e["date"] = pd.to_datetime(e["published"]).dt.normalize()
        daily = e.groupby("date").agg(geopolitical_risk=("geopolitical_score", "sum"), institution_score=("institution_score", "sum")).reset_index()
        base = base.merge(daily, on="date", how="left")
    for col in ["geopolitical_risk", "institution_score"]:
        if col not in base.columns:
            base[col] = 0.0
    base[["geopolitical_risk", "institution_score"]] = base[["geopolitical_risk", "institution_score"]].fillna(0.0)

    base["wti_return"] = np.log(base["wti"]).diff()
    base["brent_return"] = np.log(base["brent"]).diff()
    base["usd_return"] = np.log(base["dxy"]).diff()
    base["us10y_change"] = base["us10y"].diff()
    base["cpi_surprise_proxy"] = base["cpi"].pct_change(30) - base["cpi"].pct_change(90).rolling(30, min_periods=5).mean()
    base["payroll_momentum"] = base["payroll"].diff(30) / 1000.0
    base["momentum_5d"] = np.log(base["wti"] / base["wti"].shift(5))
    base["target_next_return"] = base["wti_return"].shift(-1)
    return base.replace([np.inf, -np.inf], np.nan)
