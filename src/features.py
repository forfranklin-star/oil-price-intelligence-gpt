from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    x = df[["date", value_col]].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.tz_localize(None)
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
    return x.dropna().sort_values("date").drop_duplicates("date", keep="last")


def _asof(base: pd.DataFrame, source: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    """Point-in-time join: use only an observation already known on/before date.

    This is not imputation: source observations remain untouched, and training rows
    with no prior observation are later discarded. It prevents look-ahead bias.
    """
    s = _prepare(source, value_col).rename(columns={value_col: out_col})
    return pd.merge_asof(base.sort_values("date"), s.sort_values("date"), on="date", direction="backward")


def build_feature_frame(
    wti: pd.DataFrame,
    brent: pd.DataFrame,
    dxy: pd.DataFrame,
    us10y: pd.DataFrame,
    cpi: pd.DataFrame,
    payroll: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    base = _prepare(wti.rename(columns={"price": "value"}), "value").rename(columns={"value": "wti"})
    for df, col in [(brent, "brent"), (dxy, "dxy"), (us10y, "us10y"), (cpi, "cpi"), (payroll, "payroll")]:
        base = _asof(base, df.rename(columns={"price": "value"}) if "price" in df.columns else df, "value", col)

    if events.empty:
        base["geopolitical_risk"] = 0.0
        base["institution_score"] = 0.0
    else:
        e = events.copy()
        e["date"] = pd.to_datetime(e["published"], errors="coerce").dt.tz_localize(None).dt.normalize()
        daily = e.dropna(subset=["date"]).groupby("date").agg(
            geopolitical_risk=("geopolitical_score", "sum"),
            institution_score=("institution_score", "sum"),
        ).reset_index()
        base = base.merge(daily, on="date", how="left")
        # Missing event observations mean no verified event was recorded that day;
        # this is a semantic zero, not a fabricated market observation.
        base[["geopolitical_risk", "institution_score"]] = base[["geopolitical_risk", "institution_score"]].fillna(0.0)

    base["wti_return"] = np.log(base["wti"]).diff()
    base["brent_return"] = np.log(base["brent"]).diff()
    base["usd_return"] = np.log(base["dxy"]).diff()
    base["us10y_change"] = base["us10y"].diff()
    base["cpi_surprise_proxy"] = base["cpi"].pct_change(30) - base["cpi"].pct_change(90).rolling(30, min_periods=5).mean()
    base["payroll_momentum"] = base["payroll"].diff(30) / 1000.0
    base["momentum_5d"] = np.log(base["wti"] / base["wti"].shift(5))
    base["target_next_return"] = base["wti_return"].shift(-1)
    return base.replace([np.inf, -np.inf], np.nan).sort_values("date")
