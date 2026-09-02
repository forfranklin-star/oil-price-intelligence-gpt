from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import math
import numpy as np
import pandas as pd

from src.explain import event_impacts, narrative
from src.features import build_feature_frame
from src.models import long_scenarios, medium_forecast, short_forecast, train_weight_models
from src.providers import ChinaDieselProvider, FredProvider, MarketProvider, NewsProvider
from src.storage import connect, save_events, save_report, upsert_series
from src.utils import ROOT, atomic_json_dump, dataframe_records, ensure_dirs, load_config


def _diesel_forecast(diesel: pd.DataFrame, wti_current: float, wti_future: dict) -> dict:
    cur = float(diesel["price"].iloc[-1])
    ratio = wti_future["mean"] / max(wti_current, 1e-9)
    # Research forecast from the observed domestic diesel series; no proxy is used.
    mean = cur * (1 + (ratio - 1) * 0.55)
    width = max(cur * 0.035, abs(mean - cur) * 0.8)
    return {"mean": mean, "low": mean - width, "high": mean + width}


def run_pipeline() -> dict:
    cfg = load_config()
    ensure_dirs()
    tz = ZoneInfo(cfg["app"]["timezone"])
    report_date = datetime.now(tz).strftime("%Y-%m-%d")

    market = MarketProvider(cfg)
    fred = FredProvider(cfg)
    news_provider = NewsProvider(cfg)

    wti, src_wti = market.history(cfg["symbols"]["wti"])
    brent, src_brent = market.history(cfg["symbols"]["brent"])
    dxy, src_dxy = market.history(cfg["symbols"]["dxy"])
    cny, src_cny = market.history(cfg["symbols"]["cnyusd"])

    us10y, src_10y = fred.series(cfg["fred_series"]["us10y"])
    cpi, src_cpi = fred.series(cfg["fred_series"]["cpi"])
    payroll, src_pay = fred.series(cfg["fred_series"]["nonfarm"])
    broad_usd, src_busd = fred.series(cfg["fred_series"]["broad_usd"])
    events, src_news = news_provider.fetch(30)
    diesel, src_diesel = ChinaDieselProvider(cfg).history(brent, cny)

    if events.empty:
        events = pd.DataFrame(columns=["published", "title", "link", "source", "geopolitical_score", "institution_score", "direction"])
    features = build_feature_frame(wti, brent, dxy, us10y, cpi, payroll, events)
    bundle = train_weight_models(features, cfg)
    short_path = short_forecast(features, bundle, cfg["app"]["forecast_short_days"])
    medium = medium_forecast(features, bundle, cfg["app"]["forecast_medium_days"])
    longterm = long_scenarios(features, bundle)

    current_wti = float(wti["price"].iloc[-1])
    current_brent = float(brent["price"].iloc[-1])
    current_diesel = float(diesel["price"].iloc[-1])
    end_short = short_path.iloc[-1]
    short_summary = {"mean": float(end_short["mean"]), "low": float(end_short["low"]), "high": float(end_short["high"])}
    short_sigma = max((short_summary["high"] - short_summary["low"]) / (2 * 1.282), 0.01)
    z = (short_summary["mean"] - current_wti) / short_sigma
    short_summary["bull_probability"] = float(np.clip(0.5 * (1 + math.erf(z / np.sqrt(2))), 0.01, 0.99))
    short_summary["bear_probability"] = 1 - short_summary["bull_probability"]
    short_summary["model"] = "SARIMAX + RF/LASSO signal tilt"

    impacts = event_impacts(events, bundle.weights, current_wti, bundle.residual_sigma)
    diesel_short = _diesel_forecast(diesel, current_wti, short_summary)
    diesel_medium = _diesel_forecast(diesel, current_wti, medium)

    payload = {
        "report_date": report_date,
        "generated_at": datetime.now(tz).isoformat(timespec="seconds"),
        "data_status": {
            "wti": src_wti, "brent": src_brent, "dxy": src_dxy, "cnyusd": src_cny,
            "us10y": src_10y, "cpi": src_cpi, "payroll": src_pay, "broad_usd": src_busd,
            "news": src_news, "china_diesel": src_diesel,
        },
        "latest": {"wti": current_wti, "brent": current_brent, "china_diesel": current_diesel},
        "forecasts": {
            "wti_short": short_summary,
            "wti_medium": medium,
            "wti_long": longterm,
            "diesel_short": diesel_short,
            "diesel_medium": diesel_medium,
        },
        "weights": bundle.weights,
        "narrative": narrative(current_wti, short_summary, medium, longterm, bundle.weights),
        "history": {
            "wti": dataframe_records(wti, 400),
            "brent": dataframe_records(brent, 400),
            "diesel": dataframe_records(diesel, 400),
            "short_path": dataframe_records(short_path),
        },
        "events": dataframe_records(impacts.head(40)),
        "disclaimer": "本系统仅使用经来源与新鲜度校验的真实观测数据；任一关键数据源缺失或过期时任务失败，不使用模拟、随机生成或隐式代理数据。预测区间、概率与事件影响均为统计模型估算。",
    }

    with connect() as con:
        for df, table, series, source in [
            (wti, "prices", "wti", src_wti), (brent, "prices", "brent", src_brent),
            (dxy, "prices", "dxy", src_dxy), (cny, "prices", "cnyusd", src_cny),
            (diesel, "prices", "china_diesel", src_diesel),
        ]:
            upsert_series(con, df, table, series, source)
        for df, series, source in [(us10y, "us10y", src_10y), (cpi, "cpi", src_cpi), (payroll, "payroll", src_pay), (broad_usd, "broad_usd", src_busd)]:
            upsert_series(con, df.rename(columns={"value": "price"}), "macro", series, source)
        save_events(con, impacts)
        for feature, weight in bundle.weights.items():
            con.execute("INSERT OR REPLACE INTO model_weights(report_date,feature,weight) VALUES(?,?,?)", (report_date, feature, float(weight)))
        for horizon, f in [("short", short_summary), ("medium", medium)]:
            con.execute("INSERT OR REPLACE INTO forecasts VALUES(?,?,?,?,?,?,?,?,?)", (report_date, horizon, "WTI", f["mean"], f["low"], f["high"], f["bull_probability"], f["bear_probability"], f["model"]))
        save_report(con, report_date, payload)
        con.commit()

    features.to_csv(ROOT / "data/features/latest_features.csv", index=False)
    atomic_json_dump(payload, ROOT / "reports" / f"{report_date}.json")
    atomic_json_dump(payload, ROOT / "reports" / "latest.json")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    report = run_pipeline()
    if args.print_summary:
        print(report["report_date"], report["latest"], report["forecasts"]["wti_short"])
