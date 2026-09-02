import pandas as pd

from src.providers.market import MarketProvider


def test_eia_mapping_works_for_configured_market_symbol(monkeypatch):
    cfg = {
        "data": {
            "user_agent": "test", "market_history_days": 30,
            "request_timeout_seconds": 1, "request_delay_seconds": 0,
            "min_market_rows": 1, "max_market_staleness_days": 99,
            "allow_eia_fallback": True, "yahoo_chart_base": "x",
            "eia_api_base": "x", "eia_api_key": "k",
        },
        "symbols": {"wti": "CL=F", "brent": "BZ=F", "eia": {"wti": "RWTC", "brent": "RBRTE"}},
    }
    p = MarketProvider(cfg)
    monkeypatch.setattr(p, "_fetch_yahoo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    seen = {}
    def fake_eia(series_id, days):
        seen["series"] = series_id
        return pd.DataFrame({"date": [pd.Timestamp.today().normalize()], "price": [80.0]})
    monkeypatch.setattr(p, "_fetch_eia", fake_eia)
    df, src = p.history("CL=F", 10)
    assert seen["series"] == "RWTC"
    assert src == "eia_spot:RWTC"
