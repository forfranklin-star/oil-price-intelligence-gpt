import pandas as pd
import pytest

from src.providers.diesel import ChinaDieselProvider, DieselDataError, SourceResult


def _cfg(min_rows=3, max_age=10):
    return {
        "data": {
            "user_agent": "test",
            "request_timeout_seconds": 1,
            "request_delay_seconds": 0,
            "min_diesel_rows": min_rows,
            "max_diesel_staleness_days": max_age,
            "china_diesel_csv_urls": [],
        },
        "diesel_sources": {
            "training_priority": ["longzhong", "business_society", "sina"],
            "crosscheck_max_gap_pct": 15,
        },
    }


def _frame(prices, provider, metric):
    today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    dates = pd.date_range(end=today, periods=len(prices), freq="D")
    return pd.DataFrame({
        "date": dates,
        "price": prices,
        "provider": provider,
        "server": "example.com",
        "metric": metric,
        "source_url": "https://example.com",
        "retrieved_at": "2026-09-02T00:00:00+00:00",
    })


def test_longzhong_article_parser_extracts_real_table_rows():
    html = """
    <table>
      <thead><tr><th>市场</th><th>规格</th><th>2026/08/13</th><th>2026/08/14</th><th>涨跌值</th></tr></thead>
      <tbody>
        <tr><td>中国</td><td>汽油</td><td>8785</td><td>8743</td><td>-42</td></tr>
        <tr><td>中国</td><td>柴油</td><td>7594</td><td>7543</td><td>-51</td></tr>
      </tbody>
    </table>
    """
    df = ChinaDieselProvider._longzhong_article_prices(html)
    assert list(df["price"]) == [7594.0, 7543.0]
    assert str(df.iloc[-1]["date"].date()) == "2026-08-14"


def test_sina_parser_uses_tonne_value_and_does_not_convert_litre():
    html = """
      <html><body>2026年08月28日
      <table><tr><th>品名规格</th><th>最高批发价</th><th>最高零售价</th><th>升价</th></tr>
      <tr><td>0号柴油（Ⅵ）</td><td>8710</td><td>9010</td><td>7.75</td></tr></table>
      </body></html>
    """
    df = ChinaDieselProvider._extract_sina_article(html, "https://finance.sina.com.cn/wm/2026-08-28/x.shtml")
    assert float(df.iloc[0]["price"]) == 9010.0


def test_failover_selects_second_complete_series_without_splicing(monkeypatch):
    p = ChinaDieselProvider(_cfg(min_rows=3))
    short = SourceResult("longzhong", "oilchem.net", "national_mainstream_wholesale", _frame([7500, 7510], "longzhong", "national_mainstream_wholesale"), "u1")
    backup = SourceResult("business_society", "100ppi.com", "benchmark_wholesale", _frame([7400, 7420, 7440], "business_society", "benchmark_wholesale"), "u2")
    monkeypatch.setattr(p, "collect_all", lambda: [short, backup])
    df, src = p.history()
    assert src.startswith("business_society:")
    assert len(df) == 3
    assert set(df["provider"]) == {"business_society"}


def test_crosscheck_rejects_implausible_wholesale_gap(monkeypatch):
    p = ChinaDieselProvider(_cfg(min_rows=3))
    primary = SourceResult("longzhong", "oilchem.net", "national_mainstream_wholesale", _frame([7500, 7520, 7540], "longzhong", "national_mainstream_wholesale"), "u1")
    peer = SourceResult("business_society", "100ppi.com", "benchmark_wholesale", _frame([11000, 11020, 11040], "business_society", "benchmark_wholesale"), "u2")
    monkeypatch.setattr(p, "collect_all", lambda: [primary, peer])
    with pytest.raises(DieselDataError):
        p.history()


def test_no_synthetic_fallback_when_all_sources_missing(monkeypatch):
    p = ChinaDieselProvider(_cfg(min_rows=3))
    monkeypatch.setattr(p, "collect_all", lambda: [])
    with pytest.raises(DieselDataError, match="No verified real domestic diesel series"):
        p.history()
