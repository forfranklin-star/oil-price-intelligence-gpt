from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import pandas as pd
import requests

BULLISH = {
    "attack": 2.0, "war": 2.2, "sanction": 1.8, "strike": 1.3, "outage": 1.6,
    "cut": 1.3, "disruption": 1.7, "tension": 1.2, "conflict": 1.8, "embargo": 2.0,
    "inventory draw": 0.9, "demand rises": 0.8,
}
BEARISH = {
    "ceasefire": -1.3, "peace": -1.0, "output rise": -1.2, "production increase": -1.2,
    "inventory build": -0.9, "demand weak": -1.0, "recession": -1.3, "oversupply": -1.4,
    "quota increase": -1.1,
}
INSTITUTIONS = ["goldman", "jpmorgan", "ubs", "iea", "eia", "opec", "morgan stanley", "barclays"]


class NewsProvider:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session = requests.Session()
        self.headers = {"User-Agent": cfg["data"]["user_agent"]}

    @staticmethod
    def score_text(text: str) -> tuple[float, float, str]:
        t = text.lower()
        geo = 0.0
        for k, v in {**BULLISH, **BEARISH}.items():
            if k in t:
                geo += v
        inst = 0.0
        if any(x in t for x in INSTITUTIONS):
            if any(x in t for x in ["raise forecast", "bullish", "upgrade", "higher oil", "price target raised"]):
                inst += 1.0
            if any(x in t for x in ["cut forecast", "bearish", "downgrade", "lower oil", "price target cut"]):
                inst -= 1.0
        direction = "看涨" if geo + inst > 0.2 else "看跌" if geo + inst < -0.2 else "中性"
        return geo, inst, direction

    @staticmethod
    def _parse_rss(xml_text: str) -> list[dict]:
        root = ET.fromstring(xml_text)
        rows = []
        for item in root.findall(".//item"):
            def txt(tag: str) -> str:
                node = item.find(tag)
                return node.text.strip() if node is not None and node.text else ""
            rows.append({
                "title": txt("title"),
                "link": txt("link"),
                "published": txt("pubDate"),
                "description": txt("description"),
                "source": txt("source") or "Google News",
            })
        return rows

    def fetch(self, lookback_days: int = 30) -> tuple[pd.DataFrame, str]:
        rows = []
        try:
            for q in self.cfg["news"]["queries"]:
                url = f'{self.cfg["data"]["google_news_rss"]}?q={quote_plus(q + " when:" + str(lookback_days) + "d")}&hl=en-US&gl=US&ceid=US:en'
                r = self.session.get(url, timeout=self.cfg["data"]["request_timeout_seconds"], headers=self.headers)
                r.raise_for_status()
                for e in self._parse_rss(r.text)[:30]:
                    published = pd.to_datetime(e.get("published") or datetime.now(timezone.utc), utc=True, errors="coerce")
                    if pd.isna(published):
                        published = pd.Timestamp.now(tz="UTC")
                    published = published.tz_convert(None)
                    title = e.get("title", "")
                    summary = e.get("description", "")
                    geo, inst, direction = self.score_text(title + " " + summary)
                    rows.append({
                        "published": published,
                        "title": title,
                        "link": e.get("link", ""),
                        "source": e.get("source", "Google News"),
                        "geopolitical_score": geo,
                        "institution_score": inst,
                        "direction": direction,
                    })
            if not rows:
                raise ValueError("no news")
            df = pd.DataFrame(rows).drop_duplicates(subset=["title"]).sort_values("published", ascending=False)
            return df, "google_news_rss"
        except Exception:
            now = pd.Timestamp.today().normalize()
            demo = [
                ("OPEC+ signals cautious supply policy amid demand uncertainty", 0.8, 0.2),
                ("Major bank keeps crude outlook broadly unchanged", 0.0, 0.0),
                ("Shipping risk premium remains elevated on key trade route", 1.3, 0.0),
            ]
            return pd.DataFrame([
                {"published": now - pd.Timedelta(days=i), "title": t, "link": "", "source": "demo", "geopolitical_score": g, "institution_score": s, "direction": "看涨" if g+s>0.2 else "中性"}
                for i, (t, g, s) in enumerate(demo)
            ]), "synthetic_fallback"
