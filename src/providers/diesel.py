from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
import re
from typing import Callable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils import polite_get


class DieselDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceResult:
    name: str
    server: str
    metric: str
    frame: pd.DataFrame
    url: str
    status: str = "ok"


class ChinaDieselProvider:
    """Multi-source real-data provider for China diesel.

    Policy:
    - Never synthesize or interpolate a missing market observation.
    - Do not splice incompatible price definitions into one training target.
    - Choose one complete source series by priority; other servers are used for
      independent cross-checks and failover.
    - Public pages are fetched politely. No CAPTCHA/login/anti-bot bypass exists.

    Normalized frame returned to the model contains ``date, price`` plus source
    metadata. ``price`` is always an observed RMB/metric-ton value.
    """

    def __init__(self, cfg: dict, session: requests.Session | None = None):
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        self.diesel_cfg = cfg.get("diesel_sources", {})
        self.session = session or requests.Session()
        self.headers = {
            "User-Agent": self.data_cfg["user_agent"],
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
        }
        self.diagnostics: list[dict] = []

    def _get(self, url: str) -> requests.Response:
        r = polite_get(
            self.session,
            url,
            timeout=int(self.data_cfg["request_timeout_seconds"]),
            delay=float(self.data_cfg["request_delay_seconds"]),
            headers=self.headers,
        )
        r.raise_for_status()
        return r

    @staticmethod
    def _clean_price(value) -> float | None:
        s = str(value).replace(",", "").replace("，", "").strip()
        m = re.search(r"(?<!\d)([3-9]\d{3}(?:\.\d+)?|1\d{4}(?:\.\d+)?)(?!\d)", s)
        if not m:
            return None
        v = float(m.group(1))
        return v if 3000 <= v <= 20000 else None

    @staticmethod
    def _parse_dates(values: pd.Series) -> pd.Series:
        now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
        out = []
        for raw in values.astype(str):
            s = raw.strip().replace("年", "-").replace("月", "-").replace("日", "")
            # MM-DD pages need the current/previous year resolved without future dates.
            if re.fullmatch(r"\d{1,2}[-/]\d{1,2}", s):
                mm, dd = map(int, re.split(r"[-/]", s))
                candidate = pd.Timestamp(year=now.year, month=mm, day=dd)
                if candidate > now + pd.Timedelta(days=3):
                    candidate = candidate.replace(year=now.year - 1)
                out.append(candidate)
            else:
                out.append(pd.to_datetime(s, errors="coerce"))
        return pd.Series(out, index=values.index)

    def _normalize(self, df: pd.DataFrame, *, name: str, server: str, metric: str, url: str) -> SourceResult:
        if df.empty or not {"date", "price"}.issubset(df.columns):
            raise DieselDataError(f"{name}: no usable date/price observations")
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
        out = out.dropna(subset=["date", "price"])
        out = out[(out["price"] >= 3000) & (out["price"] <= 20000)]
        out = out.drop_duplicates("date", keep="last").sort_values("date")
        if out.empty:
            raise DieselDataError(f"{name}: all observations failed validation")
        out["provider"] = name
        out["server"] = server
        out["metric"] = metric
        out["source_url"] = url
        out["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return SourceResult(name, server, metric, out, url)

    def _tables(self, html: str) -> list[pd.DataFrame]:
        try:
            return pd.read_html(StringIO(html))
        except ValueError:
            return []

    @staticmethod
    def _longzhong_article_prices(html: str) -> pd.DataFrame:
        """Extract 中国/柴油 daily observations from a public 隆众 article table."""
        rows = []
        try:
            tables = pd.read_html(StringIO(html))
        except ValueError:
            tables = []
        for t in tables:
            if t.empty:
                continue
            t = t.copy()
            # Flatten MultiIndex headers while preserving dates.
            if isinstance(t.columns, pd.MultiIndex):
                t.columns = [" ".join(str(x) for x in c if str(x) != "nan") for c in t.columns]
            else:
                t.columns = [str(c) for c in t.columns]
            if t.shape[1] < 3:
                continue
            first = t.iloc[:, 0].astype(str).replace("nan", np.nan).ffill()
            second = t.iloc[:, 1].astype(str)
            mask = first.str.contains("中国", na=False) & second.str.contains("柴油", na=False)
            if not mask.any():
                continue
            row = t.loc[mask].iloc[-1]
            for col, val in row.items():
                m = re.search(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", str(col))
                if not m:
                    continue
                price = ChinaDieselProvider._clean_price(val)
                if price is not None:
                    rows.append({"date": pd.Timestamp(*map(int, m.groups())), "price": price})
        # Text fallback matches a line-like sequence around 中国 柴油 and dated values.
        if not rows:
            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            date_tokens = re.findall(r"20\d{2}[/-]\d{1,2}[/-]\d{1,2}", text)
            m = re.search(r"中国\s*[^。]{0,50}?柴油\s*([3-9]\d{3}(?:\.\d+)?)\s*([3-9]\d{3}(?:\.\d+)?)", text)
            if m and len(date_tokens) >= 2:
                rows.extend([{"date": pd.to_datetime(date_tokens[-2]), "price": float(m.group(1))}, {"date": pd.to_datetime(date_tokens[-1]), "price": float(m.group(2))}])
        return pd.DataFrame(rows, columns=["date", "price"])

    def _fetch_longzhong(self) -> SourceResult:
        cfg = self.diesel_cfg.get("longzhong", {})
        url = cfg.get("url", "https://oil.oilchem.net/445/")
        r = self._get(url)
        frames: list[pd.DataFrame] = []
        for t in self._tables(r.text):
            if t.shape[1] < 2:
                continue
            cols = [str(c) for c in t.columns]
            date_idx = next((i for i, c in enumerate(cols) if "日期" in c), None)
            price_idx = next((i for i, c in enumerate(cols) if any(k in c for k in ["报价", "价格"])), None)
            if date_idx is None or price_idx is None:
                continue
            x = pd.DataFrame({"date": self._parse_dates(t.iloc[:, date_idx]), "price": t.iloc[:, price_idx].map(self._clean_price)})
            frames.append(x)

        # Public 7-day page is not enough for an initial model. Backfill from
        # public 隆众 article list pages, but never enter login/subscriber pages.
        target_rows = int(self.data_cfg.get("min_diesel_rows", 30))
        if sum(len(x) for x in frames) < target_rows and cfg.get("backfill_public_articles", True):
            archive_tmpl = cfg.get("archive_url_template", "https://list.oilchem.net/445/{page}.html")
            max_pages = int(cfg.get("archive_pages", 8))
            max_articles = int(cfg.get("max_article_fetches", 55))
            links = []
            candidate_pages = [url] + [archive_tmpl.format(page=page) for page in range(1, max_pages + 1)]
            for page_url in candidate_pages:
                try:
                    lr = self._get(page_url)
                    soup = BeautifulSoup(lr.text, "html.parser")
                    for a in soup.find_all("a", href=True):
                        title = a.get_text(" ", strip=True)
                        href = urljoin(lr.url, a["href"])
                        if "oilchem.net" not in href:
                            continue
                        # Only public oil-product articles are candidates.
                        if not any(k in title for k in ["成品油", "汽柴", "柴油"]):
                            continue
                        if not any(k in title for k in ["早间", "提示", "预期", "价格", "日评", "汇总", "汽柴"]):
                            continue
                        if href not in links:
                            links.append(href)
                        if len(links) >= max_articles:
                            break
                except Exception as exc:
                    self.diagnostics.append({"source": "longzhong_archive", "status": "page_failed", "detail": str(exc), "page": page_url})
                if len(links) >= max_articles:
                    break
            for href in links[:max_articles]:
                try:
                    ar = self._get(href)
                    x = self._longzhong_article_prices(ar.text)
                    if not x.empty:
                        frames.append(x)
                    merged_rows = len(pd.concat(frames, ignore_index=True).drop_duplicates("date")) if frames else 0
                    if merged_rows >= target_rows:
                        break
                except Exception as exc:
                    self.diagnostics.append({"source": "longzhong_archive", "status": "article_failed", "detail": str(exc), "url": href})
        if not frames:
            raise DieselDataError("longzhong: national diesel price table not found")
        df = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
        return self._normalize(df, name="longzhong", server="oilchem.net", metric="national_mainstream_wholesale", url=url)

    def _fetch_100ppi(self) -> SourceResult:
        cfg = self.diesel_cfg.get("business_society", {})
        url = cfg.get("url", "https://m1.100ppi.com/vane/490-%E6%9F%B4%E6%B2%B9")
        r = self._get(url)
        frames = []
        for t in self._tables(r.text):
            if t.shape[1] < 2:
                continue
            cols = [str(c) for c in t.columns]
            date_idx = next((i for i, c in enumerate(cols) if "日期" in c), None)
            price_idx = next((i for i, c in enumerate(cols) if "价格" in c), None)
            if date_idx is None or price_idx is None:
                continue
            x = pd.DataFrame({"date": self._parse_dates(t.iloc[:, date_idx]), "price": t.iloc[:, price_idx].map(self._clean_price)})
            frames.append(x)
        if not frames:
            raise DieselDataError("100ppi: diesel benchmark table not found")
        df = max(frames, key=len)
        return self._normalize(df, name="business_society", server="100ppi.com", metric="benchmark_wholesale", url=url)

    @staticmethod
    def _extract_sina_article(html: str, url: str) -> pd.DataFrame:
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        # Prefer tonne observations; per-litre figures are intentionally not converted.
        date_match = re.search(r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})", text)
        if not date_match:
            date_match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", url)
        date = pd.Timestamp(*map(int, date_match.groups())) if date_match else pd.NaT
        patterns = [
            r"0\s*[号#]\s*柴油[^。；\n]{0,120}?(?:最高零售价|零售价|吨价)[^\d]{0,20}([3-9]\d{3,4}(?:\.\d+)?)\s*元\s*/?\s*吨",
            r"0\s*[号#]\s*柴油[^。；\n]{0,80}?([3-9]\d{3,4}(?:\.\d+)?)\s*元\s*/?\s*吨",
            r"柴油（?标准品）?[^。；\n]{0,80}?([3-9]\d{3,4}(?:\.\d+)?)\s*元\s*/?\s*吨",
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return pd.DataFrame({"date": [date], "price": [float(m.group(1))]})
        # HTML tables often separate product name and tonne price into cells.
        try:
            for t in pd.read_html(StringIO(html)):
                for _, row in t.iterrows():
                    joined = " ".join(map(str, row.tolist()))
                    if re.search(r"0\s*[号#]\s*柴油", joined):
                        vals = [ChinaDieselProvider._clean_price(v) for v in row.tolist()]
                        vals = [v for v in vals if v is not None]
                        if vals:
                            # Retail tables usually list wholesale columns first and retail tonne last;
                            # use the maximum as the max-retail tonne observation.
                            return pd.DataFrame({"date": [date], "price": [max(vals)]})
        except ValueError:
            pass
        return pd.DataFrame(columns=["date", "price"])

    def _fetch_sina(self) -> SourceResult:
        cfg = self.diesel_cfg.get("sina", {})
        urls = cfg.get("article_urls", [])
        if isinstance(urls, str):
            urls = [urls]
        rows = []
        used = []
        for url in urls:
            try:
                r = self._get(url)
                x = self._extract_sina_article(r.text, url)
                if not x.empty:
                    rows.append(x)
                    used.append(url)
            except Exception as exc:
                self.diagnostics.append({"source": "sina", "status": "error", "detail": str(exc), "url": url})
        if not rows:
            raise DieselDataError("sina: no configured article yielded a tonne diesel observation")
        df = pd.concat(rows, ignore_index=True)
        return self._normalize(df, name="sina", server="finance.sina.com.cn", metric="provincial_max_retail", url=used[-1])

    def _fetch_zhuochuang(self) -> SourceResult:
        """Parse explicit public SCI99 URLs when they contain a diesel RMB/t observation.

        SCI99 has a mixture of free and subscriber-only pages. We do not bypass
        access controls. URLs are configurable because public article paths change.
        """
        cfg = self.diesel_cfg.get("zhuochuang", {})
        urls = cfg.get("public_urls", [])
        if isinstance(urls, str):
            urls = [urls]
        rows = []
        used = []
        for url in urls:
            r = self._get(url)
            text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
            d = re.search(r"(20\d{2})[-年](\d{1,2})[-月](\d{1,2})", text)
            date = pd.Timestamp(*map(int, d.groups())) if d else pd.NaT
            # Only accept an explicit diesel price, not SCI99's broader petroleum index.
            m = re.search(r"柴油[^。；\n]{0,100}?([3-9]\d{3,4}(?:\.\d+)?)\s*元\s*/?\s*吨", text)
            if m:
                rows.append(pd.DataFrame({"date": [date], "price": [float(m.group(1))]}))
                used.append(url)
        if not rows:
            raise DieselDataError("zhuochuang: no public configured page exposed an explicit diesel RMB/t price")
        return self._normalize(pd.concat(rows), name="zhuochuang", server="sci99.com", metric="industry_wholesale", url=used[-1])

    def _fetch_csv(self) -> SourceResult:
        urls = self.data_cfg.get("china_diesel_csv_urls", [])
        if isinstance(urls, str):
            urls = [urls] if urls.strip() else []
        errors = []
        for url in urls:
            try:
                r = self._get(url)
                df = pd.read_csv(StringIO(r.text))
                if not {"date", "price"}.issubset(df.columns):
                    raise DieselDataError("CSV must contain date,price")
                return self._normalize(df[["date", "price"]], name="configured_csv", server=url.split('/')[2], metric="configured_real_series", url=url)
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        raise DieselDataError("configured_csv: " + " | ".join(errors or ["no URLs configured"]))

    def _validate_candidate(self, result: SourceResult, *, for_training: bool) -> SourceResult:
        df = result.frame
        min_rows = int(self.data_cfg["min_diesel_rows"] if for_training else 1)
        if len(df) < min_rows:
            raise DieselDataError(f"{result.name}: {len(df)} rows < required {min_rows}")
        latest = pd.Timestamp(df["date"].max()).normalize()
        today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
        age = int((today - latest).days)
        max_age = int(self.data_cfg["max_diesel_staleness_days"])
        if age > max_age:
            raise DieselDataError(f"{result.name}: latest observation is {age} days old; limit={max_age}")
        if age < -1:
            raise DieselDataError(f"{result.name}: future-dated observation {latest.date()}")
        return result

    def collect_all(self) -> list[SourceResult]:
        """Collect independent servers without requiring model-level row counts."""
        methods: list[tuple[str, Callable[[], SourceResult]]] = [
            ("longzhong", self._fetch_longzhong),
            ("business_society", self._fetch_100ppi),
            ("zhuochuang", self._fetch_zhuochuang),
            ("sina", self._fetch_sina),
            ("configured_csv", self._fetch_csv),
        ]
        enabled = self.diesel_cfg.get("enabled", {})
        results = []
        for name, fn in methods:
            if enabled and not enabled.get(name, True):
                continue
            try:
                result = fn()
                result = self._validate_candidate(result, for_training=False)
                results.append(result)
                self.diagnostics.append({"source": name, "server": result.server, "status": "ok", "rows": len(result.frame), "latest": str(result.frame['date'].max().date()), "metric": result.metric})
            except Exception as exc:
                self.diagnostics.append({"source": name, "status": "failed", "detail": str(exc)})
        return results

    def history(self, *_args) -> tuple[pd.DataFrame, str]:
        results = self.collect_all()
        priorities = self.diesel_cfg.get("training_priority", ["longzhong", "business_society", "zhuochuang", "configured_csv", "sina"])
        by_name = {r.name: r for r in results}
        errors = []
        for name in priorities:
            r = by_name.get(name)
            if not r:
                continue
            try:
                r = self._validate_candidate(r, for_training=True)
                df = r.frame.copy()
                # Cross-check only against same/nearby wholesale metric; do not blend.
                peers = [p for p in results if p.name != r.name and "wholesale" in p.metric]
                checks = []
                if peers:
                    primary_latest = float(df.iloc[-1]["price"])
                    for p in peers:
                        peer_latest = float(p.frame.iloc[-1]["price"])
                        gap = abs(peer_latest / primary_latest - 1)
                        checks.append({"peer": p.name, "gap_pct": round(gap * 100, 3)})
                    limit = float(self.diesel_cfg.get("crosscheck_max_gap_pct", 15.0)) / 100.0
                    if checks and min(c["gap_pct"] for c in checks) / 100.0 > limit:
                        raise DieselDataError(f"{r.name}: all wholesale cross-checks exceed {limit*100:.1f}%: {checks}")
                df.attrs["diagnostics"] = self.diagnostics
                df.attrs["crosscheck"] = checks
                return df, f"{r.name}:{r.server}:{r.metric}"
            except Exception as exc:
                errors.append(str(exc))
                self.diagnostics.append({"source": name, "status": "rejected_for_training", "detail": str(exc)})
        detail = " | ".join(errors) if errors else "no source met training requirements"
        raise DieselDataError("No verified real domestic diesel series passed failover policy. " + detail + f" diagnostics={self.diagnostics}")
