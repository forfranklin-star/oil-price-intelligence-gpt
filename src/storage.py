from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from src.utils import ROOT

DB_PATH = ROOT / "data" / "oil_intelligence.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
  date TEXT NOT NULL, series TEXT NOT NULL, value REAL, source TEXT,
  PRIMARY KEY(date, series)
);
CREATE TABLE IF NOT EXISTS macro (
  date TEXT NOT NULL, series TEXT NOT NULL, value REAL, source TEXT,
  PRIMARY KEY(date, series)
);
CREATE TABLE IF NOT EXISTS events (
  published TEXT, title TEXT PRIMARY KEY, source TEXT, link TEXT,
  geopolitical_score REAL, institution_score REAL, direction TEXT
);
CREATE TABLE IF NOT EXISTS forecasts (
  report_date TEXT, horizon TEXT, asset TEXT, mean REAL, low REAL, high REAL,
  bull_probability REAL, bear_probability REAL, model TEXT,
  PRIMARY KEY(report_date, horizon, asset)
);
CREATE TABLE IF NOT EXISTS model_weights (
  report_date TEXT, feature TEXT, weight REAL,
  PRIMARY KEY(report_date, feature)
);
CREATE TABLE IF NOT EXISTS reports (
  report_date TEXT PRIMARY KEY, payload_json TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def upsert_series(con, df: pd.DataFrame, table: str, series: str, source: str, value_col: str = "price") -> None:
    x = df[["date", value_col]].copy().rename(columns={value_col: "value"})
    x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")
    x["series"] = series
    x["source"] = source
    con.executemany(
        f"INSERT OR REPLACE INTO {table}(date, series, value, source) VALUES(?,?,?,?)",
        x[["date", "series", "value", "source"]].itertuples(index=False, name=None),
    )


def save_events(con, events: pd.DataFrame) -> None:
    rows = []
    for _, r in events.iterrows():
        rows.append((str(r["published"]), r["title"], r.get("source", ""), r.get("link", ""), float(r.get("geopolitical_score", 0)), float(r.get("institution_score", 0)), r.get("direction", "中性")))
    con.executemany("INSERT OR REPLACE INTO events VALUES(?,?,?,?,?,?,?)", rows)


def save_report(con, report_date: str, payload: dict) -> None:
    con.execute("INSERT OR REPLACE INTO reports(report_date,payload_json) VALUES(?,?)", (report_date, json.dumps(payload, ensure_ascii=False)))


def list_report_dates() -> list[str]:
    if not DB_PATH.exists():
        return []
    with connect() as con:
        return [r[0] for r in con.execute("SELECT report_date FROM reports ORDER BY report_date DESC").fetchall()]


def load_report(report_date: str) -> dict | None:
    if not DB_PATH.exists():
        return None
    with connect() as con:
        row = con.execute("SELECT payload_json FROM reports WHERE report_date=?", (report_date,)).fetchone()
    return json.loads(row[0]) if row else None
