"""
etl.py
------
Extract -> Transform -> Load pipeline that turns raw per-source records
into a clean, analysis-ready DataFrame, plus a small SQLite "warehouse"
the agent's tools can query.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from pipeline.sources import SOURCE_NAMES, aggregate_all_sources

DB_PATH = Path(__file__).parent.parent / "data" / "warehouse.db"


def extract(source_names: list[str] | None = None) -> pd.DataFrame:
    """Pull raw records from every configured source."""
    return aggregate_all_sources(source_names)


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean, normalize, and enrich the raw multi-source records."""
    df = raw.copy()

    # Normalize dtypes
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    numeric_cols = ["unit_price", "quantity", "revenue", "customer_rating"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop hard nulls / dedupe on record_id
    df = df.dropna(subset=["revenue", "quantity", "timestamp"])
    df = df.drop_duplicates(subset=["record_id"])

    # Enrichment
    df["date"] = df["timestamp"].dt.date
    df["week"] = df["timestamp"].dt.isocalendar().week
    df["hour"] = df["timestamp"].dt.hour
    df["channel_type"] = df["source"].apply(_classify_channel)

    return df.reset_index(drop=True)


def _classify_channel(source: str) -> str:
    if source.startswith("web_store"):
        return "Owned Web"
    if source.startswith("mobile_app"):
        return "Mobile App"
    if source.startswith("marketplace"):
        return "Marketplace"
    if source.startswith("social"):
        return "Paid Social"
    if source in ("affiliate_network", "email_campaigns"):
        return "Marketing"
    return "Retail Partner"


def load(df: pd.DataFrame) -> None:
    """Persist the clean dataset into a lightweight SQLite warehouse."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql("transactions", conn, if_exists="replace", index=False)
    finally:
        conn.close()


def run_pipeline(source_names: list[str] | None = None) -> pd.DataFrame:
    """Full ETL run: extract -> transform -> load. Returns the clean frame."""
    raw = extract(source_names)
    clean = transform(raw)
    load(clean)
    return clean


def query_warehouse(sql: str) -> pd.DataFrame:
    """Run an arbitrary read-only SQL query against the warehouse (used by agent tools)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    clean = run_pipeline()
    print(f"ETL complete: {len(clean)} clean records from {clean['source'].nunique()} sources")
    print(clean.groupby("channel_type")["revenue"].sum().sort_values(ascending=False))
