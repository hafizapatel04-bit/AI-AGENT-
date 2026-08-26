"""
sources.py
----------
Registry of data "sources" the ETL pipeline aggregates per query.

Each source is a small connector function that returns a DataFrame slice.
In this project the sources are simulated by filtering the unified
transactions dataset by its `source` column (12 distinct feeds), which
mirrors how the real system would work if each connector hit a different
API/marketplace/CRM export -- swap `_load_base()` for real HTTP/DB calls
and the rest of the pipeline is unchanged.

To wire in a REAL external API, add an entry to LIVE_SOURCES with a
callable that returns a DataFrame with at least the columns:
    timestamp, source, region, category, revenue, quantity
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "transactions.csv"

# The 12 simulated feeds (>= 10 web sources requirement)
SOURCE_NAMES = [
    "web_store_us", "web_store_eu", "web_store_apac", "mobile_app_ios",
    "mobile_app_android", "marketplace_amazon", "marketplace_etsy",
    "social_instagram_ads", "social_tiktok_ads", "affiliate_network",
    "email_campaigns", "retail_pos_partners",
]


def _load_base() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from data.sample_data_generator import generate
        df = generate()
        df.to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


def fetch_source(name: str) -> pd.DataFrame:
    """Simulated connector: pulls the slice of the base dataset for one feed."""
    df = _load_base()
    return df[df["source"] == name].copy()


# Example of a REAL external connector a user could plug in (kept inert/off
# by default so the app runs with zero network dependencies out of the box).
def fetch_live_example(url: str) -> pd.DataFrame:
    """Template for a real HTTP-backed source. Not called unless enabled."""
    import requests  # local import: optional dependency
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


LIVE_SOURCES: dict[str, Callable[[], pd.DataFrame]] = {
    # "my_api": lambda: fetch_live_example("https://api.example.com/sales"),
}


def aggregate_all_sources(source_names: list[str] | None = None) -> pd.DataFrame:
    """Pulls and concatenates every configured source into one frame."""
    names = source_names or SOURCE_NAMES
    frames = [fetch_source(n) for n in names]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
