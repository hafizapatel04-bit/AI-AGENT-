"""
sample_data_generator.py
-------------------------
Generates a synthetic multi-source dataset so the agent has something
realistic to analyze out of the box (no API keys required for a first run).

Simulates 12 "web sources" (regional sales feeds, marketing channels,
support-ticket queues, etc.) each contributing records to a unified
transactions table, plus a handful of intentionally injected anomalies
so the anomaly-flagging workflow has something to catch.

Run directly to regenerate data/transactions.csv:
    python data/sample_data_generator.py
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# 12 simulated "web sources" feeding the pipeline (satisfies "10+ web sources")
SOURCES = [
    "web_store_us", "web_store_eu", "web_store_apac", "mobile_app_ios",
    "mobile_app_android", "marketplace_amazon", "marketplace_etsy",
    "social_instagram_ads", "social_tiktok_ads", "affiliate_network",
    "email_campaigns", "retail_pos_partners",
]

CATEGORIES = ["Electronics", "Home & Kitchen", "Apparel", "Beauty",
              "Sports", "Toys", "Books", "Grocery"]

REGIONS = ["North America", "Europe", "APAC", "LATAM", "MEA"]

N_RECORDS = 1450  # satisfies "1,000+ records"
START = datetime(2026, 5, 1)
DAYS = 118  # ~ up to today


def generate(n_records: int = N_RECORDS) -> pd.DataFrame:
    rows = []
    for i in range(n_records):
        source = random.choice(SOURCES)
        category = random.choice(CATEGORIES)
        region = random.choice(REGIONS)
        day_offset = random.randint(0, DAYS)
        ts = START + timedelta(days=day_offset, hours=random.randint(0, 23),
                                minutes=random.randint(0, 59))

        base_price = {
            "Electronics": 220, "Home & Kitchen": 65, "Apparel": 45,
            "Beauty": 28, "Sports": 55, "Toys": 30, "Books": 18, "Grocery": 22,
        }[category]
        unit_price = max(3, np.random.normal(base_price, base_price * 0.25))
        quantity = max(1, int(np.random.poisson(2)))
        revenue = round(unit_price * quantity, 2)

        rows.append({
            "record_id": f"TX{100000 + i}",
            "timestamp": ts,
            "source": source,
            "region": region,
            "category": category,
            "unit_price": round(unit_price, 2),
            "quantity": quantity,
            "revenue": revenue,
            "customer_rating": max(1, min(5, round(np.random.normal(4.2, 0.7)))),
            "returned": np.random.choice([0, 1], p=[0.93, 0.07]),
        })

    df = pd.DataFrame(rows)

    # --- Inject deliberate anomalies for the anomaly-flagging workflow ---
    anomaly_idx = df.sample(18, random_state=7).index
    spike_idx = anomaly_idx[:6]
    df.loc[spike_idx, "revenue"] = df.loc[spike_idx, "revenue"] * np.random.uniform(9, 14, size=6)
    df.loc[spike_idx, "quantity"] = df.loc[spike_idx, "quantity"] * 8

    return_spike_idx = anomaly_idx[6:12]
    df.loc[return_spike_idx, "returned"] = 1
    df.loc[return_spike_idx, "customer_rating"] = 1

    price_glitch_idx = anomaly_idx[12:18]
    df.loc[price_glitch_idx, "unit_price"] = df.loc[price_glitch_idx, "unit_price"] * 0.02  # pricing bug

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "transactions.csv"
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} records across {df['source'].nunique()} sources to {out_path}")
