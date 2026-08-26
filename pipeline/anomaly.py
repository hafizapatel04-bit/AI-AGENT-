"""
anomaly.py
----------
Statistical anomaly flagging for the analytics pipeline.

Uses a hybrid of:
  - Z-score on revenue/quantity per category (catches global outliers)
  - IQR fencing per source (catches source-specific pricing/volume glitches)
  - Rolling-window z-score on daily revenue (catches sudden real-time spikes/drops)

Returns a DataFrame of flagged rows plus a human-readable reason per flag,
so both the dashboard and the LLM agent can consume it directly.
"""

from __future__ import annotations

import pandas as pd


def _zscore_flags(df: pd.DataFrame, col: str, group_col: str, threshold: float = 3.0) -> pd.Series:
    grouped = df.groupby(group_col)[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, pd.NA)
    z = (df[col] - mean) / std
    return z.abs() > threshold


def _iqr_flags(df: pd.DataFrame, col: str, group_col: str, k: float = 3.0) -> pd.Series:
    def flag(group):
        q1, q3 = group.quantile(0.25), group.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        return (group < lower) | (group > upper)
    return df.groupby(group_col)[col].transform(flag)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the subset of df flagged as anomalous, with a `flag_reason` column."""
    if df.empty:
        return df.assign(flag_reason=pd.Series(dtype=str))

    working = df.copy()
    reasons = pd.Series([""] * len(working), index=working.index)

    rev_z = _zscore_flags(working, "revenue", "category")
    reasons[rev_z] += "revenue z-score outlier for category; "

    qty_iqr = _iqr_flags(working, "quantity", "source")
    reasons[qty_iqr] += "quantity IQR outlier for source; "

    price_iqr = _iqr_flags(working, "unit_price", "category")
    reasons[price_iqr] += "unit price IQR outlier for category (possible pricing bug); "

    return_flag = working["returned"] == 1
    high_return_and_low_rating = return_flag & (working["customer_rating"] <= 1)
    reasons[high_return_and_low_rating] += "return + rock-bottom rating; "

    working["flag_reason"] = reasons
    flagged = working[working["flag_reason"] != ""].copy()
    flagged["flag_reason"] = flagged["flag_reason"].str.rstrip("; ")
    return flagged.sort_values("timestamp", ascending=False)


def daily_revenue_anomalies(df: pd.DataFrame, window: int = 7, threshold: float = 2.5) -> pd.DataFrame:
    """Rolling-window anomaly detection over daily aggregated revenue (real-time drift/spikes)."""
    daily = df.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
    daily["rolling_mean"] = daily["revenue"].rolling(window, min_periods=3).mean()
    daily["rolling_std"] = daily["revenue"].rolling(window, min_periods=3).std()
    daily["z"] = (daily["revenue"] - daily["rolling_mean"]) / daily["rolling_std"]
    daily["is_anomaly"] = daily["z"].abs() > threshold
    return daily
