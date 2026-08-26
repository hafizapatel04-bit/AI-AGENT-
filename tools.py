"""
tools.py
--------
Defines the tools exposed to the LLM in the tool-calling loop, plus the
Python functions that actually execute them against the current DataFrame.

Each tool is declared in the Ollama/OpenAI-style JSON schema format and
paired with a Python implementation in TOOL_IMPLEMENTATIONS. The agent
loop (agent_loop.py) is what wires "the model asked for tool X" to
"actually run TOOL_IMPLEMENTATIONS[X]".
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd

from pipeline.anomaly import daily_revenue_anomalies, detect_anomalies

# ---------------------------------------------------------------------------
# Tool schemas (passed to the LLM so it knows what it can call)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_stats",
            "description": "Get aggregate stats (sum/mean/count of revenue, quantity, rating) "
                            "optionally grouped by a column such as source, region, category, or channel_type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "description": "Column to group by: source, region, category, channel_type, or 'none'.",
                    }
                },
                "required": ["group_by"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_n",
            "description": "Get the top N groups by total revenue for a given column (e.g. top 5 categories).",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string", "description": "Column to rank: source, region, category, channel_type."},
                    "n": {"type": "integer", "description": "How many top rows to return."},
                    "metric": {"type": "string", "description": "revenue, quantity, or customer_rating"},
                },
                "required": ["group_by", "n"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_data_anomalies",
            "description": "Run the anomaly-detection pipeline and return flagged transactions "
                            "(pricing bugs, revenue/quantity outliers, high-return low-rating clusters).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_revenue_trend",
            "description": "Get the daily revenue trend and flag days where revenue deviated sharply "
                            "from its recent rolling average (real-time drift/spike detection).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_records",
            "description": "Filter the dataset by source, region, or category and return summary counts/revenue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "region": {"type": "string"},
                    "category": {"type": "string"},
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations (executed locally; results fed back to the LLM)
# ---------------------------------------------------------------------------

def _get_summary_stats(df: pd.DataFrame, group_by: str = "none") -> str:
    if group_by in (None, "none", ""):
        out = {
            "total_revenue": round(df["revenue"].sum(), 2),
            "total_orders": int(len(df)),
            "avg_order_value": round(df["revenue"].mean(), 2),
            "avg_rating": round(df["customer_rating"].mean(), 2),
            "return_rate_pct": round(df["returned"].mean() * 100, 2),
        }
        return json.dumps(out)

    g = df.groupby(group_by).agg(
        total_revenue=("revenue", "sum"),
        orders=("revenue", "count"),
        avg_rating=("customer_rating", "mean"),
        return_rate_pct=("returned", "mean"),
    ).reset_index()
    g["total_revenue"] = g["total_revenue"].round(2)
    g["avg_rating"] = g["avg_rating"].round(2)
    g["return_rate_pct"] = (g["return_rate_pct"] * 100).round(2)
    g = g.sort_values("total_revenue", ascending=False)
    return g.to_json(orient="records")


def _get_top_n(df: pd.DataFrame, group_by: str, n: int = 5, metric: str = "revenue") -> str:
    n = int(n)
    metric = metric if metric in ("revenue", "quantity", "customer_rating") else "revenue"
    g = df.groupby(group_by)[metric].sum().sort_values(ascending=False).head(n)
    return g.round(2).to_json()


def _detect_data_anomalies(df: pd.DataFrame) -> str:
    flagged = detect_anomalies(df)
    if flagged.empty:
        return json.dumps({"anomaly_count": 0, "records": []})
    cols = ["record_id", "source", "region", "category", "revenue", "quantity", "flag_reason"]
    return json.dumps({
        "anomaly_count": int(len(flagged)),
        "records": json.loads(flagged[cols].head(15).to_json(orient="records")),
    })


def _get_daily_revenue_trend(df: pd.DataFrame) -> str:
    daily = daily_revenue_anomalies(df)
    daily = daily.copy()
    daily["date"] = daily["date"].astype(str)
    return daily.round(2).to_json(orient="records")


def _filter_records(df: pd.DataFrame, source: str = None, region: str = None, category: str = None) -> str:
    filtered = df
    if source:
        filtered = filtered[filtered["source"] == source]
    if region:
        filtered = filtered[filtered["region"] == region]
    if category:
        filtered = filtered[filtered["category"] == category]
    return json.dumps({
        "matching_records": int(len(filtered)),
        "total_revenue": round(float(filtered["revenue"].sum()), 2),
        "avg_rating": round(float(filtered["customer_rating"].mean()), 2) if len(filtered) else None,
    })


def build_tool_implementations(df: pd.DataFrame) -> dict[str, Callable[..., str]]:
    """Bind each tool implementation to the current DataFrame."""
    return {
        "get_summary_stats": lambda **kwargs: _get_summary_stats(df, **kwargs),
        "get_top_n": lambda **kwargs: _get_top_n(df, **kwargs),
        "detect_data_anomalies": lambda **kwargs: _detect_data_anomalies(df),
        "get_daily_revenue_trend": lambda **kwargs: _get_daily_revenue_trend(df),
        "filter_records": lambda **kwargs: _filter_records(df, **kwargs),
    }
