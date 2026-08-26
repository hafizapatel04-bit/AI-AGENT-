"""
report_generator.py
--------------------
Automated report synthesis: runs the agent over a fixed set of analytical
prompts, stitches the answers into a structured Markdown report alongside
the concrete stats/anomaly tables, and saves it to disk.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from agent.agent_loop import run_agent
from agent.llm_client import DEFAULT_MODEL
from pipeline.anomaly import detect_anomalies

REPORT_QUESTIONS = [
    "Summarize overall performance: total revenue, orders, and average rating.",
    "Which sources and regions are driving the most revenue, and which are underperforming?",
    "What anomalies were detected, and what's the likely business explanation for each cluster?",
    "What should the team investigate or act on this week based on this data?",
]


def generate_report(df: pd.DataFrame, model: str = DEFAULT_MODEL) -> str:
    """Runs the agent across a fixed prompt set and assembles a Markdown report."""
    sections = []
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections.append(f"# Automated Analytics Report\n\n_Generated {generated_at} · model: {model}_\n")

    for question in REPORT_QUESTIONS:
        run = run_agent(question, df, model=model)
        sections.append(f"## {question}\n\n{run.answer}\n")

    flagged = detect_anomalies(df)
    sections.append("## Flagged Records (raw)\n")
    if flagged.empty:
        sections.append("No anomalies detected in the current dataset.\n")
    else:
        cols = ["record_id", "timestamp", "source", "category", "revenue", "flag_reason"]
        table = flagged[cols].head(20).to_markdown(index=False)
        sections.append(table + "\n")

    return "\n".join(sections)


def save_report(markdown_text: str, out_dir: str | Path = "reports/generated") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path = out_dir / filename
    path.write_text(markdown_text, encoding="utf-8")
    return path
