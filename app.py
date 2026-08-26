"""
app.py
------
Streamlit dashboard for the autonomous AI analytics agent.

Run with:
    streamlit run app.py

Panels:
  1. Overview KPIs + interactive charts (Plotly)
  2. Real-time anomaly flagging table
  3. Chat with the agent (autonomous tool-calling loop over Ollama)
  4. One-click automated report synthesis (Markdown, downloadable)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from agent.agent_loop import run_agent
from agent.llm_client import DEFAULT_MODEL, is_ollama_running, list_models
from pipeline.anomaly import daily_revenue_anomalies, detect_anomalies
from pipeline.etl import run_pipeline
from reports.report_generator import generate_report, save_report

st.set_page_config(page_title="AI Analytics Agent", layout="wide", page_icon="📊")


@st.cache_data(show_spinner="Running ETL pipeline across sources...")
def load_data() -> pd.DataFrame:
    return run_pipeline()


def kpi_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", f"${df['revenue'].sum():,.0f}")
    c2.metric("Orders", f"{len(df):,}")
    c3.metric("Avg Order Value", f"${df['revenue'].mean():,.2f}")
    c4.metric("Avg Rating", f"{df['customer_rating'].mean():.2f} / 5")
    c5.metric("Return Rate", f"{df['returned'].mean()*100:.1f}%")


def charts(df: pd.DataFrame) -> None:
    left, right = st.columns(2)

    with left:
        daily = df.groupby("date", as_index=False)["revenue"].sum()
        fig = px.line(daily, x="date", y="revenue", title="Daily Revenue Trend", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        by_channel = df.groupby("channel_type", as_index=False)["revenue"].sum().sort_values("revenue")
        fig2 = px.bar(by_channel, x="revenue", y="channel_type", orientation="h",
                       title="Revenue by Channel Type")
        st.plotly_chart(fig2, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        by_region = df.groupby("region", as_index=False)["revenue"].sum()
        fig3 = px.pie(by_region, names="region", values="revenue", title="Revenue Share by Region")
        st.plotly_chart(fig3, use_container_width=True)

    with right2:
        by_source = df.groupby("source", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        fig4 = px.bar(by_source, x="source", y="revenue", title="Revenue by Source (all 12 feeds)")
        fig4.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig4, use_container_width=True)


def anomaly_panel(df: pd.DataFrame) -> None:
    flagged = detect_anomalies(df)
    st.subheader(f"🚩 Flagged Anomalies ({len(flagged)})")
    if flagged.empty:
        st.success("No anomalies detected in the current dataset.")
    else:
        st.dataframe(
            flagged[["record_id", "timestamp", "source", "region", "category",
                     "revenue", "quantity", "flag_reason"]],
            use_container_width=True, hide_index=True,
        )

    daily = daily_revenue_anomalies(df)
    anomalous_days = daily[daily["is_anomaly"]]
    if not anomalous_days.empty:
        st.warning(f"{len(anomalous_days)} day(s) deviated sharply from the rolling revenue average.")
        st.dataframe(anomalous_days[["date", "revenue", "rolling_mean", "z"]], use_container_width=True, hide_index=True)


def chat_panel(df: pd.DataFrame, model: str) -> None:
    st.subheader("💬 Ask the Agent")
    st.caption("Autonomous tool-calling loop: the agent decides which data tools to call, "
               "executes them, and reasons over the results.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander("Agent reasoning / tool calls"):
                    for step in msg["steps"]:
                        if step.role == "tool_call":
                            st.code(f"CALL {step.tool_name}({step.content})", language="json")
                        elif step.role == "tool_result":
                            st.code(f"RESULT [{step.tool_name}] {step.content}", language="json")

    question = st.chat_input("e.g. Which region has the highest return rate, and why?")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Agent is reasoning and calling tools..."):
                try:
                    run = run_agent(question, df, model=model)
                    st.markdown(run.answer)
                    with st.expander("Agent reasoning / tool calls"):
                        for step in run.steps:
                            if step.role == "tool_call":
                                st.code(f"CALL {step.tool_name}({step.content})", language="json")
                            elif step.role == "tool_result":
                                st.code(f"RESULT [{step.tool_name}] {step.content}", language="json")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": run.answer, "steps": run.steps}
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Agent error: {e}")


def report_panel(df: pd.DataFrame, model: str) -> None:
    st.subheader("📝 Automated Report Synthesis")
    st.caption("Runs the agent over a fixed analytical checklist and assembles a Markdown report.")
    if st.button("Generate Report", type="primary"):
        with st.spinner("Agent is synthesizing the report..."):
            try:
                report_md = generate_report(df, model=model)
                path = save_report(report_md)
                st.session_state["last_report"] = report_md
                st.session_state["last_report_path"] = str(path)
            except Exception as e:  # noqa: BLE001
                st.error(f"Report generation failed: {e}")

    if st.session_state.get("last_report"):
        st.markdown(st.session_state["last_report"])
        st.download_button(
            "Download Report (.md)",
            data=st.session_state["last_report"],
            file_name=Path(st.session_state["last_report_path"]).name,
            mime="text/markdown",
        )


def sidebar(models: list[str]) -> str:
    st.sidebar.title("📊 AI Analytics Agent")
    st.sidebar.caption("Python · Pandas · Streamlit · Ollama")

    ollama_ok = is_ollama_running()
    if ollama_ok:
        st.sidebar.success("Ollama connected")
    else:
        st.sidebar.error("Ollama not reachable at localhost:11434")
        st.sidebar.markdown(
            "Start it with `ollama serve`, then pull a tool-capable model, "
            "e.g. `ollama pull llama3.1`."
        )

    default_idx = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
    model = st.sidebar.selectbox(
        "Model", options=models or [DEFAULT_MODEL], index=default_idx if models else 0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Pipeline:** 12 simulated web sources → ETL → SQLite warehouse "
        "→ anomaly detector → tool-calling agent."
    )
    if st.sidebar.button("↻ Refresh data / re-run ETL"):
        st.cache_data.clear()
        st.rerun()

    return model


def main() -> None:
    df = load_data()
    models = list_models() or [DEFAULT_MODEL]
    model = sidebar(models)

    st.title("Autonomous AI Analytics Agent")
    st.caption(
        f"{len(df):,} records aggregated across {df['source'].nunique()} sources "
        f"({df['date'].min()} → {df['date'].max()})"
    )

    tab_overview, tab_anomalies, tab_chat, tab_report = st.tabs(
        ["Overview", "Anomalies", "Chat with Agent", "Report Synthesis"]
    )

    with tab_overview:
        kpi_row(df)
        st.markdown("---")
        charts(df)

    with tab_anomalies:
        anomaly_panel(df)

    with tab_chat:
        chat_panel(df, model)

    with tab_report:
        report_panel(df, model)


if __name__ == "__main__":
    main()
