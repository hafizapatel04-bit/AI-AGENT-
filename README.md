# Autonomous AI Analytics Agent & Dashboard

An autonomous analytics agent + Streamlit dashboard built with **Python, Pandas, Streamlit,
and Ollama LLMs**. The agent runs a tool-calling loop over a live ETL pipeline that
aggregates records from 12 simulated web sources, flags anomalies in real time, and
synthesizes analytical reports on demand.

## What it does

- **ETL pipeline** (`pipeline/etl.py`, `pipeline/sources.py`) — extracts records from 12
  simulated source feeds (web storefronts, mobile apps, marketplaces, paid social, affiliate,
  email, retail partners), cleans/dedupes/enriches them, and loads them into a SQLite warehouse.
  Swap `sources.py`'s connectors for real HTTP/API calls and nothing else changes.
- **Anomaly detection** (`pipeline/anomaly.py`) — z-score and IQR-based flagging per
  category/source, plus rolling-window detection on daily revenue for real-time drift/spikes.
- **Tool-calling agent** (`agent/`) — an autonomous loop: the LLM (via a local Ollama server)
  decides which data tools to call (`get_summary_stats`, `get_top_n`, `detect_data_anomalies`,
  `get_daily_revenue_trend`, `filter_records`), the app executes them against the live
  DataFrame, and the results are fed back until the model produces a grounded answer.
- **Report synthesis** (`reports/report_generator.py`) — runs the agent over a fixed
  analytical checklist and assembles a downloadable Markdown report.
- **Dashboard** (`app.py`) — Streamlit UI with KPIs, interactive Plotly charts, an anomaly
  table, a chat interface to the agent, and one-click report generation.

## Setup

### 1. Install Ollama and pull a tool-capable model
```bash
# https://ollama.com/download
ollama serve                 # starts the local server on :11434
ollama pull llama3.1         # or qwen2.5, mistral-nemo, etc. (must support tool calling)
```

### 2. Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Generate sample data (optional — auto-generates on first run if missing)
```bash
python data/sample_data_generator.py
```

### 4. Run the dashboard
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually http://localhost:8501).
Architecture
 12 web sources ──▶ ETL (extract/clean/enrich) ──▶ SQLite warehouse
                                                          │
                                                          ▼
                                  ┌──────────── Anomaly Detector
                                  │                       │
                          Agent Tools ◀── tool calls ── Ollama LLM
                                  │                       │
                                  └──────── answers ──────┘
                                                          │
                                                          ▼
                                          Streamlit Dashboard / Report
Layer	Module	Responsibility
Sources	pipeline/sources.py	12 simulated web-source connectors
ETL	pipeline/etl.py	Extract → clean/dedupe/enrich → load into SQLite
Analytics	pipeline/anomaly.py	Z-score, IQR, rolling-window anomaly detection
Agent	agent/llm_client.py, tools.py, agent_loop.py	Autonomous tool-calling loop against a local Ollama model
Reporting	reports/report_generator.py	Agent-driven Markdown report synthesis
UI	app.py	Streamlit dashboard

## Project structure
```
app.py                        # Streamlit dashboard entry point
agent/
  llm_client.py                # Ollama /api/chat wrapper with tool-calling
  tools.py                     # Tool schemas + implementations
  agent_loop.py                # Autonomous tool-calling loop
pipeline/
  sources.py                   # 12 simulated web-source connectors (swap for real APIs)
  etl.py                       # Extract -> Transform -> Load into SQLite
  anomaly.py                   # Z-score / IQR / rolling-window anomaly detection
reports/
  report_generator.py          # Automated report synthesis via the agent
data/
  sample_data_generator.py     # Generates 1,450 synthetic multi-source records
requirements.txt
```

## Using real data / real sources

Point `pipeline/sources.py` at real APIs: add entries to `LIVE_SOURCES` with callables that
return a DataFrame with at least `timestamp, source, region, category, revenue, quantity`
columns, and swap `aggregate_all_sources()` to pull from `LIVE_SOURCES` instead of the CSV.
The ETL, anomaly detection, agent tools, and dashboard all work unchanged against any
DataFrame with that schema — or point it at your own CSV by replacing `data/transactions.csv`.

## Notes on model choice

Tool calling requires a model that supports Ollama's `tools` parameter (Llama 3.1+, Qwen2.5,
Mistral Nemo, etc.). If a model ignores the tools and just free-answers, the dashboard's
sidebar model picker lets you switch models without touching code.
