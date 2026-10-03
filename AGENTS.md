# TradingAgents — Base44 Development Environment

## Project Overview
TradingAgents is a multi-agent LLM financial trading framework (Python CLI tool).
It deploys specialized LLM-powered agents (analysts, researchers, trader, risk
management) that collaboratively evaluate market conditions and produce a
trading decision. The framework is a Python package (`tradingagents/`) with a
CLI entry point (`cli/`).

## Web Interface
A Flask web app (`web_app.py`) was added to provide a browser UI for the preview.
It wraps `TradingAgentsGraph.propagate()` — the same API `main.py` uses — and
runs analysis in a background thread with status polling. The UI is a
single-page app in `templates/index.html`.

## Running the App
```bash
docker compose -f docker-compose.base44.yml up -d
```
The web app is served on port 3000 with live reload (Flask debug mode).

## Dependencies
- Python 3.13 (from the `python:3.13-slim` base image)
- Flask (installed at container startup, not in pyproject.toml)
- Project dependencies (installed via `pip install -e .` at startup)

## API Keys
The app needs an LLM provider API key to run analysis. Set one via the Base44
secrets dashboard. Supported providers and their key env vars:
- `OPENAI_API_KEY` (platform.openai.com/api-keys)
- `ANTHROPIC_API_KEY` (console.anthropic.com/settings/keys)
- `GOOGLE_API_KEY` (aistudio.google.com/apikey)
- `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`, etc.
- Optional: `FRED_API_KEY` for macro data, `SEC_EDGAR_USER_AGENT` for fundamentals.
Without a key, the form loads but analysis returns an error.

## Verification
```bash
curl http://localhost:3000/   # should return the HTML form page
```

## Architecture
- `web_app.py` — Flask web server (port 3000, debug mode + threaded)
- `templates/index.html` — Single-page UI (form, status polling, results)
- `tradingagents/` — Core framework (agents, graph, data flows, LLM clients)
- `cli/` — CLI interface (not used by the web app, but installed as a dependency)
- Data persisted in `tradingagents_data` Docker volume at `/root/.tradingagents`
