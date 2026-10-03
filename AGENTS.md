# TradingAgents — Base44 Development Environment

## What this is
TradingAgents is a Python CLI tool (not a web app) for multi-agent LLM financial
trading analysis. It has no built-in web interface.

## Base44 additions
- `web/dashboard.py` — a minimal web dashboard (Python stdlib only, no extra
  dependencies) that serves on port 3000 so the project is visible in the preview.
  It shows project status, API key presence, and a form to run an analysis.
- `docker-compose.base44.yml` — development compose that bind-mounts the source
  and runs the dashboard with live-reload-friendly setup.
- `.env.base44-defaults` — non-secret defaults (SEC EDGAR user agent, etc.).

## Running
```bash
docker compose -f docker-compose.base44.yml up -d
```
The dashboard is on http://localhost:3000. The CLI is also available inside the
container: `docker compose -f docker-compose.base44.yml exec web tradingagents --help`.

## Dependencies
Installed via `pip install -e .` on container startup (editable mode, so source
changes are reflected). All dependencies have wheels — no compilation needed.

## API keys
At least one LLM provider key is needed to run an analysis. The default provider
is OpenAI (`OPENAI_API_KEY`). Set keys via the Base44 secrets dashboard; they're
delivered to `/run/base44/app.env`. Data vendor keys (FRED, SEC EDGAR, TypeSafe)
are optional.

## Tests
```bash
docker compose -f docker-compose.base44.yml exec web python -m pytest -m "not integration"
```
