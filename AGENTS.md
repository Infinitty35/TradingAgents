# AGENTS.md

## Cursor Cloud specific instructions

TradingAgents is a single Python package (no separate frontend/backend). It ships a library (`tradingagents/`) and an interactive Typer/Rich CLI (`cli/`). The `## News`/README describe the product; there are no other services to run.

### Environment

- A virtualenv is created at `.venv` by the startup update script (`pip install -e ".[dev]"`). Activate it before running anything: `source .venv/bin/activate`. Dependencies are editable-installed, so source edits are picked up without reinstalling.
- Requires Python >= 3.10 (the VM uses 3.12). `requirements.txt` is a placeholder (`.`); the real dependency list lives in `pyproject.toml`.

### Lint / test / run (see also `.github/workflows/ci.yml`)

- Lint: `ruff check .` (config in `pyproject.toml`; `results/` and `worklog/` are excluded).
- Tests: `pytest -q`. Tests are hermetic and need no network or real API keys — `tests/conftest.py` injects placeholder API keys and resets global config between tests. Expected skips: `test_bedrock_provider` (optional `bedrock` extra not installed) and `test_deepseek_reasoning` (needs a real `DEEPSEEK_API_KEY`).
- Run the app: `tradingagents` (or `python -m cli.main`). The library entrypoint is `TradingAgentsGraph(...).propagate(ticker, date)` (see `main.py`).

### Non-obvious gotchas

- Running an actual analysis requires an LLM provider API key (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, ...). Without one, the CLI walks through ticker/date/analyst/depth/provider selection and then stops at the API-key prompt; the library raises when constructing the LLM client. Set the key as an env var or in a `.env` file (`cp .env.example .env`). No key is bundled.
- Market/data flows (yfinance, etc.) hit the live network and need no key — e.g. `python test.py` fetches real indicator data and is a good keyless smoke check.
- Optional AWS Bedrock support needs the extra: `pip install -e ".[bedrock]"`.
- Runtime state persists under `~/.tradingagents/` (decision log + checkpoint SQLite DBs); override with `TRADINGAGENTS_MEMORY_LOG_PATH` / `TRADINGAGENTS_CACHE_DIR`.
