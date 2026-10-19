"""HTTP API for TradingAgents.

Exposes the analysis graph, backtest harness, memory log and report tree as
REST endpoints so the framework can be driven from another process, a
scheduler, or a web UI instead of the interactive CLI.

Run with ``tradingagents-api`` (installed console script) or
``python -m api``. The server binds to 127.0.0.1 by default; set
``TRADINGAGENTS_API_HOST`` / ``TRADINGAGENTS_API_PORT`` to change that, and
``TRADINGAGENTS_API_TOKEN`` to require a bearer token.
"""

from api.app import create_app

__all__ = ["create_app"]
