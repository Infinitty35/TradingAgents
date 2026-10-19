"""``python -m api`` entry point: serve the TradingAgents API with uvicorn."""

import os

import uvicorn

from api.app import app  # noqa: F401  (imported so uvicorn can serve it by object)


def main() -> None:
    host = os.environ.get("TRADINGAGENTS_API_HOST", "127.0.0.1")
    port = int(os.environ.get("TRADINGAGENTS_API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
