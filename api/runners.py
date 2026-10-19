"""Bridge between API requests and the tradingagents machinery.

Each function turns an API model into the arguments the library functions take,
runs them, and returns a plain JSON-able dict. LLM clients read their keys
from the environment, so the server process must be started with the same
environment (or .env) the CLI would use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from api.jobs import Job
from api.models import AnalyzeRequest, BacktestRequest
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.memory import TradingMemoryLog
from tradingagents.portfolio import PortfolioContext

logger = logging.getLogger(__name__)


def build_config(overrides) -> dict:
    """DEFAULT_CONFIG plus a request's overrides (None fields are ignored)."""
    config = DEFAULT_CONFIG.copy()
    if overrides is not None:
        for key, value in overrides.model_dump(exclude_none=True).items():
            config[key] = value
    return config


def build_portfolio(portfolio) -> PortfolioContext | None:
    if portfolio is None:
        return None
    return PortfolioContext.model_validate(portfolio.model_dump())


def run_analysis(request: AnalyzeRequest, job: Job) -> dict[str, Any]:
    """Run one analysis and return its decision and reports."""
    config = build_config(request.config)
    config["checkpoint_enabled"] = bool(config.get("checkpoint_enabled"))
    analysts = tuple(request.analysts) if request.analysts else (
        "market", "social", "news", "fundamentals"
    )
    graph = TradingAgentsGraph(selected_analysts=analysts, config=config)
    portfolio = build_portfolio(request.portfolio)
    final_state, rating = graph.propagate(
        request.ticker, request.date, asset_type=request.asset_type, portfolio=portfolio,
    )
    if job.cancel_event.is_set():
        raise RuntimeError("cancelled")

    report_path = None
    if request.save_report:
        report_path = str(graph.save_reports(final_state, request.ticker))

    return {
        "ticker": request.ticker,
        "date": request.date,
        "rating": rating,
        "decision": final_state.get("final_trade_decision", ""),
        "reports": {
            "market": final_state.get("market_report", ""),
            "sentiment": final_state.get("sentiment_report", ""),
            "news": final_state.get("news_report", ""),
            "fundamentals": final_state.get("fundamentals_report", ""),
            "investment_plan": final_state.get("investment_plan", ""),
            "trader_plan": final_state.get("trader_investment_plan", ""),
            "final_decision": final_state.get("final_trade_decision", ""),
        },
        "report_path": report_path,
        "run_settings": graph.run_settings(),
    }


def run_backtest_job(request: BacktestRequest, job: Job) -> dict[str, Any]:
    """Run a backtest sweep, reporting per-cell progress to the job."""
    config = build_config(request.config)
    dates = iter_grid(request.start, request.end, request.every)
    analysts = tuple(request.analysts) if request.analysts else (
        "market", "social", "news", "fundamentals"
    )
    portfolio = build_portfolio(request.portfolio)

    def progress(done: int, total: int, ticker: str, date: str) -> None:
        if job.cancel_event.is_set():
            raise RuntimeError("cancelled")
        from api.app import get_registry

        get_registry().set_progress(job, f"[{done}/{total}] {ticker} {date}")

    result = run_backtest(
        request.tickers,
        dates,
        config,
        asset_type=request.asset_type,
        portfolio=portfolio,
        selected_analysts=analysts,
        run_id=request.run_id,
        progress=progress,
    )
    summary = summarize(result)
    return {
        "run_id": result.run_id,
        "cells_run": result.cells_run,
        "skipped": result.skipped,
        "failures": [[t, d, r] for t, d, r in result.failures],
        "settlement_failures": [[t, r] for t, r in result.settlement_failures],
        "log_path": str(result.log_path),
        "summary": summary.render(),
    }


def list_history(config: dict | None = None) -> list[dict[str, Any]]:
    """The server's memory log entries, newest first."""
    cfg = config or DEFAULT_CONFIG
    log = TradingMemoryLog(cfg)
    entries = log.load_entries()
    history = []
    for entry in reversed(entries):
        history.append({
            "date": entry.get("date", ""),
            "ticker": entry.get("ticker", ""),
            "rating": entry.get("rating", ""),
            "pending": bool(entry.get("pending")),
            "raw_return": entry.get("raw"),
            "alpha": entry.get("alpha"),
            "holding": entry.get("holding"),
            "resolved": entry.get("resolved"),
            "decision": entry.get("decision", ""),
            "reflection": entry.get("reflection"),
        })
    return history


def list_reports(config: dict | None = None) -> list[dict[str, str]]:
    """Saved report trees under results_dir/reports, newest first."""
    cfg = config or DEFAULT_CONFIG
    reports_root = Path(cfg["results_dir"]) / "reports"
    reports = []
    if reports_root.is_dir():
        for path in sorted(reports_root.iterdir(), reverse=True):
            complete = path / "complete_report.md"
            if path.is_dir() and complete.is_file():
                reports.append({
                    "path": str(path),
                    "ticker": safe_ticker_component(path.name.rsplit("_", 2)[0])
                    if "_" in path.name else path.name,
                    "report_path": str(complete),
                })
    return reports


def read_report(path: str, config: dict | None = None) -> str:
    """Read a saved complete report, confined to the configured results dir."""
    cfg = config or DEFAULT_CONFIG
    root = Path(cfg["results_dir"]).resolve()
    target = Path(path).expanduser().resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"report path {path} is outside the results directory")
    if not target.is_file():
        raise FileNotFoundError(f"no report at {path}")
    if target.suffix != ".md":
        raise ValueError(f"report path {path} is not a markdown file")
    return target.read_text(encoding="utf-8")
