"""Request and response schemas for the TradingAgents HTTP API."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ANALYSTS = ("market", "social", "news", "fundamentals")
ASSET_TYPES = ("stock", "crypto")


class Position(BaseModel):
    ticker: str = Field(description="Instrument symbol, e.g. AAPL")
    quantity: float = Field(description="Signed units held; negative is short")
    average_price: float | None = Field(default=None, description="Average entry price per unit")


class Portfolio(BaseModel):
    cash: float | None = Field(default=None, description="Free cash available")
    currency: str | None = Field(default=None, description="Currency label for cash and prices")
    positions: list[Position] = Field(default_factory=list)


class RunConfigOverrides(BaseModel):
    """Per-run overrides folded on top of DEFAULT_CONFIG.

    Only the documented keys are accepted; anything else is rejected so a typo
    fails the request instead of silently running with defaults.
    """

    model_config = ConfigDict(extra="forbid")

    llm_provider: str | None = None
    deep_think_llm: str | None = None
    quick_think_llm: str | None = None
    backend_url: str | None = None
    output_language: str | None = None
    max_debate_rounds: int | None = Field(default=None, ge=1)
    max_risk_discuss_rounds: int | None = Field(default=None, ge=1)
    max_tool_rounds: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    llm_max_retries: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    checkpoint_enabled: bool | None = None


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date(value: str) -> str:
    """Reject anything the graph would reject later, at request time."""
    from tradingagents.dataflows.date_window import get_current_date

    if not _DATE_RE.match(value):
        raise ValueError(f"date must be in YYYY-MM-DD format, got {value!r}")
    if value > get_current_date():
        raise ValueError(f"date cannot be in the future: {value}")
    return value


class AnalyzeRequest(BaseModel):
    ticker: str = Field(description="Ticker to analyze, e.g. NVDA or 0700.HK", min_length=1)
    date: str = Field(description="Analysis date, YYYY-MM-DD; must not be in the future")

    @field_validator("date")
    @classmethod
    def _check_date(cls, value: str) -> str:
        return _validate_date(value)
    analysts: list[Literal["market", "social", "news", "fundamentals"]] | None = Field(
        default=None, description="Analysts to run; omit for all four"
    )
    asset_type: Literal["stock", "crypto"] = "stock"
    portfolio: Portfolio | None = Field(
        default=None, description="Current holdings and cash, so decisions size against the real book"
    )
    save_report: bool = Field(default=True, description="Write the markdown report tree under results_dir")
    config: RunConfigOverrides | None = None


class BacktestRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, description="Tickers to analyze, e.g. [\"NVDA\", \"AAPL\"]")
    start: str = Field(description="First analysis date, YYYY-MM-DD")
    end: str = Field(description="Last analysis date, YYYY-MM-DD")
    every: int = Field(default=7, ge=1, description="Days between analysis dates")

    @field_validator("start", "end")
    @classmethod
    def _check_dates(cls, value: str) -> str:
        return _validate_date(value)
    analysts: list[Literal["market", "social", "news", "fundamentals"]] | None = None
    asset_type: Literal["stock", "crypto"] = "stock"
    portfolio: Portfolio | None = None
    run_id: str | None = Field(
        default=None, description="Continue an earlier sweep: its cells are skipped and its log reused"
    )
    config: RunConfigOverrides | None = None


class JobStatus(BaseModel):
    id: str
    kind: Literal["analyze", "backtest"]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    ticker: str | None = None
    date: str | None = None
    tickers: list[str] | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: str | None = Field(
        default=None, description="Last progress line, e.g. \"[3/12] NVDA 2026-07-06\""
    )
    error: str | None = None
    result: dict | None = None


class AnalyzeResult(BaseModel):
    ticker: str
    date: str
    rating: str
    decision: str
    reports: dict[str, str] = Field(default_factory=dict)
    report_path: str | None = None
    run_settings: dict = Field(default_factory=dict)


class BacktestResultModel(BaseModel):
    run_id: str
    cells_run: int
    skipped: int
    failures: list[list[str]]
    settlement_failures: list[list[str]]
    log_path: str
    summary: str


class HistoryEntry(BaseModel):
    date: str
    ticker: str
    rating: str
    pending: bool
    raw_return: str | None = None
    alpha: str | None = None
    holding: str | None = None
    resolved: str | None = None
    decision: str
    reflection: str | None = None


class ReportInfo(BaseModel):
    path: str
    ticker: str
    report_path: str


class ProviderInfo(BaseModel):
    name: str
    api_key_env: str | None = Field(description="Env var holding this provider's API key")
    key_configured: bool = Field(description="Whether that env var is set in the server process")
    models: dict[str, list[dict[str, str]]] = Field(
        default_factory=dict, description="Curated model options per tier (quick/deep)"
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    auth_required: bool
