"""FastAPI application exposing TradingAgents over HTTP.

Endpoints:
  GET  /health                     liveness, version, auth requirement
  GET  /providers                  LLM providers, key status, curated models
  GET  /config                     the server's effective run configuration
  POST /analyze                    run one analysis (background job)
  POST /backtest                   run a ticker x date sweep (background job)
  GET  /jobs                       list jobs
  GET  /jobs/{id}                  job status and result
  GET  /jobs/{id}/events           server-sent events stream of a job
  POST /jobs/{id}/cancel           cancel a queued or running job
  GET  /history                    the memory log, newest first
  GET  /reports                    saved report trees
  GET  /reports/content?path=...   one saved complete report as markdown

Long runs execute on background threads; submit with POST and poll (or stream)
the job. The server binds to 127.0.0.1 by default and requires no token unless
``TRADINGAGENTS_API_TOKEN`` is set, in which case every request must carry it
as ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import tradingagents
from api import runners
from api.jobs import JobRegistry
from api.models import (
    AnalyzeRequest,
    BacktestRequest,
    HealthResponse,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS

_bearer = HTTPBearer(auto_error=False)
_security = Security(_bearer)  # module-level so ruff's B008 (no calls in defaults) holds
_registry = JobRegistry()

# Config keys safe to show over the API: no endpoints that can carry credentials,
# no local paths beyond the results/cache roots, no secrets.
_PUBLIC_CONFIG_KEYS = (
    "llm_provider",
    "deep_think_llm",
    "quick_think_llm",
    "output_language",
    "max_debate_rounds",
    "max_risk_discuss_rounds",
    "max_tool_rounds",
    "max_recur_limit",
    "temperature",
    "checkpoint_enabled",
    "news_article_limit",
    "global_news_article_limit",
    "global_news_lookback_days",
    "holding_period_days",
    "benchmark_ticker",
    "data_vendors",
    "tool_vendors",
    "results_dir",
    "data_cache_dir",
    "memory_log_path",
)


def get_registry() -> JobRegistry:
    return _registry


def _expected_token() -> str | None:
    token = os.environ.get("TRADINGAGENTS_API_TOKEN", "").strip()
    return token or None


async def require_token(
    credentials: HTTPAuthorizationCredentials | None = _security,
) -> None:
    """Enforce the optional bearer token; constant-time compare."""
    import hmac

    expected = _expected_token()
    if expected is None:
        return
    provided = credentials.credentials if credentials else None
    if provided is None or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="a valid bearer token is required (set TRADINGAGENTS_API_TOKEN on the server)",
            headers={"WWW-Authenticate": "Bearer"},
        )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="TradingAgents API",
        description="Multi-Agents LLM Financial Trading Framework, over HTTP.",
        version=tradingagents.__version__,
        lifespan=_lifespan,
    )

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def _not_found(request: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health(_: None = Depends(require_token)) -> HealthResponse:
        return HealthResponse(
            status="ok", version=tradingagents.__version__, auth_required=_expected_token() is not None
        )

    @app.get("/providers", tags=["meta"])
    async def providers(_: None = Depends(require_token)):
        result = []
        for name in sorted(PROVIDER_API_KEY_ENV):
            env_var = PROVIDER_API_KEY_ENV[name]
            key_configured = bool(env_var and os.environ.get(env_var))
            models = {
                mode: [{"label": label, "id": value} for label, value in options]
                for mode, options in MODEL_OPTIONS.get(name, {}).items()
            }
            result.append({
                "name": name,
                "api_key_env": env_var,
                "key_configured": key_configured,
                "models": models,
            })
        return {"providers": result}

    @app.get("/config", tags=["meta"])
    async def config(_: None = Depends(require_token)):
        return {key: DEFAULT_CONFIG.get(key) for key in _PUBLIC_CONFIG_KEYS}

    @app.post("/analyze", status_code=202, tags=["runs"])
    async def analyze(request: AnalyzeRequest, _: None = Depends(require_token)):
        job = _registry.create("analyze", ticker=request.ticker, date=request.date)

        def target(j):
            return runners.run_analysis(request, j)

        _registry.run_async(job, target)
        return job.to_dict()

    @app.post("/backtest", status_code=202, tags=["runs"])
    async def backtest(request: BacktestRequest, _: None = Depends(require_token)):
        job = _registry.create("backtest", tickers=list(request.tickers))

        def target(j):
            return runners.run_backtest_job(request, j)

        _registry.run_async(job, target)
        return job.to_dict()

    @app.get("/jobs", tags=["runs"])
    async def jobs(_: None = Depends(require_token)):
        return {"jobs": [j.to_dict() for j in _registry.list()]}

    @app.get("/jobs/{job_id}", tags=["runs"])
    async def job_status(job_id: str, _: None = Depends(require_token)):
        job = _registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return job.to_dict()

    @app.post("/jobs/{job_id}/cancel", tags=["runs"])
    async def job_cancel(job_id: str, _: None = Depends(require_token)):
        job = _registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        if not _registry.cancel(job):
            raise HTTPException(status_code=409, detail=f"job {job_id} already finished")
        return job.to_dict()

    @app.get("/jobs/{job_id}/events", tags=["runs"])
    async def job_events(job_id: str, _: None = Depends(require_token)):
        job = _registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")

        async def event_stream() -> AsyncIterator[str]:
            last_status = None
            while True:
                current = _registry.get(job_id)
                if current is None:
                    yield _sse({"detail": "job disappeared"})
                    return
                if current.status != last_status or current.progress:
                    last_status = current.status
                    yield _sse(current.to_dict())
                if current.status in ("completed", "failed", "cancelled"):
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/history", tags=["data"])
    async def history(
        ticker: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
        _: None = Depends(require_token),
    ):
        entries = runners.list_history()
        if ticker:
            entries = [e for e in entries if e["ticker"].upper() == ticker.upper()]
        return {"entries": entries[:limit]}

    @app.get("/reports", tags=["data"])
    async def reports(_: None = Depends(require_token)):
        return {"reports": runners.list_reports()}

    @app.get("/reports/content", response_class=PlainTextResponse, tags=["data"])
    async def report_content(
        path: str = Query(description="Path of a saved complete_report.md"),
        _: None = Depends(require_token),
    ):
        return runners.read_report(path)

    return app


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


app = create_app()
