"""The HTTP API: endpoints, job lifecycle, auth, and path confinement.

The graph and backtest harness are replaced with fakes, so no LLM call and no
network happens; the tests pin the API's own behaviour (routing, validation,
job state transitions, token enforcement, report reading).
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from api.app import create_app

DECISION = "Rating: Buy\n\nFINAL TRANSACTION PROPOSAL: **BUY**"


class _FakeGraph:
    """Stands in for TradingAgentsGraph with a canned decision."""

    instances: list = []

    def __init__(self, selected_analysts=None, config=None, **kwargs):
        self.selected_analysts = list(selected_analysts) if selected_analysts else None
        self.config = config or {}
        _FakeGraph.instances.append(self)

    def propagate(self, ticker, trade_date, asset_type="stock", portfolio=None):
        self.last_call = {
            "ticker": ticker,
            "trade_date": trade_date,
            "asset_type": asset_type,
            "portfolio": portfolio,
        }
        return (
            {
                "company_of_interest": ticker,
                "trade_date": trade_date,
                "market_report": "market report",
                "sentiment_report": "sentiment report",
                "news_report": "news report",
                "fundamentals_report": "fundamentals report",
                "investment_plan": "investment plan",
                "trader_investment_plan": "trader plan",
                "final_trade_decision": DECISION,
            },
            "Buy",
        )

    def run_settings(self):
        return {"llm_provider": "openai", "analysts": self.selected_analysts}

    def save_reports(self, final_state, ticker, save_path=None):
        from pathlib import Path

        path = Path(save_path or self.config["results_dir"]) / "reports" / f"{ticker}_test"
        path.mkdir(parents=True, exist_ok=True)
        (path / "complete_report.md").write_text("# report\n", encoding="utf-8")
        return path / "complete_report.md"


class _FakeBacktestResult:
    def __init__(self, run_id, log_path):
        self.run_id = run_id
        self.log_path = log_path
        self.cells_run = 2
        self.skipped = 0
        self.failures = []
        self.settlement_failures = []


class _FakeSummary:
    def render(self):
        return "Resolved cells: 2"


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A TestClient with the graph, backtest and memory log faked out."""
    _FakeGraph.instances = []
    monkeypatch.setattr("api.runners.TradingAgentsGraph", _FakeGraph)
    monkeypatch.setattr(
        "api.runners.iter_grid", lambda start, end, every: ["2026-01-05", "2026-01-12"]
    )

    def fake_run_backtest(tickers, dates, config, **kwargs):
        kwargs["progress"](1, 2, tickers[0], dates[0])
        kwargs["progress"](2, 2, tickers[0], dates[1])
        from pathlib import Path

        log_path = Path(config["results_dir"]) / "backtest" / "run1" / "trading_memory.md"
        return _FakeBacktestResult("run1", log_path)

    monkeypatch.setattr("api.runners.run_backtest", fake_run_backtest)
    monkeypatch.setattr("api.runners.summarize", lambda result: _FakeSummary())
    monkeypatch.setattr("api.runners.list_history", lambda config=None: [
        {
            "date": "2026-01-05",
            "ticker": "NVDA",
            "rating": "Buy",
            "pending": False,
            "raw_return": "+3.2%",
            "alpha": "+1.1%",
            "holding": "5d",
            "resolved": "2026-01-12",
            "decision": DECISION,
            "reflection": "held up",
        },
    ])
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _wait_for_job(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        status = response.json()["status"]
        if status in ("completed", "failed", "cancelled"):
            return response.json()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


@pytest.mark.unit
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["auth_required"] is False


@pytest.mark.unit
def test_providers_lists_key_status_without_values(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    response = client.get("/providers")
    assert response.status_code == 200
    providers = {p["name"]: p for p in response.json()["providers"]}
    assert providers["openai"]["api_key_env"] == "OPENAI_API_KEY"
    assert providers["openai"]["key_configured"] is True
    assert providers["bedrock"]["key_configured"] is False
    assert providers["openai"]["models"]["deep"]
    # The key value itself must never appear in the response.
    assert "sk-test" not in response.text


@pytest.mark.unit
def test_config_exposes_only_public_keys(client):
    response = client.get("/config")
    assert response.status_code == 200
    config = response.json()
    assert config["llm_provider"] == "openai"
    assert "backend_url" not in config
    for key in ("llm_provider", "max_debate_rounds", "data_vendors", "results_dir"):
        assert key in config


@pytest.mark.unit
def test_analyze_runs_and_returns_decision(client):
    response = client.post("/analyze", json={
        "ticker": "NVDA",
        "date": "2026-01-09",
        "analysts": ["market", "news"],
        "portfolio": {"cash": 25000.0, "currency": "USD",
                      "positions": [{"ticker": "NVDA", "quantity": 10, "average_price": 150.0}]},
        "config": {"max_debate_rounds": 2, "llm_provider": "anthropic"},
    })
    assert response.status_code == 202
    job = response.json()
    assert job["kind"] == "analyze"
    assert job["status"] in ("queued", "running")

    finished = _wait_for_job(client, job["id"])
    assert finished["status"] == "completed"
    result = finished["result"]
    assert result["ticker"] == "NVDA"
    assert result["rating"] == "Buy"
    assert "BUY" in result["decision"]
    assert result["reports"]["market"] == "market report"
    assert result["report_path"].endswith("complete_report.md")

    # The request's overrides reached the graph, and the portfolio was parsed.
    graph = _FakeGraph.instances[-1]
    assert graph.selected_analysts == ["market", "news"]
    assert graph.config["max_debate_rounds"] == 2
    assert graph.config["llm_provider"] == "anthropic"
    portfolio = graph.last_call["portfolio"]
    assert portfolio is not None
    assert portfolio.cash == 25000.0
    assert portfolio.positions[0].ticker == "NVDA"


@pytest.mark.unit
def test_analyze_rejects_unknown_config_key(client):
    response = client.post("/analyze", json={
        "ticker": "NVDA",
        "date": "2026-01-09",
        "config": {"not_a_real_key": 1},
    })
    assert response.status_code == 422


@pytest.mark.unit
def test_analyze_rejects_bad_date_and_analyst(client):
    response = client.post("/analyze", json={"ticker": "NVDA", "date": "09-01-2026"})
    assert response.status_code == 422
    response = client.post("/analyze", json={
        "ticker": "NVDA", "date": "2026-01-09", "analysts": ["warren_buffett"],
    })
    assert response.status_code == 422


@pytest.mark.unit
def test_backtest_runs_and_reports_progress(client):
    response = client.post("/backtest", json={
        "tickers": ["NVDA", "AAPL"],
        "start": "2026-01-05",
        "end": "2026-01-12",
        "every": 7,
    })
    assert response.status_code == 202
    job = response.json()
    assert job["kind"] == "backtest"
    assert job["tickers"] == ["NVDA", "AAPL"]

    finished = _wait_for_job(client, job["id"])
    assert finished["status"] == "completed"
    result = finished["result"]
    assert result["run_id"] == "run1"
    assert result["cells_run"] == 2
    assert "Resolved cells: 2" in result["summary"]
    assert finished["progress"] == "[2/2] NVDA 2026-01-12"


@pytest.mark.unit
def test_backtest_requires_a_ticker(client):
    response = client.post("/backtest", json={
        "tickers": [], "start": "2026-01-05", "end": "2026-01-12",
    })
    assert response.status_code == 422


@pytest.mark.unit
def test_jobs_list_and_unknown_job(client):
    response = client.post("/analyze", json={"ticker": "NVDA", "date": "2026-01-09"})
    job_id = response.json()["id"]
    _wait_for_job(client, job_id)

    listing = client.get("/jobs")
    assert listing.status_code == 200
    ids = [j["id"] for j in listing.json()["jobs"]]
    assert job_id in ids

    assert client.get("/jobs/does-not-exist").status_code == 404


@pytest.mark.unit
def test_job_events_stream_ends_on_completion(client):
    response = client.post("/analyze", json={"ticker": "NVDA", "date": "2026-01-09"})
    job_id = response.json()["id"]
    with client.stream("GET", f"/jobs/{job_id}/events") as stream:
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        payloads = []
        for line in stream.iter_lines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[len("data: "):]))
    statuses = [p["status"] for p in payloads]
    assert statuses[-1] == "completed"
    assert "completed" in statuses


@pytest.mark.unit
def test_history_filters_by_ticker(client):
    response = client.get("/history")
    assert response.status_code == 200
    entries = response.json()["entries"]
    assert entries[0]["ticker"] == "NVDA"
    assert entries[0]["rating"] == "Buy"
    assert entries[0]["raw_return"] == "+3.2%"

    response = client.get("/history", params={"ticker": "aapl"})
    assert response.json()["entries"] == []


@pytest.mark.unit
def test_reports_listing_and_content(client, tmp_path, monkeypatch):
    from tradingagents.default_config import DEFAULT_CONFIG

    reports_root = tmp_path / "logs" / "reports" / "NVDA_20260109_120000"
    reports_root.mkdir(parents=True)
    (reports_root / "complete_report.md").write_text("# Trading Analysis Report: NVDA\n",
                                                     encoding="utf-8")
    monkeypatch.setitem(DEFAULT_CONFIG, "results_dir", str(tmp_path / "logs"))

    listing = client.get("/reports")
    assert listing.status_code == 200
    reports = listing.json()["reports"]
    assert len(reports) == 1
    assert reports[0]["ticker"] == "NVDA"

    content = client.get("/reports/content", params={"path": str(reports_root / "complete_report.md")})
    assert content.status_code == 200
    assert "Trading Analysis Report: NVDA" in content.text

    # A path outside the results dir is refused, not read.
    outside = tmp_path / "elsewhere.md"
    outside.write_text("secret", encoding="utf-8")
    response = client.get("/reports/content", params={"path": str(outside)})
    assert response.status_code == 400

    missing_inside = tmp_path / "logs" / "reports" / "nope" / "complete_report.md"
    assert client.get("/reports/content", params={"path": str(missing_inside)}).status_code == 404


@pytest.mark.unit
def test_auth_required_when_token_set(client, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_API_TOKEN", "s3cret")
    try:
        assert client.get("/health").status_code == 401
        assert client.get("/config").status_code == 401
        assert client.post("/analyze", json={"ticker": "NVDA", "date": "2026-01-09"}).status_code == 401

        headers = {"Authorization": "Bearer wrong"}
        assert client.get("/health", headers=headers).status_code == 401

        headers = {"Authorization": "Bearer s3cret"}
        response = client.get("/health", headers=headers)
        assert response.status_code == 200
        assert response.json()["auth_required"] is True
        # The token value is never echoed back.
        assert "s3cret" not in response.text
    finally:
        monkeypatch.delenv("TRADINGAGENTS_API_TOKEN")


@pytest.mark.unit
def test_cancel_a_running_job(client, monkeypatch):
    import threading

    started = threading.Event()
    release = threading.Event()

    class _SlowGraph(_FakeGraph):
        def propagate(self, ticker, trade_date, asset_type="stock", portfolio=None):
            started.set()
            release.wait(timeout=30)
            return super().propagate(ticker, trade_date, asset_type, portfolio)

    monkeypatch.setattr("api.runners.TradingAgentsGraph", _SlowGraph)
    response = client.post("/analyze", json={"ticker": "NVDA", "date": "2026-01-09"})
    job_id = response.json()["id"]
    assert started.wait(timeout=5)

    cancel = client.post(f"/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelled"
    release.set()  # let the worker thread exit; the cancel already marked the job

    # Cancelling a finished job is a conflict.
    monkeypatch.setattr("api.runners.TradingAgentsGraph", _FakeGraph)
    finished = client.post("/analyze", json={"ticker": "NVDA", "date": "2026-01-09"})
    _wait_for_job(client, finished.json()["id"])
    assert client.post(f"/jobs/{finished.json()['id']}/cancel").status_code == 409


@pytest.mark.unit
def test_job_registry_lifecycle():
    from api.jobs import JobRegistry

    registry = JobRegistry()
    job = registry.create("analyze", ticker="NVDA", date="2026-01-09")
    assert job.status == "queued"
    assert registry.get(job.id) is job

    registry.start(job)
    assert job.status == "running"
    registry.set_progress(job, "[1/2] NVDA 2026-01-05")
    assert job.progress == "[1/2] NVDA 2026-01-05"

    registry.complete(job, {"rating": "Buy"})
    assert job.status == "completed"
    assert job.result == {"rating": "Buy"}
    assert job.finished_at is not None
    # A finished job cannot be cancelled.
    assert registry.cancel(job) is False

    pending = registry.create("backtest", tickers=["NVDA"])
    assert registry.cancel(pending) is True
    assert pending.status == "cancelled"
    assert pending.cancel_event.is_set()

    # run_async records a target's outcome on the job.
    ran = registry.create("analyze", ticker="AAPL", date="2026-01-09")
    registry.run_async(ran, lambda j: {"ok": True})
    deadline = time.time() + 5
    while ran.status == "running" and time.time() < deadline:
        time.sleep(0.01)
    assert ran.status == "completed"
    assert ran.result == {"ok": True}

    failed = registry.create("analyze", ticker="MSFT", date="2026-01-09")

    def boom(j):
        raise RuntimeError("vendor exploded")

    registry.run_async(failed, boom)
    deadline = time.time() + 5
    while failed.status == "running" and time.time() < deadline:
        time.sleep(0.01)
    assert failed.status == "failed"
    assert "vendor exploded" in failed.error
