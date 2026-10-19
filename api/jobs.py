"""In-memory job registry for long-running API runs.

Analysis and backtest runs take minutes, so the API runs them on background
threads and exposes their state here. The registry lives in the server
process: a restart forgets queued and running jobs, which is the honest
behaviour for an in-process executor (a queued job is not persisted to disk
and pretending otherwise would strand it).
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Job:
    kind: str
    status: str = "queued"
    ticker: str | None = None
    date: str | None = None
    tickers: list[str] | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    progress: str | None = None
    error: str | None = None
    result: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "ticker": self.ticker,
            "date": self.date,
            "tickers": self.tickers,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "error": self.error,
            "result": self.result,
        }

    # Set by the registry when the job is registered.
    id: str = ""


class JobRegistry:
    """Thread-safe registry of background jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(
        self,
        kind: str,
        *,
        ticker: str | None = None,
        date: str | None = None,
        tickers: list[str] | None = None,
    ) -> Job:
        job = Job(kind=kind, ticker=ticker, date=date, tickers=tickers)
        job.id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def start(self, job: Job) -> None:
        with self._lock:
            job.status = "running"
            job.started_at = _now()

    def set_progress(self, job: Job, progress: str) -> None:
        with self._lock:
            job.progress = progress

    def complete(self, job: Job, result: dict) -> None:
        with self._lock:
            job.status = "completed"
            job.result = result
            job.finished_at = _now()

    def fail(self, job: Job, error: str) -> None:
        with self._lock:
            job.status = "failed"
            job.error = error
            job.finished_at = _now()

    def cancel(self, job: Job) -> bool:
        """Ask a running job to stop. Returns False when it already finished."""
        with self._lock:
            if job.status not in ("queued", "running"):
                return False
            job.cancel_event.set()
            job.status = "cancelled"
            job.finished_at = _now()
            return True

    def run_async(self, job: Job, target: Callable[[Job], dict]) -> threading.Thread:
        """Run ``target(job)`` on a daemon thread, recording the outcome."""

        def runner() -> None:
            self.start(job)
            try:
                result = target(job)
            except Exception as exc:  # a failed run is a result, not a crash
                if job.cancel_event.is_set():
                    return  # cancel() already marked it
                self.fail(job, f"{type(exc).__name__}: {exc}")
                return
            if job.cancel_event.is_set():
                return
            self.complete(job, result)

        thread = threading.Thread(target=runner, name=f"tradingagents-job-{job.id}", daemon=True)
        thread.start()
        return thread
