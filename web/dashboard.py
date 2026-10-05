#!/usr/bin/env python3
"""Web dashboard for TradingAgents — serves on port 3000.

A minimal, zero-dependency (stdlib only) web UI that wraps the TradingAgents
programmatic API so the project is visible in the Base44 preview. Shows project
status, API key presence, and a form to run an analysis.
"""

import html
import json
import os
import threading
import time
import traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

PORT = int(os.environ.get("PORT", "3000"))
HOST = "0.0.0.0"

# ---------------------------------------------------------------------------
# Job tracking for background analyses
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# API key registry — checked at request time so dashboard reflects env changes
# ---------------------------------------------------------------------------
LLM_KEYS = [
    ("OPENAI_API_KEY", "OpenAI"),
    ("GOOGLE_API_KEY", "Google Gemini"),
    ("ANTHROPIC_API_KEY", "Anthropic"),
    ("XAI_API_KEY", "xAI Grok"),
    ("DEEPSEEK_API_KEY", "DeepSeek"),
    ("OPENROUTER_API_KEY", "OpenRouter"),
    ("GROQ_API_KEY", "Groq"),
    ("MISTRAL_API_KEY", "Mistral"),
    ("NVIDIA_API_KEY", "NVIDIA"),
]

DATA_KEYS = [
    ("FRED_API_KEY", "FRED (macro data)"),
    ("SEC_EDGAR_USER_AGENT", "SEC EDGAR (filings)"),
    ("TYPESAFE_API_KEY", "TypeSafe Jev (social)"),
]

VERSION = "0.5.2"

# ---------------------------------------------------------------------------
# CSS (regular string — braces are literal, not f-string placeholders)
# ---------------------------------------------------------------------------
CSS = """
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
    padding: 2rem; max-width: 900px; margin: 0 auto;
  }
  header { text-align: center; margin-bottom: 2rem; }
  h1 { font-size: 2rem; color: #fff; }
  header p { color: var(--muted); margin-top: 0.25rem; }
  .version {
    display: inline-block; background: var(--card); border: 1px solid var(--border);
    border-radius: 999px; padding: 0.15rem 0.75rem; font-size: 0.8rem;
    color: var(--muted); margin-top: 0.5rem;
  }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;
  }
  .card h2 { font-size: 1.1rem; margin-bottom: 1rem; color: #fff; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
  code { font-family: 'SF Mono', Monaco, monospace; font-size: 0.85rem; color: var(--accent); }
  .present { color: var(--green); }
  .absent { color: var(--muted); }
  .running { color: var(--yellow); }
  .warning {
    background: rgba(210,153,34,0.1); border: 1px solid var(--yellow);
    border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; color: var(--yellow);
  }
  form { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: flex-end; }
  label { display: block; font-size: 0.8rem; color: var(--muted); margin-bottom: 0.25rem; }
  input {
    background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    padding: 0.5rem 0.75rem; color: var(--text); font-size: 0.95rem; font-family: inherit;
  }
  input:focus { outline: none; border-color: var(--accent); }
  button {
    background: var(--accent); color: #fff; border: none; border-radius: 6px;
    padding: 0.5rem 1.5rem; font-size: 0.95rem; cursor: pointer; font-weight: 600;
  }
  button:hover { opacity: 0.9; }
  .signal {
    font-size: 2.5rem; font-weight: 700; text-align: center;
    padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;
  }
  .signal.Buy, .signal.Overweight { background: rgba(63,185,80,0.15); color: var(--green); }
  .signal.Sell, .signal.Underweight { background: rgba(248,81,73,0.15); color: var(--red); }
  .signal.Hold, .signal.REVIEW { background: rgba(210,153,34,0.15); color: var(--yellow); }
  .report {
    background: var(--card); border: 1px solid var(--border); border-radius: 6px;
    padding: 1rem; margin-bottom: 1rem;
  }
  .report h3 { color: var(--accent); margin-bottom: 0.5rem; font-size: 0.95rem; }
  .report pre {
    white-space: pre-wrap; word-wrap: break-word; font-size: 0.9rem;
    font-family: 'SF Mono', Monaco, monospace; color: var(--text);
  }
  .error {
    background: rgba(248,81,73,0.1); border: 1px solid var(--red);
    border-radius: 8px; padding: 1rem; color: var(--red); margin-bottom: 1rem;
  }
  .spinner {
    display: inline-block; width: 2rem; height: 2rem;
    border: 3px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.8s linear infinite; margin-bottom: 1rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .running-info { text-align: center; padding: 3rem; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .cli-block {
    background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    padding: 1rem; font-family: 'SF Mono', Monaco, monospace; font-size: 0.85rem;
    color: var(--text);
  }
"""


def _key_status():
    """Return list of (env_name, label, present) for all keys."""
    result = []
    for name, label in LLM_KEYS + DATA_KEYS:
        result.append((name, label, bool(os.environ.get(name, "").strip())))
    return result


def _has_llm_key():
    return any(os.environ.get(name, "").strip() for name, _ in LLM_KEYS)


def _run_analysis_job(job_id, ticker, date):
    """Run analysis in a background thread."""
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = DEFAULT_CONFIG.copy()
        config["checkpoint_enabled"] = False
        graph = TradingAgentsGraph(debug=False, config=config)
        final_state, signal = graph.propagate(ticker, date)
        with _lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["final_state"] = final_state
            _jobs[job_id]["signal"] = signal
    except Exception as exc:
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"
            _jobs[job_id]["traceback"] = traceback.format_exc()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress default request logging

    # -- helpers -----------------------------------------------------------
    def _send_text(self, body, content_type, code=200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def _send_html(self, body, code=200):
        self._send_text(body, "text/html; charset=utf-8", code)

    # -- routing -----------------------------------------------------------
    def do_GET(self):
        if self.path == "/health":
            self._send_text("ok", "text/plain")
        elif self.path in ("/", ""):
            self._serve_landing()
        elif self.path.startswith("/results/"):
            self._serve_results(self.path.split("/")[-1])
        elif self.path.startswith("/api/status/"):
            self._serve_status_json(self.path.split("/")[-1])
        else:
            self._send_text("Not found", "text/plain", 404)

    def do_POST(self):
        if self.path == "/analyze":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode()
            params = parse_qs(body)
            ticker = params.get("ticker", [""])[0].strip().upper()
            date = params.get("date", [""])[0].strip()

            if not ticker or not date:
                self._send_text("Ticker and date are required", "text/plain", 400)
                return

            job_id = str(int(time.time() * 1000))
            with _lock:
                _jobs[job_id] = {
                    "status": "running",
                    "ticker": ticker,
                    "date": date,
                    "started_at": time.time(),
                }

            thread = threading.Thread(
                target=_run_analysis_job, args=(job_id, ticker, date), daemon=True
            )
            thread.start()

            self.send_response(302)
            self.send_header("Location", f"/results/{job_id}")
            self.end_headers()
        else:
            self._send_text("Not found", "text/plain", 404)

    # -- pages -------------------------------------------------------------
    def _serve_landing(self):
        keys = _key_status()
        has_key = _has_llm_key()

        llm_rows = ""
        for name, label, present in keys[: len(LLM_KEYS)]:
            cls = "present" if present else "absent"
            dot = "\u2705" if present else "\u26aa"
            llm_rows += (
                f"<tr><td>{html.escape(label)}</td>"
                f"<td><code>{html.escape(name)}</code></td>"
                f'<td class="{cls}">{dot}</td></tr>'
            )

        data_rows = ""
        for name, label, present in keys[len(LLM_KEYS) :]:
            cls = "present" if present else "absent"
            dot = "\u2705" if present else "\u26aa"
            data_rows += (
                f"<tr><td>{html.escape(label)}</td>"
                f"<td><code>{html.escape(name)}</code></td>"
                f'<td class="{cls}">{dot}</td></tr>'
            )

        warning = ""
        if not has_key:
            warning = (
                '<div class="warning">\u26a0\ufe0f No LLM API key detected. Set at least one '
                "(e.g. <code>OPENAI_API_KEY</code>) via the Base44 secrets dashboard "
                "to run analyses.</div>"
            )

        # Recent jobs
        with _lock:
            recent = sorted(
                _jobs.items(), key=lambda x: x[1].get("started_at", 0), reverse=True
            )[:5]
        recent_html = ""
        for jid, job in recent:
            st = job["status"]
            cls = {"done": "present", "error": "absent", "running": "running"}.get(st, "")
            recent_html += (
                f"<tr><td><a href='/results/{jid}'>{html.escape(job['ticker'])}</a></td>"
                f"<td>{html.escape(job['date'])}</td>"
                f'<td class="{cls}">{st}</td></tr>'
            )
        recent_card = ""
        if recent_html:
            recent_card = (
                '<div class="card"><h2>Recent Analyses</h2>'
                "<table><thead><tr><th>Ticker</th><th>Date</th><th>Status</th></tr></thead>"
                f"<tbody>{recent_html}</tbody></table></div>"
            )

        page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TradingAgents</title>
<style>{CSS}</style>
</head>
<body>
  <header>
    <h1>TradingAgents</h1>
    <p>Multi-Agents LLM Financial Trading Framework</p>
    <span class="version">v{VERSION}</span>
  </header>

  {warning}

  <div class="card">
    <h2>Run Analysis</h2>
    <form method="POST" action="/analyze">
      <div>
        <label for="ticker">Ticker</label>
        <input type="text" id="ticker" name="ticker" placeholder="NVDA" required>
      </div>
      <div>
        <label for="date">Analysis Date</label>
        <input type="text" id="date" name="date" placeholder="2026-09-01" required>
      </div>
      <button type="submit">Run Analysis</button>
    </form>
  </div>

  <div class="card">
    <h2>LLM API Keys</h2>
    <table>
      <thead><tr><th>Provider</th><th>Env Var</th><th>Status</th></tr></thead>
      <tbody>{llm_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Data Vendor Keys (Optional)</h2>
    <table>
      <thead><tr><th>Source</th><th>Env Var</th><th>Status</th></tr></thead>
      <tbody>{data_rows}</tbody>
    </table>
  </div>

  {recent_card}

  <div class="card">
    <h2>CLI Usage</h2>
    <p style="color:var(--muted);font-size:0.9rem;margin-bottom:0.75rem">
      The CLI is also available inside the container:
    </p>
    <div class="cli-block">
      tradingagents --ticker NVDA --date 2026-09-01<br>
      tradingagents --help
    </div>
  </div>
</body>
</html>"""
        self._send_html(page)

    def _serve_results(self, job_id):
        with _lock:
            job = _jobs.get(job_id)
        if not job:
            self._send_html(
                "<h1>Job not found</h1><p><a href='/'>\u2190 Back</a></p>", 404
            )
            return

        ticker = html.escape(job.get("ticker", ""))
        date = html.escape(job.get("date", ""))
        status = job["status"]

        if status == "running":
            page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analyzing {ticker} \u2014 TradingAgents</title>
<style>{CSS}</style>
</head>
<body>
  <div class="running-info">
    <div class="spinner"></div>
    <h1>Analyzing {ticker}</h1>
    <p>Analysis date: {date}</p>
    <p id="status">Running multi-agent analysis\u2026</p>
    <p><a href="/">\u2190 Cancel and go back</a></p>
  </div>
  <script>
    async function poll() {{
      try {{
        const res = await fetch('/api/status/{job_id}');
        const data = await res.json();
        if (data.status === 'done' || data.status === 'error') {{
          window.location.reload();
        }}
        if (data.status === 'running' && data.elapsed) {{
          document.getElementById('status').textContent =
            'Running multi-agent analysis\u2026 (' + Math.round(data.elapsed) + 's)';
        }}
      }} catch(e) {{}}
    }}
    setInterval(poll, 3000);
  </script>
</body>
</html>"""
            self._send_html(page)
            return

        if status == "error":
            error = html.escape(job.get("error", "Unknown error"))
            tb = html.escape(job.get("traceback", ""))
            page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>Error \u2014 TradingAgents</title>
<style>{CSS}</style>
</head>
<body>
  <header><h1>Analysis Failed: {ticker}</h1><p>Date: {date}</p></header>
  <div class="error">{error}</div>
  <div class="report"><pre>{tb}</pre></div>
  <p style="margin-top:1rem"><a href="/">\u2190 Run another analysis</a></p>
</body>
</html>"""
            self._send_html(page)
            return

        # status == done
        final_state = job.get("final_state", {})
        signal = html.escape(job.get("signal", "N/A"))
        valid_signals = ("Buy", "Overweight", "Hold", "Underweight", "Sell", "REVIEW")
        signal_class = signal if signal in valid_signals else "Hold"

        report_sections = []
        for key, title in [
            ("final_trade_decision", "Final Trade Decision"),
            ("market_report", "Market Analyst"),
            ("sentiment_report", "Sentiment Analyst"),
            ("news_report", "News Analyst"),
            ("fundamentals_report", "Fundamentals Analyst"),
        ]:
            content = final_state.get(key, "")
            if content:
                report_sections.append(
                    f'<div class="report"><h3>{title}</h3><pre>{html.escape(content)}</pre></div>'
                )
        reports_html = "\n".join(report_sections) if report_sections else "<p>No reports available.</p>"

        page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{ticker} \u2014 TradingAgents</title>
<style>{CSS}</style>
</head>
<body>
  <header>
    <h1>{ticker}</h1>
    <p>Analysis date: {date}</p>
  </header>
  <div class="signal {signal_class}">{signal}</div>
  {reports_html}
  <p style="margin-top:1.5rem"><a href="/">\u2190 Run another analysis</a></p>
</body>
</html>"""
        self._send_html(page)

    def _serve_status_json(self, job_id):
        with _lock:
            job = _jobs.get(job_id)
        if not job:
            self._send_text(json.dumps({"error": "not found"}), "application/json", 404)
            return
        elapsed = time.time() - job.get("started_at", time.time())
        data = {"status": job["status"], "elapsed": round(elapsed, 1)}
        self._send_text(json.dumps(data), "application/json")


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"TradingAgents dashboard on http://{HOST}:{PORT}", flush=True)
    server.serve_forever()
