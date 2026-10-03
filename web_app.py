"""Flask web interface for TradingAgents — wraps the CLI framework for browser access.

This is a thin layer over tradingagents.graph.trading_graph.TradingAgentsGraph.
It does not modify any framework logic; it only builds a config dict from form
input and calls propagate(), the same API main.py uses.
"""

import datetime
import os
import threading
import traceback
import uuid

from flask import Flask, jsonify, render_template, request

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS

app = Flask(__name__)

# Provider display names and default backend URLs.
# Mirrors cli/prompts._llm_provider_table so the web UI resolves endpoints the
# same way the CLI does.
PROVIDER_URLS: dict[str, str | None] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/",
    "google": None,
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "qwen-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "glm": "https://api.z.ai/api/paas/v4/",
    "glm-cn": "https://open.bigmodel.cn/api/paas/v4/",
    "minimax": "https://api.minimax.io/v1",
    "minimax-cn": "https://api.minimaxi.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral": "https://api.mistral.ai/v1",
    "kimi": "https://api.moonshot.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "openai_compatible": None,
    "bedrock": None,
}

PROVIDER_DISPLAY: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "xai": "xAI",
    "deepseek": "DeepSeek",
    "qwen": "Qwen (International)",
    "qwen-cn": "Qwen (China)",
    "glm": "GLM (Z.AI)",
    "glm-cn": "GLM (BigModel China)",
    "minimax": "MiniMax (Global)",
    "minimax-cn": "MiniMax (China)",
    "openrouter": "OpenRouter",
    "mistral": "Mistral",
    "kimi": "Kimi (Moonshot)",
    "groq": "Groq",
    "nvidia": "NVIDIA NIM",
    "ollama": "Ollama (Local)",
    "openai_compatible": "OpenAI-compatible",
    "bedrock": "Amazon Bedrock",
}

# In-memory task store (sufficient for a single-instance dev preview).
_tasks: dict[str, dict] = {}


def _serialize_state(state: dict) -> dict:
    """Extract JSON-serializable report fields from the final graph state."""
    return {
        "company_of_interest": state.get("company_of_interest", ""),
        "trade_date": state.get("trade_date", ""),
        "market_report": state.get("market_report", ""),
        "sentiment_report": state.get("sentiment_report", ""),
        "news_report": state.get("news_report", ""),
        "fundamentals_report": state.get("fundamentals_report", ""),
        "investment_plan": state.get("investment_plan", ""),
        "trader_investment_plan": state.get("trader_investment_plan", ""),
        "final_trade_decision": state.get("final_trade_decision", ""),
        "investment_debate_state": state.get("investment_debate_state", {}) or {},
        "risk_debate_state": state.get("risk_debate_state", {}) or {},
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/providers")
def get_providers():
    """List all LLM providers with their API-key status."""
    result = []
    for key, display in PROVIDER_DISPLAY.items():
        if key not in MODEL_OPTIONS:
            continue
        key_env = PROVIDER_API_KEY_ENV.get(key)
        has_key = True if key_env is None else bool(os.environ.get(key_env))
        result.append({
            "key": key,
            "name": display,
            "key_env": key_env,
            "has_key": has_key,
        })
    return jsonify(result)


@app.route("/api/models/<provider>")
def get_models(provider):
    """Return the quick/deep model options for a provider."""
    provider = provider.lower()
    if provider not in MODEL_OPTIONS:
        return jsonify({"error": "Unknown provider"}), 400
    return jsonify({
        "quick": [{"label": label, "value": value}
                  for label, value in MODEL_OPTIONS[provider]["quick"]],
        "deep": [{"label": label, "value": value}
                 for label, value in MODEL_OPTIONS[provider]["deep"]],
    })


@app.route("/api/analyze", methods=["POST"])
def start_analysis():
    """Start a background analysis task. Returns a task_id for polling."""
    data = request.get_json(force=True)
    provider = (data.get("provider") or "openai").lower()

    # Validate API key before attempting graph construction.
    key_env = PROVIDER_API_KEY_ENV.get(provider)
    if key_env and not os.environ.get(key_env):
        return jsonify({
            "error": f"{key_env} is not set. Please add it in the Base44 secrets dashboard."
        }), 400

    # Build config from form input, starting from DEFAULT_CONFIG.
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = provider
    config["deep_think_llm"] = data.get("deep_model") or config["deep_think_llm"]
    config["quick_think_llm"] = data.get("quick_model") or config["quick_think_llm"]
    config["backend_url"] = PROVIDER_URLS.get(provider)

    depth = int(data.get("research_depth", 1))
    config["max_debate_rounds"] = depth
    config["max_risk_discuss_rounds"] = depth
    config["output_language"] = data.get("output_language", "English")

    analysts = data.get("analysts") or ["market", "social", "news", "fundamentals"]
    ticker = (data.get("ticker") or "SPY").strip().upper()
    date = data.get("date") or datetime.date.today().isoformat()

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "running",
        "result": None,
        "error": None,
        "ticker": ticker,
        "date": date,
    }

    def _run():
        try:
            graph = TradingAgentsGraph(
                selected_analysts=analysts, config=config, debug=False,
            )
            final_state, signal = graph.propagate(ticker, date)
            _tasks[task_id].update({
                "status": "completed",
                "result": _serialize_state(final_state),
                "signal": signal,
                "error": None,
            })
        except Exception as exc:
            _tasks[task_id].update({
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"task_id": task_id})


@app.route("/api/status/<task_id>")
def get_status(task_id):
    task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Unknown task"}), 404
    return jsonify({
        "status": task["status"],
        "error": task.get("error"),
        "ticker": task.get("ticker"),
        "date": task.get("date"),
    })


@app.route("/api/result/<task_id>")
def get_result(task_id):
    task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Unknown task"}), 404
    if task["status"] != "completed":
        return jsonify({"error": "Analysis not completed yet"}), 400
    return jsonify({
        "result": task.get("result"),
        "signal": task.get("signal"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True, threaded=True)
