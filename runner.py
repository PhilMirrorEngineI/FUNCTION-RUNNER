# runner.py — Function Runner (PMEi safe minimal)
# Flask app with health, echo chat, and optional Memory API passthrough.
# Works with: gunicorn -w 1 -k gthread -t 120 -b 0.0.0.0:$PORT runner:app

import os
import json
import time
from typing import Optional, Dict, Any

import requests
from flask import Flask, request, jsonify

# -----------------------------
# Config (env-driven, all optional)
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
# Strip trailing "/" so we can safely add paths
MEMORY_BASE_URL = (os.getenv("MEMORY_BASE_URL") or "").rstrip("/")
MEMORY_API_KEY = os.getenv("MEMORY_API_KEY", "")

# If you use PostgreSQL/Neon later, add its URL here (not required to boot)
DATABASE_URL = os.getenv("DATABASE_URL", "")

# -----------------------------
# App
# -----------------------------
app = Flask(__name__)

BOOT_TS = int(time.time())

def ok() -> Dict[str, Any]:
    return {"ok": True}

def _mem_enabled() -> bool:
    return bool(MEMORY_BASE_URL and MEMORY_API_KEY)

# -----------------------------
# Health & root
# -----------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "PMEi Function Runner",
        "status": "alive",
        "since_epoch": BOOT_TS,
        "memory_api_enabled": _mem_enabled(),
    })

@app.route("/health", methods=["GET"])
@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "uptime_seconds": int(time.time()) - BOOT_TS,
        "memory_api_enabled": _mem_enabled(),
    })

# -----------------------------
# Simple chat echo (text in -> text out)
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    """
    Minimal, predictable echo endpoint.
    Body: {"message": "...", "userEmail": "...", "meta": {...}}
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    message = (data.get("message") or "").strip()
    user_email = (data.get("userEmail") or "").strip()
    meta = data.get("meta") or {}

    if not message:
        return jsonify({"ok": False, "error": "message is required"}), 400

    # Echo response (no OpenAI call by default — predictable + cheap)
    reply = {
        "reply": f"🪞 Echo: {message}",
        "userEmail": user_email,
        "meta": meta,
        "ts": int(time.time()),
    }
    return jsonify({"ok": True, "data": reply})

# -----------------------------
# Optional Memory API passthroughs
# Only used if MEMORY_BASE_URL + MEMORY_API_KEY are set.
# -----------------------------
def _mem_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        # Use the exact header your API expects:
        "X-API-KEY": MEMORY_API_KEY,
    }

@app.route("/memory/save", methods=["POST"])
def memory_save():
    if not _mem_enabled():
        return jsonify({"ok": False, "error": "Memory API not configured"}), 503
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    url = f"{MEMORY_BASE_URL}/save_memory"
    try:
        r = requests.post(url, headers=_mem_headers(), data=json.dumps(payload), timeout=10)
        return jsonify({"upstream_status": r.status_code, "ok": r.ok, "data": safe_json(r)})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Upstream error: {e}"}), 502

@app.route("/memory/get", methods=["POST"])
def memory_get():
    if not _mem_enabled():
        return jsonify({"ok": False, "error": "Memory API not configured"}), 503
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    # Support both POST body and query args in your upstream.
    url = f"{MEMORY_BASE_URL}/get_memory"
    try:
        r = requests.post(url, headers=_mem_headers(), data=json.dumps(payload), timeout=10)
        return jsonify({"upstream_status": r.status_code, "ok": r.ok, "data": safe_json(r)})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Upstream error: {e}"}), 502

def safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text[:2000]}  # avoid huge payloads

# -----------------------------
# Local run
# -----------------------------
if __name__ == "__main__":
    # Local dev: python runner.py
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
