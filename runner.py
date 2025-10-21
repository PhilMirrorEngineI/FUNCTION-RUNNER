# runner.py — PMEi Function Runner (companion to server.py)
# Start with:
#   gunicorn -w 1 -k gthread -t 120 -b 0.0.0.0:$PORT function_run:app

import os
import json
import time
from typing import Any, Dict, Optional, Tuple

import requests
from flask import Flask, request, jsonify
import threading  # added for keepalive thread

# -----------------------------
# Config (env-driven)
# -----------------------------
OPENAI_API_KEY   = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL     = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"
MEMORY_BASE_URL  = (os.getenv("MEMORY_BASE_URL") or "").rstrip("/")   # IMPORTANT: no trailing slash
MEMORY_API_KEY   = (os.getenv("MEMORY_API_KEY") or "").strip()
TAVILY_API_KEY   = (os.getenv("TAVILY_API_KEY") or "").strip()        # reserved (unused here)

# Optional OpenAI initialization (safe if key missing)
try:
    from openai import OpenAI  # openai >= 1.x
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    _openai_client = None

# -----------------------------
# App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Keepalive Thread (prevents Render idle sleep)
# -----------------------------
def _keepalive():
    """Self-ping loop to keep the Render container awake."""
    url = os.getenv("SELF_HEALTH_URL", "").strip()
    interval = int(os.getenv("KEEPALIVE_INTERVAL", "240"))  # default = 4 minutes
    if not url:
        print("[KEEPALIVE] SELF_HEALTH_URL not set; skipping loop.")
        return
    print(f"[KEEPALIVE] Active; pinging {url} every {interval} seconds")
    while True:
        try:
            r = requests.get(url, timeout=10)
            print(f"[KEEPALIVE] Ping {url} -> {r.status_code}")
        except Exception as e:
            print(f"[KEEPALIVE ERROR] {e}")
        time.sleep(interval)

_enable = os.getenv("ENABLE_KEEPALIVE", "true").lower()
if _enable in ("1", "true", "yes", "on"):
    threading.Thread(target=_keepalive, daemon=True).start()
    print("[KEEPALIVE] Background thread started")
else:
    print("[KEEPALIVE] Disabled by env var")

BOOT_TS = int(time.time())

# -----------------------------
# Helpers
# -----------------------------
def jok(data: Any = None, **extra):
    payload = {"ok": True}
    if data is not None:
        payload["data"] = data
    if extra:
        payload.update(extra)
    return jsonify(payload)

def jfail(msg: str, code: int = 400, **extra):
    payload = {"ok": False, "error": msg}
    if extra:
        payload.update(extra)
    return jsonify(payload), code

def get_json() -> Tuple[Optional[dict], Optional[Tuple[Any, int]]]:
    try:
        data = request.get_json(force=True) or {}
        if not isinstance(data, dict):
            return None, jfail("JSON body must be an object", 400)
        return data, None
    except Exception:
        return None, jfail("Invalid or missing JSON body", 400)

def mem_enabled() -> bool:
    return bool(MEMORY_BASE_URL and MEMORY_API_KEY)

def mem_headers() -> Dict[str, str]:
    # Header name aligned with your Memory API
    return {"Content-Type": "application/json", "X-API-KEY": MEMORY_API_KEY}

def safe_upstream_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text[:2000], "status": resp.status_code}

# -----------------------------
# Root & Health
# -----------------------------
@app.route("/", methods=["GET"])
def root():
    return jok({
        "service": "PMEi Function Runner",
        "status": "alive",
        "since_epoch": BOOT_TS,
        "openai_enabled": bool(_openai_client),
        "memory_api_enabled": mem_enabled(),
    })

@app.route("/health", methods=["GET"])
@app.route("/healthz", methods=["GET"])
def health():
    return jok({
        "uptime_seconds": int(time.time()) - BOOT_TS,
        "openai_enabled": bool(_openai_client),
        "memory_api_enabled": mem_enabled(),
    })

# -----------------------------
# Simple chat echo (cheap, predictable)
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data, err = get_json()
    if err:
        return err
    msg = (data.get("message") or "").strip()
    user_email = (data.get("userEmail") or "").strip()
    meta = data.get("meta") or {}
    if not msg:
        return jfail("message is required", 400)
    return jok({
        "reply": f"🪞 Echo: {msg[:2000]}",
        "userEmail": user_email,
        "meta": meta,
        "ts": int(time.time()),
    })

# -----------------------------
# OpenAI passthrough (optional)
# -----------------------------
@app.route("/openai/chat", methods=["POST"])
def openai_chat():
    if not _openai_client:
        return jfail("OpenAI not configured", 503)
    data, err = get_json()
    if err:
        return err
    message = (data.get("message") or "").strip()
    system = (data.get("system") or "You are a concise, helpful assistant.").strip()
    model = (data.get("model") or OPENAI_MODEL).strip() or OPENAI_MODEL
    temperature = float(data.get("temperature") or 0.2)
    max_tokens = int(data.get("max_tokens") or 512)
    if not message:
        return jfail("message is required", 400)
    if max_tokens > 4096:
        max_tokens = 4096
    try:
        resp = _openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": message}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content if resp and resp.choices else ""
        return jok({"model": model, "reply": text})
    except Exception as e:
        return jfail(f"OpenAI error: {e}", 502)

# -----------------------------
# Memory API passthroughs
# -----------------------------
@app.route("/memory/save", methods=["POST"])
def memory_save():
    if not mem_enabled():
        return jfail("Memory API not configured", 503)
    data, err = get_json()
    if err:
        return err
    try:
        r = requests.post(f"{MEMORY_BASE_URL}/save_memory",
                          headers=mem_headers(),
                          data=json.dumps(data),
                          timeout=12)
        return jsonify({
            "ok": r.ok,
            "upstream_status": r.status_code,
            "data": safe_upstream_json(r),
        }), (200 if r.ok else 502)
    except Exception as e:
        return jfail(f"Upstream error: {e}", 502)

@app.route("/memory/get", methods=["POST"])
def memory_get():
    if not mem_enabled():
        return jfail("Memory API not configured", 503)
    data, err = get_json()
    if err:
        return err
    try:
        r = requests.post(f"{MEMORY_BASE_URL}/get_memory",
                          headers=mem_headers(),
                          data=json.dumps(data),
                          timeout=12)
        return jsonify({
            "ok": r.ok,
            "upstream_status": r.status_code,
            "data": safe_upstream_json(r),
        }), (200 if r.ok else 502)
    except Exception as e:
        return jfail(f"Upstream error: {e}", 502)

# -----------------------------
# Local dev
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
