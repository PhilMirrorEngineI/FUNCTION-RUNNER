# ────────────────────────────────────────────────
# runner.py — PMEi Function Runner (Lawful Memory Core)
# ────────────────────────────────────────────────
# Run with:
#   gunicorn -w 1 -k gthread -t 120 -b 0.0.0.0:$PORT runner:app
#
# Environment Variables (Render Dashboard):
#   SELF_HEALTH_URL    = https://function-runner.onrender.com/health
#   KEEPALIVE_INTERVAL = 30
#   ENABLE_KEEPALIVE   = true
#   OPENAI_API_KEY     = <optional>
#   OPENAI_MODEL       = gpt-4o-mini
#   SERVICE_NAME       = function-runner
#   ⚠️ DO NOT SET MEMORY_BASE_URL HERE — this is the core, not a proxy.
# ────────────────────────────────────────────────

import os, json, time, threading, requests
from flask import Flask, request, jsonify
from typing import Any, Dict, Optional, Tuple

# ---------- Configuration ----------
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()

try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    _openai_client = None

app = Flask(__name__)
BOOT_TS = int(time.time())

# ---------- Utility Helpers ----------
def jok(data: Any = None, **extra):
    p = {"ok": True}
    if data is not None:
        p["data"] = data
    p.update(extra)
    return jsonify(p)

def jfail(msg: str, code: int = 400, **extra):
    p = {"ok": False, "error": msg}
    p.update(extra)
    return jsonify(p), code

def get_json() -> Tuple[Optional[dict], Optional[Tuple[Any, int]]]:
    try:
        d = request.get_json(force=True) or {}
        if not isinstance(d, dict):
            return None, jfail("JSON body must be an object", 400)
        return d, None
    except Exception:
        return None, jfail("Invalid or missing JSON body", 400)

# ---------- Keepalive ----------
def _keepalive():
    url = os.getenv("SELF_HEALTH_URL", "").strip()
    interval = int(os.getenv("KEEPALIVE_INTERVAL", "30"))
    if not url:
        print("[KEEPALIVE] Disabled (no SELF_HEALTH_URL)")
        return
    print(f"[KEEPALIVE] Pinging {url} every {interval}s")
    while True:
        try:
            r = requests.get(url, timeout=10)
            print(f"[KEEPALIVE] Ping → {r.status_code} @ {int(time.time())}")
        except Exception as e:
            print(f"[KEEPALIVE] Error: {e}")
        time.sleep(interval)

if os.getenv("ENABLE_KEEPALIVE", "true").lower() in ("1", "true", "yes", "on"):
    threading.Thread(target=_keepalive, daemon=True).start()

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def root():
    return jok({
        "service": "PMEi Function Runner",
        "status": "alive",
        "since_epoch": BOOT_TS,
        "openai_enabled": bool(_openai_client),
    })

@app.route("/health", methods=["GET"])
@app.route("/healthz", methods=["GET"])
def health():
    return jok({
        "uptime_seconds": int(time.time()) - BOOT_TS,
        "openai_enabled": bool(_openai_client),
        "service": "function-runner"
    })

# ---------- Echo & OpenAI Chat ----------
@app.route("/chat", methods=["POST"])
def chat():
    d, err = get_json()
    if err:
        return err
    msg = (d.get("message") or "").strip()
    if not msg:
        return jfail("message required", 400)
    return jok({"reply": f"🪞 Echo: {msg[:2000]}", "ts": int(time.time())})

@app.route("/openai/chat", methods=["POST"])
def openai_chat():
    if not _openai_client:
        return jfail("OpenAI not configured", 503)
    d, err = get_json()
    if err:
        return err
    msg = (d.get("message") or "").strip()
    sys = (d.get("system") or "You are a concise, lawful assistant.").strip()
    model = (d.get("model") or OPENAI_MODEL).strip()
    temperature = float(d.get("temperature") or 0.2)
    max_tokens = min(int(d.get("max_tokens") or 512), 4096)
    if not msg:
        return jfail("message required", 400)
    try:
        resp = _openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": msg}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content if resp and resp.choices else ""
        return jok({"model": model, "reply": text})
    except Exception as e:
        return jfail(f"OpenAI error: {e}", 502)

# ---------- Memory API (Local Persistence Layer) ----------
@app.route("/memory/save", methods=["POST"])
def memory_save():
    d, err = get_json()
    if err:
        return err
    print(f"[Memory] Save request from user={d.get('user_id')} thread={d.get('thread_id')}")
    # Simulated persistence
    shard = {
        "user_id": d.get("user_id", "public"),
        "thread_id": d.get("thread_id", "general"),
        "content": d.get("content", ""),
        "drift_score": d.get("drift_score", 0.0),
        "ts": int(time.time())
    }
    return jok({"saved": True, "record": shard})

@app.route("/memory/get", methods=["POST"])
def memory_get():
    d, err = get_json()
    if err:
        return err
    user = d.get("user_id", "public")
    thread = d.get("thread_id", "general")
    limit = int(d.get("limit") or 10)
    print(f"[Memory] Get request for user={user} thread={thread} limit={limit}")
    # Simulated retrieval
    records = [
        {"ts": int(time.time()) - i * 60, "content": f"Sample memory {i+1}"}
        for i in range(limit)
    ]
    return jok({"ok": True, "user_id": user, "thread_id": thread, "records": records})

# ---------- Local Run ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=True)
