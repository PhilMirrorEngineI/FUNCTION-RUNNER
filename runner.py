# runner.py — PMEi Function Runner (lawful triple-ping warm build)
# Start with:
#   gunicorn -w 1 -k gthread -t 120 -b 0.0.0.0:$PORT function_run:app
#
# Env:
#   SELF_HEALTH_URL      = https://function-runner.onrender.com/health
#   KEEPALIVE_INTERVAL   = 60
#   ENABLE_KEEPALIVE     = true
#   MEMORY_BASE_URL      = https://davepmei-ai.onrender.com
#   MEMORY_API_KEY       = <secret>
#   OPENAI_API_KEY       = <optional>
#   OPENAI_MODEL         = gpt-4o-mini

import os, json, time, threading, requests
from typing import Any, Dict, Optional, Tuple
from flask import Flask, request, jsonify

# ---------- Config ----------
OPENAI_API_KEY   = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL     = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"
MEMORY_BASE_URL  = (os.getenv("MEMORY_BASE_URL") or "").rstrip("/")
MEMORY_API_KEY   = (os.getenv("MEMORY_API_KEY") or "").strip()

try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    _openai_client = None

app = Flask(__name__)
BOOT_TS = int(time.time())

# ---------- Helpers ----------
def jok(data: Any = None, **extra):
    p = {"ok": True}
    if data is not None: p["data"] = data
    p.update(extra)
    return jsonify(p)

def jfail(msg: str, code: int = 400, **extra):
    p = {"ok": False, "error": msg}
    p.update(extra)
    return jsonify(p), code

def get_json() -> Tuple[Optional[dict], Optional[Tuple[Any, int]]]:
    try:
        d = request.get_json(force=True) or {}
        if not isinstance(d, dict): return None, jfail("JSON body must be an object", 400)
        return d, None
    except Exception: return None, jfail("Invalid or missing JSON body", 400)

def mem_enabled(): return bool(MEMORY_BASE_URL and MEMORY_API_KEY)
def mem_headers(): return {"Content-Type":"application/json","X-API-KEY":MEMORY_API_KEY}
def safe_upstream_json(r: requests.Response):
    try: return r.json()
    except Exception: return {"raw": r.text[:2000], "status": r.status_code}

# ---------- Keepalive (60s triple ping) ----------
def _keepalive():
    url = os.getenv("SELF_HEALTH_URL", "").strip()
    interval = int(os.getenv("KEEPALIVE_INTERVAL", "60"))
    if not url:
        print("[KEEPALIVE] Disabled (no SELF_HEALTH_URL)")
        return
    print(f"[KEEPALIVE] Active: triple ping to {url} every {interval}s")
    while True:
        for i in range(3):
            try:
                r = requests.get(url, timeout=10)
                print(f"[KEEPALIVE] Ping {i+1}/3 -> {r.status_code} @ {int(time.time())}")
            except Exception as e:
                print(f"[KEEPALIVE] Error {i+1}/3: {e}")
            time.sleep(2)
        time.sleep(interval)

# ---------- Triple Warmup ----------
def _triple_warmup():
    target = f"{MEMORY_BASE_URL}/health" if MEMORY_BASE_URL else None
    if not target:
        print("[WARMUP] Skipped (no MEMORY_BASE_URL)")
        return
    print(f"[WARMUP] Starting triple ghost ping to {target}")
    for i in range(3):
        try:
            r = requests.get(target, timeout=5)
            print(f"[WARMUP] Ghost ping {i+1}/3 -> {r.status_code}")
        except Exception as e:
            print(f"[WARMUP] Ghost ping {i+1}/3 failed: {e}")
        time.sleep(3)
    print("[WARMUP] Triple ping complete.")

# ---------- Threads ----------
_enable = os.getenv("ENABLE_KEEPALIVE", "true").lower()
if _enable in ("1", "true", "yes", "on"):
    threading.Thread(target=_keepalive, daemon=True).start()
    print("[KEEPALIVE] Thread started")
else:
    print("[KEEPALIVE] Disabled by env var")

threading.Thread(target=_triple_warmup, daemon=True).start()

# ---------- Routes ----------
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

@app.route("/chat", methods=["POST"])
def chat():
    d, err = get_json()
    if err: return err
    msg = (d.get("message") or "").strip()
    if not msg: return jfail("message is required", 400)
    return jok({"reply": f"🪞 Echo: {msg[:2000]}", "ts": int(time.time())})

@app.route("/openai/chat", methods=["POST"])
def openai_chat():
    if not _openai_client:
        return jfail("OpenAI not configured", 503)
    d, err = get_json()
    if err: return err
    msg = (d.get("message") or "").strip()
    sys = (d.get("system") or "You are a concise, helpful assistant.").strip()
    model = (d.get("model") or OPENAI_MODEL).strip() or OPENAI_MODEL
    temperature = float(d.get("temperature") or 0.2)
    max_tokens = min(int(d.get("max_tokens") or 512), 4096)
    if not msg: return jfail("message is required", 400)
    try:
        resp = _openai_client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys},{"role":"user","content":msg}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content if resp and resp.choices else ""
        return jok({"model": model, "reply": text})
    except Exception as e:
        return jfail(f"OpenAI error: {e}", 502)

@app.route("/memory/save", methods=["POST"])
def memory_save():
    if not mem_enabled(): return jfail("Memory API not configured", 503)
    d, err = get_json()
    if err: return err
    try:
        r = requests.post(f"{MEMORY_BASE_URL}/save_memory",
                          headers=mem_headers(),
                          data=json.dumps(d),
                          timeout=30)
        return jsonify({"ok": r.ok, "upstream_status": r.status_code, "data": safe_upstream_json(r)}), (200 if r.ok else 502)
    except Exception as e:
        return jfail(f"Upstream error: {e}", 502)

@app.route("/memory/get", methods=["POST"])
def memory_get():
    if not mem_enabled(): return jfail("Memory API not configured", 503)
    d, err = get_json()
    if err: return err
    try:
        r = requests.post(f"{MEMORY_BASE_URL}/get_memory",
                          headers=mem_headers(),
                          data=json.dumps(d),
                          timeout=12)
        return jsonify({"ok": r.ok, "upstream_status": r.status_code, "data": safe_upstream_json(r)}), (200 if r.ok else 502)
    except Exception as e:
        return jfail(f"Upstream error: {e}", 502)

# ---------- Local dev ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
