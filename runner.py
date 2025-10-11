import os, time, json
from flask import Flask, request, jsonify
from functools import wraps
from openai import OpenAI

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "").strip()
ASSISTANT_ID     = os.environ.get("ASSISTANT_ID", "").strip()  # optional; ok if blank
RUNNER_ACTION_KEY= os.environ.get("RUNNER_ACTION_KEY", "").strip()
RUN_MAX_SECS     = int(os.environ.get("RUN_MAX_SECS", "25"))

app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

def require_runner_key(fn):
    @wraps(fn)
    def w(*a, **k):
        if request.method == "OPTIONS" or request.path in ("/", "/health"):
            return fn(*a, **k)
        auth = request.headers.get("Authorization", "")
        key  = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else request.headers.get("X-API-KEY","")
        if not key or key != RUNNER_ACTION_KEY:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        return fn(*a, **k)
    return w

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-KEY"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r

@app.route("/")
def root():
    return jsonify({"ok": True, "service": "Function-Runner", "routes": ["/health","/chat"]})

@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": int(time.time())})

@app.errorhandler(Exception)
def oops(e):
    return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/chat", methods=["POST", "OPTIONS"])
@require_runner_key
def chat():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    body = request.get_json(silent=True) or {}
    user_msg = str(body.get("message", "")).strip()
    if not user_msg:
        return jsonify({"ok": False, "error": "Missing 'message'"}), 400
    if not OPENAI_API_KEY:
        return jsonify({"ok": False, "error": "OPENAI_API_KEY not set"}), 500

    # Use Completions directly (simple, reliable). You can swap to Assistants later.
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are Dave, a concise helpful assistant."},
                      {"role":"user","content":user_msg}],
            temperature=0.3,
            timeout=RUN_MAX_SECS
        )
        text = (resp.choices[0].message.content or "").strip()
        return jsonify({"ok": True, "type": "assistant_message", "content": text})
    except Exception as e:
        return jsonify({"ok": False, "error": f"OpenAI call failed: {e}"}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","8000")))
