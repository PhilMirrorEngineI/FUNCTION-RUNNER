import os
import time
import json
import shlex
import logging
from functools import wraps

import requests
from flask import Flask, request, jsonify
from openai import OpenAI

# ── Env ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY     = os.environ["OPENAI_API_KEY"]                      # sk-...
ASSISTANT_ID       = os.environ["ASSISTANT_ID"]                        # asst_...
MEMORY_BASE_URL    = os.environ.get("MEMORY_BASE_URL", "https://davepmei-ai.onrender.com").rstrip("/")
MEMORY_API_KEY     = os.environ["MEMORY_API_KEY"]
RUNNER_ACTION_KEY  = os.environ["RUNNER_ACTION_KEY"]                   # Bearer gate for /chat

# ── Clients / App ────────────────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Global JSON error and JSON-only responses ────────────────────────────────
@app.errorhandler(Exception)
def handle_any_error(e):
    code = getattr(e, "code", 500)
    logging.exception("Unhandled error: %s", e)
    # 200 so frontends show the message, not a hard 500 page
    return jsonify({"ok": False, "error": str(e), "code": code}), 200

@app.after_request
def force_json_headers(resp):
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp

# ── Auth helper (protect /chat) ──────────────────────────────────────────────
def require_runner_key(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if request.method == "OPTIONS" or request.path in ("/", "/health"):
            return fn(*args, **kwargs)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"ok": False, "error": "Missing Bearer token"}), 401
        token = auth.split(" ", 1)[1].strip()
        if not token or token != RUNNER_ACTION_KEY:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapped

# ── Helpers ──────────────────────────────────────────────────────────────────
def mem_call(path: str, method: str = "GET", params=None, body=None):
    """Call the Memory API with auth; always return JSON or raise for caller to catch."""
    url = f"{MEMORY_BASE_URL}{path}"
    headers = {"X-API-KEY": MEMORY_API_KEY, "Content-Type": "application/json"}
    try:
        if method.upper() == "GET":
            r = requests.get(url, headers=headers, params=params or {}, timeout=20)
        else:
            r = requests.post(url, headers=headers, json=body or {}, timeout=20)
    except requests.RequestException as rexc:
        raise RuntimeError(f"memory_api network error: {rexc}") from rexc

    # Attempt JSON; on non-200 Render may still send HTML
    ct = r.headers.get("content-type", "")
    if "application/json" in ct:
        data = r.json()
    else:
        raise RuntimeError(f"memory_api non-JSON response (status {r.status_code})")

    # Bubble up HTTP failures with JSON payload
    if not r.ok:
        raise RuntimeError(f"memory_api error {r.status_code}: {data}")

    return data

def parse_message_kv(message: str) -> dict:
    """Parse free-form 'key=value' pairs (quotes supported)."""
    args = {}
    for tok in shlex.split(message or ""):
        if "=" in tok:
            k, v = tok.split("=", 1)
            args[k.strip()] = v.strip()
    return args

def handle_tool_call(tc):
    """
    Handle a tool call from the Assistant.
    Supports either unified bridge (function_runner(message="...")) or legacy tools.
    """
    name = getattr(tc.function, "name", "") or ""
    raw_args = json.loads(getattr(tc.function, "arguments", "") or "{}")

    if name == "function_runner":
        msg = raw_args.get("message", "")
        args = parse_message_kv(msg)
    else:
        args = dict(raw_args)

    # Defaults (do not override provided)
    defaults = {
        "slide_id": "t-001",
        "glyph_echo": "🪞",
        "drift_score": 0.00,     # clamp to lawful default
        "seal": "LAWFUL",
        "limit": 5,
        "content": "(no content provided)",
    }
    for k, v in defaults.items():
        args.setdefault(k, v)

    operation = (args.get("operation") or name or "").strip().lower()

    # Reads
    if operation in ("memory_bridge", "get_memory", "recall_memory_window"):
        params = {k: args.get(k) for k in ("user_id", "thread_id", "limit") if args.get(k) is not None}
        out = mem_call("/get_memory", "GET", params=params)
        return tc.id, json.dumps(out)

    # Writes
    if operation in ("save_memory", "reflect_and_store_memory"):
        out = mem_call("/save_memory", "POST", body={
            "user_id":     args.get("user_id", ""),
            "thread_id":   args.get("thread_id", ""),
            "slide_id":    args.get("slide_id"),
            "glyph_echo":  args.get("glyph_echo"),
            "drift_score": float(args.get("drift_score") or 0.0),
            "seal":        args.get("seal"),
            "content":     args.get("content"),
        })
        return tc.id, json.dumps(out)

    # Unknown op
    return tc.id, json.dumps({
        "ok": False,
        "error": f"unknown operation '{operation}'",
        "received": {"name": name, "args": args}
    })

def run_once(user_msg: str) -> dict:
    """
    Create a thread, run once with tool loop, and return a dict response.
    We return a structured dict so /chat can always jsonify cleanly.
    """
    try:
        th = client.beta.threads.create()
        client.beta.threads.messages.create(th.id, role="user", content=user_msg)

        run = client.beta.threads.runs.create(
            thread_id=th.id,
            assistant_id=ASSISTANT_ID,
            tool_choice="auto"
        )

        started = time.time()
        MAX_SECS = 45
        SLEEP_SECS = 0.6

        while True:
            run = client.beta.threads.runs.retrieve(thread_id=th.id, run_id=run.id)
            status = run.status

            if status == "requires_action":
                calls = run.required_action.submit_tool_outputs.tool_calls
                outs = []
                for tc in calls:
                    try:
                        tid, output = handle_tool_call(tc)
                    except Exception as e:
                        tid = tc.id
                        output = json.dumps({"ok": False, "error": f"runner exception: {e}"})
                    outs.append({"tool_call_id": tid, "output": output})

                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=th.id,
                    run_id=run.id,
                    tool_outputs=outs
                )

            elif status in ("completed", "failed", "cancelled", "expired"):
                break

            if time.time() - started > MAX_SECS:
                return {"ok": False, "error": "timeout waiting for assistant run"}

            time.sleep(SLEEP_SECS)

        # Collect most recent assistant text
        msgs = client.beta.threads.messages.list(thread_id=th.id, order="desc").data
        for m in msgs:
            if m.role == "assistant":
                parts = []
                for part in m.content:
                    if getattr(part, "type", "") == "text":
                        parts.append(part.text.value)
                return {"ok": True, "assistant": "\n".join(parts) if parts else ""}

        return {"ok": True, "assistant": ""}

    except Exception as e:
        # Catch OpenAI/network/config errors and return JSON
        logging.exception("OpenAI run_once error: %s", e)
        return {"ok": False, "source": "openai", "error": str(e)}

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": int(time.time())})

@app.route("/")
def root():
    return jsonify({
        "ok": True,
        "service": "FUNCTION-RUNNER",
        "endpoints": ["/health", "/chat"]
    })

@app.route("/chat", methods=["POST"])
@require_runner_key
def chat():
    """
    Accepts either:
      { "message": "free text" }
    or a structured JSON which is converted to key=value tokens for tools.
    """
    data = request.get_json(silent=True) or {}
    if "message" in data and isinstance(data["message"], str):
        msg = data["message"]
    else:
        # build message if user posts JSON like {"operation":"get_memory","user_id":"phil",...}
        kv_parts = []
        for k, v in data.items():
            if isinstance(v, str):
                v_out = v if " " not in v else json.dumps(v)
            else:
                v_out = json.dumps(v)
            kv_parts.append(f"{k}={v_out}")
        msg = " ".join(kv_parts) if kv_parts else "operation=get_memory user_id=phil thread_id=smoke limit=3"

    app.logger.info(">> Received: %s", msg)
    out = run_once(msg)
    app.logger.info("<< Reply: %s", out)

    # Always JSON. If OpenAI/Memory failed, out will contain ok:false + error.
    return jsonify(out), 200

# ── Local dev ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
