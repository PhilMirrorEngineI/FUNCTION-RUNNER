import os
import time
import json
from flask import Flask, request, jsonify
import requests
from openai import OpenAI

# ── Env ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
ASSISTANT_ID    = os.environ["ASSISTANT_ID"]  # your “Dave v1 api” Assistant ID
MEMORY_BASE_URL = os.environ.get("MEMORY_BASE_URL", "https://dave-runner.onrender.com")
MEMORY_API_KEY  = os.environ["MEMORY_API_KEY"]

# OpenAI client (SDK v1.x reads key from env; passing explicitly is fine too)
client = OpenAI(api_key=OPENAI_API_KEY)

# Flask app
app = Flask(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────
def mem_call(path: str, method: str = "GET", params=None, body=None):
    """Call Dave Runner memory API with auth header; always return JSON or raise."""
    url = f"{MEMORY_BASE_URL}{path}"
    headers = {
        "X-API-KEY": MEMORY_API_KEY,
        "Content-Type": "application/json",
    }
    if method.upper() == "GET":
        r = requests.get(url, headers=headers, params=params or {}, timeout=20)
    else:
        r = requests.post(url, headers=headers, json=body or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def handle_tool_call(tc):
    """
    Route tool/function calls from the Assistant to the memory API.
    - Accepts either the Playground `operation` arg or the function name.
    - Fills in reasonable defaults so Playground 'strict' doesn’t break runs.
    """
    name = getattr(tc.function, "name", None) or ""
    args = json.loads(getattr(tc.function, "arguments", "") or "{}")

    # Auto-fill defaults for optional keys (won’t override explicitly provided values)
    defaults = {
        "slide_id": "t-001",
        "glyph_echo": "🪞",
        "drift_score": 0.05,
        "seal": "lawful",
        "limit": 5,
        "content": "(no content provided)"
    }
    for k, v in defaults.items():
        args.setdefault(k, v)

    # Either explicit 'operation' (Playground freeform) or the tool's function name
    operation = (args.get("operation") or name or "").strip()

    # ── Read routes ───────────────────────────────────────────────────────────
    if operation in ("memory_bridge", "get_memory", "recall_memory_window"):
        params = {k: args.get(k) for k in ("user_id", "thread_id", "limit") if args.get(k) is not None}
        out = mem_call("/get_memory", "GET", params=params)
        return tc.id, json.dumps(out)

    # ── Write routes ──────────────────────────────────────────────────────────
    if operation in ("save_memory", "reflect_and_store_memory"):
        # For reflect_and_store_memory, system prompt should have composed a summary in `content`.
        out = mem_call("/save_memory", "POST", body=args)
        return tc.id, json.dumps(out)

    # Unknown / unhandled
    return tc.id, json.dumps({"ok": False, "error": f"unknown operation '{operation}'", "received": {"name": name, "args": args}})

def run_once(user_msg: str) -> str:
    """Create a thread, run once with tool loop, return last assistant text."""
    # 1) create thread + user message
    th = client.beta.threads.create()
    client.beta.threads.messages.create(th.id, role="user", content=user_msg)

    # 2) start run
    run = client.beta.threads.runs.create(
        thread_id=th.id,
        assistant_id=ASSISTANT_ID,
        tool_choice="auto"
    )

    # 3) tool loop
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
        else:
            time.sleep(0.6)

    # 4) return the latest assistant text
    msgs = client.beta.threads.messages.list(thread_id=th.id, order="desc").data
    for m in msgs:
        if m.role == "assistant":
            parts = []
            for part in m.content:
                if getattr(part, "type", "") == "text":
                    parts.append(part.text.value)
            return "\n".join(parts)
    return ""

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": int(time.time())})

@app.route("/")
def root():
    return jsonify({"ok": True, "service": "FUNCTION-RUNNER", "endpoints": ["/health", "/chat"]})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = data.get("message") or "operation=get_memory user_id=phil thread_id=smoke limit=3"
    try:
        reply = run_once(msg)
        return jsonify({"ok": True, "assistant": reply})
    except requests.HTTPError as http_err:
        # bubble up memory API errors nicely
        code = getattr(http_err.response, "status_code", 500)
        try:
            payload = http_err.response.json()
        except Exception:
            payload = {"error": http_err.response.text}
        return jsonify({"ok": False, "source": "memory_api", "code": code, **payload}), code
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ── Local dev ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Local only; Render uses gunicorn per Procfile
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
