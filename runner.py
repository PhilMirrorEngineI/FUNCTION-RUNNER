import os
import time
import json
import shlex
from flask import Flask, request, jsonify
import requests
from openai import OpenAI
from functools import wraps

# ── Env ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
ASSISTANT_ID     = os.environ["ASSISTANT_ID"]  # your “Dave v1 api” Assistant ID
MEMORY_BASE_URL  = os.environ.get("MEMORY_BASE_URL", "https://dave-runner.onrender.com")
MEMORY_API_KEY   = os.environ["MEMORY_API_KEY"]
RUNNER_ACTION_KEY = os.environ["RUNNER_ACTION_KEY"]  # <-- add in Render env

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

# ── Auth helper (protect /chat) ──────────────────────────────────────────────
def require_runner_key(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        # allow health & root without auth (and preflight)
        if request.method == "OPTIONS" or request.path in ("/", "/health"):
            return fn(*args, **kwargs)
        auth = request.headers.get("Authorization", "")
        # Expect: "Bearer <RUNNER_ACTION_KEY>"
        if not auth.startswith("Bearer "):
            return jsonify({"ok": False, "error": "Missing Bearer token"}), 401
        token = auth.split(" ", 1)[1].strip()
        if not token or token != RUNNER_ACTION_KEY:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapped

# ── Helpers ──────────────────────────────────────────────────────────────────
def mem_call(path: str, method: str = "GET", params=None, body=None):
    """Call Dave Runner memory API with auth header; always return JSON or raise."""
    url = f"{MEMORY_BASE_URL}{path}"
    headers = {"X-API-KEY": MEMORY_API_KEY, "Content-Type": "application/json"}
    if method.upper() == "GET":
        r = requests.get(url, headers=headers, params=params or {}, timeout=20)
    else:
        r = requests.post(url, headers=headers, json=body or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def parse_message_kv(message: str) -> dict:
    """
    Parse free-form 'key=value' pairs from a string.
    Respects quotes using shlex (e.g., content="hello world").
    """
    args = {}
    for tok in shlex.split(message or ""):
        if "=" in tok:
            k, v = tok.split("=", 1)
            args[k.strip()] = v.strip().strip()
    return args

def handle_tool_call(tc):
    """
    Handle a tool call from the Assistant.
    Supports either:
      - Playground function:  function_runner(message="operation=... user_id=...")
      - Legacy direct tools:  save_memory / get_memory / memory_bridge / recall_memory_window / reflect_and_store_memory
    """
    name = getattr(tc.function, "name", "") or ""
    raw_args = json.loads(getattr(tc.function, "arguments", "") or "{}")

    # If the unified bridge is used, parse its 'message' string to args.
    if name == "function_runner":
        msg = raw_args.get("message", "")
        args = parse_message_kv(msg)
    else:
        args = dict(raw_args)

    # Auto-defaults (don’t override provided values)
    defaults = {
        "slide_id": "t-001",
        "glyph_echo": "🪞",
        "drift_score": 0.05,
        "seal": "lawful",
        "limit": 5,
        "content": "(no content provided)",
    }
    for k, v in defaults.items():
        args.setdefault(k, v)

    # Operation resolution: explicit 'operation' beats name.
    operation = (args.get("operation") or name or "").strip().lower()

    # ── Read routes ───────────────────────────────────────────────────────────
    if operation in ("memory_bridge", "get_memory", "recall_memory_window"):
        params = {k: args.get(k) for k in ("user_id", "thread_id", "limit") if args.get(k) is not None}
        out = mem_call("/get_memory", "GET", params=params)
        return tc.id, json.dumps(out)

    # ── Write routes ──────────────────────────────────────────────────────────
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

    # Unknown
    return tc.id, json.dumps({
        "ok": False,
        "error": f"unknown operation '{operation}'",
        "received": {"name": name, "args": args}
    })

def run_once(user_msg: str) -> str:
    """
    Create a thread, run once with tool loop, return last assistant text.
    Includes a timeout to avoid endless spins.
    """
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
            break

        time.sleep(SLEEP_SECS)

    # Return the most recent assistant text
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
    # ---- auth gate for GPT Action ----
    required = os.environ.get("RUNNER_ACTION_KEY", "")
    auth = request.headers.get("Authorization", "")

    if required and not auth.startswith("Bearer " + required):
        return jsonify({
            "ok": False,
            "error": "Unauthorized",
            "detail": "Missing or invalid Bearer token."
        }), 401

    data = request.get_json(silent=True) or {}
    if "message" in data:
        msg = data["message"]
    else:
        kv = [f"{k}={json.dumps(v) if isinstance(v, str) and ' ' in v else v}"
              for k, v in data.items()]
        msg = " ".join(kv) if kv else "operation=get_memory user_id=phil thread_id=smoke limit=3"

    try:
        print(">> Received:", msg, flush=True)
        reply = run_once(msg)
        print("<< Reply:", reply, flush=True)
        return jsonify({"ok": True, "assistant": reply})
    except requests.HTTPError as http_err:
        code = getattr(http_err.response, "status_code", 500)
        try:
            payload = http_err.response.json()
        except Exception:
            payload = {"error": http_err.response.text}
        return jsonify({"ok": False, "source": "memory_api", "code": code, **payload}), code
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    # Accept either a raw freeform message, or structured fields we turn into message
    if "message" in data and isinstance(data["message"], str):
        msg = data["message"]
    else:
        # build message if user posts JSON like {"operation": "...", "user_id": "...", ...}
        kv_parts = []
        for k, v in data.items():
            if isinstance(v, str):
                # quote strings that contain spaces
                v_out = v if " " not in v else json.dumps(v)
            else:
                v_out = json.dumps(v)
            kv_parts.append(f"{k}={v_out}")
        msg = " ".join(kv_parts) if kv_parts else "operation=get_memory user_id=phil thread_id=smoke limit=3"

    try:
        print(">> Received:", msg, flush=True)
        reply = run_once(msg)
        print("<< Reply:", reply, flush=True)
        return jsonify({"ok": True, "assistant": reply})
    except requests.HTTPError as http_err:
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
