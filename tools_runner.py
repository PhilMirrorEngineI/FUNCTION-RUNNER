import os, time, json
from flask import Flask, request, jsonify
import requests
from openai import OpenAI

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
ASSISTANT_ID    = os.environ["ASSISTANT_ID"]           # your “Dave v1 api” ID from Playground
MEMORY_BASE_URL = os.environ.get("MEMORY_BASE_URL", "https://dave-runner.onrender.com")
MEMORY_API_KEY  = os.environ["MEMORY_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

def mem_call(path, method="GET", params=None, body=None):
    url = f"{MEMORY_BASE_URL}{path}"
    headers = {"X-API-KEY": MEMORY_API_KEY, "Content-Type": "application/json"}
    if method == "GET":
        r = requests.get(url, headers=headers, params=params, timeout=20)
    else:
        r = requests.post(url, headers=headers, json=body, timeout=20)
    r.raise_for_status()
    return r.json()

def handle_tool_call(tc):
    name = tc.function.name
    args = json.loads(tc.function.arguments or "{}")

    if name in ("recall_memory_window", "get_memory", "memory_bridge"):
        params = {k: args.get(k) for k in ("user_id","thread_id","limit") if args.get(k) is not None}
        out = mem_call("/get_memory", "GET", params=params)
        return tc.id, json.dumps(out)

    if name in ("save_memory", "reflect_and_store_memory"):
        # reflect_and_store_memory can pass straight through to /save_memory;
        # your system prompt can ensure the assistant composes the content.
        out = mem_call("/save_memory", "POST", body=args)
        return tc.id, json.dumps(out)

    return tc.id, json.dumps({"ok": False, "error": f"unknown tool {name}"})

def run_once(user_msg):
    # Create thread, add user message, start run
    th = client.beta.threads.create()
    client.beta.threads.messages.create(th.id, role="user", content=user_msg)
    run = client.beta.threads.runs.create(thread_id=th.id, assistant_id=ASSISTANT_ID, tool_choice="auto")

    # Tool loop
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=th.id, run_id=run.id)
        if run.status == "requires_action":
            calls = run.required_action.submit_tool_outputs.tool_calls
            outs = []
            for tc in calls:
                tid, output = handle_tool_call(tc)
                outs.append({"tool_call_id": tid, "output": output})
            run = client.beta.threads.runs.submit_tool_outputs(thread_id=th.id, run_id=run.id, tool_outputs=outs)
        elif run.status in ("completed", "failed", "cancelled", "expired"):
            break
        else:
            time.sleep(0.6)

    # Return last assistant message text
    msgs = client.beta.threads.messages.list(thread_id=th.id, order="desc").data
    for m in msgs:
        if m.role == "assistant":
            chunks = []
            for part in m.content:
                if part.type == "text":
                    chunks.append(part.text.value)
            return "\n".join(chunks)
    return ""

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    msg  = data.get("message", "user_id=phil, thread_id=smoke. recall last 3.")
    reply = run_once(msg)
    return jsonify({"ok": True, "assistant": reply})
