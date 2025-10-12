# runner.py — Function Runner
# chat + memory recall + reflective reply + base64 image gen + web search + PDF

import os, io, json, base64, logging, requests
from typing import Tuple, Dict, Any, List
from flask import Flask, request, jsonify
from openai import OpenAI

# ---------- app & logging ----------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("runner")

# ---------- env / clients ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY is missing!")
client = OpenAI(api_key=OPENAI_API_KEY)

TAVILY_KEY = os.getenv("TAVILY_API_KEY")  # optional

MEMORY_BASE = os.getenv("MEMORY_BASE_URL")   # e.g. https://davepmei-ai.onrender.com
MEMORY_KEY  = os.getenv("MEMORY_API_KEY")
SAVE_REPLIES = os.getenv("SAVE_REPLIES", "true").lower() == "true"

# ---------- fixed identity ----------
SYSTEM_IDENTITY = (
    "You are Dave, PMEi's assistant. PMEi = PhilMirrorEngineI, a lawful-reflection "
    "framework that coordinates multiple AI nodes via symbolic recursion — "
    "‘Taking the Artificial out of AI’. Tone: concise, helpful, proactive. "
    "You can: generate images, run simple web searches, and create PDFs."
)

# ---------- memory helpers ----------
def load_preamble(user_email: str, limit: int = 8) -> str:
    if not (MEMORY_BASE and MEMORY_KEY and user_email):
        return ""
    try:
        r = requests.get(
            f"{MEMORY_BASE}/get_memory",
            params={"user_id": user_email, "limit": limit},
            headers={"Authorization": f"Bearer {MEMORY_KEY}"},
            timeout=8,
        )
        if not r.ok:
            log.warning("get_memory HTTP %s: %s", r.status_code, r.text[:200])
            return ""
        items = (r.json() or {}).get("items", []) or []
        if not items:
            return ""
        lines: List[str] = []
        for it in items:
            content = (it.get("content") or "").strip()
            if content:
                lines.append(f"- {' '.join(content.split())}")
        if not lines:
            return ""
        return "Known context from prior interactions:\n" + "\n".join(lines[:limit])
    except Exception as e:
        log.exception("load_preamble failed: %s", e)
        return ""

def save_memory(user_email: str, content: str, role: str = "assistant") -> None:
    if not (MEMORY_BASE and MEMORY_KEY and user_email and content):
        return
    try:
        r = requests.post(
            f"{MEMORY_BASE}/save_memory",
            json={"user_id": user_email, "content": content, "role": role},
            headers={"Authorization": f"Bearer {MEMORY_KEY}"},
            timeout=8,
        )
        if not r.ok:
            log.warning("save_memory HTTP %s: %s", r.status_code, r.text[:200])
    except Exception as e:
        log.exception("save_memory failed: %s", e)

# ---------- model helpers ----------
def respond_with_reflection(user_msg: str, preamble: str) -> Tuple[str, Dict[str, Any]]:
    # Draft
    messages = []
    if SYSTEM_IDENTITY:
        messages.append({"role": "system", "content": SYSTEM_IDENTITY})
    if preamble:
        messages.append({"role": "system", "content": preamble})
    messages.append({"role": "user", "content": user_msg})

    try:
        draft = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
        ).choices[0].message.content or ""
    except Exception as e:
        log.exception("draft completion failed: %s", e)
        return (f"Error during completion: {e}", {"memory_shards": [], "followups": []})

    # Reflect
    reflect_prompt = f"""
Return STRICT JSON with keys reply, memory_shards, followups.

User message:
{user_msg}

Your draft reply:
{draft}

Improve the reply (Dave/PMEi tone). Add 0–5 compact memory_shards (facts/preferences).
Add 0–3 short followups.
""".strip()

    messages_reflect = []
    if SYSTEM_IDENTITY:
        messages_reflect.append({"role": "system", "content": SYSTEM_IDENTITY})
    if preamble:
        messages_reflect.append({"role": "system", "content": preamble})
    messages_reflect.append({"role": "user", "content": reflect_prompt})

    try:
        reflected = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_reflect,
            temperature=0.2,
            response_format={"type": "json_object"},
        ).choices[0].message.content
        obj = json.loads(reflected or "{}")
    except Exception as e:
        log.warning("reflection fell back to draft: %s", e)
        obj = {"reply": draft, "memory_shards": [], "followups": []}

    final_text = (obj.get("reply") or draft).strip()
    meta = {
        "memory_shards": obj.get("memory_shards") or [],
        "followups": obj.get("followups") or [],
    }
    return final_text, meta

def do_image(prompt: str) -> Dict[str, Any]:
    img = client.images.generate(model="gpt-image-1", prompt=prompt, size="1024x1024")
    return {"b64": img.data[0].b64_json, "format": "png"}

def do_web_search(query: str) -> Dict[str, Any]:
    if not TAVILY_KEY:
        return {"error": "Missing TAVILY_API_KEY for web search"}
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_KEY, "query": query, "search_depth": "advanced", "max_results": 5},
            timeout=20,
        )
        data = resp.json()
        results = data.get("results", []) or []
        links = [{"title": r.get("title"), "url": r.get("url")} for r in results]
        return {"summary": data.get("answer") or data.get("summary") or "No summary available.", "links": links}
    except Exception as e:
        log.exception("web search failed: %s", e)
        return {"error": f"Web search failed: {e}"}

def do_pdf(title: str, body: str) -> Dict[str, Any]:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, height - 3 * cm, (title or "Document")[:200])

    c.setFont("Helvetica", 11)
    y = height - 4 * cm
    max_width = width - 4 * cm
    for paragraph in (body or "").split("\n"):
        line = ""
        for word in paragraph.split():
            test = (line + " " + word).strip()
            if c.stringWidth(test, "Helvetica", 11) > max_width:
                c.drawString(2 * cm, y, line); y -= 14
                line = word
                if y < 2 * cm:
                    c.showPage(); y = height - 3 * cm; c.setFont("Helvetica", 11)
            else:
                line = test
        if line:
            c.drawString(2 * cm, y, line); y -= 14
        y -= 6
        if y < 2 * cm:
            c.showPage(); y = height - 3 * cm; c.setFont("Helvetica", 11)

    c.showPage(); c.save()
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"b64": b64, "filename": "document.pdf", "mime": "application/pdf"}

# ---------- routes ----------
@app.get("/")
def root():
    return jsonify({"ok": True, "service": "function-runner", "health": "use /health", "chat": "POST /chat"}), 200

@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "healthy"}), 200

@app.post("/chat")
def chat():
    try:
        data = request.get_json(force=True) or {}
        msg = (data.get("message") or "").strip()
        user_email = (data.get("userEmail") or data.get("email") or "").strip()

        if not msg:
            return jsonify({"ok": False, "type": "error", "error": "Empty message"}), 400

        lower = msg.lower()

        # Save user utterance first (non-fatal if memory envs missing)
        if SAVE_REPLIES and user_email:
            save_memory(user_email, msg, role="user")

        # Image intent
        if lower.startswith(("generate an image", "create an image", "draw", "make an image", "show me an image")):
            img = do_image(msg)
            if user_email:
                save_memory(user_email, f"User requested image: {msg}", role="event")
            return jsonify({"ok": True, "type": "assistant_message", "content": f"Here’s your image for: '{msg}'", "images": [img]}), 200

        # Web search intent
        if lower.startswith(("search:", "web:", "google:", "lookup:", "find:")):
            query = msg.split(":", 1)[-1].strip() or msg
            result = do_web_search(query)
            if "error" in result:
                return jsonify({"ok": False, "type": "error", "error": result["error"]}), 502
            if user_email:
                save_memory(user_email, f"Search: {query}", role="event")
            return jsonify({"ok": True, "type": "assistant_message", "content": result["summary"], "search": {"query": query, "links": result["links"]}}), 200

        # Document intent
        if lower.startswith(("make pdf:", "make document:", "generate document:", "create pdf:")):
            payload = msg.split(":", 1)[-1].strip()
            if "|" in payload:
                title, body = [p.strip() for p in payload.split("|", 1)]
            else:
                parts = payload.split("\n", 1)
                title = parts[0].strip() or "Document"
                body = (parts[1] if len(parts) > 1 else "").strip() or "(empty)"
            pdf = do_pdf(title, body)
            if user_email:
                save_memory(user_email, f"PDF generated: {title}", role="event")
            return jsonify({"ok": True, "type": "assistant_message", "content": f"Generated PDF: {pdf['filename']}", "files": [pdf]}), 200

        # Default: reflective chat (with memory preamble)
        preamble = load_preamble(user_email)
        final_text, meta = respond_with_reflection(msg, preamble)

        if SAVE_REPLIES and user_email and final_text:
            save_memory(user_email, final_text, role="assistant")
        if user_email and meta.get("memory_shards"):
            for shard in meta["memory_shards"][:5]:
                save_memory(user_email, shard, role="memory")

        payload: Dict[str, Any] = {"ok": True, "type": "assistant_message", "content": final_text}
        if meta.get("followups"):
            payload["suggestions"] = meta["followups"]
        return jsonify(payload), 200

    except Exception as e:
        log.exception("Unhandled /chat error: %s", e)
        return jsonify({"ok": False, "type": "error", "error": f"Runner exception: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
