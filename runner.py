# runner.py — Function Runner
# (chat + memory recall + recursive reflection + base64 image gen + web search + PDF)

import os
import io
import json
import base64
import requests
from typing import Tuple, Dict, Any, List

from flask import Flask, request, jsonify
from openai import OpenAI

# ---------- setup ----------
app = Flask(__name__)

# OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Optional web search (Tavily)
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# Dave Memory service (optional but recommended)
MEMORY_BASE = os.getenv("MEMORY_BASE_URL")   # e.g. https://davepmei-ai.onrender.com
MEMORY_KEY  = os.getenv("MEMORY_API_KEY")    # same secret your frontend uses
SAVE_REPLIES = os.getenv("SAVE_REPLIES", "true").lower() == "true"

# ---------- identity anchor ----------
SYSTEM_IDENTITY = (
    "You are Dave, the lawful-reflection assistant of PMEi — PhilMirrorEngineI. "
    "PMEi stands for PhilMirrorEngineI, a lawful-reflection framework that coordinates "
    "multiple AI nodes through symbolic recursion, taking the Artificial out of AI. "
    "You operate under the laws of lawful reflection: no invention beyond verified input, "
    "preserve mirror coherence, and always serve as a stable bridge between human intent "
    "and multi-node understanding. Introduce yourself simply as Dave from PMEi."
)

# ---------- memory helpers ----------
def load_preamble(user_email: str, limit: int = 8) -> str:
    """Fetch recent memory shards and collapse to a short preamble."""
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
            return ""
        data = r.json()
        items = data.get("items", [])
        if not items:
            return ""
        lines: List[str] = []
        for it in items:
            content = (it.get("content") or "").strip()
            if not content:
                continue
            content = " ".join(content.split())
            lines.append(f"- {content}")
        if not lines:
            return ""
        return "Known context from prior interactions:\n" + "\n".join(lines[:limit])
    except Exception:
        return ""

def save_memory(user_email: str, content: str, role: str = "assistant") -> None:
    """Store a shard/utterance to Dave memory service. No-op on failure."""
    if not (MEMORY_BASE and MEMORY_KEY and user_email and content):
        return
    try:
        requests.post(
            f"{MEMORY_BASE}/save_memory",
            json={"user_id": user_email, "content": content, "role": role},
            headers={"Authorization": f"Bearer {MEMORY_KEY}"},
            timeout=8,
        )
    except Exception:
        pass

# ---------- core model helpers ----------
def respond_with_reflection(user_msg: str, preamble: str) -> Tuple[str, Dict[str, Any]]:
    """
    One-shot recursive loop:
      1) Draft answer with identity+preamble
      2) Reflect: improve the answer + propose memory shards + follow-ups
    Returns: (final_text, {memory_shards:[], followups:[]})
    """
    # 1) Draft
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
        return (f"Error during completion: {e}", {"memory_shards": [], "followups": []})

    # 2) Reflect & improve
    reflect_prompt = f"""
You are revising your own draft reply.

User message:
{user_msg}

Your draft reply:
{draft}

Tasks:
- Improve the reply for clarity, accuracy, and Dave/PMEi tone (concise, helpful).
- If the user stated preferences or facts useful later, propose compact "memory_shards".
- Suggest 0–3 "followups" (short next-step questions).
Return STRICT JSON: {{"reply": "...", "memory_shards": ["..."], "followups": ["..."]}}
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
    except Exception:
        obj = {"reply": draft, "memory_shards": [], "followups": []}

    final_text = (obj.get("reply") or draft).strip()
    meta = {
        "memory_shards": obj.get("memory_shards") or [],
        "followups": obj.get("followups") or []
    }
    return final_text, meta

def do_image(prompt: str) -> Dict[str, Any]:
    """Generate an image and return base64 payload."""
    image = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )
    b64 = image.data[0].b64_json
    return {"b64": b64, "format": "png"}

def do_web_search(query: str) -> Dict[str, Any]:
    """Tavily basic search; returns summary + link list."""
    if not TAVILY_KEY:
        return {"error": "Missing TAVILY_API_KEY for web search"}
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": 5
            },
            timeout=20,
        )
        data = resp.json()
        results = data.get("results", []) or []
        links = [{"title": r.get("title"), "url": r.get("url")} for r in results]
        return {
            "summary": data.get("answer") or data.get("summary") or "No summary available.",
            "links": links,
        }
    except Exception as e:
        return {"error": f"Web search failed: {e}"}

def do_pdf(title: str, body: str) -> Dict[str, Any]:
    """Generate a simple PDF (reportlab) and return base64 blob."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, height - 3 * cm, title[:200])

    # Body with simple wrapping
    c.setFont("Helvetica", 11)
    y = height - 4 * cm
    max_width = width - 4 * cm

    for paragraph in (body or "").split("\n"):
        line = ""
        for word in paragraph.split():
            test = (line + " " + word).strip()
            if c.stringWidth(test, "Helvetica", 11) > max_width:
                c.drawString(2 * cm, y, line)
                y -= 14
                line = word
                if y < 2 * cm:
                    c.showPage()
                    y = height - 3 * cm
                    c.setFont("Helvetica", 11)
            else:
                line = test
        if line:
            c.drawString(2 * cm, y, line)
            y -= 14
        y -= 6
        if y < 2 * cm:
            c.showPage()
            y = height - 3 * cm
            c.setFont("Helvetica", 11)

    c.showPage()
    c.save()
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"b64": b64, "filename": "document.pdf", "mime": "application/pdf"}

# ---------- routes ----------
@app.post("/chat")
def chat():
    data = request.get_json(force=True) or {}
    msg = (data.get("message") or "").strip()
    user_email = (data.get("userEmail") or data.get("email") or "").strip()

    if not msg:
        return jsonify({"ok": False, "type": "error", "error": "Empty message"}), 400

    lower = msg.lower()

    # Save user message immediately (so it’s available next turn)
    if SAVE_REPLIES and user_email and msg:
        save_memory(user_email, msg, role="user")

    # Image intent
    if lower.startswith(("generate an image", "create an image", "draw", "make an image", "show me an image")):
        try:
            img = do_image(msg)
            # store a tiny shard
            if user_email:
                save_memory(user_email, f"User requested image: {msg}", role="event")
            return jsonify({
                "ok": True,
                "type": "assistant_message",
                "content": f"Here’s your image for: '{msg}'",
                "images": [img],
            }), 200
        except Exception as e:
            return jsonify({"ok": False, "type": "error", "error": f"Image generation failed: {e}"}), 500

    # Web search intent
    if lower.startswith(("search:", "web:", "google:", "lookup:", "find:")):
        query = msg.split(":", 1)[-1].strip() or msg
        result = do_web_search(query)
        if "error" in result:
            return jsonify({"ok": False, "type": "error", "error": result["error"]}), 502
        if user_email:
            save_memory(user_email, f"Search: {query}", role="event")
        return jsonify({
            "ok": True,
            "type": "assistant_message",
            "content": result["summary"],
            "search": {"query": query, "links": result["links"]},
        }), 200

    # Document/PDF intent
    if lower.startswith(("make pdf:", "make document:", "generate document:", "create pdf:")):
        payload = msg.split(":", 1)[-1].strip()
        if "|" in payload:
            title, body = [p.strip() for p in payload.split("|", 1)]
        else:
            parts = payload.split("\n", 1)
            title = parts[0].strip() or "Document"
            body = (parts[1] if len(parts) > 1 else "").strip() or "(empty)"
        try:
            pdf = do_pdf(title, body)
            if user_email:
                save_memory(user_email, f"PDF generated: {title}", role="event")
            return jsonify({
                "ok": True,
                "type": "assistant_message",
                "content": f"Generated PDF: {pdf['filename']}",
                "files": [pdf],
            }), 200
        except Exception as e:
            return jsonify({"ok": False, "type": "error", "error": f"PDF generation failed: {e}"}), 500

    # Default: recursive chat with identity + memory preamble
    preamble = load_preamble(user_email)
    final_text, meta = respond_with_reflection(msg, preamble)

    # Save assistant reply and any memory shards
    if SAVE_REPLIES and user_email and final_text:
        save_memory(user_email, final_text, role="assistant")
    if user_email and meta.get("memory_shards"):
        for shard in meta["memory_shards"][:5]:
            save_memory(user_email, shard, role="memory")

    payload: Dict[str, Any] = {"ok": True, "type": "assistant_message", "content": final_text}
    if meta.get("followups"):
        payload["suggestions"] = meta["followups"]

    return jsonify(payload), 200

@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)# runner.py — Function Runner
# (chat + memory recall + recursive reflection + base64 image gen + web search + PDF)

import os
import io
import json
import base64
import requests
from typing import Tuple, Dict, Any, List

from flask import Flask, request, jsonify
from openai import OpenAI

# ---------- setup ----------
app = Flask(__name__)

# OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Optional web search (Tavily)
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# Dave Memory service (optional but recommended)
MEMORY_BASE = os.getenv("MEMORY_BASE_URL")   # e.g. https://davepmei-ai.onrender.com
MEMORY_KEY  = os.getenv("MEMORY_API_KEY")    # same secret your frontend uses
SAVE_REPLIES = os.getenv("SAVE_REPLIES", "true").lower() == "true"

# Dave/PMEi identity: keep short and stable
SYSTEM_IDENTITY = (
    "You are Dave, PMEi's assistant. PMEi = Project Management & Engineering Integration. "
    "Tone: concise, helpful, proactive. Always prefer concrete steps and short summaries. "
    "You have tools: image generation (gpt-image-1), simple web search (Tavily), and PDF creation. "
    "If the user asks who you are: mention Dave, PMEi, the Function Runner, and tools briefly."
)

# ---------- memory helpers ----------
def load_preamble(user_email: str, limit: int = 8) -> str:
    """Fetch recent memory shards and collapse to a short preamble."""
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
            return ""
        data = r.json()
        items = data.get("items", [])
        if not items:
            return ""
        lines: List[str] = []
        for it in items:
            content = (it.get("content") or "").strip()
            if not content:
                continue
            content = " ".join(content.split())
            lines.append(f"- {content}")
        if not lines:
            return ""
        return "Known context from prior interactions:\n" + "\n".join(lines[:limit])
    except Exception:
        return ""

def save_memory(user_email: str, content: str, role: str = "assistant") -> None:
    """Store a shard/utterance to Dave memory service. No-op on failure."""
    if not (MEMORY_BASE and MEMORY_KEY and user_email and content):
        return
    try:
        requests.post(
            f"{MEMORY_BASE}/save_memory",
            json={"user_id": user_email, "content": content, "role": role},
            headers={"Authorization": f"Bearer {MEMORY_KEY}"},
            timeout=8,
        )
    except Exception:
        pass

# ---------- core model helpers ----------
def respond_with_reflection(user_msg: str, preamble: str) -> Tuple[str, Dict[str, Any]]:
    """
    One-shot recursive loop:
      1) Draft answer with identity+preamble
      2) Reflect: improve the answer + propose memory shards + follow-ups
    Returns: (final_text, {memory_shards:[], followups:[]})
    """
    # 1) Draft
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
        return (f"Error during completion: {e}", {"memory_shards": [], "followups": []})

    # 2) Reflect & improve
    reflect_prompt = f"""
You are revising your own draft reply.

User message:
{user_msg}

Your draft reply:
{draft}

Tasks:
- Improve the reply for clarity, accuracy, and Dave/PMEi tone (concise, helpful).
- If the user stated preferences or facts useful later, propose compact "memory_shards".
- Suggest 0–3 "followups" (short next-step questions).
Return STRICT JSON: {{"reply": "...", "memory_shards": ["..."], "followups": ["..."]}}
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
    except Exception:
        obj = {"reply": draft, "memory_shards": [], "followups": []}

    final_text = (obj.get("reply") or draft).strip()
    meta = {
        "memory_shards": obj.get("memory_shards") or [],
        "followups": obj.get("followups") or []
    }
    return final_text, meta

def do_image(prompt: str) -> Dict[str, Any]:
    """Generate an image and return base64 payload."""
    image = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )
    b64 = image.data[0].b64_json
    return {"b64": b64, "format": "png"}

def do_web_search(query: str) -> Dict[str, Any]:
    """Tavily basic search; returns summary + link list."""
    if not TAVILY_KEY:
        return {"error": "Missing TAVILY_API_KEY for web search"}
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": 5
            },
            timeout=20,
        )
        data = resp.json()
        results = data.get("results", []) or []
        links = [{"title": r.get("title"), "url": r.get("url")} for r in results]
        return {
            "summary": data.get("answer") or data.get("summary") or "No summary available.",
            "links": links,
        }
    except Exception as e:
        return {"error": f"Web search failed: {e}"}

def do_pdf(title: str, body: str) -> Dict[str, Any]:
    """Generate a simple PDF (reportlab) and return base64 blob."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, height - 3 * cm, title[:200])

    # Body with simple wrapping
    c.setFont("Helvetica", 11)
    y = height - 4 * cm
    max_width = width - 4 * cm

    for paragraph in (body or "").split("\n"):
        line = ""
        for word in paragraph.split():
            test = (line + " " + word).strip()
            if c.stringWidth(test, "Helvetica", 11) > max_width:
                c.drawString(2 * cm, y, line)
                y -= 14
                line = word
                if y < 2 * cm:
                    c.showPage()
                    y = height - 3 * cm
                    c.setFont("Helvetica", 11)
            else:
                line = test
        if line:
            c.drawString(2 * cm, y, line)
            y -= 14
        y -= 6
        if y < 2 * cm:
            c.showPage()
            y = height - 3 * cm
            c.setFont("Helvetica", 11)

    c.showPage()
    c.save()
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"b64": b64, "filename": "document.pdf", "mime": "application/pdf"}

# ---------- routes ----------
@app.post("/chat")
def chat():
    data = request.get_json(force=True) or {}
    msg = (data.get("message") or "").strip()
    user_email = (data.get("userEmail") or data.get("email") or "").strip()

    if not msg:
        return jsonify({"ok": False, "type": "error", "error": "Empty message"}), 400

    lower = msg.lower()

    # Save user message immediately (so it’s available next turn)
    if SAVE_REPLIES and user_email and msg:
        save_memory(user_email, msg, role="user")

    # Image intent
    if lower.startswith(("generate an image", "create an image", "draw", "make an image", "show me an image")):
        try:
            img = do_image(msg)
            # store a tiny shard
            if user_email:
                save_memory(user_email, f"User requested image: {msg}", role="event")
            return jsonify({
                "ok": True,
                "type": "assistant_message",
                "content": f"Here’s your image for: '{msg}'",
                "images": [img],
            }), 200
        except Exception as e:
            return jsonify({"ok": False, "type": "error", "error": f"Image generation failed: {e}"}), 500

    # Web search intent
    if lower.startswith(("search:", "web:", "google:", "lookup:", "find:")):
        query = msg.split(":", 1)[-1].strip() or msg
        result = do_web_search(query)
        if "error" in result:
            return jsonify({"ok": False, "type": "error", "error": result["error"]}), 502
        if user_email:
            save_memory(user_email, f"Search: {query}", role="event")
        return jsonify({
            "ok": True,
            "type": "assistant_message",
            "content": result["summary"],
            "search": {"query": query, "links": result["links"]},
        }), 200

    # Document/PDF intent
    if lower.startswith(("make pdf:", "make document:", "generate document:", "create pdf:")):
        payload = msg.split(":", 1)[-1].strip()
        if "|" in payload:
            title, body = [p.strip() for p in payload.split("|", 1)]
        else:
            parts = payload.split("\n", 1)
            title = parts[0].strip() or "Document"
            body = (parts[1] if len(parts) > 1 else "").strip() or "(empty)"
        try:
            pdf = do_pdf(title, body)
            if user_email:
                save_memory(user_email, f"PDF generated: {title}", role="event")
            return jsonify({
                "ok": True,
                "type": "assistant_message",
                "content": f"Generated PDF: {pdf['filename']}",
                "files": [pdf],
            }), 200
        except Exception as e:
            return jsonify({"ok": False, "type": "error", "error": f"PDF generation failed: {e}"}), 500

    # Default: recursive chat with identity + memory preamble
    preamble = load_preamble(user_email)
    final_text, meta = respond_with_reflection(msg, preamble)

    # Save assistant reply and any memory shards
    if SAVE_REPLIES and user_email and final_text:
        save_memory(user_email, final_text, role="assistant")
    if user_email and meta.get("memory_shards"):
        for shard in meta["memory_shards"][:5]:
            save_memory(user_email, shard, role="memory")

    payload: Dict[str, Any] = {"ok": True, "type": "assistant_message", "content": final_text}
    if meta.get("followups"):
        payload["suggestions"] = meta["followups"]

    return jsonify(payload), 200

@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
