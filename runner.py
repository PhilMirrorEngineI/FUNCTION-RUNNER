# runner.py — Function Runner (chat + base64 images + web search + PDF + Dave memory)

import os, io, base64, requests
from flask import Flask, request, jsonify
from openai import OpenAI

# ---------- setup ----------
app = Flask(__name__)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Optional: Web search (Tavily)
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# Optional: Dave memory service (same service your frontend hits)
# Accept either the new names or the old Dave_* names
MEMORY_BASE = os.getenv("MEMORY_BASE_URL") or os.getenv("DAVE_API_BASE")
MEMORY_KEY  = os.getenv("MEMORY_API_KEY")  or os.getenv("DAVE_API_KEY")
SAVE_REPLIES = os.getenv("SAVE_REPLIES", "true").lower() == "true"

# ---------- Dave Memory helpers ----------
def load_preamble(user_email: str, limit: int = 8) -> str:
    """
    Fetch recent memory shards and turn them into a compact system preamble.
    No-ops (returns "") if memory is not configured or user not provided.
    """
    identity = (
        "You are Dave — Phil’s assistant. Be brief, practical, action-focused. "
        "You know about Phil’s projects (Function Runner, Web Dave, Document Gen). "
        "Use prior context if available; otherwise, ask minimally."
    )

    if not (MEMORY_BASE and MEMORY_KEY and user_email):
        return identity

    try:
        r = requests.get(
            f"{MEMORY_BASE}/get_memory",
            params={"user_id": user_email, "limit": limit},
            headers={"Authorization": f"Bearer {MEMORY_KEY}"},
            timeout=8,
        )
        if not r.ok:
            return identity
        data = r.json()
        items = data.get("items") or []
        lines = []
        for it in items:
            txt = (it.get("content") or "").strip()
            if not txt:
                continue
            txt = " ".join(txt.split())
            lines.append(f"- {txt}")
        if not lines:
            return identity
        return identity + "\nKnown context from prior interactions:\n" + "\n".join(lines)
    except Exception:
        return identity

def save_memory(user_email: str, content: str, role: str = "assistant") -> None:
    """
    Fire-and-forget save. No-ops if memory service/envs are missing.
    """
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
        # Never break chat on memory issues
        pass

# ---------- core helpers ----------
def do_text_completion(prompt: str, memory_preamble: str = "") -> str:
    try:
        messages = []
        if memory_preamble:
            messages.append({"role": "system", "content": memory_preamble})
        messages.append({"role": "user", "content": prompt})

        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error during completion: {e}"

def do_image(prompt: str) -> dict:
    image = client.images.generate(model="gpt-image-1", prompt=prompt, size="1024x1024")
    b64 = image.data[0].b64_json
    return {"b64": b64, "format": "png"}

def do_web_search(query: str) -> dict:
    """
    Uses Tavily. Set TAVILY_API_KEY in Render.
    """
    if not TAVILY_KEY:
        return {"error": "Missing TAVILY_API_KEY for web search"}
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": 5,
            },
            timeout=20,
        )
        data = resp.json()
        results = data.get("results", [])
        links = [{"title": r.get("title"), "url": r.get("url")} for r in results]
        return {
            "summary": data.get("answer") or data.get("summary") or "No summary available.",
            "links": links,
        }
    except Exception as e:
        return {"error": f"Web search failed: {e}"}

def do_pdf(title: str, body: str) -> dict:
    """
    Generates a simple PDF and returns base64 (reportlab).
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm, height - 3*cm, title[:200])

    # Body (basic wrapping)
    c.setFont("Helvetica", 11)
    y = height - 4*cm
    max_width = width - 4*cm
    for paragraph in body.split("\n"):
        line = ""
        for word in paragraph.split():
            test = (line + " " + word).strip()
            if c.stringWidth(test, "Helvetica", 11) > max_width:
                c.drawString(2*cm, y, line)
                y -= 14
                line = word
                if y < 2*cm:
                    c.showPage()
                    y = height - 3*cm
                    c.setFont("Helvetica", 11)
            else:
                line = test
        if line:
            c.drawString(2*cm, y, line)
            y -= 14
        y -= 6  # paragraph spacing
        if y < 2*cm:
            c.showPage(); y = height - 3*cm; c.setFont("Helvetica", 11)

    c.showPage()
    c.save()
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"b64": b64, "filename": "document.pdf", "mime": "application/pdf"}

# ---------- routes ----------
@app.post("/chat")
def chat():
    try:
        data = request.get_json(force=True) or {}
        msg = (data.get("message") or "").strip()
        user_email = (data.get("userEmail") or data.get("email") or "").strip()

        if not msg:
            return jsonify({"ok": False, "type": "error", "error": "Empty message"}), 400

        # save the user's message first (optional)
        if SAVE_REPLIES and user_email and msg:
            save_memory(user_email, msg, role="user")

        # memory preamble
        memory_preamble = load_preamble(user_email)
        lower = msg.lower()

        # --- image intent ---
        if lower.startswith(("generate an image", "create an image", "draw", "make an image", "show me an image")):
            img = do_image(msg)
            return jsonify({"ok": True, "type": "assistant_message",
                            "content": f"Here’s your image for: '{msg}'",
                            "images": [img]}), 200

        # --- web search intent ---
        if lower.startswith(("search:", "web:", "google:", "lookup:", "find:")):
            query = msg.split(":", 1)[-1].strip() or msg
            result = do_web_search(query)
            if "error" in result:
                return jsonify({"ok": False, "type": "error", "error": result["error"]}), 502
            return jsonify({"ok": True, "type": "assistant_message",
                            "content": result["summary"],
                            "search": {"query": query, "links": result["links"]}}), 200

        # --- document (PDF) intent ---
        if lower.startswith(("make pdf:", "make document:", "generate document:", "create pdf:")):
            payload = msg.split(":", 1)[-1].strip()
            if "|" in payload:
                title, body = [p.strip() for p in payload.split("|", 1)]
            else:
                parts = payload.split("\n", 1)
                title = parts[0].strip() or "Document"
                body = (parts[1] if len(parts) > 1 else "").strip() or "(empty)"
            pdf = do_pdf(title, body)
            return jsonify({"ok": True, "type": "assistant_message",
                            "content": f"Generated PDF: {pdf['filename']}",
                            "files": [pdf]}), 200

        # --- default: text completion (with memory preamble) ---
        reply = do_text_completion(msg, memory_preamble=memory_preamble)
        if SAVE_REPLIES and reply and user_email:
            save_memory(user_email, reply, role="assistant")
        return jsonify({"ok": True, "type": "assistant_message", "content": reply}), 200

    except Exception as e:
        return jsonify({"ok": False, "type": "error", "error": f"Server exception: {e}"}), 500

    # --- web search intent ---
    if lower.startswith(("search:", "web:", "google:", "lookup:", "find:")):
        query = msg.split(":", 1)[-1].strip() or msg
        result = do_web_search(query)
        if "error" in result:
            return jsonify({"ok": False, "type": "error", "error": result["error"]}), 502

        if user_email:
            try:
                save_memory(user_email, f"Search query: {query}", role="user")
                if SAVE_REPLIES:
                    save_memory(user_email, f"Assistant search summary: {result['summary']}", role="assistant")
            except Exception:
                pass

        return jsonify({
            "ok": True,
            "type": "assistant_message",
            "content": result["summary"],
            "search": {"query": query, "links": result["links"]},
        }), 200

    # --- document generation intent ---
    # Patterns: "make pdf: Title | body..." or "make document: Title ... \n\n Body ..."
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
                try:
                    save_memory(user_email, f"User generated PDF '{title}'", role="user")
                    if SAVE_REPLIES:
                        save_memory(user_email, f"Assistant: PDF generated '{title}'", role="assistant")
                except Exception:
                    pass
            return jsonify({
                "ok": True,
                "type": "assistant_message",
                "content": f"Generated PDF: {pdf['filename']}",
                "files": [pdf],  # base64 artifact
            }), 200
        except Exception as e:
            return jsonify({"ok": False, "type": "error", "error": f"PDF generation failed: {e}"}), 500

    # --- default: plain text chat with memory injection ---
    reply = do_text_completion(msg, memory_preamble=memory_preamble)

    # Save assistant reply (if enabled)
    if SAVE_REPLIES and reply and user_email:
        save_memory(user_email, reply, role="assistant")

    return jsonify({"ok": True, "type": "assistant_message", "content": reply}), 200

@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
