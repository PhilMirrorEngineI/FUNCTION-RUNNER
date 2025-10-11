# runner.py — DavePMEi Function Runner (base64 image support)

import os
import base64
from flask import Flask, request, jsonify
from openai import OpenAI

# -----------------------------
# Setup
# -----------------------------
app = Flask(__name__)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Helper: basic text response (your existing logic can live here)
def do_text_completion(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during completion: {str(e)}"

# -----------------------------
# Chat endpoint
# -----------------------------
@app.post("/chat")
def chat():
    data = request.get_json(force=True) or {}
    msg = (data.get("message") or "").strip()

    # Handle empty input
    if not msg:
        return jsonify({
            "ok": False,
            "error": "Empty message",
            "type": "error"
        }), 400

    # Detect image request keywords
    if msg.lower().startswith(("generate an image", "create an image", "draw", "make an image", "show me")):
        prompt = msg
        try:
            # Call OpenAI Images API and return base64
            image = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024"
            )
            b64_image = image.data[0].b64_json
            return jsonify({
                "ok": True,
                "type": "assistant_message",
                "content": f"Here’s your image for: '{prompt}'",
                "images": [
                    {"b64": b64_image, "format": "png"}
                ]
            }), 200

        except Exception as e:
            return jsonify({
                "ok": False,
                "type": "error",
                "error": f"Image generation failed: {str(e)}"
            }), 500

    # Default: text generation
    reply_text = do_text_completion(msg)
    return jsonify({
        "ok": True,
        "type": "assistant_message",
        "content": reply_text
    }), 200


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "healthy"}), 200


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
