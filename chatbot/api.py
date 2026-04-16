"""
chatbot/api.py
Flask Blueprint for KisanCredit AI chatbot REST API.
Mount using: app.register_blueprint(chatbot_bp)

Endpoints:
  POST /api/chat              — send a message, get a reply
  GET  /api/chat/suggestions  — get suggested questions (language-aware)
  POST /api/chat/reset        — clear conversation history
  GET  /api/chat/history      — retrieve full conversation history
  POST /api/chat/transcribe   — speech-to-text via Web Speech API helper
  GET  /api/chat/languages    — list supported languages
"""

from flask import Blueprint, request, jsonify, session
from chatbot.chatbot import KisanChatbot, detect_language

chatbot_bp = Blueprint("chatbot", __name__)

# Global instance (single-user / demo). For multi-user, key by session["user_id"].
_bot = KisanChatbot()


# ── HELPERS ────────────────────────────────────────────────────────────────────

def _get_bot() -> KisanChatbot:
    """Return the chatbot instance. Falls back to global for single-user demo."""
    try:
        if "chat_id" not in session:
            import uuid
            session["chat_id"] = str(uuid.uuid4())
        return _bot   # Swap for per-session dict in multi-user production
    except RuntimeError:
        return _bot


# ── ROUTES ─────────────────────────────────────────────────────────────────────

@chatbot_bp.route("/api/chat", methods=["POST"])
def chat():
    """
    Send a message and receive a response.

    Request JSON:
        { "message": "Your question", "language": "en" }   ← language optional

    Response JSON:
        {
            "reply":       "...",
            "intent":      "kcc",
            "confidence":  0.85,
            "language":    "en",
            "source":      "rule",
            "history_len": 4
        }
    """
    data    = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    req_lang = data.get("language")

    if not message:
        return jsonify({"error": "Field 'message' is required."}), 400

    if len(message) > 1500:
        return jsonify({"error": "Message too long (max 1500 chars)."}), 400

    bot    = _get_bot()
    result = bot.respond(message, force_language=req_lang)

    return jsonify({
        "reply":       result["reply"],
        "intent":      result["intent"],
        "confidence":  result["confidence"],
        "language":    result["language"],
        "source":      result["source"],
        "history_len": len(bot.get_history()),
    })


@chatbot_bp.route("/api/chat/suggestions", methods=["GET"])
def suggestions():
    """
    Get language-aware suggested questions.

    Query params:
        ?lang=en | hi | mr   (optional, defaults to last detected language)

    Response JSON:
        { "suggestions": ["...", ...], "language": "hi" }
    """
    bot  = _get_bot()
    lang = request.args.get("lang") or bot.detected_language or "en"
    return jsonify({
        "suggestions": bot.get_suggestions(lang),
        "language":    lang,
    })


@chatbot_bp.route("/api/chat/reset", methods=["POST"])
def reset():
    """Clear the conversation history."""
    _get_bot().reset()
    return jsonify({"status": "ok", "message": "Conversation reset."})


@chatbot_bp.route("/api/chat/history", methods=["GET"])
def history():
    """Return full conversation history."""
    bot  = _get_bot()
    hist = bot.get_history()
    return jsonify({"history": hist, "count": len(hist)})


@chatbot_bp.route("/api/chat/detect-language", methods=["POST"])
def detect_lang_endpoint():
    """
    Detect language of a given text snippet.

    Request JSON:
        { "text": "..." }

    Response JSON:
        { "language": "hi", "label": "Hindi" }
    """
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400

    lang   = detect_language(text)
    labels = {"en": "English", "hi": "Hindi (हिंदी)", "mr": "Marathi (मराठी)"}
    return jsonify({"language": lang, "label": labels.get(lang, "Unknown")})


@chatbot_bp.route("/api/chat/languages", methods=["GET"])
def languages():
    """List all supported languages."""
    return jsonify({
        "languages": [
            {"code": "en", "label": "English",        "native": "English"},
            {"code": "hi", "label": "Hindi",          "native": "हिंदी"},
            {"code": "mr", "label": "Marathi",        "native": "मराठी"},
        ]
    })
