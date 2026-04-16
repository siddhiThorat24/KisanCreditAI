"""
chatbot/api.py
Flask Blueprint that exposes the chatbot as REST endpoints.
Mount this in backend/app.py using app.register_blueprint(chatbot_bp).

Endpoints:
POST /api/chat            — send a message, get a reply
GET  /api/chat/suggestions — get suggested questions
POST /api/chat/reset       — clear conversation history
GET  /api/chat/history     — retrieve full conversation history
"""

from flask import Blueprint, request, jsonify, session
from chatbot.chatbot import KisanChatbot

chatbot_bp = Blueprint("chatbot", __name__)

# One chatbot instance per Flask session (keyed by session ID).
# For simplicity with stateless APIs a single global instance works fine
# for single-user / demo usage. For multi-user production use a dict
# keyed on session["user_id"] or a request header token.
_bot = KisanChatbot()


# ── HELPERS ────────────────────────────────────────────────────────────────────

def _bot_for_session() -> KisanChatbot:
    """
    Return the chatbot instance for the current session.
    Falls back to the global instance when sessions are not configured.
    """
    try:
        uid = session.get("chat_id")
        if uid is None:
            import uuid
            session["chat_id"] = str(uuid.uuid4())
        # For a production multi-user app, maintain a dict of bots here.
        # For a single-user demo the global _bot is fine.
        return _bot
    except RuntimeError:
        # Outside of request context (e.g. tests)
        return _bot


# ── ROUTES ─────────────────────────────────────────────────────────────────────

@chatbot_bp.route("/api/chat", methods=["POST"])
def chat():
    """
    Send a message to the chatbot.

    Request body (JSON):
        { "message": "Your question here" }

    Response (JSON):
        {
            "reply":      "...",
            "intent":     "credit_history",
            "confidence": 0.85,
            "history_len": 4
        }
    """
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Field 'message' is required."}), 400

    if len(message) > 1000:
        return jsonify({"error": "Message too long (max 1000 chars)."}), 400

    bot = _bot_for_session()
    result = bot.respond(message)

    return jsonify({
        "reply":       result["reply"],
        "intent":      result["intent"],
        "confidence":  result["confidence"],
        "history_len": len(bot.get_history()),
    })


@chatbot_bp.route("/api/chat/suggestions", methods=["GET"])
def suggestions():
    """
    Get a list of suggested questions to show the user.

    Response (JSON):
        { "suggestions": ["...", "..."] }
    """
    bot = _bot_for_session()
    return jsonify({"suggestions": bot.get_suggestions()})


@chatbot_bp.route("/api/chat/reset", methods=["POST"])
def reset():
    """
    Clear the conversation history.

    Response (JSON):
        { "status": "ok", "message": "Conversation reset." }
    """
    bot = _bot_for_session()
    bot.reset()
    return jsonify({"status": "ok", "message": "Conversation reset."})


@chatbot_bp.route("/api/chat/history", methods=["GET"])
def history():
    """
    Retrieve full conversation history.

    Response (JSON):
        {
            "history": [
                {"role": "user",      "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ],
            "count": 4
        }
    """
    bot = _bot_for_session()
    hist = bot.get_history()
    return jsonify({"history": hist, "count": len(hist)})
