"""
chatbot/chatbot.py
Core chatbot engine for KisanCredit AI.
Handles intent detection and response generation.
No external AI API needed — fully rule-based with fuzzy matching.
"""

import re
from chatbot.knowledge_base import INTENTS, RESPONSES, SUGGESTIONS


class KisanChatbot:
    """
    Rule-based NLU chatbot for farmer loan queries.

    Intent detection uses keyword matching with priority ordering.
    Multiple intents can match; highest-priority one wins.
    """

    # Priority order — more specific intents should rank higher
    PRIORITY = [
        "credit_history", "income", "loan_amount", "pca", "overfitting",
        "features", "model", "approval", "rejection", "schemes",
        "dataset", "property", "education", "crop", "weather",
        "thanks", "bye", "greeting",
    ]

    def __init__(self):
        self.history: list[dict] = []   # [{role, content}, ...]

    # ── PUBLIC ─────────────────────────────────────────────────────────────────

    def respond(self, user_message: str) -> dict:
        """
        Process a user message and return a response dict:
        {
            "reply":    str,        # formatted response text
            "intent":   str,        # detected intent name
            "confidence": float,    # 0–1 match confidence
        }
        """
        clean = self._normalise(user_message)
        intent, confidence = self._detect_intent(clean)
        reply = RESPONSES.get(intent, RESPONSES["fallback"])

        # Store in history
        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant",  "content": reply})

        return {
            "reply":      reply,
            "intent":     intent,
            "confidence": round(confidence, 2),
        }

    def get_suggestions(self) -> list[str]:
        """Return the static list of suggested questions."""
        return SUGGESTIONS

    def reset(self):
        """Clear conversation history."""
        self.history.clear()

    def get_history(self) -> list[dict]:
        return self.history.copy()

    # ── PRIVATE ────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, remove punctuation, collapse whitespace."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s₹]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _detect_intent(self, text: str) -> tuple[str, float]:
        """
        Score every intent by counting keyword matches.
        Returns (intent_name, confidence_score).
        """
        scores: dict[str, float] = {}

        for intent, keywords in INTENTS.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches:
                # Normalise: matches / total_keywords (capped at 1.0)
                scores[intent] = min(1.0, matches / max(1, len(keywords) * 0.3))

        if not scores:
            return "fallback", 0.0

        # Resolve ties using PRIORITY list
        best_intent = max(
            scores,
            key=lambda k: (scores[k], -self.PRIORITY.index(k)
                        if k in self.PRIORITY else -99),
        )
        return best_intent, scores[best_intent]
