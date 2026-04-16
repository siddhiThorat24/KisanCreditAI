"""
chatbot/chatbot.py
Enhanced KisanCredit AI chatbot engine.
- Multilingual support: English, Hindi, Marathi
- Groq API integration (FREE)
- Rule-based fallback when API unavailable
- Language auto-detection
"""

import re
import os
import json
import logging
from groq import Groq
from chatbot.knowledge_base import INTENTS, RESPONSES, SUGGESTIONS, MULTILINGUAL_SUGGESTIONS

logger = logging.getLogger(__name__)


# ── LANGUAGE DETECTION ─────────────────────────────────────────────────────────

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

MARATHI_MARKERS = [
    "आहे", "आहेत", "नाही", "केले", "करा", "मला", "तुम्ही", "सांगा",
    "कसा", "काय", "कुठे", "शेती", "पीक", "कर्ज", "मंजुरी", "विमा",
]
HINDI_MARKERS = [
    "है", "हैं", "नहीं", "करें", "मुझे", "आप", "बताएं", "क्या",
    "कैसे", "कहां", "खेती", "फसल", "लोन", "मंजूरी", "बीमा",
]


def detect_language(text: str) -> str:
    if not DEVANAGARI_RE.search(text):
        return "en"

    marathi_score = sum(1 for m in MARATHI_MARKERS if m in text)
    hindi_score   = sum(1 for m in HINDI_MARKERS if m in text)

    return "mr" if marathi_score > hindi_score else "hi"


# ── GROQ API INTEGRATION ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are KisanCredit AI, a friendly agricultural loan assistant for Indian farmers.

Help with:
- Loan eligibility (credit history, income, loan amount)
- Government schemes (PM Kisan, KCC, PMFBY, NABARD)
- Crop guidance & risk
- Loan documents & approval tips

Important facts:
- Credit history = most important factor
- KCC: up to ₹3 lakh at ~4%
- PM Kisan: ₹6000/year
- PMFBY: 2% (kharif), 1.5% (rabi)

Respond in SAME language (English/Hindi/Marathi).
Keep answers simple, short, and farmer-friendly.
Use bullet points and ₹ symbol.
"""


def call_groq_api(user_message: str, history: list, language: str) -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        client = Groq(api_key=api_key)

        sys_prompt = SYSTEM_PROMPT + f"\nUSER REQUESTED LANGUAGE: {language}. You MUST respond entirely in {language}."
        messages = [{"role": "system", "content": sys_prompt}]

        # Add last 6 conversation turns
        for turn in history[-6:]:
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })

        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=512
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning("Groq API failed: %s", e)
        return None


# ── CHATBOT ENGINE ─────────────────────────────────────────────────────────────

class KisanChatbot:

    PRIORITY = [
        "kcc", "pmfby", "pm_kisan", "nabard", "soil",
        "credit_history", "income", "loan_amount", "interest_rate",
        "documents", "approval", "rejection", "schemes",
        "crop", "weather", "dataset",
        "language", "thanks", "bye", "greeting",
    ]

    def __init__(self, use_api: bool = True):
        self.history = []
        self.use_api = use_api and bool(os.getenv("GROQ_API_KEY"))
        self.detected_language = "en"

    def respond(self, user_message: str, force_language: str = None) -> dict:
        lang = force_language if force_language else detect_language(user_message)
        self.detected_language = lang

        clean = self._normalise(user_message)
        intent, confidence = self._detect_intent(clean)

        api_reply = None
        if self.use_api and (intent == "fallback" or confidence < 0.35):
            api_reply = call_groq_api(user_message, self.history, lang)

        if api_reply:
            reply = api_reply
            source = "api"
        else:
            reply = RESPONSES.get(intent, RESPONSES["fallback"])
            
            # Split language base on current selection
            if lang == "hi" and "**हिंदी:**" in reply:
                reply = reply.split("**हिंदी:**")[1].split("**मराठी:**")[0].strip()
            elif lang == "mr" and "**मराठी:**" in reply:
                reply = reply.split("**मराठी:**")[1].strip()
            elif lang == "en":
                reply = reply.split("**हिंदी:**")[0].split("**मराठी:**")[0].strip()

            source = "rule"

        # Save history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": reply})

        if len(self.history) > 40:
            self.history = self.history[-40:]

        return {
            "reply": reply,
            "intent": intent,
            "confidence": round(confidence, 2),
            "language": lang,
            "source": source,
        }

    def get_suggestions(self, language=None):
        lang = language or self.detected_language
        return MULTILINGUAL_SUGGESTIONS.get(lang, SUGGESTIONS)

    def reset(self):
        self.history.clear()
        self.detected_language = "en"

    def get_history(self):
        return self.history.copy()

    @staticmethod
    def _normalise(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s₹\u0900-\u097F]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _detect_intent(self, text: str):
        scores = {}

        for intent, keywords in INTENTS.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches:
                scores[intent] = min(1.0, matches / max(1, len(keywords) * 0.25))

        if not scores:
            return "fallback", 0.0

        best_intent = max(
            scores,
            key=lambda k: (
                scores[k],
                -self.PRIORITY.index(k) if k in self.PRIORITY else -99,
            ),
        )
        return best_intent, scores[best_intent]