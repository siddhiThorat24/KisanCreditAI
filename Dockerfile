# ─────────────────────────────────────────────────────────────
#  KisanCredit AI  —  Dockerfile
# ─────────────────────────────────────────────────────────────

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (including standard build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and upgrade pip beforehand to avoid sha256 mismatch bugs
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend/   ./backend/
COPY chatbot/   ./chatbot/
COPY frontend/  ./frontend/
COPY data/      ./data/
COPY models/    ./models/
COPY outputs/   ./outputs/

# ── Runtime settings ──────────────────────────────────────────
# Ensure Python can find the local packages
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV FLASK_ENV=production
ENV SECRET_KEY=change-me-in-production

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

# Start the Flask app
CMD ["python", "backend/app.py"]