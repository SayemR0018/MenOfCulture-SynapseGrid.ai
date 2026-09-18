FROM python:3.12-slim AS base

# Prevent .pyc files and force unbuffered stdout/stderr (so logs show up
# immediately in `docker logs`).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source.
COPY app/ app/
COPY ml/ ml/
COPY optimizer/ optimizer/
COPY scripts/ scripts/
COPY docs/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json docs/

# Run as a non-root user.
RUN useradd --create-home --uid 1000 synapsegrid \
    && chown -R synapsegrid:synapsegrid /app
USER synapsegrid

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:" + os.environ.get("PORT", "8000") + "/health", timeout=2)' || exit 1

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
