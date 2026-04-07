# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY env/          ./env/
COPY server/       ./server/
COPY server_api.py .
COPY app.py        .
COPY inference.py  .
COPY openenv.yaml  .

# ── Environment variables (overridable at run-time) ───────────────────────────
# These are the three variables required by the OpenEnv hackathon spec.
# Pass them via:  docker run -e HF_TOKEN=hf_... -e API_BASE_URL=... -e MODEL_NAME=...
ENV API_BASE_URL="https://api-inference.huggingface.co/v1/"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""

# Hugging Face Spaces default port is 7860
EXPOSE 7860

# Health check — used by both Docker and the automated ping gate
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -f http://localhost:7860/health || exit 1

# Entry point — `app:app` works because app.py re-exports the FastAPI app object
# --proxy-headers  : trust X-Forwarded-* headers set by the HF Spaces reverse proxy
# --forwarded-allow-ips='*': allow any upstream IP (HF proxy IPs are not fixed)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips=*", "--log-level", "info"]