# ============================================================
# SPIRO Backend — production image
# Serves FastAPI + in-process ONNX ML inference.
# Build from the repo root:  docker build -f backend/Dockerfile backend
# Or via the root docker-compose.yml.
# ============================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/app

# System deps (psycopg2-binary, pillow and opencv-headless ship wheels,
# so no compiler is needed).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code + configs + exported ONNX model(s).
COPY app ./app
COPY configs ./configs
COPY models ./models

# Writable directory for uploaded report images.
# In production this should be backed by a persistent volume
# (see docker-compose.yml) — container filesystems are ephemeral.
RUN mkdir -p uploads/reports

EXPOSE 8000

# NOTE: no --reload here. Uvicorn reload is for local development only.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
