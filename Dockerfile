# Single-container build: React UI is compiled once, then served by FastAPI on the same origin.
#   docker build -t sdp .
#   docker run --rm -p 8000:8000 sdp          -> http://localhost:8000  (API docs at /docs)

# ---- 1. build the frontend --------------------------------------------------
FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 2. runtime -------------------------------------------------------------
FROM python:3.13-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SDP_STATIC_DIR=/app/static
WORKDIR /app

# Fonts for native-script PDFs: Noto Naskh/Sans Arabic, Sans Devanagari, Noto Sans (fonts-noto-core) and a TrueType CJK
# face (WenQuanYi Micro Hei). reportlab cannot embed the CFF-based Noto Sans CJK, hence the WenQuanYi choice for Chinese.
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-noto-core fonts-wqy-microhei \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY --from=ui /ui/dist ./static

# run as an unprivileged user; /data holds job manifests and outputs (mount a volume to keep them)
ENV SDP_DATA_DIR=/data
RUN useradd --create-home --uid 10001 sdp && mkdir -p /data && chown -R sdp:sdp /app /data
VOLUME ["/data"]
USER sdp

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "sdp.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
