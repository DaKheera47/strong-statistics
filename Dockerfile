FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# System deps (add build tools only if needed for future libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY dashboard_layout.yaml ./
COPY icon.svg ./
COPY readme.md ./

# Create data & logs volume mount points
RUN mkdir -p data/uploads logs
VOLUME ["/app/data", "/app/logs"]

# Expose default port
EXPOSE 8069

# Provide a non-root user for better security
RUN useradd -u 10001 -m appuser
USER appuser

# Default envs (can be overridden at runtime)
ENV HOST=0.0.0.0 \
    PORT=8069 \
    LOG_LEVEL=INFO

# Healthcheck: hit /health
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8069/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8069"]
